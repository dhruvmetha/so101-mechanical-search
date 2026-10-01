"""Claude as the teacher, one fresh call per move (2026-10-01, dm1487: collect demos through Claude with as few
tokens as possible).

The program keeps the episode alive in memory, looks, and does the routine moves itself: it picks the red when the
front camera sees it well and act allows the pick, and sets a held gray down on the best checked spot. Every other
move is Claude's. Each call gets only what this move needs: the teacher rules and the strategy card (the same text in
every call, so the prompt cache holds them), the picture from the last look, a short summary of what the cameras
see, Claude's own belief note from the previous call, what changed, and checked candidate moves that act allows.
Claude answers with an updated belief and one move. The call has no tools, so it sees only what the cameras showed.

How the five teacher rules (docs/knowledge/mechanical-search-learnings.md, IMPORTANT) are kept:
  1. briefs come only from observe on recorded looks; Claude's belief note is built from earlier briefs only;
  2. every picture follows a recorded park, or is taken from the holding pose, which is in the demo too;
  3. one frozen prompt per batch (rules and card versions stored in the record), routine moves done by the program,
     Claude choosing among checked candidates;
  4. the belief note goes to demonstrator/beliefs/EP.md, which export never reads;
  5. the miss draw and spill undos are the program's, never Claude's choice; three moves in a row without progress
     stop the episode for review.

    python demonstrator/driver.py EP [--seed S --kind buried|partial] [--effort xhigh]
"""
import argparse
import base64
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "experiments"))

import tools  # noqa: E402  (imports so101sim before mujoco)
from live import Live  # noqa: E402
from so101sim.mechsearch import TARGET  # noqa: E402

import push_goal  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LEARNINGS = ROOT / "docs/knowledge/mechanical-search-learnings.md"
CARD = HERE / "strategy.md"
BELIEFS = HERE / "beliefs"
MODEL = "claude-opus-5-5"
STALL = 3  # Claude moves in a row without progress before the episode stops for review
SHORT = tools.short


def teacher_rules():
    text = LEARNINGS.read_text()
    start = text.index("## IMPORTANT: how the teacher plays")
    end = text.index("\n## ", start + 5)
    return text[start:end].strip()


def system_prompt():
    return f"""You are the teacher in a robot task in simulation, deciding one move per call.

The task: an SO-101 arm beside a sunken bin, 22 cm square, centre (0.200, 0.000), floor z -0.096, rim z -0.046,
holding 29 gray 30 mm cubes and one red one, up to three layers (layer 1 centres z -0.081, layer 2 -0.051, layer 3
-0.021). Lift the red clear of the bin without any cube leaving it, in at most 15 moves. x grows away from the arm's
base toward the far wall; y grows to the left seen from the base. Cube names: g0 to g28, and red.

{teacher_rules()}

{CARD.read_text().strip()}

Each call gives you the picture from the last look (left panel: the front camera, top of the panel is the base side,
left is -y; right panel: the wrist camera, left is the base side, top is +y; labels are yellow for a well-seen cube,
cyan for a rough one), a summary of what the cameras see, your own belief note from your last call, what changed
since, and numbered candidate moves that the program has checked and act allows. The program makes routine moves
itself: it picks a well-seen red when act allows it, and sets a held gray down on the best checked spot.

Reply with one JSON object and nothing else:
{{"belief": "<at most 80 words: where the red is and how sure, what blocks it, your plan>",
 "choice": <a candidate number, or null>,
 "custom": null or {{"kind": "pick|place|push|sweep", "xyz": [x, y, z], "deg": d, "stroke": s_or_null}},
 "why": "<one sentence>",
 "expect": "<one sentence: what the cameras should show after the move>"}}
Use custom only when no candidate fits; the program checks it and tells you if act refuses it."""


# ---- what the cameras see, in short --------------------------------------------------------------

def access(v):
    if TARGET not in v["cubes"]:
        return None
    a = tools.red_access(v["cubes"])
    return {"covers": len(a["covers"]), "blocked": min(n for _, n in a["axes"]),
            "trusted": v["cubes"][TARGET]["cov"] >= tools.TRUST}


def progress(before, after):
    """Did the red become better seen or freer? Judged from the two views only."""
    if after is None:
        return False
    if before is None:
        return True
    return (after["trusted"] and not before["trusted"]) or after["covers"] < before["covers"] \
        or after["blocked"] < before["blocked"]


def summary(ep, v):
    cubes = v["cubes"]
    grip = float(ep.env.joint_state[5])
    lines = [f"Move {ep.moves() + 1} of {tools.MAX_MOVES}. Gripper {grip:.1f} ({tools.holding(grip)})."]
    px = v["red_px"]
    if len(px):
        lines.append(f"Red pixels: {len(px)}, around {tools.fmt(np.median(px, 0))}.")
    else:
        lines.append("Red pixels: none.")
    red = cubes.get(TARGET)
    if red is None:
        lines.append("Red: not listed.")
        centre = np.median(px[:, :2], 0) if len(px) else np.asarray(ep.bin["center"])
    else:
        how = "well seen (top {:.0%})".format(red["cov"]) if red["cov"] >= tools.TRUST else tools.rough_note(red)
        lines.append(f"Red: {how}, at {tools.fmt(red['pos'])}, layer {red['layer']}, yaw {red['yaw']:.0f}.")
        acc = tools.red_access(cubes)
        lines.append("Covers on the red: " + (", ".join(SHORT(n) for n in acc["covers"]) or "none") + ".")
        for yaw, ends in tools.axis_ends(v, TARGET):
            parts = []
            for sign in (1, -1):
                at = ", ".join(f"{SHORT(n)} (layer {layer}, gap {max(gap, 0) * 1000:.0f} mm)"
                               for gap, n, layer in ends[sign]) or "clear"
                parts.append(f"{'+' if sign > 0 else '-'} end: {at}")
            lines.append(f"Jaw axis {yaw:.0f} deg: " + "; ".join(parts) + ".")
        centre = np.asarray(red["pos"][:2])
    near = sorted((n for n in cubes if n != TARGET),
                  key=lambda n: np.linalg.norm(np.asarray(cubes[n]["pos"][:2]) - centre))[:10]
    lines.append("Nearest cubes: " + "; ".join(
        f"{SHORT(n)} {tools.fmt(cubes[n]['pos'])} L{cubes[n]['layer']} yaw {cubes[n]['yaw']:.0f}"
        f"{'' if cubes[n]['cov'] >= tools.TRUST else ' rough'}" for n in near) + ".")
    lines.append(f"Unseen: {v['unseen'].mean():.0%} of the bin floor.")
    return "\n".join(lines)


def changes(v0, v1):
    if v0 is None:
        return "First look."
    moved = [n for n in v1["cubes"] if n in v0["cubes"]
             and np.linalg.norm(v1["cubes"][n]["pos"] - v0["cubes"][n]["pos"]) > 0.01]
    out = [f"Listed cubes that moved over 1 cm: {len(moved)}"
           + (f" ({', '.join(SHORT(n) for n in moved[:8])})" if moved else "") + "."]
    r0, r1 = v0["cubes"].get(TARGET), v1["cubes"].get(TARGET)
    if r0 is not None and r1 is not None:
        out.append(f"Red moved {np.linalg.norm(r1['pos'] - r0['pos']) * 1000:.0f} mm; top seen {r0['cov']:.0%} -> "
                   f"{r1['cov']:.0%}.")
    elif r1 is not None:
        out.append("Red now listed.")
    elif r0 is not None:
        out.append("Red no longer listed.")
    return " ".join(out)


# ---- candidates -----------------------------------------------------------------------------------

def checked(ep, v, kind, xyz, deg, stroke=None):
    """(flip, check result) if act would allow the move, else None."""
    ep.view = v
    flip, results = ep.choose(kind, [float(c) for c in xyz], float(deg), stroke)
    if flip is None or tools.refusal(kind, results[flip]):
        return None
    return flip, results[flip]


def pick_of(ep, v, name):
    cb = v["cubes"][name]
    for yaw in (cb["yaw"] % 180.0, (cb["yaw"] + 90.0) % 180.0):
        ok = checked(ep, v, "pick", cb["pos"], yaw)
        if ok and ok[1].get("target") == name:
            return {"kind": "pick", "xyz": [float(c) for c in cb["pos"]], "deg": float(yaw), "flip": ok[0],
                    "r": ok[1]}
    return None


def stroke_note(r):
    p = r["push"]
    movers = sorted(p["moves"])
    note = f"moves {len(movers)} ({', '.join(SHORT(n) for n in movers[:6])})"
    note += ", red moves too" if p["moves_red"] else ", red stays (as predicted; predictions often miss chains)"
    ra = p.get("red_access")
    if ra:
        note += (f"; predicted covers {len(ra['before']['covers'])}->{len(ra['after']['covers'])}, blocked ends "
                 f"{[n for _, n in ra['before']['axes']]}->{[n for _, n in ra['after']['axes']]}")
    if r["unseen_mm"] >= tools.PUSH_UNSEEN_MM:
        note += f"; passes {r['unseen_mm']:.0f} mm of unseen space"
    return note


def candidates(ep, v):
    cubes = v["cubes"]
    out = []
    red = cubes.get(TARGET)
    if red is not None:
        acc = tools.red_access(cubes)
        ends = tools.axis_ends(v, TARGET)
        far_clear = {}
        for yaw, e in ends:
            for sign, found in e.items():
                for _, n, _ in found[:1]:
                    far_clear[n] = far_clear.get(n, False) or not e[-sign]
        rp = pick_of(ep, v, TARGET)
        if rp:
            rp["text"] = (f"pick the red at yaw {rp['deg']:.0f}"
                          + ("" if red["cov"] >= tools.TRUST else " (rough pose: such picks lifted it 1 in 11)"))
            out.append(rp)
        names = list(acc["covers"]) + [n for _, e in ends for f in e.values() for _, n, _ in f[:1]]
        for n in dict.fromkeys(names):
            pk = pick_of(ep, v, n)
            if pk:
                role = "cover" if n in acc["covers"] else "axis blocker"
                pk["text"] = (f"pick {SHORT(n)} ({role}, layer {cubes[n]['layer']}) at yaw {pk['deg']:.0f}; the program "
                              f"then sets it down far from the red")
                out.append(pk)
        strokes = []
        for s in push_goal.strokes_for(v):
            stroke = s["stroke"] if s["kind"] == "sweep" else None
            ok = checked(ep, v, s["kind"], s["xyz"], s["deg"], stroke)
            if not ok:
                continue
            what = {"sideways": f"{SHORT(s['about'])} sideways off the red's axis"
                                + (" (far end of that axis clear)" if far_clear.get(s["about"]) else ""),
                    "cover": f"sweep cover {SHORT(s['about'])} off the red",
                    "red": "push the red itself"}[s["what"]]
            strokes.append(((s["what"] == "sideways" and far_clear.get(s["about"], False), s["what"] == "cover",
                             -s["stroke"]),
                            {"kind": s["kind"], "xyz": s["xyz"], "deg": s["deg"], "stroke": s["stroke"],
                             "flip": ok[0], "r": ok[1],
                             "text": f"{s['kind']} {what}, heading {s['deg']:.0f} deg, {s['stroke'] * 100:.0f} cm: "
                                     + stroke_note(ok[1])}))
        strokes.sort(key=lambda t: t[0], reverse=True)
        out += [c for _, c in strokes[:10]]
    else:
        px = v["red_px"]
        if len(px):
            centre = np.median(px[:, :2], 0)
            near = sorted((n for n in cubes if np.linalg.norm(np.asarray(cubes[n]["pos"][:2]) - centre) < 0.03),
                          key=lambda n: -cubes[n]["pos"][2])
            for n in near[:4]:
                pk = pick_of(ep, v, n)
                if pk:
                    pk["text"] = (f"pick {SHORT(n)} (at the red pixels, layer {cubes[n]['layer']}) at yaw "
                                  f"{pk['deg']:.0f}; the program then sets it down far from there")
                    out.append(pk)
                cp = np.asarray(cubes[n]["pos"])
                for deg in (0.0, 90.0, 180.0, 270.0):
                    d = np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg))])
                    start = cp[:2] - d * (tools.HALF + 0.015)
                    xyz = [float(start[0]), float(start[1]), float(cp[2])]
                    ok = checked(ep, v, "sweep", xyz, deg, 0.08)
                    if ok:
                        out.append({"kind": "sweep", "xyz": xyz, "deg": deg, "stroke": 0.08, "flip": ok[0],
                                    "r": ok[1], "text": f"sweep {SHORT(n)} off the red pixels, heading {deg:.0f} deg, "
                                                        f"8 cm: " + stroke_note(ok[1])})
    for k, c in enumerate(out, 1):
        c["n"] = k
    return out


# ---- Claude ---------------------------------------------------------------------------------------

def ask(system, image_png, text, effort, workdir):
    content = [{"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                            "data": base64.b64encode(image_png).decode()}},
               {"type": "text", "text": text}]
    msg = json.dumps({"type": "user", "message": {"role": "user", "content": content}})
    cmd = ["claude", "-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
           "--model", MODEL, "--effort", effort, "--system-prompt", system, "--tools", "",
           "--no-session-persistence", "--setting-sources", "", "--strict-mcp-config", "--mcp-config",
           '{"mcpServers":{}}', "--disable-slash-commands", "--exclude-dynamic-system-prompt-sections"]
    t0 = time.time()
    out = subprocess.run(cmd, input=msg, capture_output=True, text=True, cwd=workdir, timeout=900).stdout
    result = None
    for line in out.splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("type") == "result":
            result = d
    if result is None:
        raise RuntimeError(f"no result from claude: {out[-500:]}")
    u = result.get("usage", {})
    usage = {"model": list(result.get("modelUsage", {}) or [MODEL]), "effort": effort,
             "input": u.get("input_tokens", 0), "cache_write": u.get("cache_creation_input_tokens", 0),
             "cache_read": u.get("cache_read_input_tokens", 0), "output": u.get("output_tokens", 0),
             "thinking": (u.get("output_tokens_details") or {}).get("thinking_tokens", 0),
             "usd": result.get("total_cost_usd"), "seconds": round(time.time() - t0, 1)}
    text = result.get("result") or ""
    m = re.search(r"\{.*\}", text, re.S)
    reply = json.loads(m.group(0)) if m else None
    return reply, text, usage


# ---- playing --------------------------------------------------------------------------------------

def look(ep):
    """Park (a recorded move of the arm) unless holding, then see. Returns the view and the picture as PNG bytes;
    the picture is also kept in demonstrator/beliefs/ for audits."""
    held = 14 < float(ep.env.joint_state[5]) < 30
    if not held and not ep.parked():
        ep.run([{"op": "park", "keep_grip": True}])
    v = ep.see()
    path = BELIEFS / f"{ep.name}_m{ep.moves() + 1:02d}.png"
    ep.picture(path)
    return v, path.read_bytes()


def act(ep, c, why):
    op = {"op": "act", "kind": c["kind"], "xyz": [float(x) for x in c["xyz"]], "deg": float(c["deg"]),
          "flip": c["flip"], "unpark": True, "why": why}
    if c["kind"] == "sweep":
        op["stroke"] = float(c["stroke"])
    return ep.run([op])


def take_back(ep, snap, why, granted):
    """Restore the scene from before the move and file the dropped ops, as tools.take_back does for replayed
    episodes."""
    ops = ep.rec["ops"]
    undone = ep.rec.setdefault("undone", [])
    undone.append({"at_move": sum(1 for op in ops[:snap["n_ops"]] if op["op"] == "act") + 1, "why": why,
                   "granted": granted, "ops": ops[snap["n_ops"]:]})
    ep.restore(snap)


def play(name, effort, workdir):
    ep = Live(name, size=64)
    BELIEFS.mkdir(exist_ok=True)
    notes = BELIEFS / f"{name}.md"
    system = system_prompt()
    ep.rec.setdefault("teacher", {"model": MODEL, "effort": effort, "card": CARD.read_text().splitlines()[0],
                                  "prompt_sha": hashlib.sha256(system.encode()).hexdigest()[:12], "calls": []})
    belief, last_note, prev_view, stall = "none yet", "", None, 0
    judged = None  # the red's access when Claude chose the last move; the next look judges progress against it
    auto_missed = False
    with notes.open("a") as f:
        f.write(f"# {name}, seed {ep.rec['seed']}, {'buried' if ep.rec['buried'] else 'partial'}\n\n")
    while ep.outcome() is None:
        v, png = look(ep)
        acc = access(v)
        if judged is not None:
            stall = 0 if progress(judged[0], acc) else stall + 1
            judged = None
            if stall >= STALL:
                ep.rec["teacher"]["stopped"] = f"no progress in {STALL} moves"
                break
        grip = float(ep.env.joint_state[5])
        snap = ep.snapshot()
        held = ep.held_cube() if 14 < grip < 30 else None
        auto, why = None, None
        if held and held != TARGET:
            _, good = tools.place_options(ep, v)
            if good:
                _, xy, z, deg, what, r = good[0]
                flip = ep.choose("place", [xy[0], xy[1], z], deg)[0]
                auto = {"kind": "place", "xyz": [xy[0], xy[1], z], "deg": deg, "flip": flip}
                why = f"program: set {SHORT(held)} down {what}, the best checked spot away from the red"
        elif acc and acc["trusted"] and not auto_missed:
            rp = pick_of(ep, v, TARGET)
            if rp:
                auto, why = rp, "program: the red is well seen and act allows the pick"
        if auto is not None:
            c, chosen_note = auto, why
        else:
            cands = candidates(ep, v)
            text = (f"{summary(ep, v)}\nSince your last move: {changes(prev_view, v)}{last_note}\n"
                    f"Your belief note from last call: {belief}\nCandidates (checked, act allows each):\n"
                    + ("\n".join(f"{c['n']}. {c['text']}" for c in cands) or "none; propose a custom move"))
            for attempt in range(3):
                reply, raw, usage = ask(system, png, text, effort, workdir)
                ep.rec["teacher"]["calls"].append({"move": ep.moves() + 1, **usage})
                if reply is None:
                    text += "\n\nYour last reply was not one JSON object. Reply with the JSON object only."
                    continue
                c = None
                if reply.get("choice"):
                    c = next((x for x in cands if x["n"] == int(reply["choice"])), None)
                elif reply.get("custom"):
                    cu = reply["custom"]
                    stroke = cu.get("stroke") if cu.get("kind") == "sweep" else None
                    ep.view = v
                    flip, results = ep.choose(cu["kind"], cu["xyz"], float(cu["deg"]), stroke)
                    refused = "unreachable" if flip is None else tools.refusal(cu["kind"], results[flip])
                    if not refused:
                        c = {**cu, "flip": flip}
                    else:
                        text += f"\n\nact refuses your custom move: {refused} Choose again."
                        continue
                if c is not None:
                    break
                text += "\n\nYour reply named no valid move. Choose a candidate number or a custom move."
            else:
                ep.rec["teacher"]["stopped"] = "no valid move after 3 replies"
                break
            belief = reply.get("belief", belief)
            chosen_note = reply.get("why", "")
            with notes.open("a") as f:
                f.write(f"## move {ep.moves() + 1}\nbelief: {belief}\nmove: {c.get('text', c)}\nwhy: {chosen_note}\n"
                        f"expect: {reply.get('expect', '')}\ntokens: {usage}\n")
        res = act(ep, c, chosen_note)
        last_note = ""
        if res is not None and "left the bin" in str(res.get("failure") or ""):
            take_back(ep, snap, f"spill: {res['failure']}", "spill")
            last_note = f" Your last move spilled ({res['failure']}) and was taken back; the scene is as before it."
        elif res is not None and c["kind"] == "pick" and not res["terminated"] and res["grip"] <= 14:
            n_acts = ep.moves()
            u = tools.draw(ep.rec["seed"], n_acts, len(ep.rec.get("undone", [])))
            if u < ep.rec.get("miss_reset_eps", tools.MISS_RESET_EPS) and len(ep.rec.get("undone", [])) < tools.MAX_UNDOS:
                take_back(ep, snap, f"missed pick, reset by the draw ({u:.3f})", "miss draw")
                last_note = " Your last pick missed and the draw reset it; the scene is as before it."
            else:
                last_note = " Your last pick missed (gripper closed on nothing); the miss stays."
        with notes.open("a") as f:
            f.write(f"result: {res}{last_note}\n\n" if auto is None else
                    f"## move {ep.moves()} (program)\n{chosen_note}\nresult: {res}{last_note}\n\n")
        ep.save()
        # A missed program pick of the red goes to Claude next, so the program never repeats a move that just failed.
        auto_missed = auto is not None and c["kind"] == "pick" and "missed" in last_note
        if auto is None:
            judged = (acc,)
        prev_view = v
    ep.save()
    calls = ep.rec["teacher"]["calls"]
    total = {k: sum(c[k] for c in calls) for k in ("input", "cache_write", "cache_read", "output")}
    usd = sum(c["usd"] or 0 for c in calls)
    print(f"{name}: {ep.outcome() or ep.rec['teacher'].get('stopped')}, {ep.moves()} moves, robot {ep.robot_s:.0f} s, "
          f"{len(calls)} Claude calls, tokens {total}, ${usd:.2f}")
    ep.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--kind", choices=["buried", "partial"])
    ap.add_argument("--effort", default="xhigh")
    a = ap.parse_args()
    if a.seed is not None:
        tools.cmd_new(a.name, a.seed, a.kind or "buried")
    # Outside the repo, so the call doesn't pick up the project's CLAUDE.md.
    play(a.name, a.effort, tempfile.mkdtemp(prefix="teacher_"))


if __name__ == "__main__":
    main()
