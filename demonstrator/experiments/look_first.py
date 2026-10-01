"""Look first, then push, with every pick tried for real (2026-09-30, after dm1487 asked "But do you know where the
red is?").

push_choice.py found the common-sense stroke opened the red in 8 of 29 piles: 3 of 4 when the red's pose was trusted,
5 of 25 when rough. "Opened" meant act would allow a pick on the perceived pose; no pick was tried. Here each pile
plays from its start, deciding from the cameras only:
  observe; if act would allow a pick of the red, pick it for real (a miss is kept, as when the draw keeps it);
  otherwise make a stroke that act allows, park, and look again.
Policies differ in the stroke and in how many moves they get:
  once  the push_choice stroke (far end clear, then open floor at the landing, then finger room), then one pick if
        allowed: the old test, now with a real pick. Afterwards the red is picked even when act would refuse, to see
        what the verdict misses.
  loop  up to MOVES moves; the freeing stroke ranked by the cue that measured (far end clear), the 5 cm push first
  look  as loop, but while the red's pose is rough, first a stroke meant to show it to the front camera: sweep a
        cover off its top, else push the red itself toward a clear end of its jaw axis
Outcomes come from the simulator: solved means the env saw the red lifted clear with nothing spilled.

    python demonstrator/experiments/look_first.py --procs 10
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tools  # noqa: E402  (imports so101sim before mujoco)
from live import Live  # noqa: E402
from so101sim.mechsearch import TARGET  # noqa: E402

import push_choice  # noqa: E402
import push_goal  # noqa: E402

MOVES = 5
SEEDS = [5103, 5105, 5107, 5109, 5110, 5111, 5112, 5113, 5114, 5115, 5117, 5119, 5121, 5123, 5125, 5127, 5129, 5131,
         5133, 5135, 5137, 5139, 5140, 5141, 5142, 5143, 5145, 5147, 5148]  # push_choice's 29 piles


def pick_plan(ep, v, allowed_only=True):
    """(yaw, flip, allowed) for a pick of the red from view v: the first jaw axis act would allow, or with
    allowed_only False, the axis whose worst neighbour hit is smallest."""
    if TARGET not in v["cubes"]:
        return None
    red = v["cubes"][TARGET]
    ep.view = v
    best = None
    for yaw in (red["yaw"] % 180.0, (red["yaw"] + 90.0) % 180.0):
        flip, results = ep.choose("pick", list(map(float, red["pos"])), yaw)
        if flip is None:
            continue
        r = results[flip]
        worst = max((x for k, x in r["hits"].items() if k != TARGET), default=0.0) * 1000
        ok = (r.get("target") == TARGET and not (r["trusted"] and r["shift_mm"] > tools.SHIFT_MAX_MM)
              and r["on_top_mm"] <= tools.MARGIN * 1000 and worst < tools.HIT_REFUSE_MM
              and r["self_mm"] >= tools.SELF_JAM_MM)
        if ok:
            return yaw, flip, True
        if best is None or worst < best[0]:
            best = (worst, yaw, flip)
    if allowed_only or best is None:
        return None
    return best[1], best[2], False


def stroke_ok(ep, v, s):
    """The flip act would use for stroke s, or None if act refuses it or it can't run."""
    ep.view = v
    flip, results = ep.choose(s["kind"], s["xyz"], s["deg"], s["stroke"] if s["kind"] == "sweep" else None)
    if flip is None:
        return None
    r = results[flip]
    if r["push"]["spill_risk"] or r["self_mm"] < tools.SELF_JAM_MM:
        return None
    return flip


def wall_room(ep, xy, deg):
    d = np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg))])
    c, h = ep.bin["center"], ep.bin["half"]
    return min(((h - np.sign(d[k]) * (xy[k] - c[k])) / abs(d[k])) if abs(d[k]) > 1e-6 else np.inf for k in (0, 1))


def sideways(ep, v, key):
    """Allowed sideways strokes at the red's blockers, best first by key."""
    ends = {round(yaw): e for yaw, e in tools.axis_ends(v, TARGET)}
    floor = ep.bin["floor_z"]
    out = []
    for s in push_goal.strokes_for(v):
        if s["what"] != "sideways":
            continue
        bp = np.asarray(v["cubes"][s["about"]]["pos"])
        d = np.array([np.cos(np.radians(s["deg"])), np.sin(np.radians(s["deg"]))])
        f = {"room": push_choice.share(ep, v, bp[:2] + d * s["stroke"], tools.HALF,
                                       lambda h, u: (not u) and h < floor + 0.008),
             "finger": push_choice.share(ep, v, np.asarray(s["xyz"][:2]), 0.006, lambda h, u: (not u) and h < bp[2]),
             "other": any(not e[-sign] for e in ends.values() for sign, found in e.items()
                          if any(n == s["about"] for _, n, _ in found))}
        out.append((key(f, s), f, s))
    out.sort(key=lambda t: t[0], reverse=True)
    return [(f, s) for _, f, s in out]


def reveals(ep, v):
    """Strokes meant to show the red to the front camera: covers swept off toward the most room, then the red pushed
    toward a clear end of a jaw axis, toward the most room."""
    red = v["cubes"][TARGET]
    clear = set()
    for yaw, e in tools.axis_ends(v, TARGET):
        for sign, found in e.items():
            if not found:
                clear.add(round((yaw + (0 if sign > 0 else 180)) % 360))
    out = []
    for s in push_goal.strokes_for(v):
        if s["what"] == "cover":
            cp = v["cubes"][s["about"]]["pos"]
            out.append(((2, wall_room(ep, cp[:2], s["deg"])), s))
        elif s["what"] == "red":
            near = min(clear, key=lambda c: abs((c - s["deg"] + 180) % 360 - 180), default=None)
            is_clear = near is not None and abs((near - s["deg"] + 180) % 360 - 180) < 20
            out.append(((1 if is_clear else 0, wall_room(ep, red["pos"][:2], s["deg"])), s))
    out.sort(key=lambda t: t[0], reverse=True)
    return [s for _, s in out]


OLD_KEY = lambda f, s: (f["other"], round(f["room"], 1), f["finger"], -s["stroke"])  # noqa: E731  push_choice's
NEW_KEY = lambda f, s: (f["other"], -s["stroke"])  # noqa: E731


def act(ep, kind, xyz, deg, flip, stroke=None):
    op = {"op": "act", "kind": kind, "xyz": [float(x) for x in xyz], "deg": float(deg), "flip": flip, "unpark": True}
    if stroke is not None and kind == "sweep":
        op["stroke"] = stroke
    return ep.run([op])


def play(seed, policy, base_rec=None):
    ep = Live(f"look{seed}{policy}", size=64, rec={"seed": seed, "buried": seed % 2 == 0, "ops": []})
    ep.run([{"op": "park", "keep_grip": True}])
    steps, outcome = [], None
    budget = 2 if policy == "once" else MOVES
    while outcome is None and len(steps) < budget:
        v = ep.see()
        if TARGET not in v["cubes"]:
            outcome = "red not listed"
            break
        red = v["cubes"][TARGET]
        rough = red["cov"] < tools.TRUST
        plan = pick_plan(ep, v)
        if plan is not None:
            yaw, flip, _ = plan
            res = act(ep, "pick", red["pos"], yaw, flip)
            steps.append({"kind": "pick", "rough": rough, "how": red["how"], "success": res["success"],
                          "grip": round(res["grip"], 1), "failure": res["failure"]})
            if res["success"]:
                outcome = "solved"
            elif res["terminated"]:
                outcome = f"failed: {res['failure']}"
            elif res["grip"] > 14:
                outcome = "holding a gray"
            elif policy == "once":
                outcome = "missed"
            ep.run([{"op": "park", "keep_grip": True}])
            continue
        if policy == "once" and steps:
            outcome = "not pickable after one stroke"
            break
        chosen = None
        if policy == "look" and rough:
            for s in reveals(ep, v):
                flip = stroke_ok(ep, v, s)
                if flip is not None:
                    chosen, why = (s, flip), "reveal"
                    break
        if chosen is None:
            for f, s in sideways(ep, v, OLD_KEY if policy == "once" else NEW_KEY):
                flip = stroke_ok(ep, v, s)
                if flip is not None:
                    chosen, why = (s, flip), "free"
                    break
        if chosen is None:
            outcome = "no allowed stroke"
            break
        s, flip = chosen
        res = act(ep, s["kind"], s["xyz"], s["deg"], flip, s["stroke"])
        steps.append({"kind": s["kind"], "why": why, "what": s["what"], "about": s["about"], "deg": round(s["deg"]),
                      "stroke": s["stroke"], "rough": rough, "failure": res["failure"]})
        if res["terminated"]:
            outcome = f"failed: {res['failure']}"
            break
        ep.run([{"op": "park", "keep_grip": True}])
    if outcome is None:
        outcome = "out of moves" if policy != "once" else "not pickable after one stroke"
    row = {"seed": seed, "policy": policy, "outcome": outcome, "moves": len(steps), "steps": steps,
           "robot_s": round(ep.robot_s, 1)}
    if policy == "once" and outcome == "not pickable after one stroke":
        # What the verdict misses: pick anyway on the axis with the shallowest neighbour hit.
        v = ep.see()
        plan = pick_plan(ep, v, allowed_only=False)
        if plan is not None:
            yaw, flip, _ = plan
            res = act(ep, "pick", v["cubes"][TARGET]["pos"], yaw, flip)
            row["refused_pick_lifted"] = bool(res["success"])
            row["refused_pick_rough"] = v["cubes"][TARGET]["cov"] < tools.TRUST
    ep.close()
    return row


def job(args):
    return play(*args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--policies", nargs="+", default=["once", "loop", "look"])
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--out", default="outputs/look_first_2026-09-30")
    a = ap.parse_args()
    jobs = [(s, p) for p in a.policies for s in a.seeds]
    with ProcessPoolExecutor(a.procs) as pool:
        rows = list(pool.map(job, jobs))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rows.json").write_text(json.dumps(rows, indent=1, default=str))
    for p in a.policies:
        rs = [r for r in rows if r["policy"] == p]
        solved = [r for r in rs if r["outcome"] == "solved"]
        outcomes = {}
        for r in rs:
            outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
        picks = [st for r in rs for st in r["steps"] if st["kind"] == "pick"]
        print(f"{p:5s}: solved {len(solved)}/{len(rs)}, moves when solved {np.mean([r['moves'] for r in solved]) if solved else 0:.1f}; "
              f"allowed picks lifted the red {sum(st['success'] for st in picks)}/{len(picks)} "
              f"(trusted {sum(st['success'] for st in picks if not st['rough'])}/{sum(not st['rough'] for st in picks)}, "
              f"rough {sum(st['success'] for st in picks if st['rough'])}/{sum(st['rough'] for st in picks)}); {outcomes}")
    once = [r for r in rows if "refused_pick_lifted" in r]
    if once:
        print(f"once, picks act would refuse, tried anyway: lifted {sum(r['refused_pick_lifted'] for r in once)}/{len(once)}")


if __name__ == "__main__":
    main()
