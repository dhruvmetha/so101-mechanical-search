"""The expert as a program: perception proposes moves, the simulator judges them on a snapshot, the best one is kept.

    python demonstrator/planner.py --seeds 1000 1001 --out outputs/planner_pilot     one process per seed

Pilot p0, 2026-09-30, following the build order in docs/knowledge/state.md. Each decision:
  1. park and look (a real motion), as the Claude players did;
  2. estimate the red from its pose if listed, else from its pixels;
  3. propose candidates from the camera data only: a pick on the red (both jaw axes, higher final lift), sweeps
     across the red's spot when cubes cover it, and relocations (pick, then place on a spot away from the red and its
     neighbours) for the covers and the cubes blocking the red's jaw axes;
  4. run each candidate on a snapshot of the live episode (demonstrator/live.py);
  5. score it with the simulator's true poses: a spill is out, success wins, otherwise count cubes taken off the red,
     flush neighbours freed, robot time spent and other cubes disturbed;
  6. keep the best, or stop when two decisions in a row find nothing that helps.
Candidates come from perception, so every kept move has a reason a camera could see; true poses only judge outcomes.
Aim-error perturbations, cost logging and the Claude rescue come later in the build order.
"""
import argparse
import json
import multiprocessing as mp
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PLANNER = "p0"
LIFT_ABOVE_TRAVEL = 0.03   # the red's final lift, above travel height
MAX_DECISIONS = 20
MAX_ROBOT_S = 200.0
FLUSH = 0.0375             # neighbour centres closer than this, on the red's level, box it in
OVER = 0.036               # a cube this close in xy and higher than the red covers it


def versions():
    import mujoco
    try:
        git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             cwd=HERE.parent).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "so101sim", "demonstrator"],
                                    capture_output=True, text=True, cwd=HERE.parent).stdout.strip())
    except OSError:
        git, dirty = "unknown", True
    return {"git": git + ("+dirty" if dirty else ""), "mujoco": mujoco.__version__, "planner": PLANNER}


# ---- truth, for judging only ----------------------------------------------------------------------------------------

def truth(env):
    """(cubes over the red, flush neighbours on its level) from true poses. Held cubes don't count."""
    from so101sim.mechsearch import TARGET
    red = env.cube_pose(TARGET)["pos"]
    over = side = 0
    for n in env.grays:
        p = env.cube_pose(n)["pos"]
        if p[2] > -0.005:
            continue
        d = p - red
        h = np.linalg.norm(d[:2])
        if d[2] > 0.015 and h < OVER:
            over += 1
        elif abs(d[2]) < 0.015 and h < FLUSH:
            side += 1
    return over, side


def poses(env):
    return {n: env.cube_pose(n)["pos"].copy() for n in env.cubes}


# ---- proposing, from perception only --------------------------------------------------------------------------------

def red_estimate(live, v):
    from so101sim.mechsearch import TARGET
    if TARGET in v["cubes"]:
        cb = v["cubes"][TARGET]
        return np.asarray(cb["pos"], float), cb
    px = v["red_px"]
    if len(px) >= 3:
        z = max(live.bin["floor_z"] + live_half(), float(np.percentile(px[:, 2], 90)) - live_half())
        return np.array([*np.median(px[:, :2], 0), z]), None
    return None, None


def live_half():
    from perception import HALF
    return HALF


def place_spots(live, v, red_xy, keep_clear, k=2):
    """Tops of trusted, level layer-1 cubes and open floor, far from the red and not beside its neighbours or any
    leaning cube (d1024, d1028, d1031 in demonstrator/notes)."""
    from tools import TRUST
    from perception import CUBE
    c, h = live.bin["center"], live.bin["half"]
    leaning = [np.asarray(cb["pos"][:2]) for cb in v["cubes"].values() if (cb["tilt"] or 0) > 10]
    cands = []
    for name, cb in v["cubes"].items():
        if cb["cov"] < TRUST or cb["layer"] != 1 or (cb["tilt"] or 0) > 5:
            continue
        cands.append((np.asarray(cb["pos"][:2]), cb["pos"][2] + CUBE + 0.003, cb["yaw"]))
    for sp in live.spots():
        cands.append((np.asarray(sp[:2]), sp[2], 0.0))
    good = []
    for xy, z, yaw in cands:
        if (h - np.abs(xy - c)).min() < 0.025:
            continue  # by a wall
        if red_xy is not None and np.linalg.norm(xy - red_xy) < 0.07:
            continue
        if any(np.linalg.norm(xy - q) < 0.045 for q in keep_clear + leaning):
            continue
        good.append((-(np.linalg.norm(xy - red_xy) if red_xy is not None else 0.0), xy, z, yaw))
    good.sort(key=lambda t: t[0])
    return [(xy, z, yaw) for _, xy, z, yaw in good[:k]]


def act(kind, xyz, deg, why, **extra):
    op = {"op": "act", "kind": kind, "xyz": [round(float(x), 4) for x in xyz], "deg": round(float(deg), 1),
          "why": why, "unpark": True}
    op.update({k: v for k, v in extra.items() if v is not None})
    return op


def pick_op(live, name, cb, why, lift_z=None, dz=0.0):
    """The better of a cube's two jaw axes by check's hits, with check's wrist roll; None if neither runs."""
    best = None
    for deg in (cb["yaw"], cb["yaw"] + 90.0):
        xyz = [cb["pos"][0], cb["pos"][1], cb["pos"][2] + dz]
        flip, res = live.choose("pick", xyz, deg)
        if flip is None:
            continue
        r = res[flip]
        bad = sum(v for k, v in r["hits"].items() if k != name) + r["unseen_mm"] / 1000 * 0.2
        if best is None or bad < best[0]:
            best = (bad, act("pick", xyz, deg % 180.0, why, flip=flip, lift_z=lift_z))
    return None if best is None else best[1]


def candidates(live, v, red, red_cb):
    """Lists of ops, each a candidate move, with a label."""
    from so101sim.mechsearch import TARGET
    from tools import axis_ends, short
    cubes, out = v["cubes"], []
    rim, floor = live.bin["rim_z"], live.bin["floor_z"]
    if red is None:
        # Nothing of the red in sight: dig where the pile is highest.
        targets = sorted((n for n in cubes if n != TARGET), key=lambda n: -cubes[n]["pos"][2])[:3]
        covers, blockers = targets, []
    else:
        covers = sorted((n for n, cb in cubes.items() if n != TARGET
                         and np.linalg.norm(np.asarray(cb["pos"][:2]) - red[:2]) < OVER
                         and cb["pos"][2] > red[2] + 0.015), key=lambda n: -cubes[n]["pos"][2])
        blockers = []
        if red_cb is not None:
            axes = axis_ends(v, TARGET)
            per_axis = [[n for sign in (1, -1) for gap, n, _ in ends[sign] if gap < 0.015] for _, ends in axes]
            blockers = min(per_axis, key=len)
    # a pick on the red, both axes, with the higher lift
    if red_cb is not None:
        dz = 0.006 if (red_cb["tilt"] or 0) > 10 else 0.0
        op = pick_op(live, TARGET, red_cb, "pick the red, lift it clear", lift_z=live.prim.travel_z + LIFT_ABOVE_TRAVEL,
                     dz=dz)
        if op:
            out.append(("red pick", [op]))
    # sweeps across the red's spot
    if red is not None and covers:
        c, h = np.asarray(live.bin["center"]), live.bin["half"]
        for deg in (0, 90, 180, 270):
            u = np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg))])
            t = min((h - 0.006 - s * (red[k] - c[k])) / (s * -u[k]) for k in range(2) for s in (1, -1)
                    if s * -u[k] > 1e-9)
            start = red[:2] - t * u
            out.append((f"sweep from the rim {deg}", [act("sweep", [*start, rim + 0.008], deg,
                                                           "sweep the covers off the red from the rim", stroke=0.12)]))
            out.append((f"sweep low {deg}", [act("sweep", [*(red[:2] - 0.05 * u), red[2] + 0.03], deg,
                                                  "sweep the covers off the red at cover height", stroke=0.09)]))
    # relocations
    keep_clear = [np.asarray(cb["pos"][:2]) for n, cb in cubes.items()
                  if red is not None and np.linalg.norm(np.asarray(cb["pos"][:2]) - red[:2]) < 0.05]
    spots = place_spots(live, v, None if red is None else red[:2], keep_clear)
    for name in list(dict.fromkeys(covers[:2] + blockers[:2])):
        why = "cover over the red" if name in covers else "flush on the red's jaw axis"
        op = pick_op(live, name, cubes[name], f"lift {short(name)}: {why}")
        if op is None:
            continue
        for xy, z, yaw in spots:
            place = act("place", [xy[0], xy[1], z], yaw % 180.0, f"set {short(name)} down away from the red")
            out.append((f"relocate {short(name)}", [op, place]))
    return out


# ---- judging and the loop -------------------------------------------------------------------------------------------

def score(before, after, res, env, moved_ok, robot_s):
    from so101sim.mechsearch import TARGET
    if res is None:
        return -1e9, "refused"
    if res["terminated"] and not res["success"]:
        return -1e9, f"spill: {res['failure']}"
    if res["success"]:
        return 1000.0 - robot_s, "success"
    (o0, s0), (o1, s1) = before["state"], truth(env)
    now = poses(env)
    disturbed = sum(1 for n in env.cubes if n not in moved_ok and n != TARGET
                    and np.linalg.norm(now[n] - before["poses"][n]) > 0.003)
    val = 10.0 * (o0 - o1) + 4.0 * (s0 - s1) - 0.05 * robot_s - 0.3 * disturbed
    return val, f"over {o0}->{o1}, flush {s0}->{s1}, disturbed {disturbed}, {robot_s:.1f} s"


def run_seed(seed, buried, out_dir):
    from live import Live
    from so101sim.mechsearch import TARGET
    name = f"p{seed}"
    rec = {"seed": int(seed), "buried": bool(buried), "ops": [], "versions": versions(), "log": []}
    live = Live(name, rec=rec)
    t0, sims, stuck, outcome = time.time(), 0, 0, "unfinished"
    try:
        while True:
            acts = [op for op in live.rec["ops"] if op["op"] == "act"]
            if len(acts) >= MAX_DECISIONS * 2 or live.robot_s > MAX_ROBOT_S:
                outcome = "out of budget"
                break
            if not live.parked() and not (14 < float(live.env.joint_state[5]) < 30):
                live.run([{"op": "park", "keep_grip": True}])
            v = live.see()
            held = live.held_cube()
            if held is not None:
                # Holding something at the start of a decision: a red that came up tilted, short of the line. Set it
                # down level where it can be grasped square again (d1004, e402).
                cands = [(f"set down {held}", [act("place", [xy[0], xy[1], z], yaw % 180.0,
                                                   "set the held cube down level to grasp it again")])
                         for xy, z, yaw in place_spots(live, v, None, [])]
                red, red_cb = None, None
            else:
                red, red_cb = red_estimate(live, v)
                cands = candidates(live, v, red, red_cb)
            base = live.snapshot()
            before = {"state": truth(live.env), "poses": poses(live.env)}
            base_ops = list(live.rec["ops"])
            tried = []
            for label, ops in cands:
                live.restore(base)
                live.rec["ops"] = list(base_ops)
                r0 = live.robot_s
                ops = [dict(op) for op in ops]
                if ops[0]["kind"] == "place" and "flip" not in ops[0]:
                    ops[0]["flip"] = live.choose("place", ops[0]["xyz"], ops[0]["deg"])[0]
                res = live.run(ops[:1])
                sims += 1
                if len(ops) > 1 and res is not None and not res["terminated"] and 14 < res["grip"] < 30:
                    # choose the place's wrist roll from where the pick left the arm
                    ops[1]["flip"] = live.choose("place", ops[1]["xyz"], ops[1]["deg"])[0]
                    res = live.run(ops[1:])
                    sims += 1
                moved_ok = set()
                for op in ops:
                    if op["kind"] == "pick":
                        near = min(live.env.cubes, key=lambda n: np.linalg.norm(before["poses"][n] - op["xyz"]))
                        moved_ok.add(near)
                val, why = score(before, None, res, live.env, moved_ok, live.robot_s - r0)
                tried.append((val, label, why, live.snapshot(), list(live.rec["ops"])))
            live.restore(base)
            live.rec["ops"] = list(base_ops)
            if not tried:
                outcome = "no candidates"
                break
            tried.sort(key=lambda t: -t[0])
            val, label, why, snap, ops = tried[0]
            rec["log"].append({"decision": len(rec["log"]) + 1, "chosen": label, "value": round(val, 2), "why": why,
                               "tried": [(t[1], round(t[0], 2), t[2]) for t in tried]})
            if val <= 0:
                stuck += 1
                if stuck >= 2:
                    outcome = "stuck"
                    break
            else:
                stuck = 0
            live.restore(snap)
            live.rec["ops"] = ops
            if val >= 900:
                outcome = "solved"
                break
    except Exception as exc:  # one seed's crash shouldn't lose the others' results
        import traceback
        outcome = f"error: {exc!r}"
        rec["traceback"] = traceback.format_exc()
    finally:
        wall = time.time() - t0
    acts = [op for op in live.rec["ops"] if op["op"] == "act"]
    summary = {"seed": seed, "buried": buried, "outcome": outcome, "moves": len(acts),
               "picks_empty": sum(1 for op in acts if op["kind"] == "pick" and op["result"]["grip"] <= 14),
               "robot_s": round(live.robot_s, 1), "wall_s": round(wall, 1), "sims": sims,
               "decisions": len(rec["log"]), "versions": rec["versions"]}
    rec["summary"] = summary
    (out_dir / f"{name}.json").write_text(json.dumps(rec, indent=1, default=float))
    live.close()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(s, s % 2 == 0, out) for s in args.seeds]  # even seeds buried, odd partial, as overnight
    with mp.get_context("spawn").Pool(min(args.procs, len(jobs))) as pool:
        rows = pool.starmap(run_seed, jobs)
    for r in rows:
        print(f"seed {r['seed']} {'buried ' if r['buried'] else 'partial'} {r['outcome']:14s} moves {r['moves']:2d} "
              f"empty picks {r['picks_empty']} robot {r['robot_s']:5.1f} s  wall {r['wall_s']:6.1f} s  sims {r['sims']}")
    (out / "summary.json").write_text(json.dumps(rows, indent=1, default=float))


if __name__ == "__main__":
    main()
