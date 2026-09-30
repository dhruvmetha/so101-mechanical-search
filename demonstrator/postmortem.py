"""After the fact, with the simulator's true poses: where each episode's moves and robot time went.

    python demonstrator/postmortem.py --out rows.json [names...]     default: every finished d* episode

This reads ground truth, so it is for studying finished episodes only, never for playing one. Per op it records the
robot time, and for an act: what was really picked (the cube lifted with the jaws), how far the other cubes and the
red moved, and how many cubes were on top of the red and flush against its sides before and after.
"""
import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

CUBE = 0.030
HELD_Z = -0.005  # a cube centre above this is in the jaws: layer 3 centres sit at -0.021


def red_state(env):
    """(cubes resting over the red, cubes flush against its sides) from true poses."""
    from so101sim.mechsearch import TARGET
    red = env.cube_pose(TARGET)["pos"]
    over = side = 0
    for n in env.grays:
        p = env.cube_pose(n)["pos"]
        if p[2] > HELD_Z:
            continue
        d = p - red
        horiz = np.linalg.norm(d[:2])
        if d[2] > CUBE * 0.5 and horiz < CUBE * 1.2:
            over += 1
        elif abs(d[2]) < CUBE * 0.5 and horiz < CUBE * 1.25:
            side += 1
    return over, side


def study(name):
    import tools
    from so101sim.mechsearch import TARGET

    frames = {"n": 0}

    def on_advance(ep, row):
        frames["n"] += 1

    rows = []
    original = tools.Episode._apply

    def spy(self, op):
        env = self.env
        before = {n: env.cube_pose(n)["pos"].copy() for n in env.cubes}
        state0 = red_state(env)
        f0 = frames["n"]
        res = original(self, op)
        after = {n: env.cube_pose(n)["pos"].copy() for n in env.cubes}
        moved = {n: float(np.linalg.norm(after[n] - before[n])) for n in env.cubes}
        r = {"op": op["op"], "seconds": round((frames["n"] - f0) / 30, 2)}
        if op["op"] == "act":
            r.update(kind=op["kind"], xyz=op["xyz"], deg=op["deg"], grip=res["grip"], success=res["success"],
                     failure=res["failure"])
            if op["kind"] == "pick":
                near = min(env.cubes, key=lambda n: np.linalg.norm(before[n] - np.asarray(op["xyz"])))
                r["aimed_at"] = near
                r["aim_error_mm"] = round(float(np.linalg.norm(before[near][:2] - np.asarray(op["xyz"][:2]))) * 1000, 1)
                lifted = [n for n in env.cubes if after[n][2] - before[n][2] > 0.02]
                r["lifted"] = lifted
            held = {n for n in env.cubes if before[n][2] > HELD_Z} | set(r.get("lifted", []))
            others = [moved[n] for n in env.cubes if n != TARGET and n not in held]
            r["others_moved_mm"] = round(sum(m for m in others if m > 0.003) * 1000, 1)
            r["cubes_moved"] = sum(1 for m in others if m > 0.003)
            r["red_moved_mm"] = round(moved[TARGET] * 1000, 1)
            r["red_over_side_before"] = state0
            r["red_over_side_after"] = red_state(env)
        rows.append(r)
        return res

    tools.Episode._apply = spy
    try:
        ep = tools.Episode(name, on_advance=on_advance)
    finally:
        tools.Episode._apply = original
    out = {"name": name, "seed": ep.rec["seed"], "buried": ep.rec["buried"], "outcome": ep.outcome(),
           "robot_s": round(ep.robot_s, 1), "undone": len(ep.rec.get("undone", [])), "ops": rows,
           "start_red_over_side": None}
    acts = [r for r in rows if r["op"] == "act"]
    if acts:
        out["start_red_over_side"] = acts[0]["red_over_side_before"]
    ep.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    names = args.names
    if not names:
        for p in sorted((HERE / "episodes").glob("d*.json")):
            rec = json.loads(p.read_text())
            acts = [op for op in rec["ops"] if op["op"] == "act"]
            if acts and (acts[-1]["result"]["success"] or acts[-1]["result"]["terminated"] or len(acts) >= 15):
                names.append(p.stem)
    with mp.get_context("spawn").Pool(min(16, len(names) or 1)) as pool:
        results = pool.map(study, names)
    Path(args.out).write_text(json.dumps(results, indent=1))
    park_s = sum(r["seconds"] for e in results for r in e["ops"] if r["op"] == "park")
    act_s = sum(r["seconds"] for e in results for r in e["ops"] if r["op"] == "act")
    picks = [r for e in results for r in e["ops"] if r.get("kind") == "pick"]
    wrong = [r for r in picks if r["lifted"] and r["aimed_at"] not in r["lifted"]]
    empty = [r for r in picks if not r["lifted"]]
    print(f"{len(results)} episodes, robot time {park_s + act_s:.0f} s: parking {park_s:.0f} s, moves {act_s:.0f} s")
    print(f"picks {len(picks)}: lifted nothing {len(empty)}, lifted a different cube {len(wrong)}")
    for kind in ("pick", "place", "push"):
        ks = [r for e in results for r in e["ops"] if r.get("kind") == kind]
        if ks:
            print(f"  {kind}: {len(ks)}, mean {np.mean([r['seconds'] for r in ks]):.1f} s, "
                  f"other cubes moved per move {np.mean([r['cubes_moved'] for r in ks]):.1f}")


if __name__ == "__main__":
    main()
