"""Calibrate check against real pick outcomes under the stiff contacts of 2026-09-30.

The finger check's thresholds (a neighbour hit under 3 mm lifted the target 96% of the time) were scored on the old
soft contacts, where a finger could press a neighbour down 10 mm and still reach the grasp. On each fresh pile this
parks the arm, observes, and tries picks on cubes observe lists, centred and deliberately shifted, on both jaw axes.
Every pick runs from a snapshot and is taken back, so one pile gives many picks. Outcomes come from the simulator's
truth, since this measures the tool, not play.

    python demonstrator/experiments/pick_calibration.py --seeds 3000 ... --procs 12
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

SHIFTS_MM = (0.0, 0.0, 2.0, -2.0, 4.0, -4.0)
PER_PILE = 14


def run(seed):
    rng = np.random.default_rng(seed)
    ep = Live(f"cal{seed}", size=64, rec={"seed": seed, "buried": seed % 2 == 0, "ops": []})
    ep.run([{"op": "park", "keep_grip": True}])
    v = ep.see()
    base = ep.snapshot()
    names = [n for n, cb in v["cubes"].items() if cb["layer"] >= 1]
    rng.shuffle(names)
    rows = []
    tried = 0
    for name in names:
        if tried >= PER_PILE:
            break
        cb = v["cubes"][name]
        for axis in (0.0, 90.0):
            shift = float(rng.choice(SHIFTS_MM))
            deg = (cb["yaw"] + axis) % 180.0
            along = np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg))])
            xyz = [float(cb["pos"][0] + along[0] * shift / 1000), float(cb["pos"][1] + along[1] * shift / 1000),
                   float(cb["pos"][2])]
            ep.restore(base)
            ep.view = v
            flip, results = ep.choose("pick", xyz, deg)
            if flip is None:
                continue
            c = results[flip]
            if c.get("target") != name:
                continue
            before = {n: ep.env.cube_pose(n)["pos"].copy() for n in ep.env.cubes}
            res = ep.run([{"op": "act", "kind": "pick", "xyz": xyz, "deg": deg, "flip": flip, "unpark": True}])
            after = {n: ep.env.cube_pose(n)["pos"] for n in ep.env.cubes}
            lifted = [n for n in before if after[n][2] - before[n][2] > 0.02]
            hits = {k: round(x * 1000, 1) for k, x in c["hits"].items() if k != name}
            rows.append({"seed": seed, "cube": name, "trusted": c["trusted"], "how": cb["how"], "tilt": cb["tilt"],
                         "layer": cb["layer"], "shift_asked_mm": shift, "shift_mm": round(c["shift_mm"], 1),
                         "on_top_mm": round(c["on_top_mm"], 1), "worst_hit_mm": max(hits.values(), default=0.0),
                         "hits": hits, "unseen_mm": round(c["unseen_mm"], 1), "wall_gap_mm": round(c["wall_gap_mm"], 1),
                         "self_mm": round(c["self_mm"], 1), "grip": round(float(res["grip"]), 1),
                         "lifted_target": name in lifted, "lifted_other": [n for n in lifted if n != name],
                         "spilled": bool(res["failure"])})
            tried += 1
    ep.close()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(3000, 3024)))
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--out", default="outputs/pick_calibration_2026-09-30")
    a = ap.parse_args()
    with ProcessPoolExecutor(a.procs) as pool:
        rows = [r for part in pool.map(run, a.seeds) for r in part]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rows.json").write_text(json.dumps(rows, indent=1))
    ok = lambda rs: f"{sum(r['lifted_target'] for r in rs)}/{len(rs)}" if rs else "0/0"
    print(f"picks {len(rows)}, lifted the target {ok(rows)}, lifted another too "
          f"{sum(bool(r['lifted_other']) for r in rows)}, spilled {sum(r['spilled'] for r in rows)}")
    for lo, hi in ((0, 3), (3, 6), (6, 10), (10, 99)):
        rs = [r for r in rows if lo <= r["worst_hit_mm"] < hi]
        print(f"  worst neighbour hit {lo}-{hi} mm: {ok(rs)}")
    for lo, hi in ((-99, 1), (1, 3), (3, 99)):
        rs = [r for r in rows if lo <= r["shift_mm"] < hi and r["trusted"]]
        print(f"  trusted, shift toward moving finger {lo}..{hi} mm: {ok(rs)}")
    for lo, hi in ((0, 10), (10, 40), (40, 999)):
        rs = [r for r in rows if lo <= r["unseen_mm"] < hi]
        print(f"  unseen {lo}-{hi} mm: {ok(rs)}")
    for t in (True, False):
        rs = [r for r in rows if r["trusted"] == t]
        print(f"  pose {'trusted' if t else 'rough'}: {ok(rs)}")


if __name__ == "__main__":
    main()
