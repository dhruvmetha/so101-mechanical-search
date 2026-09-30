"""Calibrate the push report against what pushes and sweeps really do (2026-09-30, after dm1487 found pushes sidelined).

On each fresh pile: park, observe, then try pushes and sweeps at cubes observe lists, each from a snapshot and taken
back. A stroke starts 1.5 cm behind the cube's near face at the cube's centre height and heads through its centre, in
one of 8 directions, 5 cm (push) or 8, 12 or 16 cm (sweep). Records what check predicted (what moves, spill risk,
the red) and what happened (spill, which cubes moved, how far the pushed cube went, the red). Outcomes come from the
simulator's truth: this measures the tool, not play. Friction and density vary per pile, as in play.

    python demonstrator/experiments/push_calibration.py --seeds 3100 ... --procs 12
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

STROKES = (0.05, 0.08, 0.12, 0.16)
PER_PILE = 16


def end_to_wall(ep, start, d, stroke):
    """Distance from where the jaws stop to the wall ahead of them, along the stroke."""
    c, h = ep.bin["center"], ep.bin["half"]
    end = start + d * stroke
    return min(((h - np.sign(d[k]) * (end[k] - c[k])) / abs(d[k])) if abs(d[k]) > 1e-6 else np.inf for k in (0, 1))


def run(seed):
    rng = np.random.default_rng(seed)
    ep = Live(f"pcal{seed}", size=64, rec={"seed": seed, "buried": seed % 2 == 0, "ops": []})
    ep.run([{"op": "park", "keep_grip": True}])
    v = ep.see()
    base = ep.snapshot()
    names = list(v["cubes"])
    rng.shuffle(names)
    rows, tries = [], 0
    for name in names * 3:
        if len(rows) >= PER_PILE or tries > 6 * PER_PILE:
            break
        tries += 1
        cb = v["cubes"][name]
        deg = float(rng.integers(0, 8)) * 45.0
        d = np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg))])
        stroke = float(rng.choice(STROKES))
        kind = "push" if stroke == 0.05 else "sweep"
        start = cb["pos"][:2] - d * (tools.HALF + 0.015)
        xyz = [float(start[0]), float(start[1]), float(cb["pos"][2])]
        ep.restore(base)
        ep.view = v
        flip, results = ep.choose(kind, xyz, deg, None if kind == "push" else stroke)
        if flip is None:
            continue
        c = results[flip]
        before = {n: ep.env.cube_pose(n)["pos"].copy() for n in ep.env.cubes}
        op = {"op": "act", "kind": kind, "xyz": xyz, "deg": deg, "flip": flip, "unpark": True}
        if kind == "sweep":
            op["stroke"] = stroke
        res = ep.run([op])
        after = {n: ep.env.cube_pose(n)["pos"] for n in ep.env.cubes}
        moved = {n: round(float(np.linalg.norm(after[n] - before[n])) * 1000, 1) for n in before
                 if np.linalg.norm(after[n] - before[n]) > 0.005}
        along = float((after[name][:2] - before[name][:2]) @ d) * 1000
        p = c["push"]
        rows.append({"seed": seed, "cube": name, "layer": cb["layer"], "trusted": cb["cov"] >= tools.TRUST,
                     "how": cb["how"], "kind": kind, "stroke": stroke, "deg": deg,
                     "predicted": sorted(p["moves"]), "moves_detail": p["moves"], "spill_risk": p["spill_risk"],
                     "end_to_wall_mm": round(end_to_wall(ep, start, d, stroke) * 1000, 1),
                     "cube_pos": [round(float(x), 4) for x in cb["pos"]],
                     "jam": [n for n, m in p["moves"].items() if m["jam"]], "moves_red_pred": p["moves_red"],
                     "unseen_mm": round(c["unseen_mm"], 1), "self_mm": round(c["self_mm"], 1),
                     "spilled": bool(res["failure"]), "failure": res["failure"], "moved": moved,
                     "cube_along_mm": round(along, 1), "red_moved_mm": moved.get(TARGET, 0.0),
                     "physics": dict(ep.env.physics)})
    ep.close()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(3100, 3124)))
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--out", default="outputs/push_calibration_2026-09-30/round3")
    a = ap.parse_args()
    with ProcessPoolExecutor(a.procs) as pool:
        rows = [r for part in pool.map(run, a.seeds) for r in part]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rows.json").write_text(json.dumps(rows, indent=1))

    def rate(rs, f):
        k = sum(bool(f(r)) for r in rs)
        return f"{k}/{len(rs)}" + (f" ({k / len(rs):.0%})" if rs else "")

    print(f"strokes {len(rows)}: spilled {rate(rows, lambda r: r['spilled'])}, pushed cube went 15 mm or more its way "
          f"{rate(rows, lambda r: r['cube_along_mm'] >= 15)}")
    for label, f in (("spill risk flagged", lambda r: r["spill_risk"]),
                     ("jam, not flagged", lambda r: r["jam"] and not r["spill_risk"]),
                     ("no jam", lambda r: not r["jam"])):
        rs = [r for r in rows if f(r)]
        print(f"  {label:18s}: spilled {rate(rs, lambda r: r['spilled'])}")
    for lo, hi in ((0, 10), (10, 30), (30, 999)):
        rs = [r for r in rows if lo <= r["unseen_mm"] < hi and not r["spill_risk"]]
        print(f"  unseen {lo}-{hi} mm, not flagged: spilled {rate(rs, lambda r: r['spilled'])}")
    for layer in (1, 2, 3):
        rs = [r for r in rows if r["layer"] == layer]
        print(f"  pushed cube on layer {layer}: spilled {rate(rs, lambda r: r['spilled'])}, went its way "
              f"{rate(rs, lambda r: r['cube_along_mm'] >= 15)}")
    for s in STROKES:
        rs = [r for r in rows if r["stroke"] == s]
        print(f"  stroke {s * 100:.0f} cm: spilled {rate(rs, lambda r: r['spilled'])}, went its way "
              f"{rate(rs, lambda r: r['cube_along_mm'] >= 15)}")
    rp = [r for r in rows if r["moves_red_pred"]]
    rn = [r for r in rows if not r["moves_red_pred"]]
    print(f"  red predicted to move: moved 5 mm+ {rate(rp, lambda r: r['red_moved_mm'] > 0)}; not predicted: moved "
          f"{rate(rn, lambda r: r['red_moved_mm'] > 0)}")
    hit = sum(len(set(r["predicted"]) & set(r["moved"])) for r in rows)
    print(f"  predicted movers that moved {hit}/{sum(len(r['predicted']) for r in rows)}; cubes that moved and were "
          f"predicted {hit}/{sum(len(r['moved']) for r in rows)}")


if __name__ == "__main__":
    main()
