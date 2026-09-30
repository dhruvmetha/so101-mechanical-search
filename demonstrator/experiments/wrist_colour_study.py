"""What the wrist camera adds once it gives colour and masks only (dm1487, 2026-09-30: the real one has no depth).

For each pile: park the arm as observe does, run Episode.see(), and compare against the simulator's truth (this is
a measurement of the tool, not play). Three settings on the same pile:
  front   the front camera alone (the wrist camera's image blanked)
  colour  the front camera plus the wrist camera's colour and masks, as observe now does
  depth   for reference, the unseen share if wrist depth were still used (cells hit by either camera's points)

    python demonstrator/experiments/wrist_colour_study.py --seeds 2000 2001 ... --procs 8
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tools  # noqa: E402  (imports so101sim before mujoco)
from so101sim.mechsearch import TARGET  # noqa: E402

TRUST = tools.TRUST


def measure(ep, v):
    truth = {n: ep.env.cube_pose(n)["pos"] for n in ep.env.cubes}
    trusted = [np.linalg.norm(cb["pos"] - truth[n]) * 1000 for n, cb in v["cubes"].items() if cb["cov"] >= TRUST]
    ray = [np.linalg.norm(cb["pos"][:2] - truth[n][:2]) * 1000 for n, cb in v["cubes"].items() if cb["how"] == "ray"]
    red = None
    if len(v["red_px"]):
        red = float(np.linalg.norm(np.median(v["red_px"][:, :2], 0) - truth[TARGET][:2]) * 1000)
    return {"unseen": round(float(v["unseen"].mean()), 3), "listed": len(v["cubes"]),
            "trusted_n": len(trusted), "trusted_err_mm_median": round(float(np.median(trusted)), 2) if trusted else None,
            "ray_n": len(ray), "ray_err_mm": [round(x, 1) for x in ray],
            "red_seen": bool(len(v["red_px"])), "red_err_mm": None if red is None else round(red, 1)}


def depth_unseen(ep):
    """Share of cells no camera point lands in, if the wrist camera still gave depth (the old observe)."""
    c, h = ep.bin["center"], ep.bin["half"]
    n = int(round(2 * h / tools.CELL))
    seen = np.zeros((n, n), bool)
    for cam in (ep.front, ep.wrist):
        _, _, geom, pts = cam.capture(ep.d)
        p = pts[geom >= 0]
        i = np.floor((p[:, 0] - (c[0] - h)) / tools.CELL).astype(int)
        j = np.floor((p[:, 1] - (c[1] - h)) / tools.CELL).astype(int)
        ok = (i >= 0) & (i < n) & (j >= 0) & (j < n) & (p[:, 2] < ep.bin["rim_z"] + 0.1)
        seen[i[ok], j[ok]] = True
    return round(float(1 - seen.mean()), 3)


def run(seed):
    buried = seed % 2 == 0
    ep = tools.Episode(f"study{seed}", size=64, rec={"seed": seed, "buried": buried, "ops": []})
    ep._apply({"op": "park", "keep_grip": True})
    ep.rec["ops"].append({"op": "park", "keep_grip": True})
    out = {"seed": seed, "buried": buried}
    out["colour"] = measure(ep, ep.see())
    blank = ep.wrist.capture_colour
    ep.wrist.capture_colour = lambda d: (np.zeros_like(blank(d)[0]), np.full(blank(d)[1].shape, -1))
    out["front"] = measure(ep, ep.see())
    ep.wrist.capture_colour = blank
    out["depth_unseen"] = depth_unseen(ep)
    ep.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(2000, 2012)))
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--out", default="outputs/wrist_colour_study_2026-09-30")
    a = ap.parse_args()
    with ProcessPoolExecutor(a.procs) as pool:
        rows = list(pool.map(run, a.seeds))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "rows.json").write_text(json.dumps(rows, indent=1))
    for key in ("front", "colour"):
        r = [x[key] for x in rows]
        errs = [x["trusted_err_mm_median"] for x in r if x["trusted_err_mm_median"] is not None]
        rays = [e for x in r for e in x["ray_err_mm"]]
        reds = [x["red_err_mm"] for x in r if x["red_err_mm"] is not None]
        print(f"{key:6s} unseen median {np.median([x['unseen'] for x in r]):.2f}, cubes listed median "
              f"{np.median([x['listed'] for x in r]):.0f}, trusted pose error median {np.median(errs):.2f} mm, "
              f"ray cubes {len(rays)} (error median {np.median(rays) if rays else float('nan'):.1f} mm, max "
              f"{max(rays) if rays else float('nan'):.1f}), red seen {sum(x['red_seen'] for x in r)}/{len(r)} "
              f"(error median {np.median(reds) if reds else float('nan'):.1f} mm)")
    print(f"depth  unseen median {np.median([x['depth_unseen'] for x in rows]):.2f} (the old observe, for reference)")


if __name__ == "__main__":
    main()
