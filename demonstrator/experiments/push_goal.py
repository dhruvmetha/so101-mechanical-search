"""Do purposeful pushes help get to the red? (2026-09-30, after dm1487 asked when pushing is useful.)

On each fresh pile where observe lists the red: generate strokes from what the cameras see, aimed at the red's access:
  sideways  each blocker at an end of the red's jaw axes, pushed sideways off the axis, 5 and 8 cm
  covers    each cube on the red's top, swept off in four directions, 8 cm
  red       the red itself pushed 5 cm along each of its four axis directions, toward whatever room there is
Each runs from a snapshot. Afterwards observe looks again, and the outcome is judged the way a player would: is the red
seen with a trusted pose, how many ends of its jaw axes are blocked, and would act allow a pick of it (no refusal on
either jaw axis). Spills come from the simulator. Picks-before is the same test on the starting view.

    python demonstrator/experiments/push_goal.py --seeds 5101 ... --procs 12
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

HALF, CUBE = tools.HALF, tools.CUBE


def pickable(ep, v):
    """Would act allow a pick of the red on either jaw axis, judged from view v? Returns (bool, axis yaw or None)."""
    if TARGET not in v["cubes"]:
        return False, None
    red = v["cubes"][TARGET]
    ep.view = v
    for yaw in (red["yaw"], red["yaw"] + 90.0):
        flip, results = ep.choose("pick", list(map(float, red["pos"])), yaw % 180.0)
        if flip is None:
            continue
        r = results[flip]
        if r.get("target") != TARGET:
            continue
        worst = max((x for k, x in r["hits"].items() if k != TARGET), default=0.0) * 1000
        if (r["trusted"] and r["shift_mm"] > tools.SHIFT_MAX_MM) or r["on_top_mm"] > tools.MARGIN * 1000:
            continue
        if worst >= tools.HIT_REFUSE_MM or r["self_mm"] < tools.SELF_JAM_MM:
            continue
        return True, yaw % 180.0
    return False, None


def strokes_for(v):
    cubes = v["cubes"]
    red = cubes[TARGET]
    rp = np.asarray(red["pos"])
    out = []
    access = tools.red_access(cubes)
    for yaw, ends in tools.axis_ends(v, TARGET):
        a = np.radians(yaw)
        u = np.array([np.cos(a), np.sin(a)])
        side = np.array([-u[1], u[0]])
        for sign, found in ends.items():
            for _, name, _ in found[:1]:
                bp = np.asarray(cubes[name]["pos"])
                for s in (1, -1):
                    d = s * side
                    start = bp[:2] - d * (HALF + 0.015)
                    deg = float(np.degrees(np.arctan2(d[1], d[0]))) % 360
                    for stroke in (0.05, 0.08):
                        out.append({"what": "sideways", "about": name, "kind": "push" if stroke == 0.05 else "sweep",
                                    "xyz": [float(start[0]), float(start[1]), float(bp[2])], "deg": deg, "stroke": stroke})
    for name in access["covers"]:
        cp = np.asarray(cubes[name]["pos"])
        for deg in (0.0, 90.0, 180.0, 270.0):
            d = np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg))])
            start = cp[:2] - d * (HALF + 0.015)
            out.append({"what": "cover", "about": name, "kind": "sweep", "xyz": [float(start[0]), float(start[1]),
                        float(cp[2])], "deg": deg, "stroke": 0.08})
    for k in range(4):
        deg = (red["yaw"] + 90.0 * k) % 360
        d = np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg))])
        start = rp[:2] - d * (HALF + 0.015)
        out.append({"what": "red", "about": TARGET, "kind": "push", "xyz": [float(start[0]), float(start[1]),
                    float(rp[2])], "deg": deg, "stroke": 0.05})
    return out


def run(seed):
    ep = Live(f"goal{seed}", size=64, rec={"seed": seed, "buried": seed % 2 == 0, "ops": []})
    ep.run([{"op": "park", "keep_grip": True}])
    v0 = ep.see()
    if TARGET not in v0["cubes"]:
        ep.close()
        return []
    base = ep.snapshot()
    ok0, _ = pickable(ep, v0)
    red0 = v0["cubes"][TARGET]
    acc0 = tools.red_access(v0["cubes"])
    rows = []
    for s in strokes_for(v0):
        ep.restore(base)
        ep.view = v0
        stroke = s["stroke"] if s["kind"] == "sweep" else None
        flip, results = ep.choose(s["kind"], s["xyz"], s["deg"], stroke)
        if flip is None:
            continue
        p = results[flip]["push"]
        op = {"op": "act", "kind": s["kind"], "xyz": s["xyz"], "deg": s["deg"], "flip": flip, "unpark": True}
        if stroke is not None:
            op["stroke"] = stroke
        res = ep.run([op])
        spilled = bool(res["failure"])
        row = {"seed": seed, **s, "refused": bool(p["spill_risk"]), "unseen_mm": round(results[flip]["unseen_mm"], 1),
               "spilled": spilled, "pickable_before": ok0, "red_trusted_before": red0["cov"] >= tools.TRUST,
               "blocked_before": sum(n for _, n in acc0["axes"]), "covers_before": len(acc0["covers"])}
        if not spilled:
            ep.run([{"op": "park", "keep_grip": True}])
            v1 = ep.see()
            ok1, _ = pickable(ep, v1)
            red1 = v1["cubes"].get(TARGET)
            acc1 = tools.red_access(v1["cubes"]) if red1 else None
            row.update(pickable_after=ok1, red_seen_after=red1 is not None,
                       red_trusted_after=bool(red1 and red1["cov"] >= tools.TRUST),
                       blocked_after=sum(n for _, n in acc1["axes"]) if acc1 else None,
                       covers_after=len(acc1["covers"]) if acc1 else None,
                       red_moved_mm=round(float(np.linalg.norm(red1["pos"] - red0["pos"])) * 1000, 1) if red1 else None)
        rows.append(row)
    ep.close()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(5101, 5149)))
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--out", default="outputs/push_goal_2026-09-30")
    a = ap.parse_args()
    with ProcessPoolExecutor(a.procs) as pool:
        rows = [r for part in pool.map(run, a.seeds) for r in part]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rows.json").write_text(json.dumps(rows, indent=1))
    piles = sorted({r["seed"] for r in rows})
    print(f"piles with the red listed: {len(piles)}; strokes {len(rows)}")
    for what in ("sideways", "cover", "red"):
        rs = [r for r in rows if r["what"] == what and not r["refused"]]
        ok = [r for r in rs if not r["spilled"]]
        print(f"{what:9s}: {len(rs)} allowed strokes, spilled {sum(r['spilled'] for r in rs)}; "
              f"made the red pickable (was not) {sum(r['pickable_after'] and not r['pickable_before'] for r in ok)}"
              f"/{sum(not r['pickable_before'] for r in ok)}; kept it pickable {sum(r['pickable_after'] and r['pickable_before'] for r in ok)}"
              f"/{sum(r['pickable_before'] for r in ok)}; red newly trusted {sum(r['red_trusted_after'] and not r['red_trusted_before'] for r in ok)}"
              f"/{sum(not r['red_trusted_before'] for r in ok)}; blocked ends fell {sum((r['blocked_after'] or 9) < r['blocked_before'] for r in ok)}"
              f", rose {sum((r['blocked_after'] or 0) > r['blocked_before'] for r in ok)}")


if __name__ == "__main__":
    main()
