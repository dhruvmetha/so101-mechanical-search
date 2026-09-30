"""Would today's check have caught the 32 misses of 2026-09-30? Each miss's scene is rebuilt from the miss study's
saved first frame (every cube's pose, the arm parked), without stepping physics, then observe's perception and check
run on the recorded pick. Reports which misses act would now refuse (aim shifted toward the moving finger, a finger on
the target's own top) and which the height-map hits flag.

    python demonstrator/experiments/check_on_misses.py --procs 8
"""
import argparse
import collections
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tools  # noqa: E402  (imports so101sim before mujoco)
import mujoco  # noqa: E402
from so101sim import units  # noqa: E402

STUDY = Path("outputs/miss_study_2026-09-30")


def classes():
    s = json.loads((STUDY / "summary.json").read_text())
    out = {}
    for cls, v in s["classes_all"].items():
        for pick in v["picks"]:
            out[pick] = cls
    return out


def run(rows):
    cls = classes()
    ep = tools.Episode("check_on_misses", size=64, rec={"seed": rows[0]["seed"], "buried": True, "ops": []})
    out = []
    for r in rows:
        z = np.load(STUDY / f"raw_{r['episode']}_{r['line']}_move{r['act']}.npz")
        d, m = ep.d, ep.m
        for k, name in enumerate(ep.env.cubes):
            a = ep.env._qadr[name]
            d.qpos[a:a + 7] = z["cubes"][0][k]
        d.qpos[ep.adr] = units.to_rad(z["js"][0], ep.grip_range)
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        ep.rec["ops"] = [{"op": "park"}]
        ep.see()
        c = ep.check("pick", r["xyz"], r["deg"], r["flip"])
        if not c["runs"]:
            out.append({"pick": f"{r['episode']}:{r['line']}:{r['act']}", "runs": False, "why": c.get("why")})
            continue
        target = c.get("target")
        hits = {k: v * 1000 for k, v in c["hits"].items() if k != target}
        refuse = bool(target) and ((c["trusted"] and c["shift_mm"] > tools.SHIFT_MAX_MM)
                                   or c["on_top_mm"] > tools.MARGIN * 1000)
        out.append({"pick": f"{r['episode']}:{r['line']}:{r['act']}",
                    "class": cls.get(f"{r['episode']}:{r['line']}:{r['act']}", "?"), "runs": True,
                    "target": target, "trusted": c.get("trusted"), "shift_mm": round(c.get("shift_mm", 0), 1),
                    "on_top_mm": round(c.get("on_top_mm", 0), 1), "worst_hit_mm": round(max(hits.values(), default=0), 1),
                    "unseen_mm": round(c["unseen_mm"], 1), "refused_now": refuse})
    ep.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    rows = [r for r in json.loads((STUDY / "rows.json").read_text()) if r["miss"]]
    chunks = [rows[k::a.procs] for k in range(a.procs)]
    with ProcessPoolExecutor(a.procs) as pool:
        res = [x for part in pool.map(run, [c for c in chunks if c]) for x in part]
    Path("outputs/check_on_misses_2026-09-30").mkdir(parents=True, exist_ok=True)
    Path("outputs/check_on_misses_2026-09-30/rows.json").write_text(json.dumps(res, indent=1))
    by = collections.defaultdict(list)
    for x in res:
        by[x.get("class", "?")].append(x)
    for k, xs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        ok = [x for x in xs if x["runs"]]
        print(f"{len(xs):2d} {k[:70]}")
        print(f"     refused now {sum(x['refused_now'] for x in ok)}; hit >= 3 mm {sum(x['worst_hit_mm'] >= 3 for x in ok)}; "
              f"unseen >= 10 mm {sum(x['unseen_mm'] >= 10 for x in ok)}; none of these "
              f"{sum(not x['refused_now'] and x['worst_hit_mm'] < 3 and x['unseen_mm'] < 10 for x in ok)}; "
              f"not runnable {len(xs) - len(ok)}")
    ok = [x for x in res if x["runs"]]
    print(f"all: {len(res)} misses, refused now {sum(x['refused_now'] for x in ok)}, flagged by a hit or unseen "
          f"{sum((not x['refused_now']) and (x['worst_hit_mm'] >= 3 or x['unseen_mm'] >= 10) for x in ok)}, silent "
          f"{sum(not x['refused_now'] and x['worst_hit_mm'] < 3 and x['unseen_mm'] < 10 for x in ok)}")


if __name__ == "__main__":
    main()
