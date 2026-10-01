"""Does common sense pick the push that works? (2026-09-30, dm1487: "Are we using common sense or no?")

push_goal.py ran every sideways push of every blocker, both ways, 5 and 8 cm: a blind list, and 1 stroke in 4 made the
red pickable. Here each pile is rebuilt to its start, observe looks, and for every one of those strokes we compute
what a person would check before pushing, from the cameras' height map only:
  room   share of the blocker's landing square (where it would end) that shows open floor
  finger share of the finger's starting spot that is clear down to the blocker's centre height
  other  whether the far end of that jaw axis is already clear, so moving this blocker opens the whole axis
Then in each pile take the stroke these favour (other end clear first, then room, then finger), and check its recorded
outcome. Compared with the blind hit rate and with the best stroke in hindsight.

    python demonstrator/experiments/push_choice.py --procs 12
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

import push_goal  # noqa: E402

ROWS = Path("outputs/push_goal_2026-09-30/rows.json")


def share(ep, v, centre, half, test):
    xs = np.linspace(-half, half, 5)
    pts = np.array([[centre[0] + a, centre[1] + b] for a in xs for b in xs])
    inside = (np.abs(pts - ep.bin["center"]) < ep.bin["half"] - 0.002).all(1)
    if not inside.any():
        return 0.0
    i, j = ep.cells(pts[inside])
    return float(np.mean([test(v["height"][a, b], v["unseen"][a, b]) for a, b in zip(i, j)]))


def features(seed):
    ep = Live(f"choice{seed}", size=64, rec={"seed": seed, "buried": seed % 2 == 0, "ops": []})
    ep.run([{"op": "park", "keep_grip": True}])
    v = ep.see()
    out = {}
    if TARGET in v["cubes"]:
        floor = ep.bin["floor_z"]
        ends = {round(yaw): ends for yaw, ends in tools.axis_ends(v, TARGET)}
        for s in push_goal.strokes_for(v):
            if s["what"] != "sideways":
                continue
            bp = np.asarray(v["cubes"][s["about"]]["pos"])
            d = np.array([np.cos(np.radians(s["deg"])), np.sin(np.radians(s["deg"]))])
            land = bp[:2] + d * s["stroke"]
            room = share(ep, v, land, tools.HALF, lambda h, u: (not u) and h < floor + 0.008)
            finger = share(ep, v, np.asarray(s["xyz"][:2]), 0.006, lambda h, u: (not u) and h < bp[2])
            # which axis end holds this blocker, and is the opposite end clear?
            other = False
            for yaw, e in ends.items():
                for sign, found in e.items():
                    if any(n == s["about"] for _, n, _ in found):
                        other = other or not e[-sign]
            out[(s["about"], round(s["deg"], 3), s["stroke"])] = {"room": room, "finger": finger, "other": other}
    ep.close()
    return seed, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=12)
    a = ap.parse_args()
    rows = [r for r in json.loads(ROWS.read_text()) if r["what"] == "sideways"]
    seeds = sorted({r["seed"] for r in rows if not r["pickable_before"]})
    with ProcessPoolExecutor(a.procs) as pool:
        feats = dict(pool.map(features, seeds))
    picked, blind, hindsight, n = 0, [], 0, 0
    out = []
    for seed in seeds:
        rs = [r for r in rows if r["seed"] == seed and not r["refused"] and not r["spilled"]]
        cand = []
        for r in rs:
            f = feats[seed].get((r["about"], round(r["deg"], 3), r["stroke"]))
            if f is not None:
                cand.append((f, r))
        if not cand:
            continue
        n += 1
        blind += [r["pickable_after"] for _, r in cand]
        hindsight += any(r["pickable_after"] for _, r in cand)
        f, best = max(cand, key=lambda fr: (fr[0]["other"], round(fr[0]["room"], 1), fr[0]["finger"], -fr[1]["stroke"]))
        picked += best["pickable_after"]
        out.append({"seed": seed, "chosen": {**f, "about": best["about"], "deg": best["deg"], "stroke": best["stroke"]},
                    "worked": best["pickable_after"], "any_worked": any(r["pickable_after"] for _, r in cand)})
        for fr in cand:
            fr[1]["features"] = fr[0]
    Path("outputs/push_goal_2026-09-30/choice.json").write_text(json.dumps(out, indent=1))
    print(f"piles {n}: common-sense pick worked in {picked}; a blind stroke works {np.mean(blind):.0%} of the time; "
          f"some stroke worked in {hindsight}")
    allc = [fr for s in seeds for fr in []]
    flat = [(r["features"], r["pickable_after"]) for r in rows if "features" in r]
    for name, test in (("other end clear", lambda f: f["other"]), ("room >= 0.5", lambda f: f["room"] >= 0.5),
                       ("finger >= 0.5", lambda f: f["finger"] >= 0.5)):
        yes = [w for f, w in flat if test(f)]
        no = [w for f, w in flat if not test(f)]
        print(f"  {name:16s}: worked {sum(yes)}/{len(yes)} when true, {sum(no)}/{len(no)} when false")


if __name__ == "__main__":
    main()
