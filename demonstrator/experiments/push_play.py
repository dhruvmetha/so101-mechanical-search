"""Hands-on push play: try several strokes from one saved state and look at what each did (2026-09-30, dm1487: "play
with pushes ... figure out when it might be useful to push and when not").

For one pile: park, observe, then for every stroke in a JSON list, restore the snapshot, run check (the push report),
make the stroke, observe again, and save a before/after picture (front camera crop and wrist camera, the stroke drawn
as an arrow). Prints what the push report predicted and what observe saw afterwards: the red's covers and blocked jaw
axis ends, spills, and how many listed cubes moved. Judged by what the cameras show, as a player would.

    python demonstrator/experiments/push_play.py --seed 5007 --kind partial --strokes strokes.json --out DIR
    strokes.json: [{"kind": "push"|"sweep", "xyz": [x, y, z], "deg": d, "stroke": s (sweeps), "why": "..."}, ...]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tools  # noqa: E402  (imports so101sim before mujoco)
from live import Live  # noqa: E402
from so101sim.mechsearch import TARGET  # noqa: E402


def access_line(v):
    if TARGET not in v["cubes"]:
        return "red not listed", None
    a = tools.red_access(v["cubes"])
    return (f"covers {len(a['covers'])}, blocked ends " + ", ".join(f"{y} deg: {n}" for y, n in a["axes"])), a


def pixel(ep, cam, p):
    pos, rot = ep.d.cam_xpos[cam.cid], ep.d.cam_xmat[cam.cid].reshape(3, 3)
    cp = rot.T @ (np.asarray(p, float) - pos)
    if cp[2] >= 0:
        return None
    return cam.w / 2 + cam.fx * cp[0] / -cp[2], cam.h / 2 - cam.fy * cp[1] / -cp[2]


def panel(ep, v, stroke=None):
    ep.view = v  # _labelled draws the labels of ep.view
    front = ep._labelled(v["rgb"], ep.front, 7).crop((225, 150, 425, 300)).resize((600, 450), Image.NEAREST)
    wrist = ep._labelled(v["wrist_rgb"], ep.wrist, 14).resize((600, 450), Image.BILINEAR)
    img = Image.new("RGB", (1200, 450))
    img.paste(front, (0, 0))
    img.paste(wrist, (600, 0))
    if stroke is not None:
        # Draw the stroke on the front crop: project start and end with the front camera.
        d = np.array([np.cos(np.radians(stroke["deg"])), np.sin(np.radians(stroke["deg"]))])
        length = stroke.get("stroke", 0.05)
        p0 = np.array(stroke["xyz"], float)
        p1 = p0.copy()
        p1[:2] += d * length
        uv = [pixel(ep, ep.front, p) for p in (p0, p1)]
        if all(u is not None for u in uv):
            (u0, v0), (u1, v1) = [((u - 225) * 3, (vv - 150) * 3) for u, vv in uv]
            dr = ImageDraw.Draw(img)
            dr.line((u0, v0, u1, v1), fill=(255, 220, 0), width=4)
            dr.ellipse((u1 - 7, v1 - 7, u1 + 7, v1 + 7), outline=(255, 220, 0), width=3)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--kind", default="partial")
    ap.add_argument("--strokes", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    strokes = json.loads(Path(a.strokes).read_text())
    ep = Live(f"play{a.seed}", size=64, rec={"seed": a.seed, "buried": a.kind == "buried", "ops": []})
    ep.run([{"op": "park", "keep_grip": True}])
    v0 = ep.see()
    base = ep.snapshot()
    line0, _ = access_line(v0)
    print(f"seed {a.seed}: before: {line0}")
    panel(ep, v0).save(out / "before.png")
    rows = []
    for k, s in enumerate(strokes):
        ep.restore(base)
        ep.view = v0
        stroke = s.get("stroke") if s["kind"] == "sweep" else None
        flip, results = ep.choose(s["kind"], s["xyz"], s["deg"], stroke)
        if flip is None:
            why = results[0].get("why") or results[1].get("why")
            print(f"[{k}] {s['why']}: refused or unreachable: {why}")
            continue
        p = results[flip]["push"]
        pred = p.get("red_access")
        op = {"op": "act", "kind": s["kind"], "xyz": s["xyz"], "deg": s["deg"], "flip": flip, "unpark": True}
        if stroke is not None:
            op["stroke"] = stroke
        res = ep.run([op])
        ep.run([{"op": "park", "keep_grip": True}])
        v1 = ep.see()
        line1, _ = access_line(v1)
        moved = sum(1 for n in v0["cubes"] if n in v1["cubes"]
                    and np.linalg.norm(v1["cubes"][n]["pos"] - v0["cubes"][n]["pos"]) > 0.01)
        refuse = " (act would refuse: spill risk)" if p["spill_risk"] else ""
        predtxt = "" if pred is None else (f"predicted covers {len(pred['before']['covers'])}->{len(pred['after']['covers'])}, "
                                          f"ends {[n for _, n in pred['before']['axes']]}->{[n for _, n in pred['after']['axes']]}")
        print(f"[{k}] {s['why']}{refuse}\n     {predtxt}\n     after: {line1}; spilled {bool(res['failure'])}; "
              f"listed cubes moved over 1 cm: {moved}")
        img = Image.new("RGB", (1200, 900))
        img.paste(panel(ep, v0, s), (0, 0))
        img.paste(panel(ep, v1), (0, 450))
        img.save(out / f"stroke{k}.png")
        rows.append({"k": k, **s, "predicted": pred, "after": line1, "spilled": bool(res["failure"]), "moved": moved})
    (out / "rows.json").write_text(json.dumps(rows, indent=1, default=str))
    ep.close()


if __name__ == "__main__":
    main()
