"""Pictures for docs/reports/2026-09-30-bin-depth.md, drawn with PIL (the env has no matplotlib).

    python scripts/bin_depth_figures.py            # everything
    python scripts/bin_depth_figures.py reach      # reach map along y = 0
    python scripts/bin_depth_figures.py grid       # which (floor depth, wall height) pairs work
    python scripts/bin_depth_figures.py coverage   # top views of the chosen bins, pass or why not
    python scripts/bin_depth_figures.py render     # the arm at the deepest grasp in the chosen bins

Reads outputs/bin_depth_2026-09-30/ (envelope_masks.npz, bins_summary*.json, bins_detail*.json, chosen_bins.json)
and writes PNGs back there.
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import bin_depth as bd  # noqa: E402

OUT = bd.OUT
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def font(size):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default(size=size)


# ---- reach map -------------------------------------------------------------------------------------------------------

REACH_COLORS = {
    "IK reaches, but the arm hits itself": (205, 205, 205),
    "clear, one jaw axis": (158, 202, 225),
    "clear, both jaw axes": (49, 130, 189),
    "added by a 15-45 deg lean": (116, 196, 118),
}


def reach(path=OUT / "reach_map.png", bins=()):
    d = np.load(OUT / "envelope_masks.npz")
    names = list(d["names"])
    masks = dict(zip(names, d["masks"]))
    xs, zs = d["x"], d["z"]
    S = 1400  # px per metre
    cell = int(round(0.005 * S))
    x0, x1, z0, z1 = -0.06, 0.45, -0.35, 0.15
    ml, mt, mr, mb = 80, 60, 330, 60
    W = int((x1 - x0) * S) + ml + mr
    H = int((z1 - z0) * S) + mt + mb
    img = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(img)
    px = lambda x: ml + (x - x0) * S
    pz = lambda z: mt + (z1 - z) * S

    ik_only = masks["IK only, any roll"]
    clear = masks["clear of itself, best roll"]
    both = masks["clear, both jaw axes"]
    lean = masks["clear, with leans"]
    hold = masks["holding a cube, clear"]
    for i, x in enumerate(xs):
        for j, z in enumerate(zs):
            c = None
            if both[i, j]:
                c = REACH_COLORS["clear, both jaw axes"]
            elif clear[i, j]:
                c = REACH_COLORS["clear, one jaw axis"]
            elif lean[i, j]:
                c = REACH_COLORS["added by a 15-45 deg lean"]
            elif ik_only[i, j]:
                c = REACH_COLORS["IK reaches, but the arm hits itself"]
            if c:
                dr.rectangle([px(x - 0.0025), pz(z + 0.0025), px(x + 0.0025) - 1, pz(z - 0.0025) - 1], fill=c)
    # Outline of the holding-a-cube region.
    for i, x in enumerate(xs):
        for j, z in enumerate(zs):
            if not hold[i, j]:
                continue
            for di, dj, seg in ((1, 0, "r"), (-1, 0, "l"), (0, 1, "b"), (0, -1, "t")):
                ii, jj = i + di, j + dj
                inside = 0 <= ii < len(xs) and 0 <= jj < len(zs) and hold[ii, jj]
                if inside:
                    continue
                l, r_ = px(x - 0.0025), px(x + 0.0025)
                t, b = pz(z + 0.0025), pz(z - 0.0025)
                line = {"r": [r_, t, r_, b], "l": [l, t, l, b], "t": [l, t, r_, t], "b": [l, b, r_, b]}[seg]
                dr.line(line, fill=(230, 85, 13), width=2)
    # The base, its stand-in collision box and a riser column under it.
    dr.rectangle([px(bd.RISER_LO[0]), pz(0.0), px(bd.RISER_HI[0]), pz(z0)], outline=(140, 100, 60), width=2)
    for k in range(12):
        zz = -0.02 - k * 0.03
        dr.line([px(bd.RISER_LO[0]), pz(zz), px(bd.RISER_HI[0]), pz(zz - 0.03)], fill=(190, 160, 120), width=1)
    dr.rectangle([px(0.01 - 0.045), pz(0.08), px(0.01 + 0.045), pz(0.0)], fill=(90, 90, 90))
    dr.ellipse([px(0.0388) - 5, pz(0.0624) - 5, px(0.0388) + 5, pz(0.0624) + 5], fill="white")
    # Bins as cross-sections at y = 0.
    for cand, color, label in bins:
        kind, Wd, cx, F, w = cand
        h, zf = Wd / 2, -F
        rim = zf + w
        pts = [(px(cx - h), pz(rim)), (px(cx - h), pz(zf)), (px(cx + h), pz(zf)), (px(cx + h), pz(rim))]
        dr.line(pts, fill=color, width=3)
        if kind == "flush":
            dr.line([(px(x0), pz(rim)), (px(cx - h), pz(rim))], fill=color, width=1)
            dr.line([(px(cx + h), pz(rim)), (px(x1), pz(rim))], fill=color, width=1)
        dr.text((px(cx + h) + 4, pz(zf) - 16), label, fill=color, font=font(13))
    # Axes.
    f = font(14)
    dr.rectangle([px(x0), pz(z1), px(x1), pz(z0)], outline="black", width=1)
    for x in np.arange(-0.05, x1 + 1e-9, 0.05):
        dr.line([px(x), pz(z0), px(x), pz(z0) + 6], fill="black")
        dr.text((px(x) - 14, pz(z0) + 8), f"{x:.2f}", fill="black", font=f)
    for z in np.arange(-0.35, z1 + 1e-9, 0.05):
        dr.line([px(x0) - 6, pz(z), px(x0), pz(z)], fill="black")
        dr.text((px(x0) - 52, pz(z) - 8), f"{z:+.2f}", fill="black", font=f)
        dr.line([px(x0), pz(z), px(x1), pz(z)], fill=(235, 235, 235) if abs(z) > 1e-9 else (120, 120, 120))
    dr.text((ml, 12), "Straight-down reach of the jaw midpoint along y = 0, SO-101 MJCF", fill="black", font=font(18))
    dr.text((px(0.15), pz(z0) + 30), "x from the base origin (m)", fill="black", font=f)
    dr.text((8, mt - 30), "z (m), 0 = base plane", fill="black", font=f)
    # Legend.
    lx, ly = W - mr + 20, mt + 10
    items = list(REACH_COLORS.items())
    for n, (name, c) in enumerate(items):
        dr.rectangle([lx, ly + n * 26, lx + 18, ly + n * 26 + 18], fill=c)
        dr.text((lx + 26, ly + n * 26), name, fill="black", font=f)
    n = len(items)
    dr.rectangle([lx, ly + n * 26, lx + 18, ly + n * 26 + 18], outline=(230, 85, 13), width=2)
    dr.text((lx + 26, ly + n * 26), "holding a 30 mm cube, clear", fill="black", font=f)
    n += 1
    dr.rectangle([lx, ly + n * 26, lx + 18, ly + n * 26 + 18], outline=(140, 100, 60), width=2)
    dr.text((lx + 26, ly + n * 26), "12 cm riser column", fill="black", font=f)
    n += 1
    dr.rectangle([lx, ly + n * 26, lx + 18, ly + n * 26 + 18], fill=(90, 90, 90))
    dr.text((lx + 26, ly + n * 26), "base box; dot = shoulder pan", fill="black", font=f)
    notes = ["Clear: the arm 10 mm or more from itself", "(demonstrator/reach.Clearance), for the", "better of the two wrist rolls.",
             "Jaw axis: 0 = along x, 90 = across.", "Cells are 5 mm. Source:", "scripts/bin_depth.py envelope."]
    for k, t in enumerate(notes):
        dr.text((lx, ly + (n + 2 + k) * 22), t, fill=(60, 60, 60), font=font(13))
    for k, (cand, color, label) in enumerate(bins):
        dr.line([lx, ly + (n + 9 + k) * 22 + 9, lx + 18, ly + (n + 9 + k) * 22 + 9], fill=color, width=3)
        kind, Wd, cx, F, w = cand
        dr.text((lx + 26, ly + (n + 9 + k) * 22), f"{label}: {kind}, floor -{F * 100:.0f}, wall {w * 100:.0f} cm",
                fill="black", font=font(13))
    img.save(path)
    print(f"wrote {path}")


# ---- which bins work -------------------------------------------------------------------------------------------------

def load_summary():
    rows = []
    seen = {}
    for p in sorted(OUT.glob("bins_summary*.json")):
        for s in json.loads(p.read_text()):
            seen[(s["kind"], s["W"], s["cx"], s["F"], s["w"])] = s
    return list(seen.values())


def works(s, strip_mm=20):
    """The rule the current bin passes: every position 2 cm or more from the walls picks, and the arm carries a cube
    over every position at travel height."""
    return s["interior_pass"] == s["interior"] and s["carry_all"]


def grid(path=OUT / "bin_grid.png", width=0.22):
    rows = [s for s in load_summary() if abs(s["W"] - width) < 1e-9]
    floors = sorted({s["F"] for s in rows})
    walls = sorted({s["w"] for s in rows})
    cellw, cellh = 64, 40
    ml, mt = 110, 90
    panels = ("free", "flush")
    pw = ml + cellw * len(floors) + 40
    img = Image.new("RGB", (pw * len(panels), mt + cellh * len(walls) + 150), "white")
    dr = ImageDraw.Draw(img)
    f, fs = font(14), font(12)
    dr.text((10, 10), f"{width * 100:.0f} cm bins: share of floor positions 2 cm or more from the walls that pick, best "
                      "centre x", fill="black", font=font(17))
    for p, kind in enumerate(panels):
        ox = p * pw
        dr.text((ox + ml, 40), {"free": "free-standing bin on a table at floor height",
                                "flush": "bin sunk into a table, rim flush with the top"}[kind], fill="black", font=f)
        for i, F in enumerate(floors):
            dr.text((ox + ml + i * cellw + 14, mt - 22), f"{F * 100:.0f}", fill="black", font=f)
        dr.text((ox + ml, mt + cellh * len(walls) + 8), "floor depth below the base plane (cm)", fill="black", font=f)
        for j, w in enumerate(walls):
            y = mt + (len(walls) - 1 - j) * cellh
            dr.text((ox + 40, y + 12), f"{w * 100:.0f}", fill="black", font=f)
            for i, F in enumerate(floors):
                cands = [s for s in rows if s["kind"] == kind and abs(s["F"] - F) < 1e-9 and abs(s["w"] - w) < 1e-9]
                x = ox + ml + i * cellw
                if not cands:
                    dr.rectangle([x, y, x + cellw - 2, y + cellh - 2], fill=(245, 245, 245))
                    continue
                best = max(cands, key=lambda s: (works(s), s["interior_pass"] / max(1, s["interior"]),
                                                 s["pass"] / s["positions"]))
                share = best["interior_pass"] / max(1, best["interior"])
                if works(best):
                    c = (49, 163, 84)
                elif share >= 0.9:
                    c = (186, 228, 179)
                elif share >= 0.5:
                    c = (254, 224, 139)
                else:
                    c = (244, 165, 130)
                dr.rectangle([x, y, x + cellw - 2, y + cellh - 2], fill=c)
                label = f"{share * 100:.0f}%"
                dr.text((x + 8, y + 4), label, fill="black", font=fs)
                dr.text((x + 8, y + 20), f"x {best['cx']:.2f}", fill=(60, 60, 60), font=fs)
        dr.text((ox + 4, mt - 44), "wall (cm)", fill="black", font=f)
    ly = mt + cellh * len(walls) + 40
    for k, (c, t) in enumerate((((49, 163, 84), "works: 100% and a cube carries over every spot"),
                                ((186, 228, 179), "90% or more"), ((254, 224, 139), "50 to 90%"),
                                ((244, 165, 130), "under 50%"), ((245, 245, 245), "not built: flush rim above base"))):
        dr.rectangle([20 + (k % 3) * 330, ly + (k // 3) * 26, 38 + (k % 3) * 330, ly + 18 + (k // 3) * 26], fill=c)
        dr.text((46 + (k % 3) * 330, ly + (k // 3) * 26), t, fill="black", font=fs)
    img.save(path)
    print(f"wrote {path}")


# ---- coverage maps ---------------------------------------------------------------------------------------------------

WHY_COLORS = {
    "pass": (49, 163, 84), "reach": (150, 150, 150), "finger": (253, 141, 60), "housing": (215, 48, 39),
    "mount": (117, 107, 177), "wrist": (231, 41, 138), "forearm": (140, 81, 10), "upper arm": (90, 60, 20),
    "shoulder": (60, 60, 60), "self": (33, 113, 181),
}


def coverage(path=OUT / "coverage.png", cands=None):
    import bin_depth_dynamic as dyn
    detail = dyn.load_detail()
    if cands is None:
        cands = [tuple(c) for c in json.loads((OUT / "chosen_bins.json").read_text())]
    S = 900
    pw = int(0.30 * S) + 40
    img = Image.new("RGB", (pw * len(cands) + 20, int(0.30 * S) + 190), "white")
    dr = ImageDraw.Draw(img)
    f, fs = font(14), font(12)
    for n, cand in enumerate(cands):
        kind, Wd, cx, F, w = cand
        res = detail[cand]
        ox, oy = 20 + n * pw, 70
        h = Wd / 2
        px = lambda x: ox + (x - (cx - 0.15)) * S
        py = lambda y: oy + (0.15 - y) * S
        dr.rectangle([px(cx - h), py(h), px(cx + h), py(-h)], outline="black", width=2)
        for r in res["rows"]:
            key = "pass" if r["pass"] else r["why"]
            c = WHY_COLORS.get(key, (0, 0, 0))
            rad = 0.0045 * S
            dr.rectangle([px(r["x"]) - rad, py(r["y"]) - rad, px(r["x"]) + rad, py(r["y"]) + rad], fill=c)
            if not r["carry"]:
                dr.line([px(r["x"]) - rad, py(r["y"]) - rad, px(r["x"]) + rad, py(r["y"]) + rad], fill="white", width=2)
        dr.text((ox, 10), f"{kind}, {Wd * 100:.0f} cm at x {cx:.2f}", fill="black", font=f)
        dr.text((ox, 30), f"floor -{F * 100:.0f} cm, wall {w * 100:.0f} cm, rim {-F + w:+.2f}", fill="black", font=f)
        dr.text((ox, 48), "base is to the left", fill=(90, 90, 90), font=fs)
    ly = int(0.30 * S) + 90
    for k, (name, c) in enumerate(WHY_COLORS.items()):
        x = 20 + (k % 5) * 150
        y = ly + (k // 5) * 24
        dr.rectangle([x, y, x + 16, y + 16], fill=c)
        dr.text((x + 22, y), name if name != "pass" else "picks", fill="black", font=fs)
    dr.text((20, ly + 54), "Cells are cube centres 1 cm apart. A colour other than green names what stopped the roll that "
                           "came closest. A white slash: no clear carry at travel height.", fill=(60, 60, 60), font=fs)
    img.save(path)
    print(f"wrote {path}")


# ---- renders ---------------------------------------------------------------------------------------------------------

def render(cand, xy, yaw_deg, flip, path, azimuth=90.0, elevation=-12.0, distance=0.95, lookat=None):
    """The arm at the bottom of the pick at `xy`, the -y half of the bin and table see-through."""
    import mujoco

    import bin_depth_dynamic as dyn
    from so101sim import primitives, units

    g = dyn.geometry(cand)
    model = dyn.load(cand, cube=(xy[0], xy[1], g["floor_z"] + primitives.CUBE_HALF, 0.0), clear_side=True)
    traj = primitives.pick(model, xy, np.radians(yaw_deg), hover_z=g["travel_z"],
                           z=g["floor_z"] + primitives.CUBE_HALF, flip=flip)
    probe = mujoco.MjData(model)
    zs = [primitives.tool_pose(model, r, probe)[0][2] for r in traj]
    k = [i for i in range(len(traj)) if zs[i] < min(zs) + 5e-4][-1]
    data = mujoco.MjData(model)
    joint_ids = [model.joint(n).id for n in units.JOINT_NAMES]
    grip_range = tuple(model.jnt_range[model.joint("gripper").id])
    data.qpos[model.jnt_qposadr[joint_ids]] = units.to_rad(traj[k], grip_range)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, 720, 960)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat if lookat is not None else (cand[2] - 0.03, 0.0, (g["floor_z"] + 0.1) / 2)
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    opt = mujoco.MjvOption()
    renderer.update_scene(data, camera=cam, scene_option=opt)
    Image.fromarray(renderer.render()).save(path)
    renderer.close()
    print(f"wrote {path}")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    chosen = []
    if (OUT / "chosen_bins.json").exists():
        chosen = [tuple(c) for c in json.loads((OUT / "chosen_bins.json").read_text())]
    colors = [(0, 0, 0), (203, 24, 29), (217, 95, 2), (117, 107, 177), (102, 166, 30), (231, 41, 138)]
    labels = "ABCDEF"
    bins = [(("free", 0.22, 0.20, 0.10, 0.05), colors[0], "now")] + \
        [(c, colors[1 + i % 5], labels[i]) for i, c in enumerate(chosen)]
    if which in ("reach", "all"):
        reach(bins=bins)
    if which in ("grid", "all"):
        grid()
    if which in ("coverage", "all") and chosen:
        coverage(cands=[b[0] for b in bins])


if __name__ == "__main__":
    main()
