"""Physics check of the bins scripts/bin_depth.py scored, and the pictures for docs/reports/2026-09-30-bin-depth.md.

    python scripts/bin_depth_dynamic.py picks      # empty-bin tracking and one-cube picks in the chosen bins
    python scripts/bin_depth_dynamic.py figures    # reach map, coverage maps, renders

Reads outputs/bin_depth_2026-09-30/envelope_masks.npz and bins_detail*.json, writes back to the same folder.

A pick here is exactly what MechSearchPrimitiveEnv would run: primitives.pick with the travel height, grasp height and
wrist roll the kinematic check chose, played at 30 Hz with 10 physics steps a frame and noslip on, starting from the
hover over the target. Each test position runs twice. Empty, to see whether the arm lands where it was planned (the jaw
midpoint, computed at the gripper value IK planned with, against the target, at the last open-jaw frame at the
bottom; 6 mm is the bar the 54 of 55 empty-bin picks of 2026-09-30 used). And with one 30 mm cube, to see whether it
comes out: both jaws on it and its lowest corner over the task's success line for the last 5 frames.
"""

import argparse
import contextlib
import json
import multiprocessing as mp
import os
import sys
import tempfile
import time
from pathlib import Path

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import so101sim  # noqa: E402,F401  before mujoco
import mujoco  # noqa: E402

import bin_depth as bd  # noqa: E402
from so101sim import grasp, ik, primitives, search_layout as sl, units  # noqa: E402

OUT = bd.OUT
TRACK_MM = 6.0
SUCCESS_FRAMES = 5
SETTLE_FRAMES = 20
HOLD_FRAMES = 15


# ---- scenes ----------------------------------------------------------------------------------------------------------

def objects_xml(cand, cube=None, clear_side=None):
    """The bin, table and riser of `cand` as boxes, plus an optional cube at (x, y, z, yaw).

    `clear_side` makes the obstacle boxes on the -y side see-through, for renders.
    """
    kind, W, cx, F, w = cand
    lo, hi = bd.bin_boxes(kind, W, cx, F, w)
    out = ["<mujoco>", "  <worldbody>"]
    for n, (a, b) in enumerate(zip(lo, hi)):
        c, e = (a + b) / 2, (b - a) / 2
        alpha = 1.0
        if clear_side is not None and b[1] <= 1e-9 and a[1] < -W / 2 + 1e-9 and n > 0:
            alpha = 0.18
        rgba = "0.3 0.3 0.35" if n == 0 or (kind == "free" and n >= 3) else "0.55 0.42 0.3"
        if kind == "free" and n == 2:
            rgba = "0.45 0.35 0.25"
        out.append(f'    <geom name="obst_{n}" type="box" size="{e[0]:.6f} {e[1]:.6f} {e[2]:.6f}" '
                   f'pos="{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}" rgba="{rgba} {alpha}"/>')
    if cube is not None:
        x, y, z, yaw = cube
        out += [f'    <body name="cube" pos="{x:.6f} {y:.6f} {z:.6f}" euler="0 0 {yaw:.6f}">',
                '      <freejoint name="cube"/>',
                '      <geom name="cube" class="cube" rgba="0.85 0.15 0.15 1"/>',
                "    </body>"]
    out += ["  </worldbody>", "</mujoco>", ""]
    return "\n".join(out)


def load(cand, cube=None, clear_side=None):
    fd, objects = tempfile.mkstemp(prefix=".candidate_bindepth_", suffix="_objects.xml", dir=sl.MODELS_DIR)
    os.close(fd)
    scene = objects[: -len("_objects.xml")] + "_scene.xml"
    try:
        Path(objects).write_text(objects_xml(cand, cube, clear_side))
        Path(scene).write_text(sl.scene_xml(Path(objects).name))
        return mujoco.MjModel.from_xml_path(scene)
    finally:
        for path in (objects, scene):
            with contextlib.suppress(FileNotFoundError):
                os.unlink(path)


def geometry(cand):
    kind, W, cx, F, w = cand
    zf = -F
    rim = zf + w
    return {"floor_z": zf, "rim_z": rim, "travel_z": sl.travel_z(zf, rim),
            "success_z": max(rim + 0.01, zf + (sl.PILE_LAYERS - 1) * sl.LAYER + 0.02)}


# ---- one pick --------------------------------------------------------------------------------------------------------

def _plan_point(model, probe, row):
    """Jaw midpoint of a frame with the gripper at GRIP_PLAN, the value IK solved with."""
    r = np.array(row, dtype=float)
    r[5] = primitives.GRIP_PLAN
    return primitives.tool_pose(model, r, probe)[0]


def run_pick(task):
    """One pick of one position in one bin, empty or with a cube. Returns a dict of measurements."""
    cand, x, y, yaw_deg, flip, with_cube, label = task
    g = geometry(cand)
    target = np.array([x, y, g["floor_z"] + primitives.CUBE_HALF])
    cube = (x, y, g["floor_z"] + primitives.CUBE_HALF + 0.0005, 0.0) if with_cube else None
    model = load(cand, cube)
    base = {"cand": cand, "x": x, "y": y, "yaw": yaw_deg, "flip": flip, "cube": with_cube, "label": label}
    try:
        traj = primitives.pick(model, (x, y), np.radians(yaw_deg), hover_z=g["travel_z"], z=float(target[2]),
                               flip=flip)
    except primitives.Unreachable as exc:
        return {**base, "planned": False, "why": str(exc)}
    data = mujoco.MjData(model)
    probe = mujoco.MjData(model)
    joint_ids = [model.joint(n).id for n in units.JOINT_NAMES]
    adr = model.jnt_qposadr[joint_ids]
    grip_range = tuple(model.jnt_range[model.joint("gripper").id])
    low, high = units.lerobot_limits(model.jnt_range[joint_ids])
    n_sub = int(round((1 / 30) / model.opt.timestep))
    arm = grasp.arm_geom_ids(model)
    scene = frozenset(g_ for g_ in range(model.ngeom) if model.geom(g_).name.startswith("obst_"))
    jaws = grasp.jaw_geom_ids(model)
    cube_gid = model.geom("cube").id if with_cube else None

    q0 = units.to_rad(np.clip(traj[0], low, high), grip_range)
    data.qpos[adr] = q0
    data.ctrl[:] = q0
    mujoco.mj_forward(model, data)
    rows = [traj[0]] * SETTLE_FRAMES + list(traj) + [traj[-1]] * HOLD_FRAMES
    first = SETTLE_FRAMES
    worst_pen, touched_scene, touched_at = 0.0, False, None
    achieved, cube_z, grasped = [], [], []
    for n, row in enumerate(rows):
        data.ctrl[:] = units.to_rad(np.clip(row, low, high), grip_range)
        mujoco.mj_step(model, data, nstep=n_sub)
        for c in data.contact[: data.ncon]:
            if (c.geom1 in arm and c.geom2 in scene) or (c.geom2 in arm and c.geom1 in scene):
                worst_pen = min(worst_pen, float(c.dist))
                if not touched_scene:
                    touched_scene, touched_at = True, n - first
        if n >= first:
            cur = units.to_lerobot(data.qpos[adr], grip_range)
            achieved.append(_plan_point(model, probe, cur))
        if with_cube:
            R = data.xmat[model.body("cube").id].reshape(3, 3)
            half = primitives.CUBE_HALF * np.abs(R).sum(1)  # half extent along world axes
            cube_z.append(float(data.xpos[model.body("cube").id][2] - half[2]))
            grasped.append(grasp.is_grasped(model, data, cube_gid, jaws))
    planned = np.array([_plan_point(model, probe, r) for r in traj])
    achieved = np.array(achieved[: len(traj)])
    zmin = planned[:, 2].min()
    bottom = [i for i in range(len(traj)) if planned[i, 2] < zmin + 5e-4 and traj[i, 5] > primitives.GRIP_OPEN - 1]
    k = bottom[-1]
    err = float(np.linalg.norm(achieved[k] - target))
    path_err = float(np.linalg.norm(achieved - planned, axis=1).max())
    out = {**base, "planned": True, "frames": len(traj), "track_mm": round(err * 1000, 2),
           "track_xy_mm": round(float(np.linalg.norm(achieved[k][:2] - target[:2])) * 1000, 2),
           "path_err_mm": round(path_err * 1000, 2), "touched_scene": touched_scene, "touched_frame": touched_at,
           "worst_pen_mm": round(worst_pen * 1000, 2)}
    if with_cube:
        tail = range(len(rows) - SUCCESS_FRAMES, len(rows))
        out["held_end"] = bool(all(grasped[i] for i in tail))
        out["lowest_corner_end"] = round(cube_z[-1], 4)
        out["success"] = bool(all(grasped[i] and cube_z[i] >= g["success_z"] for i in tail))
        out["success_z"] = round(g["success_z"], 4)
    return out


# ---- choosing what to test -------------------------------------------------------------------------------------------

def load_detail():
    detail = {}
    for path in sorted(OUT.glob("bins_detail*.json")):
        detail.update({tuple(json.loads(k)): v for k, v in json.loads(path.read_text()).items()})
    return detail


def test_positions(cand, res, rng, n_random=12):
    """Positions to run: the closest passing spot to each wall and corner, the failing spot just outside it, and
    random passing spots. Each with the roll the kinematic check chose (or came closest with)."""
    rows = res["rows"]
    kind, W, cx, F, w = cand
    picks = {}

    def add(r, label):
        key = (r["x"], r["y"])
        if key not in picks and r["option"] is not None:
            a, f = r["option"]
            picks[key] = (r["x"], r["y"], bd.YAWS[a], f, label + (" pass" if r["pass"] else " fail"))

    for side, axis_ok in (("near", lambda r: abs(r["y"]) < 1e-6), ("far", lambda r: abs(r["y"]) < 1e-6),
                          ("left", lambda r: abs(r["x"] - cx) < 1e-6), ("right", lambda r: abs(r["x"] - cx) < 1e-6)):
        line = sorted([r for r in rows if axis_ok(r)], key=lambda r: r["gaps"][side])
        ok = [r for r in line if r["pass"]]
        if ok:
            add(ok[0], f"{side} wall closest")
            closer = [r for r in line if r["gaps"][side] < ok[0]["gaps"][side]]
            if closer:
                add(closer[-1], f"{side} wall next")
    for sx in ("near", "far"):
        for sy in ("left", "right"):
            corner = sorted(rows, key=lambda r: max(r["gaps"][sx], r["gaps"][sy]))
            ok = [r for r in corner if r["pass"]]
            if ok:
                add(ok[0], f"{sx}-{sy} corner closest")
            add(corner[0], f"{sx}-{sy} corner")
    passing = [r for r in rows if r["pass"] and (r["x"], r["y"]) not in picks]
    for i in rng.permutation(len(passing))[:n_random]:
        add(passing[i], "random")
    return list(picks.values())


CHOSEN = OUT / "chosen_bins.json"


def picks(workers):
    detail = load_detail()
    chosen = [tuple(c) for c in json.loads(CHOSEN.read_text())]
    rng = np.random.default_rng(0)
    tasks = []
    for cand in chosen:
        for x, y, yaw, flip, label in test_positions(cand, detail[cand], rng):
            for with_cube in (False, True):
                tasks.append((cand, x, y, yaw, flip, with_cube, label))
    print(f"{len(tasks)} pick runs over {len(chosen)} bins", flush=True)
    started = time.time()
    results = []
    with mp.get_context("spawn").Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(run_pick, tasks, chunksize=1)):
            results.append(r)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(tasks)}, {time.time() - started:.0f} s", flush=True)
    results.sort(key=lambda r: (r["cand"], r["x"], r["y"], r["cube"]))
    (OUT / "picks.json").write_text(json.dumps(results, indent=0))
    summarize_picks(results)


def summarize_picks(results=None):
    if results is None:
        results = json.loads((OUT / "picks.json").read_text())
        for r in results:
            r["cand"] = tuple(r["cand"])
    by = {}
    for r in results:
        by.setdefault(tuple(r["cand"]), []).append(r)
    lines = []
    for cand, rs in by.items():
        empty = [r for r in rs if not r["cube"]]
        full = [r for r in rs if r["cube"]]
        kp = [r for r in empty if "pass" in r["label"]]
        kf = [r for r in empty if "fail" in r["label"]]
        track = lambda rr: sum(1 for r in rr if r.get("planned") and r["track_mm"] <= TRACK_MM and not r["touched_scene"])
        succ = lambda rr: sum(1 for r in rr if r.get("success"))
        fp = [r for r in full if "pass" in r["label"]]
        ff = [r for r in full if "fail" in r["label"]]
        lines.append(f"{cand}: kinematic pass, empty bin landed within {TRACK_MM:.0f} mm with no scene contact "
                     f"{track(kp)}/{len(kp)}, cube lifted over the line {succ(fp)}/{len(fp)}; kinematic fail, "
                     f"landed {track(kf)}/{len(kf)}, lifted {succ(ff)}/{len(ff)}")
    text = "\n".join(lines)
    print(text)
    (OUT / "picks_summary.txt").write_text(text + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=("picks", "summary", "figures"))
    parser.add_argument("--workers", type=int, default=bd.WORKERS)
    args = parser.parse_args()
    if args.stage == "picks":
        picks(args.workers)
    elif args.stage == "summary":
        summarize_picks()
    else:
        import bin_depth_figures
        bin_depth_figures.main()


if __name__ == "__main__":
    main()
