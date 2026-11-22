"""Study 06: is there room for the level gripper to enter ABOVE the 8 cm cylinders and drop behind one?

    experiments/sahal/.venv/bin/python experiments/sahal/studies/06_entry_above/run.py [--quick] [--workers 10]

Kinematic only, no physics, the sweep skill unchanged. For jaw-midpoint heights between the cylinder tops and the top
board, and targets across the board, solve level IK with the jaws closed (the repo's `ik.solve_side`, accepted with
side_primitives' own tolerances). For each pose that solves, take every point of every arm geom (mesh vertices, box
corners, sphere and capsule extremes) that lies over the board (front_x <= x <= back_x) and measure the gap from the
lowest one down to the cylinder tops and from the highest one up to the top board's underside. Entry above works at a
height where IK solves and both gaps are at least the path check's 6 mm. --quick runs a coarse grid into
outputs/quick/. See README.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # experiments/sahal, for `common`

import common  # noqa: E402,F401
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from common import measures, plotting, runner  # noqa: E402
from common.shelf_env import SingleRedShelf  # noqa: E402
from so101sim import ik, side_primitives, units  # noqa: E402
from so101sim.primitives import GRIP_CLOSE, _grip_rad  # noqa: E402
from so101sim.shelf_layout import CYL_HALF_HEIGHT, CYL_RADIUS  # noqa: E402

# ---- settings ------------------------------------------------------------------------------------------------
OUT_DIR = Path(__file__).resolve().parent / "outputs"
CYLINDER_HEIGHT_M = 2 * CYL_HALF_HEIGHT
PATH_CHECK_MARGIN_M = 0.006  # the env's path check keeps the arm this far from the shelf
JAW_HEIGHTS_MM = list(range(80, 151, 5))  # jaw midpoint above the board
GRID_STEP_M = 0.01
QUICK_JAW_HEIGHTS_MM = [90, 110, 130]
QUICK_GRID_STEP_M = 0.04

_ENV: SingleRedShelf | None = None


def _init() -> None:
    global _ENV
    _ENV = SingleRedShelf()


def arm_points(model: mujoco.MjModel, data: mujoco.MjData, geoms: frozenset[int]) -> tuple[np.ndarray, list[str]]:
    """World points bounding every arm geom (mesh vertices, box corners, sphere and capsule/cylinder extremes), and
    the label of the geom each point belongs to."""
    points, labels = [], []
    for g in sorted(geoms):
        centre, rotation, size = data.geom_xpos[g], data.geom_xmat[g].reshape(3, 3), model.geom_size[g]
        kind = model.geom_type[g]
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[g]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            points.append(centre + model.mesh_vert[start:start + count] @ rotation.T)
        elif kind == mujoco.mjtGeom.mjGEOM_BOX:
            corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * size
            points.append(centre + corners @ rotation.T)
        elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            points.append(centre + np.vstack([np.eye(3), -np.eye(3)]) * size[0])
        elif kind in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
            ends = centre + np.outer([-1, 1], rotation[:, 2] * size[1])
            points.append((ends[:, None, :] + np.vstack([np.eye(3), -np.eye(3)])[None] * size[0]).reshape(-1, 3))
        else:
            continue
        labels += [measures.geom_label(model, g)] * len(points[-1])
    return np.vstack(points), labels


def pose_job(job: tuple[float, float, int]) -> dict:
    """Level IK at one target; if it solves, the arm's lowest and highest points over the board."""
    x, y, height_mm = job
    env = _ENV
    model, layout = env.model, env.env.layout
    target = np.array([x, y, layout.board_z + height_mm / 1000])
    grip = _grip_rad(model, GRIP_CLOSE)
    q, info = ik.solve_side(model, target, gripper=grip)
    record = {"x_m": round(x, 4), "y_m": round(y, 4), "jaw_height_mm": height_mm, "ik_ok": bool(side_primitives._ok(info)),
              "pos_err_mm": round(info["pos_err"] * 1000, 2), "tilt_deg": round(info["tilt"], 2)}
    if not record["ik_ok"]:
        return record
    data = mujoco.MjData(model)
    joints = model.jnt_qposadr[[model.joint(n).id for n in units.JOINT_NAMES]]
    data.qpos[joints[:5]], data.qpos[joints[5]] = q, grip
    mujoco.mj_kinematics(model, data)
    points, labels = arm_points(model, data, env.env._arm_ids)
    inside = np.flatnonzero((points[:, 0] >= layout.front_x) & (points[:, 0] <= layout.back_x))
    low_index, high_index = inside[points[inside, 2].argmin()], inside[points[inside, 2].argmax()]
    lowest, highest = float(points[low_index, 2]), float(points[high_index, 2])
    record.update({
        "lowest_over_board_mm": round((lowest - layout.board_z) * 1000, 1),
        "highest_over_board_mm": round((highest - layout.board_z) * 1000, 1),
        "gap_to_cylinder_tops_mm": round((lowest - layout.board_z - CYLINDER_HEIGHT_M) * 1000, 1),
        "gap_to_top_board_mm": round((layout.opening - (highest - layout.board_z)) * 1000, 1),
        "jaw_midpoint_to_lowest_mm": round((target[2] - lowest) * 1000, 1),
        "highest_to_jaw_midpoint_mm": round((highest - target[2]) * 1000, 1),
        "lowest_geom": labels[low_index], "highest_geom": labels[high_index],
    })
    margin_mm = PATH_CHECK_MARGIN_M * 1000
    record["fits"] = (record["gap_to_cylinder_tops_mm"] >= margin_mm and record["gap_to_top_board_mm"] >= margin_mm)
    return record


def build_jobs(layout, quick: bool) -> list[tuple[float, float, int]]:
    """Targets where a cylinder centre could stand (CYL_RADIUS in from the front edge, back and side walls)."""
    step = QUICK_GRID_STEP_M if quick else GRID_STEP_M
    xs = np.arange(layout.front_x + CYL_RADIUS, layout.back_x - CYL_RADIUS + 1e-9, step)
    ys = np.arange(-layout.half_width + CYL_RADIUS, layout.half_width - CYL_RADIUS + 1e-9, step)
    heights = QUICK_JAW_HEIGHTS_MM if quick else JAW_HEIGHTS_MM
    return [(float(x), float(y), h) for h in heights for x in xs for y in ys]


def summarise(rows: list[dict], layout, action_box_front_x_m: float) -> dict:
    by_height = {}
    for h in sorted({r["jaw_height_mm"] for r in rows}):
        rs = [r for r in rows if r["jaw_height_mm"] == h]
        ok = [r for r in rs if r["ik_ok"]]
        by_height[h] = {
            "targets": len(rs), "ik_ok": len(ok), "fits": sum(r.get("fits", False) for r in ok),
            "gap_to_cylinder_tops_mm_min": min((r["gap_to_cylinder_tops_mm"] for r in ok), default=None),
            "gap_to_top_board_mm_min": min((r["gap_to_top_board_mm"] for r in ok), default=None),
            "fits_x_m_range": ([min(r["x_m"] for r in ok if r["fits"]), max(r["x_m"] for r in ok if r["fits"])]
                               if any(r["fits"] for r in ok) else None),
            "limits_top_board": _count(r["highest_geom"] for r in ok if r["gap_to_top_board_mm"] < 6),
            "limits_cylinder_tops": _count(r["lowest_geom"] for r in ok if r["gap_to_cylinder_tops_mm"] < 6),
        }
    ok = [r for r in rows if r["ik_ok"]]
    deep = [r for r in ok if r["x_m"] >= action_box_front_x_m - 1e-9]
    return {
        "board_top_z_m": layout.board_z, "cylinder_tops_above_board_mm": round(CYLINDER_HEIGHT_M * 1000, 1),
        "top_board_underside_above_board_mm": round(layout.opening * 1000, 1),
        "room_between_cylinder_tops_and_top_board_mm": round((layout.opening - CYLINDER_HEIGHT_M) * 1000, 1),
        "arm_extent_below_jaw_midpoint_mm_range": [min(r["jaw_midpoint_to_lowest_mm"] for r in ok),
                                                   max(r["jaw_midpoint_to_lowest_mm"] for r in ok)] if ok else None,
        "arm_extent_above_jaw_midpoint_mm_range": [min(r["highest_to_jaw_midpoint_mm"] for r in ok),
                                                   max(r["highest_to_jaw_midpoint_mm"] for r in ok)] if ok else None,
        "arm_vertical_extent_over_board_mm_range": _range([round(r["highest_over_board_mm"] - r["lowest_over_board_mm"], 1)
                                                           for r in ok]),
        "in_action_box": {  # targets at least as deep as the env's action box, where the cylinders start
            "front_x_m": round(action_box_front_x_m, 4), "solved_poses": len(deep),
            "arm_extent_below_jaw_midpoint_mm_range": _range([r["jaw_midpoint_to_lowest_mm"] for r in deep]),
            "arm_extent_above_jaw_midpoint_mm_range": _range([r["highest_to_jaw_midpoint_mm"] for r in deep]),
            "arm_vertical_extent_over_board_mm_range": _range([round(r["highest_over_board_mm"]
                                                                     - r["lowest_over_board_mm"], 1) for r in deep]),
            "highest_geoms": _count(r["highest_geom"] for r in deep),
            "fits": sum(r["fits"] for r in deep),
            "best_gap_to_top_board_mm_with_cylinder_gap_6mm": max((r["gap_to_top_board_mm"] for r in deep
                                                                   if r["gap_to_cylinder_tops_mm"] >= 6), default=None),
        },
        "ik_failures_x_m_min": min((r["x_m"] for r in rows if not r["ik_ok"]), default=None),
        "targets_per_height": len(rows) // len(by_height),
        "any_fit": any(r.get("fits", False) for r in rows),
        "by_height": by_height,
    }


def _range(values: list[float]) -> list[float] | None:
    return [min(values), max(values)] if values else None


def _count(values) -> dict:
    counts: dict = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def plot(path: Path, rows: list[dict], layout) -> None:
    """Left: share of targets where level IK solves, and where it also fits, per jaw height. Right: the arm's lowest
    and highest point over the board (median over solved targets) against jaw height, with cylinder tops and top
    board."""
    heights = sorted({r["jaw_height_mm"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ok_share = [100 * np.mean([r["ik_ok"] for r in rows if r["jaw_height_mm"] == h]) for h in heights]
    fit_share = [100 * np.mean([r.get("fits", False) for r in rows if r["jaw_height_mm"] == h]) for h in heights]
    axes[0].plot(heights, ok_share, color=plotting.BLUE_RAMP[-1], marker="o", ms=6, lw=2, mec="white",
                 label="level IK solves")
    fit_depths = [r["x_m"] - layout.front_x for r in rows if r.get("fits")]
    fit_label = "solves and clears both by 6 mm" + (f" (all ≤ {100 * max(fit_depths):.1f} cm past the front edge)"
                                                    if fit_depths else "")
    axes[0].plot(heights, fit_share, color=plotting.OUTCOME_STYLE["slid"][0], marker="o", ms=6, lw=2, mec="white",
                 label=fit_label)
    axes[0].set(xlabel="jaw midpoint above the board (mm)", ylabel="% of targets on the board", ylim=(-3, 103),
                title="Reach inside the shelf, level gripper, jaws closed")
    axes[0].legend(frameon=False, fontsize=8)
    for key, label, colour in (("lowest_over_board_mm", "arm's lowest point over the board", plotting.BLUE_RAMP[-1]),
                               ("highest_over_board_mm", "arm's highest point over the board", plotting.BLUE_RAMP[2])):
        values = [[r[key] for r in rows if r["jaw_height_mm"] == h and r["ik_ok"]] for h in heights]
        shown = [(h, np.median(v), np.min(v), np.max(v)) for h, v in zip(heights, values) if v]
        if shown:
            hs, med, low, high = map(np.array, zip(*shown))
            axes[1].fill_between(hs, low, high, color=colour, alpha=0.2, lw=0)
            axes[1].plot(hs, med, color=colour, marker="o", ms=5, lw=2, mec="white", label=f"{label} (median, range)")
    axes[1].axhline(CYLINDER_HEIGHT_M * 1000, color=plotting.OUTCOME_STYLE["toppled"][0], ls="--", lw=1.2,
                    label="cylinder tops")
    axes[1].axhline(layout.opening * 1000, color=plotting.REFERENCE_COLOURS["neutral"], ls="--", lw=1.2,
                    label="top board underside")
    axes[1].set(xlabel="jaw midpoint above the board (mm)", ylabel="height above the board (mm)",
                title="Room above the cylinders")
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        plotting.tidy(ax)
    fig.tight_layout()
    plotting.save(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="coarse grid into outputs/quick/")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    out = OUT_DIR / "quick" if args.quick else OUT_DIR
    probe = SingleRedShelf()
    layout, action_box_front_x_m = probe.env.layout, float(probe.strict_action_box.low[0])
    probe.close()
    rows = runner.run_parallel(build_jobs(layout, args.quick), pose_job, _init, args.workers, "poses: ")
    runner.write_csv(out / "poses.csv", rows)
    summary = summarise(rows, layout, action_box_front_x_m)
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    plot(out / "entry_above.png", rows, layout)
    runner.save_config(out, {"quick": args.quick, "jaw_heights_mm": QUICK_JAW_HEIGHTS_MM if args.quick else JAW_HEIGHTS_MM,
                             "grid_step_m": QUICK_GRID_STEP_M if args.quick else GRID_STEP_M,
                             "cylinder_height_m": CYLINDER_HEIGHT_M, "path_check_margin_m": PATH_CHECK_MARGIN_M,
                             "grip": GRIP_CLOSE})


if __name__ == "__main__":
    main()
