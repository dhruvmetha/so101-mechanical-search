"""Study 01: on an empty shelf, where can the red cylinder be grasped and pulled out, and how exact must the grasp be?

    experiments/sahal/.venv/bin/python experiments/sahal/studies/01_grasp_pull/run.py [--quick] [--parts ...]
        [--workers 10] [--video]

Parts: empty (images of the shelf with only the red), single (one pick and pull at the centre, frame trace),
heatmap (grasp and pull-out success over the whole board floor), offsets (grasp point moved off the true axis).
--quick runs a small subset into outputs/quick/ for checking. See README.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # experiments/sahal, for `common`

import common  # noqa: E402,F401
import imageio  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import measures, plotting, runner  # noqa: E402
from common.shelf_env import MID_RANGE_PHYSICS, REGION_CENTRE_XY_M, SAMPLE_PHYSICS, SingleRedShelf  # noqa: E402
from so101sim.shelf import TARGET  # noqa: E402
from so101sim.shelf_layout import CYL_RADIUS  # noqa: E402

# ---- settings ------------------------------------------------------------------------------------------------
OUT_DIR = Path(__file__).resolve().parent / "outputs"
EMPTY_SHELF_SPOTS_M = [(0.36, 0.10), (0.46, -0.12), (0.32, 0.0)]
SINGLE_RED_XY_M = REGION_CENTRE_XY_M
HEATMAP_CELL_M = 0.01
HEATMAP_SEEDS = [0, 1, 2, 3, 4]  # physics drawn from the env's ranges as ShelfEnv draws them for each seed
QUICK_HEATMAP_X_M = [0.315, 0.405, 0.465]
QUICK_HEATMAP_Y_M = [-0.145, -0.125, -0.115, 0.005, 0.125, 0.135, 0.145]
QUICK_HEATMAP_SEEDS = [0, 1]
OFFSET_SEED = 0  # offsets run at MID_RANGE_PHYSICS
OFFSET_SPOTS_M = [(0.4075, 0.0), (0.38, 0.07), (0.435, -0.07), (0.435, 0.07)]
OFFSET_SMALL_MM = [2, 5]  # at every spot
OFFSET_LARGE_MM = [8, 10, 12, 15, 20]  # at the centre only
QUICK_OFFSETS = [((0.4075, 0.0), "back", 2), ((0.4075, 0.0), "front", 10), ((0.4075, 0.0), "right", 12)]
OFFSET_DIRECTIONS = {"back": (1, 0), "front": (-1, 0), "left": (0, 1), "right": (0, -1)}  # (along ray, robot's left)

_ENV: SingleRedShelf | None = None


# ---- one run ---------------------------------------------------------------------------------------------------

def ray_frame(pan_xy: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit vectors at `xy`: back (away from the robot along the approach ray) and the robot's left."""
    back = np.asarray(xy, dtype=float) - pan_xy
    back /= np.linalg.norm(back)
    return back, np.array([-back[1], back[0]])


def classify_refusal(message: str) -> str:
    if "outside the action box" in message:
        return "refused: outside action box"
    if "shelf_left" in message:
        return "refused: left wall"
    if "shelf_right" in message:
        return "refused: right wall"
    if "would hit" in message:
        return "refused: " + message.split("would hit ")[1].split(" at")[0]
    if "unreachable" in message:
        return "refused: IK"
    return "refused: " + message[:40]


def outcome_reason(record: dict) -> str:
    """One label per run: success, or where it failed and why."""
    if record["pick_refused"]:
        return classify_refusal(record["pick_refused"])
    if record["pick_failure"]:
        return "grasp: " + record["pick_failure"]
    if record["pull_success"]:
        return "success"
    if not record["grasp_success"]:
        return "grasp: " + measures.first_failed_grasp_condition(record)
    if record["pull_refused"]:
        return "pull refused: not holding"
    return "pull: " + (record["pull_failure"] or "no success, no failure")


def pick_and_pull(env: SingleRedShelf, red_xy, seed: int, physics, offset_along_ray_m: float = 0.0,
                  offset_left_m: float = 0.0) -> dict:
    """Reset with the red at `red_xy`, then ("pick", its true centre plus an offset) and ("pull", ...)."""
    placed = env.reset(seed=seed, red_xy=np.asarray(red_xy, dtype=float), physics=physics)
    true_xy = placed["pos"][:2].copy()
    back, left = ray_frame(env.geo.pan_xy, true_xy)
    grasp_xy = true_xy + offset_along_ray_m * back + offset_left_m * left
    inner = env.env
    record = {"x": round(float(red_xy[0]), 4), "y": round(float(red_xy[1]), 4), "seed": seed,
              "friction": inner.physics["friction"], "density": inner.physics["density"],
              "settled_x": round(float(true_xy[0]), 4), "settled_y": round(float(true_xy[1]), 4),
              "start_tilt_deg": round(placed["tilt"], 2), "in_action_box": env.in_strict_action_box(true_xy),
              "offset_along_ray_mm": round(offset_along_ray_m * 1000, 1), "offset_left_mm": round(offset_left_m * 1000, 1),
              "grays_on_board": placed["grays_on_board"]}
    env.phase = "pick"
    _, _, terminated, _, info = env.step(("pick", grasp_xy, 0.0))
    conditions = measures.grasp_conditions(env, info["refused"], info["failure"])
    record.update(conditions)
    record.update({"pick_refused": info["refused"], "pick_failure": info["failure"], "pick_frames": info["frames"],
                   "gripper_reading": round(float(inner.joint_state[5]), 2), "env_says_holding": env.holding(),
                   "grasp_success": measures.grasp_succeeded(conditions)})
    pose = inner.cylinder_pose(TARGET)
    record.update({"after_pick_x": round(float(pose["pos"][0]), 4), "after_pick_y": round(float(pose["pos"][1]), 4)})
    record.update({"pull_refused": None, "pull_failure": None, "pull_frames": 0, "pull_success": False})
    if info["refused"] is None and not terminated:
        env.phase = "pull"
        _, _, _, _, info = env.step(("pull", grasp_xy, 0.0))
        record.update({"pull_refused": info["refused"], "pull_failure": info["failure"],
                       "pull_frames": info["frames"], "pull_success": bool(info["is_success"])})
    pose = inner.cylinder_pose(TARGET)
    tilts = [f["tilt"] for f in env.log.frames] or [0.0]
    record.update({"final_x": round(float(pose["pos"][0]), 4), "final_z": round(float(pose["pos"][2]), 4),
                   "final_tilt_deg": round(pose["tilt"], 2), "max_tilt_deg": round(max(tilts), 2),
                   "max_contact_force_n": round(env.log.max_contact_force_n, 2),
                   "max_red_speed_m_s": round(env.log.max_red_speed_m_s, 3),
                   "glitch": measures.is_glitch(env.log.max_contact_force_n, env.log.max_red_speed_m_s, grasping=True)})
    record["reason"] = outcome_reason(record)
    return record


# ---- workers ---------------------------------------------------------------------------------------------------

def _init_widened() -> None:
    global _ENV
    _ENV = SingleRedShelf(widen_action_box=True)


def _heatmap_job(job: tuple[tuple[float, float], int]) -> dict:
    xy, seed = job
    return pick_and_pull(_ENV, xy, seed, SAMPLE_PHYSICS)


# ---- parts -----------------------------------------------------------------------------------------------------

def part_empty(out: Path) -> dict:
    """The shelf with only the red: images, and proof from the state that no gray stands on the board."""
    env = SingleRedShelf()
    try:
        placed = env.reset(seed=0)
        imageio.imwrite(out / "empty_shelf.png", env.snapshot())
        reports = [{"asked_xy": list(REGION_CENTRE_XY_M), "pos": placed["pos"].round(4).tolist(),
                    "tilt_deg": round(placed["tilt"], 3), "resting": placed["resting"],
                    "grays_on_board": placed["grays_on_board"]}]
        for xy in EMPTY_SHELF_SPOTS_M:
            p = env.place_red(xy)
            reports.append({"asked_xy": list(xy), "pos": p["pos"].round(4).tolist(), "tilt_deg": round(p["tilt"], 3),
                            "resting": p["resting"], "grays_on_board": p["grays_on_board"]})
        imageio.imwrite(out / "empty_shelf_moved.png", env.snapshot())
    finally:
        env.close()
    (out / "empty_shelf.json").write_text(json.dumps(reports, indent=1) + "\n")
    print(f"empty: grays on board {[r['grays_on_board'] for r in reports]}")
    return {"grays_on_board_max": max(r["grays_on_board"] for r in reports)}


def part_single(out: Path, video: bool) -> dict:
    """One pick and pull at the centre, with its 30 Hz trace (and a front | wrist video with --video)."""
    env = SingleRedShelf(record_video=video)
    try:
        record = pick_and_pull(env, SINGLE_RED_XY_M, OFFSET_SEED, MID_RANGE_PHYSICS)
        frames = env.log.frames
        if video:
            imageio.mimwrite(out / "single_grasp_pull.mp4", env.log.video, fps=30, quality=7)
    finally:
        env.close()
    (out / "single_record.json").write_text(json.dumps(record, indent=1, default=str) + "\n")
    runner.write_csv(out / "single_trace.csv", frames)
    plot_single_trace(out / "single_trace.png", frames)
    print(f"single: {record['reason']}, pick {record['pick_frames']} + pull {record['pull_frames']} frames, "
          f"max tilt {record['max_tilt_deg']}")
    return {k: record[k] for k in ("reason", "pick_frames", "pull_frames", "max_tilt_deg", "after_pick_x", "final_x")}


def plot_single_trace(path: Path, frames: list[dict]) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    t = np.arange(len(frames))
    axes[0].plot(t, [f["x"] for f in frames], color=plotting.DIRECTION_COLOURS["away"])
    axes[0].axhline(0.30, ls="--", c=plotting.REFERENCE_COLOURS["neutral"])
    axes[0].axhline(0.28, ls=":", c=plotting.OUTCOME_STYLE["slid"][0])
    axes[0].text(0, 0.30, " front edge", va="bottom", fontsize=8)
    axes[0].text(0, 0.28, " success line", va="top", fontsize=8)
    axes[0].set_ylabel("red x (m)")
    axes[1].plot(t, [f["tilt"] for f in frames], color=plotting.DIRECTION_COLOURS["away"])
    axes[1].set_ylabel("red tilt (deg)")
    axes[2].plot(t, [f["gripper"] for f in frames], color=plotting.DIRECTION_COLOURS["away"], label="gripper joint")
    axes[2].plot(t, [12 * f["held"] for f in frames], color=plotting.DIRECTION_COLOURS["left"],
                 label="both jaws on red (×12)")
    axes[2].set(ylabel="lerobot units", xlabel="30 Hz frame")
    axes[2].legend(fontsize=8, frameon=False)
    split = sum(1 for f in frames if f["phase"] == "pick")
    for ax in axes:
        ax.axvline(split, c="k", lw=0.8)
        plotting.tidy(ax)
    axes[0].set_title("pick | pull", fontsize=10)
    fig.tight_layout()
    plotting.save(fig, path, dpi=120)


def floor_grid(layout) -> list[tuple[float, float]]:
    """1 cm cells over every centre where an upright cylinder fits on the board."""
    xs = np.arange(layout.front_x + CYL_RADIUS, layout.back_x - CYL_RADIUS + 1e-9, HEATMAP_CELL_M)
    ys = np.arange(-layout.half_width + CYL_RADIUS, layout.half_width - CYL_RADIUS + 1e-9, HEATMAP_CELL_M)
    return [(round(float(x), 4), round(float(y), 4)) for x in xs for y in ys]


def part_heatmap(out: Path, quick: bool, workers: int) -> dict:
    """Grasp and pull-out success rates per 1 cm cell over the floor, several physics draws per cell."""
    probe = SingleRedShelf()
    layout, region = probe.env.layout, probe.env.region
    probe.close()
    if quick:
        cells = [(x, y) for x in QUICK_HEATMAP_X_M for y in QUICK_HEATMAP_Y_M]
        seeds = QUICK_HEATMAP_SEEDS
    else:
        cells, seeds = floor_grid(layout), HEATMAP_SEEDS
    jobs = [(xy, seed) for xy in cells for seed in seeds]
    rows = runner.run_parallel(jobs, _heatmap_job, _init_widened, workers, "heatmap: ")
    runner.write_csv(out / "heatmap_runs.csv", rows)
    by_cell: dict[tuple[float, float], list[dict]] = {}
    for r in rows:
        by_cell.setdefault((r["x"], r["y"]), []).append(r)

    def rate(rs: list[dict], key: str) -> float:
        valid = [r for r in rs if not r["glitch"]]
        return float(np.mean([r[key] for r in valid])) if valid else float("nan")

    grasp = {k: rate(rs, "grasp_success") for k, rs in by_cell.items()}
    pull = {k: rate(rs, "pull_success") for k, rs in by_cell.items()}
    common_reason = {k: Counter(r["reason"] for r in rs).most_common(1)[0][0] for k, rs in by_cell.items()}
    n_seeds = len(seeds)
    plotting.rate_map(out / "heatmap_grasp.png", grasp, HEATMAP_CELL_M, layout, region,
                      f"Grasp success rate by red start position ({n_seeds} physics draws per cell)")
    plotting.rate_map(out / "heatmap_pull.png", pull, HEATMAP_CELL_M, layout, region,
                      f"Pull-out success rate by red start position ({n_seeds} physics draws per cell)")
    plotting.category_map(out / "heatmap_failures.png", common_reason, HEATMAP_CELL_M, layout, region,
                          f"Most common outcome per cell ({n_seeds} runs each)")
    in_box = {k for k, rs in by_cell.items() if rs[0]["in_action_box"]}
    summary = {
        "cells": len(by_cell), "runs": len(rows), "cells_in_action_box": len(in_box),
        "glitch_runs": sum(r["glitch"] for r in rows),
        "grasp_100_cells": sum(v == 1.0 for v in grasp.values()), "pull_100_cells": sum(v == 1.0 for v in pull.values()),
        "pull_partial_cells": sum(0 < v < 1 for v in pull.values()),
        "pull_100_cells_in_box": sum(pull[k] == 1.0 for k in in_box),
        "grasp_ok_pull_failed_runs": sum(r["grasp_success"] and not r["pull_success"] for r in rows),
        "reasons": dict(Counter(r["reason"] for r in rows).most_common()),
        "max_contact_force_n": max(r["max_contact_force_n"] for r in rows),
        "max_red_speed_m_s": max(r["max_red_speed_m_s"] for r in rows),
    }
    print("heatmap:", json.dumps(summary))
    return summary


def offset_jobs(quick: bool) -> list[tuple[tuple[float, float], str, int]]:
    if quick:
        return QUICK_OFFSETS
    jobs = []
    for spot in OFFSET_SPOTS_M:
        jobs.append((spot, "back", 0))
        jobs += [(spot, name, mm) for mm in OFFSET_SMALL_MM for name in OFFSET_DIRECTIONS]
    jobs += [(OFFSET_SPOTS_M[0], name, mm) for mm in OFFSET_LARGE_MM for name in OFFSET_DIRECTIONS]
    return jobs


def part_offsets(out: Path, quick: bool) -> dict:
    """Grasp point moved off the true axis, in the approach ray's frame, at mid-range physics."""
    env = SingleRedShelf()
    rows = []
    try:
        for spot, name, mm in offset_jobs(quick):
            along, left = OFFSET_DIRECTIONS[name]
            r = pick_and_pull(env, spot, OFFSET_SEED, MID_RANGE_PHYSICS, along * mm / 1000, left * mm / 1000)
            rows.append({"offset_direction": name, "offset_mm": mm, **r})
    finally:
        env.close()
    runner.write_csv(out / "offsets_runs.csv", rows)
    plot_offsets(out / "offsets.png", rows)
    failed = [(r["x"], r["y"], r["offset_direction"], r["offset_mm"], r["reason"]) for r in rows
              if not r["pull_success"]]
    summary = {"runs": len(rows), "pull_success": sum(r["pull_success"] for r in rows), "failed": failed,
               "glitch_runs": sum(r["glitch"] for r in rows),
               "holding_but_not_both_jaws": sum(r["env_says_holding"] and not r["g_both_jaws"] for r in rows)}
    print("offsets:", json.dumps(summary))
    return summary


def plot_offsets(path: Path, rows: list[dict]) -> None:
    """Pull-out result per offset direction and size, at the centre (all sizes) and the other spots (2, 5 mm)."""
    fig, ax = plt.subplots(figsize=(8, 4.6))
    centre = [r for r in rows if (r["x"], r["y"]) == SINGLE_RED_XY_M]
    for i, name in enumerate(OFFSET_DIRECTIONS):
        rs = sorted((r for r in centre if r["offset_direction"] == name or r["offset_mm"] == 0),
                    key=lambda r: r["offset_mm"])
        for r in rs:
            ok = r["pull_success"]
            colour, marker, _ = plotting.OUTCOME_STYLE["slid" if ok else "toppled"]
            ax.scatter(r["offset_mm"], i, s=110, c=colour, marker=marker, edgecolors="white", zorder=3)
            ax.annotate(f"{r['max_tilt_deg']:.0f}°", (r["offset_mm"], i), textcoords="offset points",
                        xytext=(0, -15), ha="center", fontsize=7, color="#555")
    ax.set_yticks(range(len(OFFSET_DIRECTIONS)), [f"{n} (toward the robot)" if n == "front" else n
                                                  for n in OFFSET_DIRECTIONS])
    ax.set_ylim(-0.6, len(OFFSET_DIRECTIONS) - 0.4)
    ax.set_xlabel("grasp point offset from the cylinder axis (mm)")
    ax.legend(handles=[plt.Line2D([], [], color=plotting.OUTCOME_STYLE["slid"][0], marker="o", ls="",
                                  label="pulled out"),
                       plt.Line2D([], [], color=plotting.OUTCOME_STYLE["toppled"][0], marker="X", ls="",
                                  label="failed")], fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title("Grasp offset at the centre (0.4075, 0); label = max tilt", fontsize=10)
    plotting.tidy(ax)
    fig.tight_layout()
    plotting.save(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="small subset into outputs/quick/")
    parser.add_argument("--parts", nargs="+", default=["empty", "single", "heatmap", "offsets"],
                        choices=["empty", "single", "heatmap", "offsets"])
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--video", action="store_true", help="save single_grasp_pull.mp4 (not in git)")
    args = parser.parse_args()
    out = OUT_DIR / "quick" if args.quick else OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    if "empty" in args.parts:
        summary["empty"] = part_empty(out)
    if "single" in args.parts:
        summary["single"] = part_single(out, args.video)
    if "heatmap" in args.parts:
        summary["heatmap"] = part_heatmap(out, args.quick, args.workers)
    if "offsets" in args.parts:
        summary["offsets"] = part_offsets(out, args.quick)
    summary_path.write_text(json.dumps(summary, indent=1, default=str) + "\n")
    runner.save_config(out, {
        "quick": args.quick, "parts": args.parts, "single_red_xy_m": SINGLE_RED_XY_M,
        "heatmap_cell_m": HEATMAP_CELL_M, "heatmap_seeds": QUICK_HEATMAP_SEEDS if args.quick else HEATMAP_SEEDS,
        "offset_seed": OFFSET_SEED, "offset_physics": MID_RANGE_PHYSICS, "offset_spots_m": OFFSET_SPOTS_M,
        "offset_small_mm": OFFSET_SMALL_MM, "offset_large_mm": OFFSET_LARGE_MM,
        "grasp_centre_tolerance_m": measures.GRASP_CENTRE_TOLERANCE_M, "grasp_min_lift_m": measures.GRASP_MIN_LIFT_M,
        "grasp_max_tilt_deg": measures.GRASP_MAX_TILT_DEG,
        "glitch_grasp_contact_force_n": measures.GLITCH_GRASP_CONTACT_FORCE_N,
        "glitch_speed_m_s": measures.GLITCH_SPEED_M_S})


if __name__ == "__main__":
    main()
