"""Study 05: does the sweep skill move a lone cylinder where it is told without toppling it, and where is it refused?

    experiments/sahal/.venv/bin/python experiments/sahal/studies/05_sweep_skill/run.py [--quick] [--parts ...]
        [--workers 10]

Parts: toward_robot (the same sweeps at 180 deg, which the skill refuses by default, planned with
allow_toward_robot=True to keep the evidence for that refusal), sanity (sweep() and sweep_points() for the same motion give the same joint trajectory) and sweeps (the red
alone at five spots, swept in the skill's 7 default directions for 3, 6 and 10 cm at friction 0.55 and 0.7, jittered
repeats). The env's checks are all on (nothing bypassed). Runs jammed against a wall are failures; solver blow-ups
(glitches) are excluded from rates and means.
--quick runs a subset into outputs/quick/. See README.md.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # experiments/sahal, for `common` and `skills`

import common  # noqa: E402,F401
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from common import measures, plotting, runner  # noqa: E402
from common.shelf_env import PUSH_START_GAP_M, SingleRedShelf, push_start_back_distance, seeded_jitter  # noqa: E402
from skills import sweep as sweep_skill  # noqa: E402
from so101sim.primitives import Unreachable  # noqa: E402
from so101sim.shelf import TARGET, TOPPLE_DEG  # noqa: E402

# ---- settings ------------------------------------------------------------------------------------------------
OUT_DIR = Path(__file__).resolve().parent / "outputs"
STUDY_SEED = 5
SPOTS_M = {"centre": (0.4075, 0.0), "near left wall": (0.4075, 0.12), "near right wall": (0.4075, -0.12),
           "front": (0.33, 0.0), "back": (0.45, 0.0)}
DIRECTIONS_DEG = list(sweep_skill.DEFAULT_DIRECTIONS_DEG)  # 0 = away from the robot, 90 = robot's left; no 180
LENGTHS_M = [0.03, 0.06, 0.10]
FRICTIONS = [0.55, 0.7]
DENSITY_KG_M3 = 900.0
TOWARD_ROBOT_DEG = 180
REPETITIONS = 3
RESET_SEED = 0
QUICK_SPOTS = ["centre", "near right wall"]
QUICK_DIRECTIONS_DEG = [0, 90, 270]
QUICK_LENGTHS_M = [0.06]
QUICK_FRICTIONS = [0.7]
QUICK_REPETITIONS = 1
# sanity: (x_start, y_start, x_end, y_end) metres
SANITY_MOTIONS_M = [(0.38, 0.0, 0.44, 0.0), (0.40, 0.05, 0.40, -0.03), (0.37, -0.06, 0.43, 0.0)]
SANITY_TOLERANCE_DEG = 1e-6

_ENV: SingleRedShelf | None = None


# ---- sanity ----------------------------------------------------------------------------------------------------

def part_sanity(out: Path) -> dict:
    """sweep() with the direction and duration of a sweep_points() motion must give the same joint trajectory."""
    env = SingleRedShelf()
    results = []
    try:
        env.reset(seed=RESET_SEED, red_xy=(0.33, 0.14))
        for x0, y0, x1, y1 in SANITY_MOTIONS_M:
            by_points = sweep_skill.sweep_points(env, x0, y0, x1, y1)
            by_direction = sweep_skill.sweep(env, x0, y0, by_points.direction_rad, by_points.duration_s)
            same_shape = by_points.trajectory.shape == by_direction.trajectory.shape
            difference = (float(np.abs(by_points.trajectory - by_direction.trajectory).max()) if same_shape
                          else float("inf"))
            results.append({"motion_m": [x0, y0, x1, y1], "direction_deg": round(math.degrees(by_points.direction_rad), 3),
                            "duration_s": round(by_points.duration_s, 4), "frames": len(by_points.trajectory),
                            "height_mm": round(by_points.height_m * 1000), "max_joint_difference_deg": difference,
                            "identical": same_shape and difference <= SANITY_TOLERANCE_DEG})
    finally:
        env.close()
    (out / "sanity.json").write_text(json.dumps(results, indent=1) + "\n")
    print("sanity:", json.dumps(results))
    return {"all_identical": all(r["identical"] for r in results), "motions": results}


# ---- sweeps ----------------------------------------------------------------------------------------------------

def _init() -> None:
    global _ENV
    _ENV = SingleRedShelf()


def sweep_job(job: tuple[str, int, float, float, int]) -> dict:
    """Reset with the red at a spot, plan a sweep from just behind it, run it, measure what the red did. Sweeps toward
    the robot are planned with allow_toward_robot=True (they only come from the toward_robot part)."""
    spot, direction_deg, length_m, friction, repetition = job
    env = _ENV
    heading = math.radians(direction_deg)
    unit = np.array([math.cos(heading), math.sin(heading)])
    placed = env.reset(seed=RESET_SEED, red_xy=SPOTS_M[spot],
                       physics={"friction": friction, "density": DENSITY_KG_M3})
    red_start = placed["pos"][:2].copy()
    record = {"spot": spot, "direction_deg": direction_deg, "length_cm": round(length_m * 100, 1),
              "friction": friction, "repetition": repetition}
    try:
        back = push_start_back_distance(env, red_start, heading, sweep_skill.SWEEP_HEIGHTS_M[0])
    except Unreachable as exc:  # level IK can't reach the point behind the red: the start itself is out of reach
        return {**record, "executed": False, "refused": f"IK, start behind the red unreachable: {exc}",
                "refusal_kind": "IK"}
    jitter = seeded_jitter([STUDY_SEED, list(SPOTS_M).index(spot), direction_deg, int(round(length_m * 1000)),
                            int(round(friction * 100)), repetition])
    start = red_start - back * unit + jitter
    record.update({"start_back_mm": round(back * 1000, 1), "jitter_x_mm": round(float(jitter[0]) * 1000, 2),
                   "jitter_y_mm": round(float(jitter[1]) * 1000, 2)})
    planning_started = time.perf_counter()
    plan = sweep_skill.sweep(env, float(start[0]), float(start[1]), heading, length_m / sweep_skill.CONTACT_SPEED_M_S,
                             allow_toward_robot=sweep_skill.toward_robot(heading))
    record.update({"duration_s": round(plan.duration_s, 3), "plan_wall_s": round(time.perf_counter() - planning_started, 3),
                   "refused": plan.refused,
                   "height_mm": round(plan.height_m * 1000) if plan.height_m is not None else None})
    if plan.refused is not None:
        return {**record, "executed": False, "refusal_kind": refusal_kind(plan.refused)}
    running_started = time.perf_counter()
    info = sweep_skill.execute(env, plan)
    log = env.log
    pose = env.env.cylinder_pose(TARGET)
    moved = pose["pos"][:2] - red_start
    intended = (length_m - PUSH_START_GAP_M) * unit
    tilts = [f["tilt"] for f in log.frames] or [0.0]
    jammed = measures.is_wall_jam(log.max_contact_force_on_wall_n, log.arm_shelf_touch)
    glitch = measures.is_glitch(log.max_contact_force_n, log.max_red_speed_m_s) and not jammed
    max_tilt = max(tilts)
    moved_norm = float(np.linalg.norm(moved))
    record.update({
        "executed": info["refused"] is None, "refused": info["refused"],
        "refusal_kind": refusal_kind(info["refused"]) if info["refused"] else None,
        "failure": info["failure"], "max_tilt_deg": round(max_tilt, 2), "final_tilt_deg": round(pose["tilt"], 2),
        "glitch": glitch, "jammed": jammed, "toppled": bool(max_tilt > TOPPLE_DEG and not glitch),
        "outcome": measures.tilt_outcome(max_tilt, glitch, jammed),
        "intended_dx_mm": round(float(intended[0]) * 1000, 1), "intended_dy_mm": round(float(intended[1]) * 1000, 1),
        "moved_dx_mm": round(float(moved[0]) * 1000, 1), "moved_dy_mm": round(float(moved[1]) * 1000, 1),
        "moved_mm": round(moved_norm * 1000, 1), "moved_along_mm": round(float(moved @ unit) * 1000, 1),
        "sideways_drift_mm": round(float(moved @ np.array([-unit[1], unit[0]])) * 1000, 1),
        "distance_ratio": round(float(moved @ unit) / float(np.linalg.norm(intended)), 3),
        "direction_error_deg": (round(math.degrees(math.atan2(moved @ np.array([-unit[1], unit[0]]), moved @ unit)), 1)
                                if moved_norm > 0.002 else None),
        "robot_time_s": round(info["frames"] / 30, 2), "slow_time_s": round(plan.slow_frames / 30, 2),
        "run_wall_s": round(time.perf_counter() - running_started, 2),
        "max_contact_force_n": round(log.max_contact_force_n, 2), "max_red_speed_m_s": round(log.max_red_speed_m_s, 3),
        "max_contact_force_on_wall_n": round(log.max_contact_force_on_wall_n, 2), "red_wall_touch": log.red_wall_touch,
        "arm_shelf_touch": log.arm_shelf_touch,
        "first_contact_part": measures.contact_part(log.arm_contacts[0].geom) if log.arm_contacts else None,
    })
    return record


def refusal_kind(reason: str) -> str:
    """A short label for a refusal reason: off the board, IK, or which shelf part the path check would hit."""
    if "toward the robot" in reason:
        return "toward the robot"
    if "off the board" in reason:
        return "start off board" if reason.startswith("start") else "end off board"
    for part in ("shelf_left", "shelf_right", "shelf_back", "shelf_board", "shelf_top", "table"):
        if part in reason:
            return f"path check: {part}"
    return "IK" if "IK" in reason else reason[:40]


def build_jobs(quick: bool, directions: list[int] | None = None) -> list[tuple]:
    spots = QUICK_SPOTS if quick else list(SPOTS_M)
    directions = directions or (QUICK_DIRECTIONS_DEG if quick else DIRECTIONS_DEG)
    lengths = QUICK_LENGTHS_M if quick else LENGTHS_M
    frictions = QUICK_FRICTIONS if quick else FRICTIONS
    reps = range(QUICK_REPETITIONS if quick else REPETITIONS)
    return [(s, d, length, mu, k) for s in spots for d in directions for length in lengths for mu in frictions
            for k in reps]


def part_sweeps(out: Path, quick: bool, workers: int) -> dict:
    rows = runner.run_parallel(build_jobs(quick), sweep_job, _init, workers, "sweeps: ")
    runner.write_csv(out / "sweep_runs.csv", rows)
    executed = [r for r in rows if r["executed"]]
    valid = [r for r in executed if not r["glitch"]]
    clean = [r for r in valid if not r["jammed"] and r["failure"] is None]  # for movement: no failure of any kind
    cells = []
    for spot in sorted({r["spot"] for r in rows}, key=list(SPOTS_M).index):
        for d in sorted({r["direction_deg"] for r in rows}):
            for length in sorted({r["length_cm"] for r in rows}):
                for mu in sorted({r["friction"] for r in rows}):
                    rs = [r for r in rows if (r["spot"], r["direction_deg"], r["length_cm"], r["friction"])
                          == (spot, d, length, mu)]
                    ex = [r for r in rs if r["executed"]]
                    cells.append({"spot": spot, "direction_deg": d, "length_cm": length, "friction": mu,
                                  "runs": len(rs), "refused": len(rs) - len(ex),
                                  "refusal_kinds": ";".join(sorted({r["refusal_kind"] for r in rs if not r["executed"]})),
                                  **({k: round(v, 3) for k, v in measures.cell_stats(ex).items()} if ex else {})})
    runner.write_csv(out / "sweep_cells.csv", cells)
    plot_rates(out / "sweep_rates.png", rows)
    plot_movement(out / "sweep_movement.png", rows)
    along_ratio = [r["distance_ratio"] for r in clean]
    jams = [r for r in executed if r["jammed"]]
    summary = {
        "runs": len(rows), "executed": len(executed), "refused": len(rows) - len(executed),
        "glitch_runs": sum(r["glitch"] for r in executed),
        "glitches": [(r["spot"], r["direction_deg"], r["length_cm"], r["friction"], r["max_contact_force_n"],
                      r["max_red_speed_m_s"]) for r in executed if r["glitch"]],
        "jammed": len(jams), "jammed_by_spot_direction": _count(f'{r["spot"]} {r["direction_deg"]}' for r in jams),
        "jammed_arm_on_wall_or_top": _count(r["arm_shelf_touch"] for r in jams if r["arm_shelf_touch"]),
        "jammed_force_on_wall_n_range": ([min(r["max_contact_force_on_wall_n"] for r in jams),
                                          max(r["max_contact_force_on_wall_n"] for r in jams)] if jams else None),
        "jammed_red_wall": _count(r["red_wall_touch"] for r in jams),
        "toppled": sum(r["toppled"] for r in valid), "toppled_and_jammed": sum(r["toppled"] and r["jammed"] for r in valid),
        "outcomes": _count(r["outcome"] for r in executed),
        "clean_runs": len(clean),
        "leaned": sum(r["outcome"] == "leaned" for r in valid), "slid": sum(r["outcome"] == "slid" for r in valid),
        "failures": _count(r["failure"] for r in executed if r["failure"]),
        "refusals_by_kind": _count(r["refusal_kind"] for r in rows if not r["executed"]),
        "refusals_by_spot": {s: sum(not r["executed"] for r in rows if r["spot"] == s) for s in SPOTS_M
                             if any(r["spot"] == s for r in rows)},
        "runs_by_spot": {s: sum(r["spot"] == s for r in rows) for s in SPOTS_M if any(r["spot"] == s for r in rows)},
        "height_mm_used": _count(r["height_mm"] for r in executed),
        "max_tilt_deg_non_glitch": max((r["max_tilt_deg"] for r in valid), default=None),
        "mean_max_tilt_deg": round(float(np.mean([r["max_tilt_deg"] for r in valid])), 2) if valid else None,
        "distance_ratio_median": round(float(np.median(along_ratio)), 3) if along_ratio else None,
        "distance_ratio_p10_p90": ([round(float(np.percentile(along_ratio, p)), 3) for p in (10, 90)]
                                   if along_ratio else None),
        "sideways_drift_mm_abs_median": _median([abs(r["sideways_drift_mm"]) for r in clean], 1),
        "sideways_drift_mm_abs_p90": (round(float(np.percentile([abs(r["sideways_drift_mm"]) for r in clean], 90)), 1)
                                      if clean else None),
        "direction_error_deg_abs_median": _median([abs(r["direction_error_deg"]) for r in clean
                                                   if r["direction_error_deg"] is not None], 1),
        "robot_time_s_by_length": {length: round(float(np.mean([r["robot_time_s"] for r in executed
                                                                 if r["length_cm"] == length])), 1)
                                   for length in sorted({r["length_cm"] for r in executed})},
        "slow_time_s_by_length": {length: round(float(np.mean([r["slow_time_s"] for r in executed
                                                                if r["length_cm"] == length])), 1)
                                  for length in sorted({r["length_cm"] for r in executed})},
        "by_direction": {d: {"executed": sum(r["direction_deg"] == d for r in executed),
                             "refused": sum(r["direction_deg"] == d and not r["executed"] for r in rows),
                             "clean_runs": sum(r["direction_deg"] == d for r in clean),
                             "distance_ratio_median": _median([r["distance_ratio"] for r in clean
                                                               if r["direction_deg"] == d], 3),
                             "distance_ratio_range": _range([r["distance_ratio"] for r in clean
                                                             if r["direction_deg"] == d]),
                             "sideways_drift_mm_median": _median([r["sideways_drift_mm"] for r in clean
                                                                  if r["direction_deg"] == d], 1),
                             "sideways_drift_mm_range": _range([r["sideways_drift_mm"] for r in clean
                                                                if r["direction_deg"] == d]),
                             "topple_rate": round(float(np.mean([r["toppled"] for r in valid if r["direction_deg"] == d])), 3),
                             "jam_rate": round(float(np.mean([r["jammed"] for r in valid if r["direction_deg"] == d])), 3),
                             "mean_max_tilt_deg": round(float(np.mean([r["max_tilt_deg"] for r in valid
                                                                       if r["direction_deg"] == d])), 2)}
                         for d in sorted({r["direction_deg"] for r in valid})},
        "plan_wall_s_median": round(float(np.median([r["plan_wall_s"] for r in rows if "plan_wall_s" in r])), 2),
        "run_wall_s_median": round(float(np.median([r["run_wall_s"] for r in executed])), 2) if executed else None,
    }
    print("sweeps:", json.dumps(summary, default=str))
    return summary


def part_toward_robot(out: Path, quick: bool, workers: int) -> dict:
    """Sweeps at 180 deg (toward the robot), which the skill refuses by default: what happens if they run."""
    rows = runner.run_parallel(build_jobs(quick, [TOWARD_ROBOT_DEG]), sweep_job, _init, workers, "toward robot: ")
    runner.write_csv(out / "toward_robot_runs.csv", rows)
    executed = [r for r in rows if r["executed"]]
    valid = [r for r in executed if not r["glitch"]]
    summary = {"runs": len(rows), "executed": len(executed), "glitch_runs": len(executed) - len(valid),
               "toppled": sum(r["toppled"] for r in valid), "jammed": sum(r["jammed"] for r in valid),
               "outcomes_by_spot": {s: _count(r["outcome"] for r in executed if r["spot"] == s)
                                    for s in SPOTS_M if any(r["spot"] == s for r in executed)},
               "refusals_by_kind": _count(r["refusal_kind"] for r in rows if not r["executed"]),
               "first_contact_part": _count(r["first_contact_part"] for r in executed)}
    print("toward robot:", json.dumps(summary, default=str))
    return summary


def _median(values: list[float], digits: int) -> float | None:
    return round(float(np.median(values)), digits) if values else None


def _range(values: list[float]) -> list[float] | None:
    return [min(values), max(values)] if values else None


def _count(values) -> dict:
    counts: dict = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def plot_rates(path: Path, rows: list[dict]) -> None:
    """Per friction (rows): refused share, topple rate and mean max tilt against direction, one line per length."""
    frictions = sorted({r["friction"] for r in rows})
    lengths = sorted({r["length_cm"] for r in rows})
    directions = sorted({r["direction_deg"] for r in rows})
    ramp = plotting.BLUE_RAMP[0::2][: len(lengths)] if len(lengths) <= 3 else plotting.BLUE_RAMP
    fig, axes = plt.subplots(len(frictions), 3, figsize=(15, 3.8 * len(frictions)), squeeze=False)
    for i, mu in enumerate(frictions):
        for length, colour in zip(lengths, ramp):
            refused, topple, tilt, error = [], [], [], []
            for d in directions:
                rs = [r for r in rows if r["friction"] == mu and r["length_cm"] == length and r["direction_deg"] == d]
                ex = [r for r in rs if r["executed"]]
                refused.append(100 * (len(rs) - len(ex)) / len(rs))
                stats = measures.cell_stats(ex) if ex else None
                failed = [r for r in ex if not r["glitch"] and (r["toppled"] or r["jammed"])]
                topple.append(100 * len(failed) / stats["n_valid"] if stats and stats["n_valid"] else np.nan)
                tilt.append(stats["mean_max_tilt_deg"] if stats and stats["n_valid"] else np.nan)
                error.append(stats["standard_error_deg"] if stats and stats["n_valid"] else np.nan)
            label = f"{length:g} cm"
            axes[i, 0].plot(directions, refused, color=colour, marker="o", ms=6, lw=2, mec="white", label=label)
            axes[i, 1].plot(directions, topple, color=colour, marker="o", ms=6, lw=2, mec="white", label=label)
            plotting.line_with_error(axes[i, 2], directions, tilt, error, colour, label)
        axes[i, 0].set(title=f"refused, μ = {mu} (all spots)", ylabel="% of runs", ylim=(-3, 103))
        axes[i, 1].set(title=f"toppled or jammed, μ = {mu} (executed runs)", ylabel="% of non-glitch runs", ylim=(-3, 103))
        axes[i, 2].set(title=f"mean max tilt, μ = {mu}", ylabel="deg")
        axes[i, 2].axhline(measures.LEAN_DEG, color=plotting.REFERENCE_COLOURS["neutral"], ls="--", lw=1)
    for ax in axes.flat:
        ax.set_xticks(directions)
        ax.set_xlabel("sweep direction (deg; 0 = away from the robot, 90 = robot's left)")
        plotting.tidy(ax)
    axes[0, 0].legend(title="sweep length", fontsize=8, frameon=False)
    fig.suptitle("Sweep at 1/3 speed, red alone at five spots (solver glitches excluded; wall jams count as failures)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    plotting.save(fig, path)


def plot_movement(path: Path, rows: list[dict]) -> None:
    """Top-down, per friction: intended movement (grey dashed) and the mean actual movement (by length) for every
    spot and direction; a cross where every run of that sweep was refused."""
    probe = SingleRedShelf()
    layout, region = probe.env.layout, probe.env.region
    probe.close()
    frictions = sorted({r["friction"] for r in rows})
    lengths = sorted({r["length_cm"] for r in rows})
    colours = dict(zip(lengths, plotting.BLUE_RAMP[0::2][: len(lengths)] if len(lengths) <= 3 else plotting.BLUE_RAMP))
    fig, axes = plt.subplots(1, len(frictions), figsize=(8.5 * len(frictions), 7), squeeze=False)
    for ax, mu in zip(axes[0], frictions):
        for spot, (x0, y0) in SPOTS_M.items():
            for d in sorted({r["direction_deg"] for r in rows}):
                for length in lengths:
                    rs = [r for r in rows if (r["spot"], r["direction_deg"], r["length_cm"], r["friction"])
                          == (spot, d, length, mu)]
                    if not rs:
                        continue
                    unit = np.array([math.cos(math.radians(d)), math.sin(math.radians(d))])
                    intended = (length / 100 - PUSH_START_GAP_M) * unit
                    ax.plot([y0, y0 + intended[1]], [x0, x0 + intended[0]], color="#b5b5b0", lw=0.8, ls="--")
                    done = [r for r in rs if r["executed"] and not r["glitch"]]
                    if not done:
                        ax.scatter(y0 + intended[1], x0 + intended[0], marker="x", s=18, c="#8a8a86", zorder=3)
                        continue
                    dx = np.mean([r["moved_dx_mm"] for r in done]) / 1000
                    dy = np.mean([r["moved_dy_mm"] for r in done]) / 1000
                    colour = (plotting.OUTCOME_STYLE["toppled"][0] if any(r["toppled"] for r in done) else
                              plotting.OUTCOME_STYLE["jammed"][0] if any(r["jammed"] for r in done) else colours[length])
                    ax.annotate("", xy=(y0 + dy, x0 + dx), xytext=(y0, x0),
                                arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.4, mutation_scale=8))
            ax.scatter(y0, x0, s=60, c=plotting.OUTCOME_STYLE["toppled"][0], edgecolors="white", zorder=4)
        plotting.draw_shelf(ax, layout, region)
        handles = [Line2D([], [], color=colours[length], lw=1.4, label=f"actual, {length:g} cm sweep (mean)")
                   for length in lengths]
        handles += [Line2D([], [], color=plotting.OUTCOME_STYLE["toppled"][0], lw=1.4, label="a run toppled"),
                    Line2D([], [], color=plotting.OUTCOME_STYLE["jammed"][0], lw=1.4, label="a run jammed against a wall"),
                    Line2D([], [], color="#b5b5b0", lw=0.8, ls="--", label="intended (length − 5 mm start gap)"),
                    Line2D([], [], color="#8a8a86", marker="x", ls="", label="refused every run")]
        ax.legend(handles=handles + ax.get_legend_handles_labels()[0], fontsize=7, loc="upper left",
                  bbox_to_anchor=(0, -0.1), ncol=2, frameon=False)
        ax.set_title(f"Red cylinder movement, intended vs actual, μ = {mu}", fontsize=10)
    fig.tight_layout()
    plotting.save(fig, path, dpi=140)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="small subset into outputs/quick/")
    parser.add_argument("--parts", nargs="+", default=["sanity", "sweeps", "toward_robot"],
                        choices=["sanity", "sweeps", "toward_robot"])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    out = OUT_DIR / "quick" if args.quick else OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    if "sanity" in args.parts:
        summary["sanity"] = part_sanity(out)
    if "sweeps" in args.parts:
        summary["sweeps"] = part_sweeps(out, args.quick, args.workers)
    if "toward_robot" in args.parts:
        summary["toward_robot"] = part_toward_robot(out, args.quick, args.workers)
    summary_path.write_text(json.dumps(summary, indent=1, default=str) + "\n")
    runner.save_config(out, {
        "quick": args.quick, "spots_m": SPOTS_M, "directions_deg": QUICK_DIRECTIONS_DEG if args.quick else DIRECTIONS_DEG,
        "lengths_m": QUICK_LENGTHS_M if args.quick else LENGTHS_M, "frictions": QUICK_FRICTIONS if args.quick else FRICTIONS,
        "repetitions": QUICK_REPETITIONS if args.quick else REPETITIONS,
        "toward_robot_deg": TOWARD_ROBOT_DEG, "toward_robot_half_angle_deg": sweep_skill.TOWARD_ROBOT_HALF_ANGLE_DEG, "density_kg_m3": DENSITY_KG_M3,
        "reset_seed": RESET_SEED, "study_seed": STUDY_SEED, "sweep_heights_m": sweep_skill.SWEEP_HEIGHTS_M,
        "contact_speed_m_s": sweep_skill.CONTACT_SPEED_M_S, "contact_speed_fraction": sweep_skill.CONTACT_SPEED_FRACTION,
        "normal_tool_speed_m_s": sweep_skill.NORMAL_TOOL_SPEED_M_S, "start_gap_m": PUSH_START_GAP_M,
        "glitch_contact_force_n": measures.GLITCH_CONTACT_FORCE_N,
        "jam_contact_force_n": measures.JAM_CONTACT_FORCE_N, "wall_geoms": measures.WALL_GEOMS, "glitch_speed_m_s": measures.GLITCH_SPEED_M_S})


if __name__ == "__main__":
    main()
