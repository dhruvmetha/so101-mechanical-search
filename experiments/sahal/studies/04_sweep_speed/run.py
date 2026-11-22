"""Study 04: what push speed keeps a lone cylinder upright, and what happens at quarter speed?

    experiments/sahal/.venv/bin/python experiments/sahal/studies/04_sweep_speed/run.py [--quick] [--parts ...]
        [--workers 10]

Parts: sweep (speed 0.25 to 1.0 of normal at 2.5 cm, mu 0.55 and 0.7, every direction, jittered repeats) and traces
(chosen quarter-speed runs recorded every physics substep: tilt, arm-cylinder force and slip, base slip, speeds).
A speed fraction s resamples the push trajectory at frame spacing s (float64 interpolation), so the same path takes
1/s as many frames. Glitch runs are excluded from rates and means. --quick runs a subset into outputs/quick/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # experiments/sahal, for `common`

import common  # noqa: E402,F401
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import measures, plotting, runner  # noqa: E402
from common.shelf_env import PUSH_DIRECTIONS_RAD, SingleRedShelf, jitter_offset, run_push  # noqa: E402

# ---- settings ------------------------------------------------------------------------------------------------
OUT_DIR = Path(__file__).resolve().parent / "outputs"
HEIGHT_CM = 2.5
SPEED_FRACTIONS = [0.25, 0.33, 0.5, 0.67, 0.75, 1.0]
FRICTIONS = [0.55, 0.7]
REPETITIONS = 8
QUICK_SPEED_FRACTIONS = [0.25, 0.5, 1.0]
QUICK_FRICTIONS = [0.7]
QUICK_REPETITIONS = 2
TRACE_SPEED_FRACTION = 0.25
TRACE_RUNS = [("left", 0.7, 1), ("left", 0.7, 2), ("right", 0.6, 0), ("left", 0.55, 0)]  # (direction, mu, repetition)
QUICK_TRACE_RUNS = TRACE_RUNS[:2]
TRACE_WINDOW_BEFORE_S = 0.3
TRACE_WINDOW_AFTER_S = 0.5
FORCE_PLOT_CAP_N = 8.0
SPEED_PLOT_CAP_MM_S = 200.0
TRACE_CSV_DECIMALS = 5

_ENV: SingleRedShelf | None = None


def _init() -> None:
    global _ENV
    _ENV = SingleRedShelf(widen_action_box=True, bypass_path_check=True, arm_touch_ends_episode=False)


def _job(job: tuple[str, float, int, float]) -> dict:
    direction, friction, repetition, speed = job
    record = run_push(_ENV, direction=direction, height_cm=HEIGHT_CM, friction=friction, speed_fraction=speed,
                      jitter_xy=jitter_offset(repetition, HEIGHT_CM, friction, direction))
    return {"repetition": repetition, "robot_time_s": round(record["frames"] / 30, 2), **record}


# ---- sweep -----------------------------------------------------------------------------------------------------

def part_sweep(out: Path, quick: bool, workers: int) -> dict:
    speeds = QUICK_SPEED_FRACTIONS if quick else SPEED_FRACTIONS
    frictions = QUICK_FRICTIONS if quick else FRICTIONS
    reps = range(QUICK_REPETITIONS if quick else REPETITIONS)
    jobs = [(d, mu, k, s) for d in PUSH_DIRECTIONS_RAD for mu in frictions for s in speeds for k in reps]
    rows = runner.run_parallel(jobs, _job, _init, workers, "sweep: ")
    runner.write_csv(out / "sweep_runs.csv", rows)
    table = {}
    for d in PUSH_DIRECTIONS_RAD:
        for mu in frictions:
            for s in speeds:
                stats = measures.cell_stats([r for r in rows if r["direction"] == d and r["friction"] == mu
                                             and r["speed_fraction"] == s])
                table[(d, mu, s)] = stats
    runner.write_csv(out / "sweep_cells.csv", [{"direction": k[0], "friction": k[1], "speed_fraction": k[2],
                                                 **{n: round(v, 3) for n, v in s.items()}} for k, s in table.items()])
    plot_sweep(out / "sweep_speed.png", table, speeds, frictions)
    valid = [r for r in rows if not r["glitch"]]
    summary = {
        "runs": len(rows), "glitch_runs": sum(r["glitch"] for r in rows),
        "glitch_runs_by_speed": {s: sum(r["glitch"] for r in rows if r["speed_fraction"] == s) for s in speeds},
        "toppled_runs_by_speed": {s: sum(r["toppled"] for r in valid if r["speed_fraction"] == s) for s in speeds},
        "worst_cell_mean_tilt_by_speed": {s: round(max(table[(d, mu, s)]["mean_max_tilt_deg"]
                                                       for d in PUSH_DIRECTIONS_RAD for mu in frictions), 2)
                                          for s in speeds},
        "max_single_tilt_by_speed": {s: max(r["max_tilt_deg"] for r in valid if r["speed_fraction"] == s)
                                     for s in speeds},
        "robot_time_s_by_speed": {s: round(float(np.mean([r["robot_time_s"] for r in valid
                                                          if r["speed_fraction"] == s])), 1) for s in speeds},
    }
    for (d, mu, s), stats in table.items():
        print(f"{d:6s} mu {mu:.2f} speed {s:4.2f}: topple {100 * stats['topple_rate']:3.0f}%  mean "
              f"{stats['mean_max_tilt_deg']:5.1f}°  median {stats['median_max_tilt_deg']:5.1f}°  glitch "
              f"{stats['n_glitch']}")
    print("sweep:", json.dumps(summary, default=str))
    return summary


def plot_sweep(path: Path, table: dict, speeds: list[float], frictions: list[float]) -> None:
    fig, axes = plt.subplots(1, 2 * len(frictions), figsize=(8.5 * len(frictions), 4.4), squeeze=False)
    axes = axes[0]
    for j, mu in enumerate(frictions):
        for d, colour in plotting.DIRECTION_COLOURS.items():
            cells = [table[(d, mu, s)] for s in speeds]
            plotting.line_with_error(axes[2 * j], speeds, [c["mean_max_tilt_deg"] for c in cells],
                                     [c["standard_error_deg"] for c in cells], colour, d)
            axes[2 * j + 1].plot(speeds, [100 * c["topple_rate"] for c in cells], color=colour, marker="o", ms=6,
                                 lw=2, mec="white", label=d)
        axes[2 * j].set(title=f"mean max tilt, μ = {mu}", xlabel="fraction of normal speed", ylabel="deg")
        axes[2 * j].axhline(measures.LEAN_DEG, color=plotting.REFERENCE_COLOURS["neutral"], ls="--", lw=1)
        axes[2 * j + 1].set(title=f"topple rate, μ = {mu}", xlabel="fraction of normal speed",
                            ylabel="% of non-glitch runs", ylim=(-3, 103))
    for ax in axes:
        ax.set_xticks(speeds, [f"{s:g}" for s in speeds])
        plotting.tidy(ax)
    axes[0].legend(title="push direction", fontsize=8, frameon=False)
    fig.suptitle(f"Push at {HEIGHT_CM} cm: tilt and topple rate vs speed (jittered repeats; bars = ±1 s.e.; "
                 "glitches excluded)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    plotting.save(fig, path)


# ---- traces ----------------------------------------------------------------------------------------------------

def trace_rows(substeps: list[dict]) -> list[dict]:
    """Substep log to CSV rows, with the jaw midpoint's speed from consecutive positions."""
    rows, previous = [], None
    for s in substeps:
        jaw_speed = 0.0 if previous is None else float(
            np.linalg.norm(s["jaw_point_xyz"] - previous["jaw_point_xyz"]) / (s["time_s"] - previous["time_s"]))
        rows.append({k: v for k, v in s.items() if k != "jaw_point_xyz"} | {"jaw_speed_m_s": jaw_speed})
        previous = s
    return rows


def contact_window(rows: list[dict]) -> tuple[float, float]:
    """(start, end) times from TRACE_WINDOW_BEFORE_S before the first arm contact to TRACE_WINDOW_AFTER_S after the last."""
    t = np.array([r["time_s"] for r in rows])
    touching = np.flatnonzero(np.array([r["arm_normal_n"] for r in rows]) > 0)
    if not len(touching):
        return float(t[0]), float(t[-1])
    return float(t[touching[0]] - TRACE_WINDOW_BEFORE_S), float(min(t[-1], t[touching[-1]] + TRACE_WINDOW_AFTER_S))


def describe_trace(rows: list[dict]) -> dict:
    """Touch episodes and their spacing, the arm's tangential/normal force ratio, and how often the base was on the
    board while the arm touched the cylinder."""
    t = np.array([r["time_s"] for r in rows])
    normal = np.array([r["arm_normal_n"] for r in rows])
    touching = normal > 0
    if not touching.any():
        return {"touch_episodes": 0}
    starts = np.flatnonzero(np.diff(touching.astype(int)) == 1)
    ratio = np.array([r["arm_tangential_n"] for r in rows])[touching] / normal[touching]
    board = np.array([r["board_normal_n"] for r in rows])
    jaw_speed = np.array([r["jaw_speed_m_s"] for r in rows])
    return {"touch_episodes": int(len(starts) + touching[0]),
            "median_gap_between_touches_s": round(float(np.median(np.diff(t[starts]))), 4) if len(starts) > 1 else None,
            "median_tangential_over_normal": round(float(np.median(ratio)), 3),
            "base_on_board_while_touching": round(float(np.mean(board[touching] > 0)), 3),
            "base_on_board_before_push": round(float(np.mean(board[: np.argmax(touching)] > 0)), 3),
            "max_jaw_speed_mm_s": round(1000 * float(jaw_speed.max()), 1),
            "max_arm_normal_force_n": round(float(normal.max()), 2)}


def part_traces(out: Path, quick: bool) -> dict:
    env = SingleRedShelf(widen_action_box=True, bypass_path_check=True, arm_touch_ends_episode=False)
    runs = []
    try:
        for direction, friction, repetition in (QUICK_TRACE_RUNS if quick else TRACE_RUNS):
            record = run_push(env, direction=direction, height_cm=HEIGHT_CM, friction=friction,
                              speed_fraction=TRACE_SPEED_FRACTION, record_substeps=True,
                              jitter_xy=jitter_offset(repetition, HEIGHT_CM, friction, direction))
            rows = trace_rows(env.log.substeps or [])
            runs.append({"direction": direction, "friction": friction, "repetition": repetition, "record": record,
                         "rows": rows, "pattern": describe_trace(rows)})
    finally:
        env.close()
    saved = []
    for r in runs:
        start, end = contact_window(r["rows"])
        saved += [{"direction": r["direction"], "friction": r["friction"], "repetition": r["repetition"],
                   **{k: round(v, TRACE_CSV_DECIMALS) for k, v in row.items()}}
                  for row in r["rows"] if start <= row["time_s"] <= end]
    runner.write_csv(out / "traces.csv", saved)
    plot_traces(out / "traces.png", runs)
    summary = [{"direction": r["direction"], "friction": r["friction"], "repetition": r["repetition"],
                "outcome": r["record"]["outcome"], "max_tilt_deg": r["record"]["max_tilt_deg"],
                "max_contact_force_n": r["record"]["max_contact_force_n"], **r["pattern"]} for r in runs]
    for s in summary:
        print("trace:", json.dumps(s))
    return {"runs": summary}


def plot_traces(path: Path, runs: list[dict]) -> None:
    fig, axes = plt.subplots(len(runs), 3, figsize=(15, 2.5 * len(runs)), squeeze=False)
    for i, run in enumerate(runs):
        rows = run["rows"]
        t = np.array([r["time_s"] for r in rows])
        normal = np.array([r["arm_normal_n"] for r in rows])
        low, high = contact_window(rows)
        window = (t >= low) & (t <= high)
        tt = t[window] - low
        tilt_ax, force_ax, speed_ax = axes[i]
        tilt_ax.plot(tt, np.array([r["tilt_deg"] for r in rows])[window], color=plotting.REFERENCE_COLOURS["danger"])
        tilt_ax.axhline(20, color=plotting.REFERENCE_COLOURS["danger"], ls="--", lw=0.8)
        tilt_ax.set_ylabel(f"{run['direction']}, μ {run['friction']}, rep {run['repetition']}\n"
                           f"({run['record']['outcome']})\ntilt (deg)", fontsize=8)
        ratio = np.where(normal > 0, np.array([r["arm_tangential_n"] for r in rows]) / np.maximum(normal, 1e-9), np.nan)
        force_ax.plot(tt, np.minimum(normal[window], FORCE_PLOT_CAP_N), color=plotting.DIRECTION_COLOURS["away"], lw=1,
                      label=f"arm normal force (N, capped at {FORCE_PLOT_CAP_N:g})")
        force_ax.plot(tt, 5 * ratio[window], color="#eda100", lw=1, label="5 × tangential/normal (5 = sliding)")
        force_ax.set_ylim(0, FORCE_PLOT_CAP_N)
        speed_ax.plot(tt, 1000 * np.array([r["jaw_speed_m_s"] for r in rows])[window], color="#6250d6", lw=1,
                      label="jaw midpoint")
        speed_ax.plot(tt, 1000 * np.array([r["red_speed_m_s"] for r in rows])[window],
                      color=plotting.DIRECTION_COLOURS["right"], lw=1, label="cylinder centre")
        speed_ax.plot(tt, 1000 * np.array([r["base_slip_m_s"] for r in rows])[window], color="#222", lw=0.8, ls=":",
                      label="cylinder base slip on board")
        speed_ax.set(ylabel="mm/s", ylim=(0, SPEED_PLOT_CAP_MM_S))
    axes[0, 0].set_title("cylinder tilt", fontsize=10)
    axes[0, 1].set_title("arm-cylinder contact", fontsize=10)
    axes[0, 2].set_title("speeds", fontsize=10)
    axes[0, 1].legend(fontsize=7, frameon=False, loc="upper right")
    axes[0, 2].legend(fontsize=7, frameon=False, loc="upper right")
    for ax in axes[-1]:
        ax.set_xlabel(f"time from {TRACE_WINDOW_BEFORE_S} s before first contact (s)")
    for ax in axes.flat:
        plotting.tidy(ax)
    fig.suptitle(f"Quarter-speed pushes at {HEIGHT_CM} cm, every physics substep", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    plotting.save(fig, path, dpi=120)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="small subset into outputs/quick/")
    parser.add_argument("--parts", nargs="+", default=["sweep", "traces"], choices=["sweep", "traces"])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    out = OUT_DIR / "quick" if args.quick else OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    if "sweep" in args.parts:
        summary["sweep"] = part_sweep(out, args.quick, args.workers)
    if "traces" in args.parts:
        summary["traces"] = part_traces(out, args.quick)
    summary_path.write_text(json.dumps(summary, indent=1, default=str) + "\n")
    runner.save_config(out, {
        "quick": args.quick, "height_cm": HEIGHT_CM,
        "speed_fractions": QUICK_SPEED_FRACTIONS if args.quick else SPEED_FRACTIONS,
        "frictions": QUICK_FRICTIONS if args.quick else FRICTIONS,
        "repetitions": QUICK_REPETITIONS if args.quick else REPETITIONS, "trace_speed_fraction": TRACE_SPEED_FRACTION,
        "trace_runs": QUICK_TRACE_RUNS if args.quick else TRACE_RUNS, "density_kg_m3": 900.0,
        "jitter": "up to 1 mm, seeded by (repetition, height, friction, direction)",
        "glitch_contact_force_n": measures.GLITCH_CONTACT_FORCE_N, "glitch_speed_m_s": measures.GLITCH_SPEED_M_S})


if __name__ == "__main__":
    main()
