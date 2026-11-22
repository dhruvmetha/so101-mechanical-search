"""Study 03: what reduces a pushed cylinder's tilt most: direction, height, jaw friction, speed or board friction?

    experiments/sahal/.venv/bin/python experiments/sahal/studies/03_sweep_factors/run.py [--quick] [--workers 10]

Every cell runs REPETITIONS times with the push start jittered up to 1 mm (fixed per cell and repetition).
Experiments: base (heights 2.25-3.0 cm, all directions and frictions), arm_friction (the same with every arm geom's
sliding friction set to the cylinder's mu, a diagnostic only), speed (2.5 cm at slower speeds; normal speed comes
from base). Glitch runs are excluded from every rate and mean. --quick runs a small subset into outputs/quick/.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # experiments/sahal, for `common`

import common  # noqa: E402,F401
import matplotlib.pyplot as plt  # noqa: E402

from common import measures, plotting, runner  # noqa: E402
from common.shelf_env import PUSH_DIRECTIONS_RAD, SingleRedShelf, jitter_offset, run_push  # noqa: E402

# ---- settings ------------------------------------------------------------------------------------------------
OUT_DIR = Path(__file__).resolve().parent / "outputs"
REPETITIONS = 5
HEIGHTS_CM = [2.25, 2.5, 2.75, 3.0]  # what the level gripper is allowed (study 02)
FRICTIONS = [0.4, 0.5, 0.55, 0.6, 0.7]
SPEED_HEIGHT_CM = 2.5
SLOW_SPEED_FRACTIONS = [0.5, 0.25]
QUICK_REPETITIONS = 2
QUICK_HEIGHTS_CM = [2.5]
QUICK_FRICTIONS = [0.55, 0.7]
QUICK_SLOW_SPEED_FRACTIONS = [0.25]

_ENV: SingleRedShelf | None = None


def _init() -> None:
    global _ENV
    _ENV = SingleRedShelf(widen_action_box=True, bypass_path_check=True, arm_touch_ends_episode=False)


def _job(job: tuple[str, str, float, float, int, float]) -> dict:
    experiment, direction, height_cm, friction, repetition, speed = job
    record = run_push(_ENV, direction=direction, height_cm=height_cm, friction=friction,
                      jitter_xy=jitter_offset(repetition, height_cm, friction, direction), speed_fraction=speed,
                      arm_friction=friction if experiment == "arm_friction" else None)
    return {"experiment": experiment, "repetition": repetition, **record}


def build_jobs(quick: bool) -> list[tuple]:
    reps = range(QUICK_REPETITIONS if quick else REPETITIONS)
    heights = QUICK_HEIGHTS_CM if quick else HEIGHTS_CM
    frictions = QUICK_FRICTIONS if quick else FRICTIONS
    speeds = QUICK_SLOW_SPEED_FRACTIONS if quick else SLOW_SPEED_FRACTIONS
    jobs = [(e, d, h, mu, k, 1.0) for e in ("base", "arm_friction") for d in PUSH_DIRECTIONS_RAD for h in heights
            for mu in frictions for k in reps]
    jobs += [("speed", d, SPEED_HEIGHT_CM, mu, k, s) for s in speeds for d in PUSH_DIRECTIONS_RAD for mu in frictions
             for k in reps]
    return jobs


def summarise(rows: list[dict], out: Path) -> dict:
    """Topple rate and max tilt per factor, the per-cell table, and the contact parts per direction."""
    cells: dict[tuple, list[dict]] = {}
    for r in rows:
        cells.setdefault((r["experiment"], r["direction"], r["speed_fraction"], r["height_cm"], r["friction"]),
                         []).append(r)
    runner.write_csv(out / "cells.csv", [
        {"experiment": k[0], "direction": k[1], "speed_fraction": k[2], "height_cm": k[3], "friction": k[4],
         **{name: round(v, 3) for name, v in measures.cell_stats(rs).items()}} for k, rs in sorted(cells.items())])
    base = [r for r in rows if r["experiment"] == "base"]
    base_at_speed_height = [r for r in base if r["height_cm"] == SPEED_HEIGHT_CM]

    def stats(rs: list[dict]) -> dict:
        return {k: round(v, 3) for k, v in measures.cell_stats(rs).items()}

    summary = {
        "runs": len(rows),
        "glitch_runs": {e: sum(r["glitch"] for r in rows if r["experiment"] == e)
                        for e in ("base", "arm_friction", "speed")},
        "by_direction": {d: stats([r for r in base if r["direction"] == d]) for d in PUSH_DIRECTIONS_RAD},
        "by_height": {h: stats([r for r in base if r["height_cm"] == h]) for h in sorted({r["height_cm"] for r in base})},
        "by_friction": {mu: stats([r for r in base if r["friction"] == mu]) for mu in sorted({r["friction"] for r in base})},
        "by_arm_friction": {d: {"default 1.0": stats([r for r in base if r["direction"] == d]),
                                "= cylinder mu": stats([r for r in rows if r["experiment"] == "arm_friction"
                                                        and r["direction"] == d])} for d in PUSH_DIRECTIONS_RAD},
        "by_speed": {d: {s: stats([r for r in base_at_speed_height if r["direction"] == d] if s == 1.0 else
                                  [r for r in rows if r["experiment"] == "speed" and r["speed_fraction"] == s
                                   and r["direction"] == d])
                         for s in [1.0] + sorted({r["speed_fraction"] for r in rows if r["experiment"] == "speed"},
                                                 reverse=True)} for d in PUSH_DIRECTIONS_RAD},
        "first_contact_part": {d: dict(Counter(r["first_contact_part"] for r in base if r["direction"] == d))
                               for d in PUSH_DIRECTIONS_RAD},
    }
    return summary


def plot(path: Path, summary: dict) -> None:
    """Mean max tilt against each factor, one line per push direction."""
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6))

    def series(table: dict, keys: list) -> tuple[list, list]:
        return [table[k]["mean_max_tilt_deg"] for k in keys], [table[k]["standard_error_deg"] for k in keys]

    for direction, colour in plotting.DIRECTION_COLOURS.items():
        by_height = summary["by_height_and_direction"][direction]
        heights = sorted(by_height)
        plotting.line_with_error(axes[0], heights, *series(by_height, heights), colour, direction)
        arm = summary["by_arm_friction"][direction]
        plotting.line_with_error(axes[1], [0, 1], *series(arm, ["default 1.0", "= cylinder mu"]), colour, direction)
        speeds = sorted(summary["by_speed"][direction])
        plotting.line_with_error(axes[2], speeds, *series(summary["by_speed"][direction], speeds), colour, direction)
        by_friction = summary["by_friction_and_direction"][direction]
        frictions = sorted(by_friction)
        plotting.line_with_error(axes[3], frictions, *series(by_friction, frictions), colour, direction)
    axes[0].set(title="Height (arm friction 1.0, full speed)", xlabel="push height (cm)", ylabel="mean max tilt (deg)")
    axes[1].set(title="Arm friction (all heights)", xticks=[0, 1], xlim=(-0.4, 1.4))
    axes[1].set_xticklabels(["1.0 (default)", "= cylinder μ"])
    axes[2].set(title=f"Speed ({SPEED_HEIGHT_CM} cm)", xlabel="fraction of normal speed")
    axes[3].set(title="Cylinder-board friction", xlabel="μ")
    top = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(0, top)
        ax.axhline(measures.LEAN_DEG, color=plotting.REFERENCE_COLOURS["neutral"], ls="--", lw=1)
        plotting.tidy(ax)
    axes[0].legend(title="push direction", fontsize=8, frameon=False)
    fig.suptitle("Mean max tilt of a lone cylinder pushed 5 cm (jittered repeats; bars = ±1 s.e.; dashed = 5° lean; "
                 "glitches excluded)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    plotting.save(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="small subset into outputs/quick/")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    out = OUT_DIR / "quick" if args.quick else OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    rows = runner.run_parallel(build_jobs(args.quick), _job, _init, args.workers, "factors: ")
    runner.write_csv(out / "runs.csv", rows)
    summary = summarise(rows, out)
    base = [r for r in rows if r["experiment"] == "base"]
    summary["by_height_and_direction"] = {
        d: {h: {k: round(v, 3) for k, v in measures.cell_stats(
            [r for r in base if r["direction"] == d and r["height_cm"] == h]).items()}
            for h in sorted({r["height_cm"] for r in base})} for d in PUSH_DIRECTIONS_RAD}
    summary["by_friction_and_direction"] = {
        d: {mu: {k: round(v, 3) for k, v in measures.cell_stats(
            [r for r in base if r["direction"] == d and r["friction"] == mu]).items()}
            for mu in sorted({r["friction"] for r in base})} for d in PUSH_DIRECTIONS_RAD}
    plot(out / "factors.png", summary)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    for factor in ("glitch_runs", "by_direction", "by_height", "by_friction", "by_arm_friction", "by_speed",
                   "first_contact_part"):
        print(factor, json.dumps(summary[factor], default=str))
    runner.save_config(out, {
        "quick": args.quick, "repetitions": QUICK_REPETITIONS if args.quick else REPETITIONS,
        "heights_cm": QUICK_HEIGHTS_CM if args.quick else HEIGHTS_CM,
        "frictions": QUICK_FRICTIONS if args.quick else FRICTIONS, "speed_height_cm": SPEED_HEIGHT_CM,
        "slow_speed_fractions": QUICK_SLOW_SPEED_FRACTIONS if args.quick else SLOW_SPEED_FRACTIONS,
        "jitter": "up to 1 mm, seeded by (repetition, height, friction, direction)", "density_kg_m3": 900.0,
        "glitch_contact_force_n": measures.GLITCH_CONTACT_FORCE_N, "glitch_speed_m_s": measures.GLITCH_SPEED_M_S})


if __name__ == "__main__":
    main()
