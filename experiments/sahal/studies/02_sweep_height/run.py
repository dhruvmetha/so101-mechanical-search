"""Study 02: how low can the level gripper push, and does a lone cylinder slide or topple at each height and friction?

    experiments/sahal/.venv/bin/python experiments/sahal/studies/02_sweep_height/run.py [--quick] [--parts ...]
        [--workers 10]

Parts: reach (lowest push height that passes level IK and the env's 6 mm path check, kinematic) and slide_topple
(one 5 cm push per height, friction and direction; the path check is bypassed and runs it would refuse are marked).
--quick runs a small subset into outputs/quick/. See README.md.
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
from common.shelf_env import PUSH_DIRECTIONS_RAD, SingleRedShelf, run_push  # noqa: E402
from so101sim import side_primitives  # noqa: E402
from so101sim.primitives import Unreachable  # noqa: E402
from so101sim.shelf_layout import CYL_RADIUS  # noqa: E402

# ---- settings ------------------------------------------------------------------------------------------------
OUT_DIR = Path(__file__).resolve().parent / "outputs"
REACH_HEIGHTS_MM = list(range(5, 36))  # push height above the board, 1 mm steps
REACH_STARTS_M = {"centre": (0.4075, 0.0), "front": (0.355, 0.0), "back": (0.46, 0.0),
                  "near left wall": (0.4075, 0.125), "near right wall": (0.4075, -0.115)}
REACH_HEADINGS_RAD = {"away": 0.0, "toward": np.pi, "left": np.pi / 2, "right": -np.pi / 2}
REACH_PARKED_RED_XY_M = (0.33, 0.14)  # out of the way; the path check ignores cylinders anyway
QUICK_REACH_STARTS = ["centre"]
SWEEP_HEIGHTS_CM = [1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5]
SWEEP_FRICTIONS = [0.4, 0.5, 0.55, 0.6, 0.7]
QUICK_SWEEP_HEIGHTS_CM = [2.0, 2.5, 3.0]
QUICK_SWEEP_FRICTIONS = [0.55, 0.7]
TILT_PLOT_CAP_DEG = 40.0

_ENV: SingleRedShelf | None = None


# ---- reach -----------------------------------------------------------------------------------------------------

def stroke_stays_on_board(layout, start_xy, heading_rad: float) -> bool:
    end = np.asarray(start_xy) + side_primitives.PUSH_STROKE * np.array([np.cos(heading_rad), np.sin(heading_rad)])
    return bool(layout.front_x + CYL_RADIUS <= end[0] <= layout.back_x - CYL_RADIUS
                and abs(end[1]) <= layout.half_width - CYL_RADIUS)


def part_reach(out: Path, quick: bool) -> dict:
    """For each start and heading: does level IK solve the push, and does the env's path check pass, per height?
    Clearance (and the arm part closest to the shelf) at the lowest passing height and 1 mm below it."""
    env = SingleRedShelf(widen_action_box=True)
    rows, trajectories = [], {}
    try:
        env.reset(seed=0, red_xy=REACH_PARKED_RED_XY_M)
        start_pose = env.env.joint_state.copy()
        layout = env.env.layout
        starts = {k: REACH_STARTS_M[k] for k in (QUICK_REACH_STARTS if quick else REACH_STARTS_M)}
        for start_name, xy in starts.items():
            for heading_name, heading in REACH_HEADINGS_RAD.items():
                if not stroke_stays_on_board(layout, xy, heading):
                    continue
                for height_mm in REACH_HEIGHTS_MM:
                    env.set_push_height(height_mm / 1000)
                    row = {"start": start_name, "x": xy[0], "y": xy[1], "heading": heading_name, "height_mm": height_mm}
                    try:
                        trajectory = side_primitives.push(env.model, env.geo, np.array(xy), heading)
                    except Unreachable as exc:
                        rows.append({**row, "ik_ok": False, "ik_reason": str(exc)[:80], "path_ok": False})
                        continue
                    full = side_primitives.bridge(env.model, env.geo, start_pose, trajectory)
                    hit = env._path_hit(full)
                    rows.append({**row, "ik_ok": True, "path_ok": hit is None, "path_hit": hit[1] if hit else None})
                    trajectories[(start_name, heading_name, height_mm)] = trajectory
        lowest = {}
        for row in rows:
            key = (row["start"], row["heading"])
            if row["path_ok"] and key not in lowest:
                lowest[key] = row["height_mm"]
        for row in rows:
            key = (row["start"], row["heading"])
            if key in lowest and row["height_mm"] in (lowest[key], lowest[key] - 1) and row["ik_ok"]:
                distance, arm, shelf = measures.clearance_to_shelf(
                    env, trajectories[(row["start"], row["heading"], row["height_mm"])])
                row.update(clearance_mm=round(distance * 1000, 1), closest_arm_geom=arm, closest_shelf_geom=shelf)
    finally:
        env.close()
    runner.write_csv(out / "reach.csv", rows)
    table = []
    for (start, heading), height in lowest.items():
        at = next(r for r in rows if r["start"] == start and r["heading"] == heading and r["height_mm"] == height)
        table.append({"start": start, "heading": heading, "lowest_passing_mm": height,
                      "lowest_ik_mm": min(r["height_mm"] for r in rows if r["start"] == start
                                          and r["heading"] == heading and r["ik_ok"]),
                      "closest_arm_geom": at["closest_arm_geom"], "clearance_mm": at["clearance_mm"]})
        print(f"reach {start:16s} {heading:7s} lowest passing {height} mm, closest {at['closest_arm_geom']} "
              f"{at['clearance_mm']} mm")
    return {"lowest_passing": table}


# ---- slide or topple -------------------------------------------------------------------------------------------

def _init() -> None:
    global _ENV
    _ENV = SingleRedShelf(widen_action_box=True, bypass_path_check=True, arm_touch_ends_episode=False)


def _push_job(job: tuple[str, float, float]) -> dict:
    direction, height_cm, friction = job
    return run_push(_ENV, direction=direction, height_cm=height_cm, friction=friction)


def lowest_passing_cm(out: Path) -> dict[str, float]:
    """Lowest height the path check allows at the centre per push direction, from reach.csv."""
    path = out / "reach.csv"
    if not path.exists():
        path = OUT_DIR / "reach.csv"
    rows = runner.read_csv(path)
    result = {}
    for direction in PUSH_DIRECTIONS_RAD:
        passing = [int(r["height_mm"]) for r in rows if r["start"] == "centre" and r["heading"] == direction
                   and r["path_ok"] == "True"]
        result[direction] = min(passing) / 10
    return result


def part_slide_topple(out: Path, quick: bool, workers: int) -> dict:
    heights = QUICK_SWEEP_HEIGHTS_CM if quick else SWEEP_HEIGHTS_CM
    frictions = QUICK_SWEEP_FRICTIONS if quick else SWEEP_FRICTIONS
    jobs = [(d, h, mu) for d in PUSH_DIRECTIONS_RAD for mu in frictions for h in heights]
    rows = runner.run_parallel(jobs, _push_job, _init, workers, "slide_topple: ")
    runner.write_csv(out / "slide_topple_runs.csv", rows)
    reach = lowest_passing_cm(out)
    outcome_map(out / "slide_topple_vs_commanded_height.png", rows, "height_cm", 1.0, reach,
                "commanded push height above the board (cm)",
                "Slide or topple vs COMMANDED push height (label = max tilt)")
    outcome_map(out / "slide_topple_vs_contact_height.png", rows, "first_contact_height_mm", 0.1, reach,
                "first contact height above the board (cm)",
                "Slide or topple vs MEASURED first contact height (label = max tilt)")
    tilt_vs_contact(out / "tilt_vs_contact_height.png", rows, frictions)
    valid = [r for r in rows if not r["glitch"]]
    below = [r for r in valid if r["first_contact_height_mm"] is not None
             and r["first_contact_height_mm"] / 10 < r["r_over_mu_cm"]]
    above = [r for r in valid if r["first_contact_height_mm"] is not None
             and r["first_contact_height_mm"] / 10 >= r["r_over_mu_cm"]]
    count = lambda rs: {o: sum(r["outcome"] == o for r in rs) for o in ("slid", "leaned", "toppled")}  # noqa: E731
    summary = {
        "runs": len(rows), "glitch_runs": sum(r["glitch"] for r in rows),
        "glitches": [(r["direction"], r["height_cm"], r["friction"], r["max_contact_force_n"], r["max_red_speed_m_s"])
                     for r in rows if r["glitch"]],
        "by_direction": {d: count([r for r in valid if r["direction"] == d]) for d in PUSH_DIRECTIONS_RAD},
        "below_r_over_mu": count(below), "above_r_over_mu": count(above),
        "arm_touched_board_by_height": {h: sum(bool(r["arm_touched_shelf"]) for r in rows if r["height_cm"] == h)
                                        for h in heights},
        "contact_minus_commanded_mm": {d: round(float(np.mean([r["first_contact_height_mm"] - 10 * r["height_cm"]
                                                                for r in valid if r["direction"] == d
                                                                and r["first_contact_height_mm"] is not None])), 1)
                                       for d in PUSH_DIRECTIONS_RAD},
        "max_contact_force_n_non_glitch": max(r["max_contact_force_n"] for r in valid),
    }
    print("slide_topple:", json.dumps(summary))
    return summary


def outcome_map(path: Path, rows: list[dict], y_key: str, y_scale: float, reach_cm: dict[str, float],
                y_label: str, title: str) -> None:
    """One panel per push direction: friction across, height up, each run marked by outcome."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.6), sharey=True)
    frictions = sorted({r["friction"] for r in rows})
    curve = np.linspace(0.38, 0.72, 100)
    for ax, direction in zip(axes, PUSH_DIRECTIONS_RAD):
        for r in (r for r in rows if r["direction"] == direction and r[y_key] is not None):
            x, y = r["friction"], r[y_key] * y_scale
            colour, marker, _ = plotting.OUTCOME_STYLE[r["outcome"]]
            refused = bool(r["would_refuse"])
            ax.scatter(x, y, s=120, c=colour, marker=marker, zorder=3, edgecolors="#222" if refused else "white",
                       linewidths=1.8 if refused else 1.0)
            if r["arm_touched_shelf"]:
                ax.scatter(x - 0.017, y, s=28, marker="v", c="#222", zorder=4)
            ax.annotate(f"{r['max_tilt_deg']:.0f}°", (x, y), textcoords="offset points", xytext=(9, -3), fontsize=7,
                        color="#555")
        ax.plot(curve, 100 * CYL_RADIUS / curve, color=plotting.REFERENCE_COLOURS["theory"], lw=2,
                label="h = r / μ (theory)")
        ax.axhline(reach_cm[direction], color=plotting.REFERENCE_COLOURS["limit"], ls="--", lw=1.5,
                   label="lowest commanded height the path check allows")
        ax.set_title(f"push {direction}" + (" from the robot" if direction == "away" else ""), fontsize=10)
        ax.set_xlabel("cylinder-board friction μ")
        ax.set_xticks(frictions)
        ax.set_xlim(0.36, 0.76)
        plotting.tidy(ax)
    axes[0].set_ylabel(y_label)
    handles = plotting.outcome_legend_handles() + axes[0].get_legend_handles_labels()[0]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    plotting.save(fig, path)


def tilt_vs_contact(path: Path, rows: list[dict], frictions: list[float]) -> None:
    """Max tilt against first contact height, one line per friction, one panel per direction; glitches left out."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    ramp = plotting.BLUE_RAMP[-len(frictions):]
    for ax, direction in zip(axes, PUSH_DIRECTIONS_RAD):
        for friction, colour in zip(frictions, ramp):
            rs = sorted((r for r in rows if r["direction"] == direction and r["friction"] == friction
                         and r["first_contact_height_mm"] is not None and not r["glitch"]),
                        key=lambda r: r["first_contact_height_mm"])
            ax.plot([r["first_contact_height_mm"] for r in rs],
                    [min(r["max_tilt_deg"], TILT_PLOT_CAP_DEG) for r in rs],
                    color=colour, lw=1.5, marker="o", ms=6, mec="white", mew=1, label=f"μ = {friction}")
            ax.axvline(1000 * CYL_RADIUS / friction, color=colour, ls=":", lw=1.2)
        ax.axhline(20, color=plotting.REFERENCE_COLOURS["danger"], lw=1, ls="--")
        ax.axhline(measures.LEAN_DEG, color=plotting.REFERENCE_COLOURS["neutral"], lw=1, ls="--")
        ax.set_title(f"push {direction}", fontsize=10)
        ax.set_xlabel("first contact height above the board (mm)\n(dotted: r/μ for each friction)")
        plotting.tidy(ax)
    axes[0].set_ylabel("max red tilt (deg; dashed: 5° lean, 20° topple)")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Max tilt vs where the gripper first touched the cylinder, per friction (glitches excluded)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    plotting.save(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="small subset into outputs/quick/")
    parser.add_argument("--parts", nargs="+", default=["reach", "slide_topple"], choices=["reach", "slide_topple"])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    out = OUT_DIR / "quick" if args.quick else OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    if "reach" in args.parts:
        summary["reach"] = part_reach(out, args.quick)
    if "slide_topple" in args.parts:
        summary["slide_topple"] = part_slide_topple(out, args.quick, args.workers)
    summary_path.write_text(json.dumps(summary, indent=1, default=str) + "\n")
    runner.save_config(out, {
        "quick": args.quick, "parts": args.parts, "reach_heights_mm": REACH_HEIGHTS_MM,
        "reach_starts_m": REACH_STARTS_M, "sweep_heights_cm": QUICK_SWEEP_HEIGHTS_CM if args.quick else SWEEP_HEIGHTS_CM,
        "sweep_frictions": QUICK_SWEEP_FRICTIONS if args.quick else SWEEP_FRICTIONS,
        "push_density_kg_m3": 900.0, "reset_seed": 0, "path_check": "bypassed, verdict recorded",
        "arm_shelf_touch": "recorded, does not end the run",
        "glitch_contact_force_n": measures.GLITCH_CONTACT_FORCE_N, "glitch_speed_m_s": measures.GLITCH_SPEED_M_S})


if __name__ == "__main__":
    main()
