"""Study 11: wider, shorter cylinders so sweeps never topple them (Dhruv's simplification).

    experiments/sahal/.venv/bin/python experiments/sahal/studies/11_wide_cylinders/run.py [--quick] [--workers 10]
        [--parts gripper sizes]

Sizes are overridden with common.cylinder_size (no repo file changes): the model's cylinders, grasp height (half the
height) and carry height (grasp + 1 cm), the side opening (diameter + 13 mm) and the planning gripper values.

gripper: how wide a cylinder the side grasp still takes. Radii GRIPPER_RADII_M at GRIPPER_HEIGHT_M tall (and the repo's
size), the repo's pick + pull at the centre and GRASP_WALL_GAP_M from each wall and the front of the action box, at
each friction, with seeded jitter. A size grasps reliably if every run pulls the cylinder out.

sizes: each candidate (radius x height) and the repo's size as a baseline. A lone cylinder at SWEEP_SPOT_M swept with
skills/sweep.py in its 7 default directions, LENGTHS_M, FRICTIONS, REPETITIONS seeded jittered starts: topples,
largest tilt, jams and glitches. Plus the gripper check at each candidate. The choice: the smallest change (smallest
radius, then the tallest) with no topple and a largest tilt under MAX_TILT_DEG over every executed sweep but glitches
(jammed ones included: a cylinder tilting while pinned against a wall is the near-topple to rule out), and every grasp
working. table: redo the size table and the choice from the saved CSVs.
Steps 3 and 4, at CHOSEN_SIZE:
reach: the repo's pick + pull of a lone cylinder on a REACH_GRID_M grid over the board, at each friction, for the
chosen size and the repo's (reach_runs.csv; reach_grid.csv is the chosen size's graspable map for the scenes).
check: CHECK_SCENES random scenes (red in the chosen size's graspable zone, 3 grays anywhere; any start, blocked or
not): the default hybrid retrieval check and the kinematic one against physics (the exact mode: pick + pull run).
scenes, trials, analyse: study 10 again with the chosen size (common.random_sweeps; scenes in scenes/wide4/).
compare: study 10 against this study: success, topple, jam and other rates (comparison.csv, comparison.png).
videos: demo videos of saved trials into experiments/sahal/videos/study11/ (top-down view with the sweep drawn,
next to the wrist camera; 2 s of physics after a failure, for the video only).
glitches: replay every trial flagged as a glitch (same seed, so the same sweep) and record what physically happened:
the env's failure, each cylinder's largest tilt and which cylinder was fast (glitch_replays.csv, glitch_summary.json),
and the same for study 10's (glitch_replays_study10.csv, glitch_summary_study10.json; its random4_0000 trials that
ran with the friction bug may replay differently, and are counted apart).
--quick: two sizes and a few runs, into outputs/quick/. See README.md.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # experiments/sahal, for `common` and `skills`

import common  # noqa: E402,F401
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from common import (cylinder_size, measures, plotting, random_sweeps, retrieval_check, runner, scenes,  # noqa: E402
                    sweep_videos)
from common.cylinder_size import CylinderSize  # noqa: E402
from common.shelf_env import RedAndGraysShelf, SingleRedShelf, push_start_back_distance, seeded_jitter  # noqa: E402
from skills import sweep as sweep_skill  # noqa: E402
from so101sim.primitives import Unreachable  # noqa: E402
from so101sim.shelf import TARGET, TOPPLE_DEG  # noqa: E402

# ---- settings ------------------------------------------------------------------------------------------------
OUT_DIR = Path(__file__).resolve().parent / "outputs"
STUDY_SEED = 11
DENSITY_KG_M3 = 900.0
FRICTIONS = [0.4, 0.7]
RESET_SEED = 0
# gripper
GRIPPER_HEIGHT_M = 0.05
GRIPPER_RADII_M = [0.015, 0.02, 0.025, 0.03, 0.035, 0.04]
GRASP_WALL_GAP_M = 0.025  # near-wall spots: cylinder surface this far from the wall (the repo size retrieves here)
GRASP_FRONT_INSET_M = 0.005  # front spot: this far inside the action box's front edge
GRASP_REPETITIONS = 3
GRASP_JITTER_M = 0.002
# sizes
CANDIDATE_RADII_M = [0.02, 0.0225, 0.025]
CANDIDATE_HEIGHTS_M = [0.04, 0.05, 0.06]
SWEEP_SPOT_M = (0.4075, 0.0)  # study 05's centre spot
DIRECTIONS_DEG = list(sweep_skill.DEFAULT_DIRECTIONS_DEG)
LENGTHS_M = [0.03, 0.06, 0.10]
REPETITIONS = 3
MAX_TILT_DEG = 5.0
# steps 3 and 4
CHOSEN_SIZE = CylinderSize(0.0225, 0.05)  # step 2's choice (size_summary.json)
REACH_GRID_M = 0.02  # grid points at (k + 0.5) * REACH_GRID_M, as scenes.graspable_cells expects
REACH_X_M = (0.33, 0.45)  # first and last grid point
REACH_Y_M = (-0.13, 0.13)
CHECK_SCENES = 50
CHECK_SEED = 111
SWEEPS = random_sweeps.Settings(
    study_seed=STUDY_SEED, scene_dir=scenes.SCENES_DIR / "wide4", out_dir=OUT_DIR, scene_prefix="wide4",
    size=CHOSEN_SIZE, friction=0.55, n_scenes=10, n_grays=3, trials_per_scene=500, quick_trials=20,
    zone_csv=OUT_DIR / "reach_grid.csv", zone_grid_m=REACH_GRID_M,
    description="random: red in the r2.25 h5 graspable zone, 3 grays anywhere, not retrievable at start")
STUDY10_OUT = Path(__file__).resolve().parents[1] / "10_random_sweeps" / "outputs"
TOPPLED_ALL = "toppled (incl. glitch-flagged that toppled)"  # of all executed trials
STUDY_COLOURS = {"study 10 (r1.5 h8)": "#8a8a86", "study 11 (r2.25 h5)": plotting.OUTCOME_STYLE["slid"][0]}
# videos: per video the first trial (lowest number) in trials.csv that matches
VIDEO_DIR = Path(__file__).resolve().parents[2] / "videos" / "study11"
VIDEO_SCENE = "wide4_0007"  # the best scene for one random sweep (scene_table.csv)
VIDEO_KEY_GRAY = "cyl_gray_2"  # its gray whose sweep most often frees the red (by_cylinder.csv)
VIDEO_JAM_SCENE = "wide4_0021"  # the scene with the most jams
LONG_STROKE_CM = 8.0
VIDEO_PICKS = [
    sweep_videos.Pick("success_key_gray", VIDEO_SCENE, "success", VIDEO_KEY_GRAY, (0.0, LONG_STROKE_CM)),
    sweep_videos.Pick("success_red", VIDEO_SCENE, "success", TARGET, (0.0, LONG_STROKE_CM)),
    sweep_videos.Pick("success_long", VIDEO_SCENE, "success", None, (LONG_STROKE_CM, 100.0)),
    sweep_videos.Pick("fail_still_blocked", VIDEO_SCENE, "still blocked", VIDEO_KEY_GRAY),
    sweep_videos.Pick("fail_toppled", VIDEO_SCENE, "toppled"),
    sweep_videos.Pick("fail_jammed", VIDEO_JAM_SCENE, "jammed"),
]
VIDEO_LAYOUT = "top+wrist"
VIDEO_AFTER_FAILURE_S = 2.0
MAX_VIDEO_S = 30.0
# quick
QUICK_SIZES = [cylinder_size.REPO, CylinderSize(0.02, 0.05)]
QUICK_DIRECTIONS_DEG = [0, 90]
QUICK_LENGTHS_M = [0.06]
QUICK_REPETITIONS = 1

_ENV: SingleRedShelf | None = None
_ENV_KEY: tuple | None = None


def _env_for(size: CylinderSize, kind: type = SingleRedShelf) -> SingleRedShelf:
    """This worker's env of `kind`, rebuilt when a job needs another cylinder size or kind."""
    global _ENV, _ENV_KEY
    if _ENV_KEY != (size, kind):
        if _ENV is not None:
            _ENV.close()
        cylinder_size.use(size)
        _ENV, _ENV_KEY = kind(), (size, kind)
    return _ENV


def _init() -> None:
    """Worker start: nothing to build until a job names its size."""


def _size_record(size: CylinderSize) -> dict:
    return {"size": size.name, "radius_cm": round(size.radius_m * 100, 2), "height_cm": round(size.height_m * 100, 2),
            "grasp_height_cm": round(size.grasp_height_m * 100, 2)}


# ---- gripper -------------------------------------------------------------------------------------------------

def grasp_spots(env: SingleRedShelf) -> dict[str, tuple[float, float]]:
    """Centre, near the left, right and back walls, and the front of the action box, for the env's cylinder size."""
    layout, radius = env.env.layout, cylinder_size.current().radius_m
    x_lo = env.env.region[0]
    centre_x = SWEEP_SPOT_M[0]
    side_y = layout.half_width - radius - GRASP_WALL_GAP_M
    return {"centre": SWEEP_SPOT_M, "near left wall": (centre_x, side_y), "near right wall": (centre_x, -side_y),
            "near back wall": (layout.back_x - radius - GRASP_WALL_GAP_M, 0.0),
            "front": (x_lo + GRASP_FRONT_INSET_M, 0.0)}


def grasp_job(job: tuple[CylinderSize, str, float, int]) -> dict:
    """Place a lone cylinder at a spot (jittered), then the repo's pick + pull on it."""
    size, spot, friction, repetition = job
    env = _env_for(size)
    xy = np.asarray(grasp_spots(env)[spot])
    rng = np.random.default_rng([STUDY_SEED, 1, int(round(size.radius_m * 1e4)), int(round(size.height_m * 1e4)),
                                 list(grasp_spots(env)).index(spot), int(round(friction * 100)), repetition])
    xy = xy + rng.uniform(-GRASP_JITTER_M, GRASP_JITTER_M, 2)
    placed = env.reset(seed=RESET_SEED, red_xy=xy, physics={"friction": friction, "density": DENSITY_KG_M3})
    result = retrieval_check.pick_and_pull(env, placed["pos"][:2].copy())
    log = env.log
    return {**_size_record(size), "spot": spot, "friction": friction, "repetition": repetition,
            "x": round(float(xy[0]), 4), "y": round(float(xy[1]), 4), "success": result["success"],
            "reason": result["reason"], "gripper_at_end": round(float(env.env.joint_state[5]), 1),
            "open_value": env.open_value, "glitch": measures.is_glitch(log.max_contact_force_n, log.max_red_speed_m_s)}


# ---- sweeps --------------------------------------------------------------------------------------------------

def sweep_job(job: tuple[CylinderSize, int, float, float, int]) -> dict:
    """A lone cylinder at SWEEP_SPOT_M, swept from just behind it (seeded jitter), and what its tilt did."""
    size, direction_deg, length_m, friction, repetition = job
    env = _env_for(size)
    heading = math.radians(direction_deg)
    unit = np.array([math.cos(heading), math.sin(heading)])
    placed = env.reset(seed=RESET_SEED, red_xy=SWEEP_SPOT_M, physics={"friction": friction, "density": DENSITY_KG_M3})
    start_xy = placed["pos"][:2].copy()
    record = {**_size_record(size), "direction_deg": direction_deg, "length_cm": round(length_m * 100, 1),
              "friction": friction, "repetition": repetition}
    try:
        back = push_start_back_distance(env, start_xy, heading, sweep_skill.SWEEP_HEIGHTS_M[0])
    except Unreachable as exc:
        return {**record, "executed": False, "refused": f"IK: {exc}"}
    jitter = seeded_jitter([STUDY_SEED, int(round(size.radius_m * 1e4)), int(round(size.height_m * 1e4)),
                            direction_deg, int(round(length_m * 1000)), int(round(friction * 100)), repetition])
    start = start_xy - back * unit + jitter
    plan = sweep_skill.sweep(env, float(start[0]), float(start[1]), heading, length_m / sweep_skill.CONTACT_SPEED_M_S)
    if plan.refused is not None:
        return {**record, "executed": False, "refused": plan.refused}
    info = sweep_skill.execute(env, plan)
    if info["refused"] is not None:
        return {**record, "executed": False, "refused": info["refused"]}
    log = env.log
    max_tilt = max([f["tilt"] for f in log.frames] or [0.0])
    jammed = measures.is_wall_jam(log.max_contact_force_on_wall_n, log.arm_shelf_touch)
    glitch = measures.is_glitch(log.max_contact_force_n, log.max_red_speed_m_s) and not jammed
    moved = env.env.cylinder_pose(TARGET)["pos"][:2] - start_xy
    return {**record, "executed": True, "refused": None, "height_mm": round(plan.height_m * 1000),
            "max_tilt_deg": round(max_tilt, 2), "toppled": bool(max_tilt > TOPPLE_DEG and not glitch),
            "jammed": jammed, "glitch": glitch, "failure": info["failure"],
            "moved_mm": round(float(np.linalg.norm(moved)) * 1000, 1),
            "max_force_n": round(log.max_contact_force_n, 1), "max_speed_m_s": round(log.max_red_speed_m_s, 3)}


# ---- outputs -------------------------------------------------------------------------------------------------

def gripper_table(rows: list[dict]) -> list[dict]:
    """Per size: grasp + pull successes over runs, overall and per spot, and the failure reasons."""
    table = []
    for size in dict.fromkeys(r["size"] for r in rows):
        mine = [r for r in rows if r["size"] == size]
        row = {k: mine[0][k] for k in ("size", "radius_cm", "height_cm", "grasp_height_cm", "open_value")}
        row.update({"successes": sum(r["success"] for r in mine), "runs": len(mine),
                    "reliable": all(r["success"] for r in mine)})
        for spot in dict.fromkeys(r["spot"] for r in mine):
            at = [r for r in mine if r["spot"] == spot]
            row[f"ok_{spot}"] = f"{sum(r['success'] for r in at)}/{len(at)}"
        row["failures"] = "; ".join(sorted({f"{r['spot']}: {r['reason']}" for r in mine if not r["success"]}))
        table.append(row)
    return table


def size_table(sweeps: list[dict], grasps: list[dict]) -> list[dict]:
    """Per size: sweeps executed, refused, topples, largest tilt (glitches and jams excluded), jams, glitches,
    tilt at each friction, grasp successes, and whether it passes."""
    table = []
    for size in dict.fromkeys(r["size"] for r in sweeps):
        mine = [r for r in sweeps if r["size"] == size]
        ran = [r for r in mine if r["executed"]]
        clean = [r for r in ran if not r["glitch"] and not r["jammed"]]
        grasp = [r for r in grasps if r["size"] == size]
        row = {k: mine[0][k] for k in ("size", "radius_cm", "height_cm", "grasp_height_cm")}
        row.update({
            "sweeps": len(mine), "executed": len(ran), "refused": len(mine) - len(ran),
            "toppled": sum(r["toppled"] for r in ran), "jammed": sum(r["jammed"] for r in ran),
            "glitches": sum(r["glitch"] for r in ran),
            "max_tilt_deg": max((r["max_tilt_deg"] for r in clean), default=None),
            "max_tilt_including_jams_deg": max((r["max_tilt_deg"] for r in ran if not r["glitch"]), default=None)})
        for friction in FRICTIONS:
            at = [r["max_tilt_deg"] for r in clean if r["friction"] == friction]
            row[f"max_tilt_mu{friction}_deg"] = max(at, default=None)
            row[f"toppled_mu{friction}"] = sum(r["toppled"] for r in ran if r["friction"] == friction)
        row["grasp_successes"] = f"{sum(r['success'] for r in grasp)}/{len(grasp)}"
        worst = row["max_tilt_including_jams_deg"]
        row["passes"] = (row["toppled"] == 0 and worst is not None and worst < MAX_TILT_DEG
                         and bool(grasp) and all(r["success"] for r in grasp))
        table.append(row)
    return table


def choose(table: list[dict]) -> dict | None:
    """The smallest change that passes: smallest radius, then the tallest (nearest the repo's 8 cm)."""
    passing = [r for r in table if r["passes"] and r["size"] != cylinder_size.REPO.name]
    return min(passing, key=lambda r: (r["radius_cm"], -r["height_cm"]), default=None)


# ---- parts ---------------------------------------------------------------------------------------------------

def _grasp_jobs(sizes: list[CylinderSize], repetitions: int) -> list:
    spots = ["centre", "near left wall", "near right wall", "near back wall", "front"]
    return [(s, spot, f, k) for s in sizes for spot in spots for f in FRICTIONS for k in range(repetitions)]


def part_gripper(out: Path, quick: bool, workers: int) -> None:
    sizes = QUICK_SIZES if quick else [cylinder_size.REPO] + [CylinderSize(r, GRIPPER_HEIGHT_M) for r in GRIPPER_RADII_M]
    started = time.perf_counter()
    rows = runner.run_parallel(_grasp_jobs(sizes, 1 if quick else GRASP_REPETITIONS), grasp_job, _init, workers,
                               "grasps: ")
    runner.write_csv(out / "gripper_runs.csv", rows)
    table = gripper_table(rows)
    runner.write_csv(out / "gripper_table.csv", table)
    widest = max((r["radius_cm"] for r in table if r["reliable"] and r["height_cm"] == GRIPPER_HEIGHT_M * 100),
                 default=None)
    summary = {"wall_s": round(time.perf_counter() - started, 1), "widest_reliable_radius_cm": widest,
               "table": table}
    (out / "gripper_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


def part_sizes(out: Path, quick: bool, workers: int) -> None:
    sizes = QUICK_SIZES if quick else [cylinder_size.REPO] + [CylinderSize(r, h) for r in CANDIDATE_RADII_M
                                                                for h in CANDIDATE_HEIGHTS_M]
    directions = QUICK_DIRECTIONS_DEG if quick else DIRECTIONS_DEG
    lengths = QUICK_LENGTHS_M if quick else LENGTHS_M
    repetitions = QUICK_REPETITIONS if quick else REPETITIONS
    sweep_jobs = [(s, d, length, f, k) for s in sizes for f in FRICTIONS for d in directions for length in lengths
                  for k in range(repetitions)]
    started = time.perf_counter()
    sweeps = runner.run_parallel(sweep_jobs, sweep_job, _init, workers, "sweeps: ")
    sweep_wall_s = time.perf_counter() - started
    grasps = runner.run_parallel(_grasp_jobs(sizes, 1), grasp_job, _init, workers, "grasps: ")
    runner.write_csv(out / "size_sweeps.csv", sweeps)
    runner.write_csv(out / "size_grasps.csv", grasps)
    write_size_summary(out, sweeps, grasps, {"sweep_wall_s": round(sweep_wall_s, 1),
                                             "total_wall_s": round(time.perf_counter() - started, 1),
                                             "sweep_jobs": len(sweep_jobs), "workers": workers})


def write_size_summary(out: Path, sweeps: list[dict], grasps: list[dict], timing: dict) -> None:
    table = size_table(sweeps, grasps)
    runner.write_csv(out / "size_table.csv", table)
    chosen = choose(table)
    summary = {**timing, "chosen": chosen["size"] if chosen else None, "table": table}
    (out / "size_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "table"}, indent=1))


def part_table(out: Path) -> None:
    """Redo the size table and the choice from size_sweeps.csv and size_grasps.csv, keeping the run's timing."""
    previous = json.loads((out / "size_summary.json").read_text())
    timing = {k: previous[k] for k in ("sweep_wall_s", "total_wall_s", "sweep_jobs", "workers") if k in previous}
    write_size_summary(out, [_typed(r) for r in runner.read_csv(out / "size_sweeps.csv")],
                       [_typed(r) for r in runner.read_csv(out / "size_grasps.csv")], timing)


def _typed(row: dict) -> dict:
    """A CSV row with booleans and numbers restored (empty cells as None)."""
    out = {}
    for k, v in row.items():
        if v == "":
            out[k] = None
        elif v in ("True", "False"):
            out[k] = v == "True"
        else:
            try:
                out[k] = int(v)
            except ValueError:
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
    return out


# ---- step 3: reach grid and the retrieval check ---------------------------------------------------------------

def reach_points() -> list[tuple[float, float]]:
    xs = np.arange(REACH_X_M[0], REACH_X_M[1] + 1e-9, REACH_GRID_M)
    ys = np.arange(REACH_Y_M[0], REACH_Y_M[1] + 1e-9, REACH_GRID_M)
    return [(round(float(x), 4), round(float(y), 4)) for x in xs for y in ys]


def reach_job(job: tuple[CylinderSize, float, float, float]) -> dict:
    """A lone cylinder at a grid point, then the repo's pick + pull."""
    size, x, y, friction = job
    env = _env_for(size, RedAndGraysShelf)
    env.reset_scene(seed=RESET_SEED, red_xy=(x, y), gray_xys={},
                    physics={"friction": friction, "density": DENSITY_KG_M3})
    red_xy = env.env.cylinder_pose(TARGET)["pos"][:2].copy()
    result = retrieval_check.pick_and_pull(env, red_xy)
    return {"size": size.name, "x": x, "y": y, "friction": friction,
            "in_action_box": env.env._in_region(np.array([x, y])), "pull_success": result["success"],
            "reason": result["reason"]}


def plot_reach(path: Path, rows: list[dict], sizes: list[CylinderSize]) -> None:
    """One top-down map per size: each grid point coloured by how many frictions pick + pull worked at."""
    fig, axes = plt.subplots(1, len(sizes), figsize=(5.2 * len(sizes), 4.2))
    colours = {0: plotting.OUTCOME_STYLE["toppled"][0], 1: plotting.OUTCOME_STYLE["leaned"][0],
               2: plotting.OUTCOME_STYLE["slid"][0]}
    for ax, size in zip(np.atleast_1d(axes), sizes):
        cylinder_size.use(size)
        env = RedAndGraysShelf()
        plotting.draw_shelf(ax, env.env.layout, env.env.region)
        env.close()
        mine = [r for r in rows if r["size"] == size.name]
        for x, y in dict.fromkeys((r["x"], r["y"]) for r in mine):
            wins = sum(r["pull_success"] for r in mine if (r["x"], r["y"]) == (x, y))
            half = REACH_GRID_M / 2 * 0.9
            ax.add_patch(Rectangle((y - half, x - half), 2 * half, 2 * half, color=colours[wins], alpha=0.85))
        full = sum(all(r["pull_success"] for r in mine if (r["x"], r["y"]) == p)
                   for p in dict.fromkeys((r["x"], r["y"]) for r in mine))
        ax.set_title(f"{size.name}: pick + pull worked at both frictions at {full} of "
                     f"{len(set((r['x'], r['y']) for r in mine))} points", fontsize=8)
    handles = [Rectangle((0, 0), 1, 1, color=c, label=f"worked at {k} of {len(FRICTIONS)} frictions")
               for k, c in colours.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    plotting.save(fig, path, dpi=120)
    plt.close(fig)
    cylinder_size.use(cylinder_size.REPO)


def part_reach(out: Path, workers: int) -> None:
    sizes = [cylinder_size.REPO, CHOSEN_SIZE]
    jobs = [(s, x, y, f) for s in sizes for x, y in reach_points() for f in FRICTIONS]
    started = time.perf_counter()
    rows = runner.run_parallel(jobs, reach_job, _init, workers, "reach: ")
    runner.write_csv(out / "reach_runs.csv", rows)
    runner.write_csv(out / "reach_grid.csv", [r for r in rows if r["size"] == CHOSEN_SIZE.name])
    summary = {"wall_s": round(time.perf_counter() - started, 1)}
    for size in sizes:
        mine = [r for r in rows if r["size"] == size.name]
        points = list(dict.fromkeys((r["x"], r["y"]) for r in mine))
        in_box = [p for p in points if all(r["in_action_box"] for r in mine if (r["x"], r["y"]) == p)]
        full = [p for p in points if all(r["pull_success"] for r in mine if (r["x"], r["y"]) == p)]
        reasons: dict[str, int] = {}
        for r in mine:
            if not r["pull_success"]:
                key = r["reason"].split(":")[0] + ": " + r["reason"].split(":")[1].split(" at ")[0].strip() \
                    if ":" in r["reason"] else r["reason"]
                reasons[key] = reasons.get(key, 0) + 1
        summary[size.name] = {"points": len(points), "in_action_box": len(in_box), "worked_at_both_frictions": len(full),
                              "worked_in_action_box": len([p for p in full if p in in_box]),
                              "runs_failed_by_reason": reasons}
    summary["zone_cells_for_scenes"] = len(scenes.graspable_cells(out / "reach_grid.csv", grid_m=REACH_GRID_M))
    (out / "reach_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    plot_reach(out / "reach.png", rows, sizes)


def check_job(job: tuple[random_sweeps.Settings, int]) -> dict:
    """A random scene at the chosen size: the hybrid and kinematic checks (timed) against physics (exact)."""
    settings, index = job
    env = _env_for(CHOSEN_SIZE, RedAndGraysShelf)
    zone = random_sweeps.graspable_zone(settings)
    scene = random_sweeps.draw_candidate(settings, index, env.env.layout, zone)
    if scene is None:
        return {"index": index, "placed": False, "why": "no room"}
    try:
        scenes.place_scene(env, scene)
    except scenes.SceneError as exc:
        return {"index": index, "placed": False, "why": str(exc)}
    record = {"index": index, "placed": True, "scene": scene.name,
              "red_x": scene.positions[TARGET][0], "red_y": scene.positions[TARGET][1]}
    clearance = retrieval_check.min_gray_clearance(env)
    record["plan_gray_clearance_mm"] = round(clearance * 1000, 1) if clearance is not None else None
    for mode in ("hybrid", "kinematic", "exact"):
        started = time.perf_counter()
        ok, reason = retrieval_check.can_retrieve(env, mode=mode)
        record.update({f"{mode}_yes": ok, f"{mode}_reason": reason,
                       f"{mode}_s": round(time.perf_counter() - started, 3)})
    record["hybrid_simulated"] = record["hybrid_reason"].startswith("simulated")
    return record


def agreement(rows: list[dict], mode: str) -> dict:
    """The 2x2 table of a check against physics (exact)."""
    cells = {"check yes, physics yes": (True, True), "check yes, physics no": (True, False),
             "check no, physics yes": (False, True), "check no, physics no": (False, False)}
    table = {k: sum((r[f"{mode}_yes"], r["exact_yes"]) == v for r in rows) for k, v in cells.items()}
    table["agree"] = table["check yes, physics yes"] + table["check no, physics no"]
    return table


def part_check(out: Path, workers: int) -> None:
    settings = dataclasses.replace(SWEEPS, study_seed=CHECK_SEED)
    started = time.perf_counter()
    rows = runner.run_parallel([(settings, i) for i in range(CHECK_SCENES)], check_job, _init, workers, "check: ")
    runner.write_csv(out / "check_scenes.csv", rows)
    placed = [r for r in rows if r["placed"]]
    summary = {"wall_s": round(time.perf_counter() - started, 1), "scenes": len(rows), "placed": len(placed),
               "physics_yes": sum(r["exact_yes"] for r in placed),
               "hybrid": agreement(placed, "hybrid"), "kinematic": agreement(placed, "kinematic"),
               "hybrid_simulated": sum(r["hybrid_simulated"] for r in placed),
               "hybrid_wrong": [{k: r[k] for k in ("scene", "plan_gray_clearance_mm", "hybrid_reason", "exact_reason")}
                                for r in placed if r["hybrid_yes"] != r["exact_yes"]],
               "median_s": {m: round(float(np.median([r[f"{m}_s"] for r in placed])), 3)
                            for m in ("hybrid", "kinematic", "exact")}}
    (out / "check_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


# ---- step 4: comparison with study 10 ------------------------------------------------------------------------

def outcome_rates(summary: dict) -> dict:
    """Rates over executed, non-glitch trials (glitches as a share of executed), with 95% intervals."""
    overall = summary["overall"]
    valid = overall["executed"] - overall["glitches"]
    rates = {}
    for label, key in (("success", "n_success"), ("still blocked", "n_still_blocked"), ("toppled", "n_toppled"),
                       ("jammed", "n_jammed"), ("gray left the board", "n_gray_left_the_board"),
                       ("red left the board", "n_red_left_the_board"), ("arm hit the shelf", "n_arm_hit_the_shelf")):
        rates[label] = {"count": overall[key], "of": valid, "rate": round(overall[key] / valid, 4),
                        "ci95": random_sweeps.wilson(overall[key], valid)}
    rates["glitch"] = {"count": overall["glitches"], "of": overall["executed"],
                       "rate": round(overall["glitches"] / overall["executed"], 4),
                       "ci95": random_sweeps.wilson(overall["glitches"], overall["executed"])}
    rates["refused before moving"] = {"count": overall["refusals"], "of": overall["refusals"] + overall["executed"],
                                      "rate": overall["refusal_rate"], "ci95": None}
    return rates


def part_compare(out: Path) -> None:
    studies = {"study 10 (r1.5 h8)": json.loads((STUDY10_OUT / "summary.json").read_text()),
               "study 11 (r2.25 h5)": json.loads((out / "summary.json").read_text())}
    rates = {name: outcome_rates(s) for name, s in studies.items()}
    glitch_files = {"study 10 (r1.5 h8)": out / "glitch_summary_study10.json",
                    "study 11 (r2.25 h5)": out / "glitch_summary.json"}
    if all(f.exists() for f in glitch_files.values()):  # from --parts glitches
        for name, path in glitch_files.items():
            toppled_glitches = json.loads(path.read_text())["env_failure"].get("toppled", 0)
            executed = studies[name]["overall"]["executed"]
            count = studies[name]["overall"]["n_toppled"] + toppled_glitches
            rates[name][TOPPLED_ALL] = {"count": count, "of": executed, "rate": round(count / executed, 4),
                                        "ci95": random_sweeps.wilson(count, executed)}
    rows = []
    for outcome in rates["study 10 (r1.5 h8)"]:
        row = {"outcome": outcome}
        for name, r in rates.items():
            short = name.split(" (")[0].replace(" ", "")
            row.update({f"{short}_count": r[outcome]["count"], f"{short}_of": r[outcome]["of"],
                        f"{short}_rate": r[outcome]["rate"], f"{short}_ci95": r[outcome]["ci95"]})
        rows.append(row)
    extra = {name: {"mean_progress_mm": s["overall"]["mean_progress_mm"],
                    "success_by_swept_cylinder": {k: v["success_rate"] for k, v in s["overall_by_swept_cylinder"].items()},
                    "success_by_length": {k: v["success_rate"] for k, v in s["by_length"].items()},
                    "success_by_scene": [r["success_rate"] for r in s["by_scene"]],
                    "trial_wall_s_median": s["trial_wall_s"]["median"]} for name, s in studies.items()}
    runner.write_csv(out / "comparison.csv", rows)
    (out / "comparison.json").write_text(json.dumps({"rates": rates, "other": extra}, indent=1) + "\n")
    print(json.dumps({"rates": rates, "other": extra}, indent=1))
    plot_compare(out / "comparison.png", rates)


def plot_compare(path: Path, rates: dict) -> None:
    """Success, topple, jam and gray-off rates of the two studies side by side, with 95% intervals."""
    outcomes = [o for o in ("success", "still blocked", "toppled", TOPPLED_ALL, "jammed", "gray left the board",
                            "glitch") if all(o in r for r in rates.values())]
    fig, ax = plt.subplots(figsize=(9.6, 4.0))
    width = 0.38
    for k, (name, r) in enumerate(rates.items()):
        xs = np.arange(len(outcomes)) + (k - 0.5) * width
        values = [r[o]["rate"] for o in outcomes]
        low = [r[o]["rate"] - r[o]["ci95"][0] for o in outcomes]
        high = [r[o]["ci95"][1] - r[o]["rate"] for o in outcomes]
        ax.bar(xs, values, width * 0.92, color=STUDY_COLOURS[name], label=name)
        ax.errorbar(xs, values, yerr=[low, high], fmt="none", ecolor="#333333", elinewidth=0.8, capsize=2)
        for x, v in zip(xs, values):
            ax.annotate(f"{v:.1%}", (x, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7)
    ax.set_xticks(np.arange(len(outcomes)), [o.replace(" (", "\n(") for o in outcomes], fontsize=7.5)
    ax.set_ylabel("share of trials (95% interval)", fontsize=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e5e2", lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("One random sweep: outcomes, repo cylinders vs wider shorter ones", fontsize=9)
    fig.tight_layout()
    plotting.save(fig, path, dpi=130)
    plt.close(fig)


# ---- glitch replays -------------------------------------------------------------------------------------------

def glitch_job(job: tuple[random_sweeps.Settings, int, str, int]) -> dict:
    """Replay one trial's sweep from the scene's settled start and record what happened to every cylinder."""
    settings, scene_index, scene_name, trial = job
    env = _env_for(settings.size, RedAndGraysShelf)
    scene = scenes.load_scene(settings.scene_dir / f"{scene_name}.json")
    scenes.place_scene(env, scene)
    names = sorted(scene.positions)
    tilts = {n: 0.0 for n in names}
    speeds = {n: 0.0 for n in names}
    inner = env.env

    def watch(_env) -> None:  # every 30 Hz frame: tilt and speed of each cylinder
        for n in names:
            tilts[n] = max(tilts[n], inner.tilt(n))
            dof = inner._dofadr[n]
            speeds[n] = max(speeds[n], float(np.linalg.norm(env.data.qvel[dof:dof + 3])))

    env.on_frame = lambda e: (env._on_frame(e), watch(e))
    try:
        refusals: list[str] = []
        sample, plan, info = random_sweeps.first_executable_sweep(
            settings, env, random_sweeps.trial_rng(settings, scene_index, trial), names, refusals)
    finally:
        env.on_frame = env._on_frame
    log = env.log
    fast = max(speeds, key=speeds.get)
    return {"scene": scene_name, "trial": trial, "cylinder": sample["cylinder"], "replayed_as": random_sweeps.classify(env, info),
            "failure": info["failure"], "max_force_n": round(max(log.max_contact_force_n, log.max_gray_contact_force_n), 1),
            "fastest_cylinder": fast, "fastest_speed_m_s": round(speeds[fast], 3),
            "fastest_was_swept": fast == sample["cylinder"],
            "any_tilt_over_topple": any(t > TOPPLE_DEG for t in tilts.values()),
            "largest_tilt_deg": round(max(tilts.values()), 1),
            "largest_tilt_cylinder": max(tilts, key=tilts.get)}


STUDY10_SWEEPS = random_sweeps.Settings(study_seed=10, scene_dir=scenes.SCENES_DIR / "random4", out_dir=STUDY10_OUT,
                                        scene_prefix="random4")  # study 10's settings (its run.py)
STUDY10_BUG_SCENE = "random4_0000"  # its scene whose first-job trials ran with the friction bug (study 10 README)


def part_glitches(out: Path, workers: int) -> None:
    replay_glitches(out, SWEEPS, out / "trials.csv", "", workers)
    replay_glitches(out, STUDY10_SWEEPS, STUDY10_OUT / "trials.csv", "_study10", workers)


def replay_glitches(out: Path, settings: random_sweeps.Settings, trials_csv: Path, suffix: str, workers: int) -> None:
    rows = [random_sweeps.typed(r) for r in runner.read_csv(trials_csv)]
    flagged = [r for r in rows if r["outcome"] == "glitch"]
    jobs = [(settings, r["scene_index"], r["scene"], r["trial"]) for r in flagged]
    replays = runner.run_parallel(jobs, glitch_job, _init, workers, "glitch replays: ")
    runner.write_csv(out / f"glitch_replays{suffix}.csv", replays)
    force_ok = [r for r in replays if r["max_force_n"] <= measures.GLITCH_CONTACT_FORCE_N]
    summary = {
        "flagged": len(flagged), "replayed_as_glitch": sum(r["replayed_as"] == "glitch" for r in replays),
        "env_failure": {k: sum((r["failure"] or "none").split(":")[0] == k for r in replays)
                        for k in sorted({(r["failure"] or "none").split(":")[0] for r in replays})},
        "a_cylinder_tilted_past_topple": sum(r["any_tilt_over_topple"] for r in replays),
        "force_at_most_glitch_limit": len(force_ok),
        "of_those_a_cylinder_tilted_past_topple": sum(r["any_tilt_over_topple"] for r in force_ok),
        "of_those_largest_tilt_deg_quantiles": np.percentile([r["largest_tilt_deg"] for r in force_ok],
                                                            [10, 50, 90]).round(1).tolist() if force_ok else None,
        "fastest_was_swept": sum(r["fastest_was_swept"] for r in replays),
        "in_scene_with_friction_bug": sum(r["scene"] == STUDY10_BUG_SCENE for r in replays) if suffix else 0}
    (out / f"glitch_summary{suffix}.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


# ---- videos --------------------------------------------------------------------------------------------------

def part_videos(out: Path) -> None:
    rows = [random_sweeps.typed(r) for r in runner.read_csv(out / "trials.csv")]
    report = sweep_videos.make_videos(SWEEPS, rows, VIDEO_PICKS, VIDEO_DIR, VIDEO_LAYOUT, VIDEO_AFTER_FAILURE_S,
                                      MAX_VIDEO_S)
    runner.write_csv(VIDEO_DIR / "videos.csv", report)
    runner.save_config(VIDEO_DIR, {"picks": [p.__dict__ for p in VIDEO_PICKS], "layout": VIDEO_LAYOUT,
                                   "after_failure_s": VIDEO_AFTER_FAILURE_S, "fps": sweep_videos.VIDEO_FPS,
                                   "slow_motion_repeat": sweep_videos.SLOW_MOTION_REPEAT,
                                   "slow_motion_pad_frames": sweep_videos.SLOW_MOTION_PAD_FRAMES,
                                   "end_hold_s": sweep_videos.END_HOLD_S, "quality": sweep_videos.QUALITY})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="two sizes, a few runs, into outputs/quick/")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--parts", nargs="+", default=["gripper", "sizes"],
                        choices=["gripper", "sizes", "table", "reach", "check", "scenes", "trials", "analyse",
                                 "compare", "glitches", "videos"])
    args = parser.parse_args()
    out = OUT_DIR / "quick" if args.quick else OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    if "gripper" in args.parts:
        part_gripper(out, args.quick, args.workers)
    if "sizes" in args.parts:
        part_sizes(out, args.quick, args.workers)
    if "table" in args.parts:
        part_table(out)
        return  # the run's config.json stays
    if "reach" in args.parts:
        part_reach(out, args.workers)
    if "check" in args.parts:
        part_check(out, args.workers)
    sweeps = dataclasses.replace(SWEEPS, out_dir=out)
    if "scenes" in args.parts:
        random_sweeps.part_scenes(sweeps, args.workers)
    if "trials" in args.parts:
        random_sweeps.part_trials(sweeps, out, args.quick, args.workers)
    if "analyse" in args.parts:
        random_sweeps.part_analyse(sweeps, out, args.quick)
    if "compare" in args.parts:
        part_compare(out)
    if "videos" in args.parts:
        part_videos(out)
        return  # the videos keep their own config.json
    if "glitches" in args.parts:
        part_glitches(out, args.workers)
        return  # a diagnostic: the run's config.json stays
    runner.save_config(out, {
        "quick": args.quick, "parts": args.parts, "study_seed": STUDY_SEED, "frictions": FRICTIONS,
        "density_kg_m3": DENSITY_KG_M3, "gripper_height_m": GRIPPER_HEIGHT_M, "gripper_radii_m": GRIPPER_RADII_M,
        "grasp_wall_gap_m": GRASP_WALL_GAP_M, "grasp_front_inset_m": GRASP_FRONT_INSET_M,
        "grasp_repetitions": GRASP_REPETITIONS, "grasp_jitter_m": GRASP_JITTER_M,
        "candidate_radii_m": CANDIDATE_RADII_M, "candidate_heights_m": CANDIDATE_HEIGHTS_M,
        "sweep_spot_m": SWEEP_SPOT_M, "directions_deg": DIRECTIONS_DEG, "lengths_m": LENGTHS_M,
        "repetitions": REPETITIONS, "max_tilt_deg": MAX_TILT_DEG, "sweep_heights_m": sweep_skill.SWEEP_HEIGHTS_M,
        "open_gap_slack_m": cylinder_size.OPEN_GAP_SLACK_M, "pick_lift_m": cylinder_size.PICK_LIFT_M,
        "chosen_size": CHOSEN_SIZE.__dict__, "reach_grid_m": REACH_GRID_M, "reach_x_m": REACH_X_M,
        "reach_y_m": REACH_Y_M, "check_scenes": CHECK_SCENES, "check_seed": CHECK_SEED,
        "random_sweeps": SWEEPS.config()})


if __name__ == "__main__":
    main()
