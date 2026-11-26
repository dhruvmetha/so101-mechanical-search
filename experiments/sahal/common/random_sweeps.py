"""Random single sweeps on random blocked scenes (studies 10 and 11): scenes, trials and their analysis.

    settings = random_sweeps.Settings(study_seed=10, scene_dir=..., out_dir=..., scene_prefix="random4")
    random_sweeps.part_scenes(settings, workers)       # keep the first N blocked candidates, JSON + picture each
    random_sweeps.part_trials(settings, out, quick, workers)
    random_sweeps.part_analyse(settings, out, quick)   # tables and plots again from trials.csv

Scenes: the red uniform over the board, kept only inside study 01's graspable zone; then the grays uniform over the
board, 5 mm or more apart; kept if the scene stands (common.scenes.place_scene) and can_retrieve(mode="exact") says
no. Trials: from the scene's settled start, sample a sweep (any cylinder, a direction within max_direction_deg of
away from the robot, a stroke in length_range_m, the skill's start moved up to start_sideways_m sideways and
start_back_m further back); a sweep refused before moving is logged and another drawn; then jammed, glitch, toppled,
off the board, arm hit, or can_retrieve(mode="exact") decides success or still blocked. Every random draw is seeded
by (study_seed, scene index, trial). `settings.size` sets the cylinder size (common.cylinder_size) in every worker.
"""

from __future__ import annotations

import functools
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import common  # noqa: F401  sets sys.path and MUJOCO_GL before mujoco loads
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from common import cylinder_size, measures, plotting, retrieval_check, runner, scenes
from common.shelf_env import RedAndGraysShelf, push_start_back_distance
from skills import sweep as sweep_skill
from so101sim import shelf_layout
from so101sim.primitives import Unreachable
from so101sim.shelf import TARGET

Z_95 = 1.959964
OUTCOMES = ("success", "still blocked", "toppled", "jammed", "gray left the board", "red left the board",
            "arm hit the shelf", "glitch")
OUTCOME_COLOURS = {"success": plotting.OUTCOME_STYLE["slid"][0], "still blocked": "#8a8a86",
                   "toppled": plotting.OUTCOME_STYLE["toppled"][0], "jammed": plotting.OUTCOME_STYLE["jammed"][0],
                   "gray left the board": plotting.OUTCOME_STYLE["leaned"][0], "red left the board": "#e87ba4",
                   "arm hit the shelf": "#222222", "glitch": "#cfcfca"}
GRAY_COLOUR = "#8a8a86"


@dataclass(frozen=True)
class Settings:
    """Everything that decides a random-sweeps study's scenes and trials."""

    study_seed: int
    scene_dir: Path
    out_dir: Path
    scene_prefix: str
    size: cylinder_size.CylinderSize = cylinder_size.REPO
    friction: float = 0.55
    n_scenes: int = 10
    n_grays: int = 3
    candidate_batch: int = 40  # scene candidates checked per round, in parallel
    max_candidates: int = 2000
    max_placement_tries: int = 500
    trials_per_scene: int = 500
    trials_per_job: int = 25
    quick_trials: int = 20
    length_range_m: tuple[float, float] = (0.02, 0.12)
    max_direction_deg: float = 180 - sweep_skill.TOWARD_ROBOT_HALF_ANGLE_DEG  # within +-157.5 deg of away
    start_sideways_m: float = 0.01  # start offset across the stroke, uniform in +-this
    start_back_m: float = 0.01  # start offset further behind the cylinder, uniform in [0, this]
    max_attempts: int = 200  # sweeps sampled per trial before giving up
    length_bins_cm: tuple = ((2, 5), (5, 8), (8, 12))
    zone_csv: Path = scenes.GRASP_ZONE_CSV  # where the red may stand: a grasp map (scenes.graspable_cells)
    zone_grid_m: float = scenes.GRASP_ZONE_GRID_M
    description: str = field(default="random: red in the graspable zone, 3 grays anywhere, not retrievable at start")

    def config(self) -> dict:
        return {k: (str(v) if isinstance(v, Path) else v.__dict__ if k == "size" else v)
                for k, v in self.__dict__.items()}


_ENV: RedAndGraysShelf | None = None


def init_worker(settings: Settings) -> None:
    """Set the cylinder size, then build this process's env."""
    global _ENV
    cylinder_size.use(settings.size)
    _ENV = RedAndGraysShelf()


def make_env(settings: Settings) -> RedAndGraysShelf:
    cylinder_size.use(settings.size)
    return RedAndGraysShelf()


# ---- scenes ----------------------------------------------------------------------------------------------------

def graspable_zone(settings: Settings) -> set[tuple[int, int]]:
    return scenes.graspable_cells(settings.zone_csv, grid_m=settings.zone_grid_m)


def draw_candidate(settings: Settings, index: int, layout, zone: set | None) -> scenes.Scene | None:
    """Red uniform over the board where a cylinder fits, accepted only inside the graspable zone (None: anywhere);
    then the grays uniform over the board, 5 mm or more apart. Seeded by the candidate's index."""
    rng = np.random.default_rng([settings.study_seed, index])
    radius = shelf_layout.CYL_RADIUS
    inset = radius + shelf_layout.WALL_GAP

    def anywhere() -> np.ndarray:
        return np.array([rng.uniform(layout.front_x + radius, layout.back_x - inset),
                         rng.uniform(-layout.half_width + inset, layout.half_width - inset)])

    positions: dict[str, tuple[float, float]] = {}
    for name in (TARGET, *(f"{scenes.GRAY_PREFIX}{k}" for k in range(settings.n_grays))):
        for _ in range(settings.max_placement_tries):
            xy = anywhere()
            if name == TARGET and zone is not None and not scenes.in_graspable_zone(xy, zone, settings.zone_grid_m):
                continue
            if all(np.linalg.norm(xy - np.asarray(p)) >= 2 * radius + scenes.MIN_SURFACE_GAP_M
                   for p in positions.values()):
                positions[name] = (round(float(xy[0]), 4), round(float(xy[1]), 4))
                break
        else:
            return None
    return scenes.Scene(name=f"{settings.scene_prefix}_{index:04d}", level="random", friction=settings.friction,
                        positions=positions, description=settings.description)


def candidate_job(job: tuple[Settings, int]) -> dict:
    """Draw, place and check one candidate scene."""
    settings, index = job
    env = _ENV
    scene = draw_candidate(settings, index, env.env.layout, graspable_zone(settings))
    if scene is None:
        return {"index": index, "kept": False, "why": "no room"}
    try:
        scenes.place_scene(env, scene)
    except scenes.SceneError as exc:
        return {"index": index, "kept": False, "why": f"placement: {exc}"}
    ok, reason = retrieval_check.can_retrieve(env, mode="exact")
    return {"index": index, "kept": not ok, "why": "retrievable at start" if ok else reason,
            "scene": scene.__dict__}


def part_scenes(settings: Settings, workers: int) -> list[scenes.Scene]:
    """Check candidates in index order until n_scenes are kept; save them with a picture each."""
    kept, log, start = [], [], 0
    while len(kept) < settings.n_scenes and start < settings.max_candidates:
        jobs = [(settings, i) for i in range(start, start + settings.candidate_batch)]
        batch = runner.run_parallel(jobs, candidate_job, functools.partial(init_worker, settings), workers,
                                    "candidates: ")
        log += [{k: v for k, v in r.items() if k != "scene"} for r in batch]
        kept += [scenes.Scene(**r["scene"]) for r in batch if r["kept"]]
        start += settings.candidate_batch
    kept = kept[:settings.n_scenes]
    settings.scene_dir.mkdir(parents=True, exist_ok=True)
    for old in settings.scene_dir.glob("*"):
        old.unlink()
    env = make_env(settings)
    try:
        for scene in kept:
            scenes.save_scene(scene, settings.scene_dir)
            scenes.place_scene(env, scene)
            picture_scene(settings.scene_dir / f"{scene.name}.png", scene, env)
    finally:
        env.close()
    last = max(int(s.name.split("_")[1]) for s in kept)
    runner.write_csv(settings.out_dir / "scene_candidates.csv", [r for r in log if r["index"] <= last])
    print(f"scenes: kept {len(kept)} of {last + 1} candidates")
    return kept


def picture_scene(path: Path, scene: scenes.Scene, env: RedAndGraysShelf) -> None:
    """Top-down layout and the front camera, for a placed scene."""
    fig = plt.figure(figsize=(10, 3.6))
    top = fig.add_axes([0.03, 0.12, 0.33, 0.76])
    draw_cylinders(top, env, scene.positions, label=True)
    top.set_title(scene.name, fontsize=9)
    image = fig.add_axes([0.38, 0.05, 0.6, 0.9])
    image.imshow(env.env._render("front"))
    image.set_axis_off()
    plotting.save(fig, path, dpi=100)
    plt.close(fig)


def draw_cylinders(ax, env: RedAndGraysShelf, positions: dict, label: bool = False) -> None:
    plotting.draw_shelf(ax, env.env.layout, env.env.region)
    for name, (x, y) in positions.items():
        colour = plotting.OUTCOME_STYLE["toppled"][0] if name == TARGET else GRAY_COLOUR
        ax.add_patch(Circle((y, x), shelf_layout.CYL_RADIUS, color=colour, zorder=3, alpha=0.9))
        if label:
            ax.annotate(name.replace("cyl_", ""), (y, x), xytext=(0, 9), textcoords="offset points", ha="center",
                        fontsize=7)


# ---- trials ----------------------------------------------------------------------------------------------------

def sample_sweep(settings: Settings, env: RedAndGraysShelf, rng: np.random.Generator,
                 names: list[str]) -> tuple[dict, object]:
    """One random sweep: (what was sampled, the plan or a refusal reason). All five draws happen every time, so the
    random stream does not depend on what was refused."""
    cylinder = names[int(rng.integers(len(names)))]
    direction_deg = float(rng.uniform(-settings.max_direction_deg, settings.max_direction_deg))
    length_m = float(rng.uniform(*settings.length_range_m))
    sideways_m = float(rng.uniform(-settings.start_sideways_m, settings.start_sideways_m))
    back_extra_m = float(rng.uniform(0, settings.start_back_m))
    heading = math.radians(direction_deg)
    unit, left = np.array([math.cos(heading), math.sin(heading)]), np.array([-math.sin(heading), math.cos(heading)])
    xy = env.env.cylinder_pose(cylinder)["pos"][:2].copy()
    sample = {"cylinder": cylinder, "direction_deg": round(direction_deg, 2), "length_cm": round(length_m * 100, 2),
              "offset_sideways_mm": round(sideways_m * 1000, 2), "offset_back_mm": round(back_extra_m * 1000, 2)}
    try:
        back = push_start_back_distance(env, xy, heading, sweep_skill.SWEEP_HEIGHTS_M[0], cylinder=cylinder)
    except Unreachable as exc:
        return sample, f"IK: start behind the cylinder unreachable: {exc}"
    start = xy - (back + back_extra_m) * unit + sideways_m * left
    plan = sweep_skill.sweep(env, float(start[0]), float(start[1]), heading, length_m / sweep_skill.CONTACT_SPEED_M_S)
    sample.update({"start_x_m": round(float(start[0]), 4), "start_y_m": round(float(start[1]), 4)})
    return sample, plan if plan.refused is None else plan.refused


def refusal_kind(reason: str) -> str:
    if "off the board" in reason:
        return "off the board"
    if "path check" in reason or "would hit" in reason:
        return "path check"
    return "IK" if "IK" in reason or "unreachable" in reason else reason[:30]


def classify(env: RedAndGraysShelf, info: dict) -> str | None:
    """What one executed sweep did, before the retrieval check: jammed, glitch, an env failure, or None."""
    log = env.log
    if measures.is_wall_jam(max(log.max_contact_force_on_wall_n, log.max_gray_force_on_wall_n), log.arm_shelf_touch):
        return "jammed"
    if measures.is_glitch(max(log.max_contact_force_n, log.max_gray_contact_force_n),
                          max(log.max_red_speed_m_s, log.max_gray_speed_m_s)):
        return "glitch"
    failure = info["failure"] or ""
    if failure.startswith("toppled"):
        return "toppled"
    if failure.startswith(f"left the shelf: {scenes.GRAY_PREFIX}"):
        return "gray left the board"
    if failure.startswith("left the shelf"):
        return "red left the board"
    if failure.startswith("hit the shelf"):
        return "arm hit the shelf"
    return None


def first_executable_sweep(settings: Settings, env: RedAndGraysShelf, rng: np.random.Generator, names: list[str],
                           refusals: list[str]) -> tuple[dict, object, dict] | None:
    """Sample sweeps until one runs, and run it: (sample, plan, the env's info), or None if max_attempts were all
    refused. Each refusal's kind is appended to `refusals`."""
    for _ in range(settings.max_attempts):
        sample, plan = sample_sweep(settings, env, rng, names)
        if isinstance(plan, str):
            refusals.append(refusal_kind(plan))
            continue
        info = sweep_skill.execute(env, plan)
        if info["refused"] is not None:  # the env refused before moving
            refusals.append(refusal_kind(info["refused"]))
            continue
        return sample, plan, info
    return None


def trial_rng(settings: Settings, scene_index: int, trial: int) -> np.random.Generator:
    return np.random.default_rng([settings.study_seed, 1, scene_index, trial])


def run_trial(settings: Settings, env: RedAndGraysShelf, scene_index: int, trial: int, names: list[str],
              before_mm: float | None) -> dict:
    """One trial from the scene's settled start; the state is put back afterwards."""
    rng = trial_rng(settings, scene_index, trial)
    started = time.perf_counter()
    refusals: list[str] = []
    record: dict = {"scene_index": scene_index, "trial": trial}
    with retrieval_check.simulation_copy(env):
        executed = first_executable_sweep(settings, env, rng, names, refusals)
        if executed is None:
            return {**record, "outcome": "no executable sweep", "refusals": len(refusals)}
        sample, plan, info = executed
        outcome = classify(env, info)
        after = None
        if outcome is None:
            ok, _ = retrieval_check.can_retrieve(env, mode="exact")
            outcome = "success" if ok else "still blocked"
            after = retrieval_check.min_gray_clearance(env)
        end = np.array(plan.end_xy_m)
        record.update({**sample, "end_x_m": round(float(end[0]), 4), "end_y_m": round(float(end[1]), 4),
                       "height_mm": round(plan.height_m * 1000), "outcome": outcome,
                       "refusals": len(refusals), **{f"refused_{k}": refusals.count(k) for k in set(refusals)},
                       "clearance_before_mm": before_mm,
                       "clearance_after_mm": round(after * 1000, 1) if after is not None else None,
                       "progress_mm": round(after * 1000 - before_mm, 1) if after is not None and before_mm is not None
                       else None,
                       "max_contact_force_n": round(max(env.log.max_contact_force_n, env.log.max_gray_contact_force_n), 1),
                       "max_speed_m_s": round(max(env.log.max_red_speed_m_s, env.log.max_gray_speed_m_s), 3)})
    record["wall_s"] = round(time.perf_counter() - started, 2)
    return record


def trials_job(job: tuple[Settings, int, str, list[int]]) -> list[dict]:
    """Place one scene once, then run a block of its trials, each from the same settled state."""
    settings, scene_index, scene_path, trials = job
    env = _ENV
    scene = scenes.load_scene(Path(scene_path))
    scenes.place_scene(env, scene)
    before = retrieval_check.min_gray_clearance(env)
    before_mm = round(before * 1000, 1) if before is not None else None
    names = sorted(scene.positions)
    return [{"scene": scene.name, **run_trial(settings, env, scene_index, t, names, before_mm)} for t in trials]


def scene_list(settings: Settings, quick: bool) -> list[scenes.Scene]:
    found = scenes.load_scenes(settings.scene_dir)
    if not found:
        raise SystemExit("no scenes: run --parts scenes first")
    return found[:1] if quick else found


def part_trials(settings: Settings, out: Path, quick: bool, workers: int) -> None:
    chosen = scene_list(settings, quick)
    n_trials = settings.quick_trials if quick else settings.trials_per_scene
    jobs = [(settings, i, str(settings.scene_dir / f"{s.name}.json"),
             list(range(t, min(t + settings.trials_per_job, n_trials))))
            for i, s in enumerate(chosen) for t in range(0, n_trials, settings.trials_per_job)]
    started = time.perf_counter()
    rows = [r for block in runner.run_parallel(jobs, trials_job, functools.partial(init_worker, settings), workers,
                                               "trial blocks: ") for r in block]
    wall = time.perf_counter() - started
    runner.write_csv(out / "trials.csv", rows)
    analyse(settings, out, rows, chosen, round(wall, 1))


# ---- outputs ---------------------------------------------------------------------------------------------------

def wilson(successes: int, n: int) -> list[float] | None:
    """95% Wilson score interval for a rate."""
    if n == 0:
        return None
    p, z2 = successes / n, Z_95 ** 2
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = Z_95 * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return [round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3)]


def rate_row(rows: list[dict]) -> dict:
    """Success rate over executed, non-glitch trials, with its interval, the outcome counts and refusals."""
    executed = [r for r in rows if r["outcome"] != "no executable sweep"]
    valid = [r for r in executed if r["outcome"] != "glitch"]
    wins = sum(r["outcome"] == "success" for r in valid)
    refusals = sum(r["refusals"] for r in rows)
    progress = [r["progress_mm"] for r in rows if r.get("progress_mm") is not None]
    return {"trials": len(rows), "executed": len(executed), "glitches": len(executed) - len(valid),
            "successes": wins, "success_rate": round(wins / len(valid), 3) if valid else None,
            "success_ci95": wilson(wins, len(valid)),
            **{f"n_{o.replace(' ', '_')}": sum(r["outcome"] == o for r in executed) for o in OUTCOMES},
            "refusals": refusals, "refusal_rate": round(refusals / (refusals + len(executed)), 3)
            if refusals + len(executed) else None,
            "mean_progress_mm": round(float(np.mean(progress)), 1) if progress else None,
            "trials_with_progress_value": len(progress)}


def summarise(settings: Settings, rows: list[dict], chosen: list[scenes.Scene]) -> dict:
    table, by_cylinder = [], []
    for scene in chosen:
        rs = [r for r in rows if r["scene"] == scene.name]
        if not rs:
            continue
        red = scene.positions[TARGET]
        table.append({"scene": scene.name, "red_x_m": red[0], "red_y_m": red[1],
                      "clearance_before_mm": rs[0].get("clearance_before_mm"), **rate_row(rs)})
        for name in sorted(scene.positions):
            cs = [r for r in rs if r.get("cylinder") == name]
            by_cylinder.append({"scene": scene.name, "cylinder": name, **rate_row(cs)} if cs else
                               {"scene": scene.name, "cylinder": name, "trials": 0})
    overall_by_kind = {kind: rate_row([r for r in rows if r.get("cylinder") and
                                       (r["cylinder"] == TARGET) == (kind == "red")])
                       for kind in ("red", "a gray")}

    def progress(outcome: str) -> dict:
        values = [r["progress_mm"] for r in rows if r["outcome"] == outcome and r.get("progress_mm") is not None]
        return {"trials": len(values), "mean_mm": round(float(np.mean(values)), 1),
                "median_mm": round(float(np.median(values)), 1),
                "share_positive": round(float(np.mean(np.array(values) > 0)), 3)} if values else {}

    last_hi = settings.length_bins_cm[-1][1]
    by_length = {f"{lo}-{hi} cm": rate_row([r for r in rows if r.get("length_cm") is not None
                                            and lo <= r["length_cm"] < hi + (1e-9 if hi == last_hi else 0)])
                 for lo, hi in settings.length_bins_cm}
    return {"overall": rate_row(rows), "by_scene": table, "by_cylinder": by_cylinder,
            "overall_by_swept_cylinder": overall_by_kind, "by_length": by_length,
            "progress_by_outcome": {o: progress(o) for o in ("success", "still blocked")},
            "trial_wall_s": {"median": round(float(np.median([r["wall_s"] for r in rows if "wall_s" in r])), 2),
                             "mean": round(float(np.mean([r["wall_s"] for r in rows if "wall_s" in r])), 2)}}


def plot_scene_sweeps(path: Path, scene: scenes.Scene, rows: list[dict], env: RedAndGraysShelf, row: dict) -> None:
    """Top-down: the scene and every executed sweep as an arrow from its start to its planned end, by outcome."""
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    draw_cylinders(ax, env, scene.positions, label=True)
    for r in rows:
        if r["outcome"] in ("no executable sweep",):
            continue
        ax.annotate("", xy=(r["end_y_m"], r["end_x_m"]), xytext=(r["start_y_m"], r["start_x_m"]), zorder=2,
                    arrowprops=dict(arrowstyle="-|>", color=OUTCOME_COLOURS[r["outcome"]], lw=0.7, alpha=0.6,
                                    mutation_scale=6))
    counts = {o: sum(r["outcome"] == o for r in rows) for o in OUTCOMES}
    handles = [Line2D([], [], color=OUTCOME_COLOURS[o], lw=2, label=f"{o} ({counts[o]})") for o in OUTCOMES
               if counts[o]]
    ax.legend(handles=handles, fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ci = row["success_ci95"]
    ax.set_title(f"{scene.name}: success {row['successes']}/{row['executed'] - row['glitches']}"
                 f" ({ci[0]:.2f} to {ci[1]:.2f})" if ci else scene.name, fontsize=9)
    fig.tight_layout()
    plotting.save(fig, path, dpi=120)
    plt.close(fig)


def typed(row: dict) -> dict:
    """A trials.csv row with its numbers back as numbers (empty cells as None)."""
    out = {}
    for k, v in row.items():
        if v == "":
            out[k] = None
        elif k in ("scene", "cylinder", "outcome"):
            out[k] = v
        else:
            out[k] = float(v) if any(c in v for c in ".e") else int(v)
    return out


def analyse(settings: Settings, out: Path, rows: list[dict], chosen: list[scenes.Scene],
            run_wall_s: float | None) -> dict:
    """Summary, tables and the per-scene sweep plots from the trial rows."""
    summary = summarise(settings, rows, chosen)
    summary["run_wall_s"] = run_wall_s
    runner.write_csv(out / "scene_table.csv", summary["by_scene"])
    runner.write_csv(out / "by_cylinder.csv", summary["by_cylinder"])
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("by_cylinder", "by_scene")}, indent=1))
    env = make_env(settings)
    try:
        for scene, row in zip(chosen, summary["by_scene"]):
            plot_scene_sweeps(out / "sweeps" / f"{scene.name}.png", scene,
                              [r for r in rows if r["scene"] == scene.name], env, row)
    finally:
        env.close()
    return summary


def part_analyse(settings: Settings, out: Path, quick: bool) -> dict:
    """Redo the summary and plots from an existing trials.csv, without running trials."""
    previous = json.loads((out / "summary.json").read_text()).get("run_wall_s") if (out / "summary.json").exists() \
        else None
    return analyse(settings, out, [typed(r) for r in runner.read_csv(out / "trials.csv")], scene_list(settings, quick),
                   previous)
