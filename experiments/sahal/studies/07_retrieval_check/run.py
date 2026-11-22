"""Study 07: does the kinematic "can the red be retrieved now?" check agree with pick + pull in physics?

    experiments/sahal/.venv/bin/python experiments/sahal/studies/07_retrieval_check/run.py [--quick] [--workers 10]
        [--mode kinematic|hybrid|exact] [--parts scenes margins otherwise] [--seeds tuning|heldout|test]

Random scenes, fixed seeds: the red somewhere in the env's action box and 2 grays standing on the board, no overlaps.
Three kinds of scene: grays anywhere on the board; one gray in the corridor the gripper comes in along (in front of
the red, near its approach ray); one gray right beside the red. For each: common.retrieval_check.can_retrieve(), timed,
then the repo's pick and pull in physics at friction 0.55 with all of the env's checks on. Output: the 2x2 agreement
table, by kind and by reason, and pictures of disagreements. --quick runs a few scenes into outputs/quick/.

Parts: scenes (the run above, with the check in --mode; hybrid writes to outputs/hybrid/), margins (from the
kinematic scenes.csv: false yes and missed successes for gray-clearance thresholds), otherwise (from the kinematic
scenes.csv: why the scenes that failed for a reason other than a gray failed, with pictures). --seeds heldout or
test draws the scenes from HELDOUT_STUDY_SEED or TEST_STUDY_SEED and writes to outputs/<mode>_<seeds>/ (part scenes
only).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # experiments/sahal, for `common`

import common  # noqa: E402,F401
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from common import measures, plotting, runner  # noqa: E402
from common import retrieval_check  # noqa: E402
from common.shelf_env import RedAndGraysShelf  # noqa: E402
from so101sim.shelf import MIN_SPACING, TARGET  # noqa: E402
from so101sim.shelf_layout import CYL_RADIUS, WALL_GAP  # noqa: E402

# ---- settings ------------------------------------------------------------------------------------------------
OUT_DIR = Path(__file__).resolve().parent / "outputs"
STUDY_SEED = 7  # the scenes the hybrid thresholds were chosen on
HELDOUT_STUDY_SEED = 70  # held-out scenes: tested the first thresholds; the fast-no cutoff was then moved
TEST_STUDY_SEED = 700  # third set: tests the moved cutoff, used for nothing else
SEED_SETS = {"tuning": STUDY_SEED, "heldout": HELDOUT_STUDY_SEED, "test": TEST_STUDY_SEED}
RESET_SEED = 0
N_SCENES = 300
QUICK_N_SCENES = 10
PHYSICS = {"friction": 0.55, "density": 900.0}
GRAYS = ("cyl_gray_0", "cyl_gray_1")
SCENE_KINDS = {"anywhere": 0.3, "in the approach corridor": 0.45, "beside the red": 0.25}  # share of scenes
CORRIDOR_HALF_WIDTH_M = 0.05  # corridor gray: at most this far sideways from the red's approach ray
BESIDE_MAX_GAP_M = 0.06  # beside gray: centre at most this far from the red's centre
MAX_PLACEMENT_TRIES = 200
MAX_SCENE_TRIES = 20
MAX_START_TILT_DEG = 2.0  # a scene whose cylinders settle tilted more than this is drawn again
TIMED_SCENES = 20  # checks timed again one at a time in the main process (the parallel times share the CPU)
QUICK_TIMED_SCENES = 3
CLEARANCE_BIN_MM = 2.0
GRAY_FAILURES = ("gray toppled", "gray left the shelf")
MARGIN_TABLE_MM = list(range(5, -31, -1))  # gray-clearance thresholds for the margin table

AGREEMENT_CELLS = ("check yes, succeeded", "check yes, failed", "check no, succeeded", "check no, failed")
_ENV: RedAndGraysShelf | None = None


def _init() -> None:
    global _ENV
    _ENV = RedAndGraysShelf()


# ---- scenes ----------------------------------------------------------------------------------------------------

def _free(xy: np.ndarray, placed: list[np.ndarray], layout) -> bool:
    """Fully on the board, clear of the walls by WALL_GAP and of every placed cylinder by MIN_SPACING."""
    inset = CYL_RADIUS + WALL_GAP
    on_board = (layout.front_x + CYL_RADIUS <= xy[0] <= layout.back_x - inset
                and abs(xy[1]) <= layout.half_width - inset)
    return on_board and all(np.linalg.norm(xy - p) >= MIN_SPACING for p in placed)


def _sample(rng: np.random.Generator, draw, placed: list[np.ndarray], layout) -> np.ndarray | None:
    for _ in range(MAX_PLACEMENT_TRIES):
        xy = draw()
        if _free(xy, placed, layout):
            return xy
    return None


def draw_scene(rng: np.random.Generator, kind: str, env: RedAndGraysShelf) -> dict | None:
    """Red in the action box, then GRAYS by `kind`: the first gray in the corridor or beside the red for those kinds,
    the rest anywhere on the board. Returns {name: xy} or None if a cylinder found no room."""
    layout, geo = env.env.layout, env.geo
    x_lo, x_hi, y_lo, y_hi = env.env.region
    red = np.array([rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi)])
    placed, scene = [red], {TARGET: red}
    toward_robot = (geo.pan_xy - red) / np.linalg.norm(geo.pan_xy - red)
    sideways = np.array([-toward_robot[1], toward_robot[0]])
    reach_to_edge = (red[0] - layout.front_x) / max(abs(toward_robot[0]), 1e-9)

    def anywhere() -> np.ndarray:
        return np.array([rng.uniform(layout.front_x, layout.back_x), rng.uniform(-layout.half_width, layout.half_width)])

    def in_corridor() -> np.ndarray:
        return (red + rng.uniform(MIN_SPACING, max(reach_to_edge, MIN_SPACING)) * toward_robot
                + rng.uniform(-CORRIDOR_HALF_WIDTH_M, CORRIDOR_HALF_WIDTH_M) * sideways)

    def beside() -> np.ndarray:
        angle = rng.uniform(0, 2 * np.pi)
        return red + rng.uniform(MIN_SPACING, BESIDE_MAX_GAP_M) * np.array([np.cos(angle), np.sin(angle)])

    first = {"anywhere": anywhere, "in the approach corridor": in_corridor, "beside the red": beside}[kind]
    for i, name in enumerate(GRAYS):
        xy = _sample(rng, first if i == 0 else anywhere, placed, layout)
        if xy is None:
            return None
        placed.append(xy)
        scene[name] = xy
    return scene


def settle_scene(env: RedAndGraysShelf, scene: dict) -> dict | None:
    """Reset with the scene; None if it didn't come to rest, touched the arm or left a cylinder tilted."""
    report = env.reset_scene(seed=RESET_SEED, red_xy=scene[TARGET],
                             gray_xys={n: scene[n] for n in GRAYS}, physics=PHYSICS)
    if not report["resting"] or report["touched_arm"] or max(report["tilt"].values()) > MAX_START_TILT_DEG:
        return None
    return report


def build_scene(index: int, env: RedAndGraysShelf, study_seed: int = STUDY_SEED) -> tuple[str, dict, dict] | None:
    """Draw scenes for this index from its own seed until one settles cleanly."""
    rng = np.random.default_rng([study_seed, index])
    kind = str(rng.choice(list(SCENE_KINDS), p=list(SCENE_KINDS.values())))
    for _ in range(MAX_SCENE_TRIES):
        scene = draw_scene(rng, kind, env)
        if scene is None:
            continue
        report = settle_scene(env, scene)
        if report is not None:
            return kind, scene, report
    return None


# ---- one scene -------------------------------------------------------------------------------------------------

def run_pick_and_pull(env: RedAndGraysShelf, red_xy: np.ndarray) -> dict:
    """The repo's pick on the red, then pull if pick ran cleanly (retrieval_check.pick_and_pull), as real_* columns."""
    result = retrieval_check.pick_and_pull(env, red_xy)
    return {"pick_frames": result["pick_frames"], "real_success": result["success"], "real_reason": result["reason"]}


def reason_kind(reason: str) -> str:
    """A short label for a check or real-run reason."""
    for pattern, label in ((r"^simulated", "simulated"), (r"^clear of the grays", "fast yes: clear"),
                           (r"^plan overlaps a gray", "fast no: deep overlap"),
                           (r"held red within .* of a wall", "held red near shelf"),
                           (r"outside the action box", "action box"), (r"^IK", "IK"),
                           (r"within 6 mm of", "arm near shelf"), (r"arm \(.*\) within .* cyl_gray", "arm near gray"),
                           (r"held red within .* cyl_gray", "held red near gray"),
                           (r"held red within .* shelf", "held red near shelf"),
                           (r"refused: .*would hit", "refused: path check"), (r"refused: unreachable", "refused: IK"),
                           (r"refused: .*action box", "refused: action box"), (r"toppled: cyl_gray", "gray toppled"),
                           (r"toppled: cyl_red", "red toppled"), (r"left the shelf: cyl_gray", "gray left the shelf"),
                           (r"left the shelf: cyl_red", "red left the shelf"), (r"hit the shelf", "arm hit the shelf"),
                           (r"without the red", "red not out and held"), (r"^ok$|^retrieved$", "ok")):
        if re.search(pattern, reason):
            return label
    return reason[:40]


def scene_job(job: tuple[int, str, int]) -> dict:
    """Build scene `index`, run the check in `mode`, then pick + pull in physics."""
    index, mode, study_seed = job
    env = _ENV
    built = build_scene(index, env, study_seed)
    if built is None:
        return {"scene": index, "study_seed": study_seed, "built": False}
    kind, scene, report = built
    started = time.perf_counter()
    check_ok, check_reason = retrieval_check.can_retrieve(env, mode=mode)
    check_s = time.perf_counter() - started
    clearance = retrieval_check.min_gray_clearance(env)
    started = time.perf_counter()
    real = run_pick_and_pull(env, report["xy"][TARGET])
    physics_s = time.perf_counter() - started
    flagged = re.search(r"^(pick|pull): .* at frame (\d+)$", check_reason)
    record = {"scene": index, "study_seed": study_seed, "built": True, "kind": kind, "check_mode": mode,
              **{f"{n}_{axis}_m": round(float(report["xy"][n][k]), 4) for n in (TARGET, *GRAYS)
                 for k, axis in enumerate("xy")},
              "check_ok": check_ok, "check_reason": check_reason, "check_kind": reason_kind(check_reason),
              "check_primitive": flagged.group(1) if flagged else None,
              "check_frame": int(flagged.group(2)) if flagged else None,
              "check_ms": round(check_s * 1000, 1),
              "plan_min_gray_clearance_mm": round(clearance * 1000, 1) if clearance is not None else None, "physics_s": round(physics_s, 2),
              **real, "real_kind": reason_kind(real["real_reason"]), "arm_touched_gray": env.log.arm_gray_touch,
              "max_contact_force_n": round(env.log.max_contact_force_n, 1),
              "max_red_speed_m_s": round(env.log.max_red_speed_m_s, 3),
              "glitch": measures.is_glitch(env.log.max_contact_force_n, env.log.max_red_speed_m_s, grasping=True)}
    record["agreement"] = f"check {'yes' if check_ok else 'no'}, {'succeeded' if real['real_success'] else 'failed'}"
    return record


# ---- outputs ---------------------------------------------------------------------------------------------------

def _count(values) -> dict:
    counts: dict = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def summarise(rows: list[dict]) -> dict:
    built = [r for r in rows if r["built"]]
    times = np.array([r["check_ms"] for r in built])
    table = {cell: sum(r["agreement"] == cell for r in built) for cell in AGREEMENT_CELLS}
    return {
        "scenes": len(rows), "built": len(built), "not_built": len(rows) - len(built),
        "agreement_table": table, "agree": table["check yes, succeeded"] + table["check no, failed"],
        "agreement_rate": round((table["check yes, succeeded"] + table["check no, failed"]) / len(built), 3),
        "real_successes": sum(r["real_success"] for r in built),
        "glitch_scenes": [r["scene"] for r in built if r["glitch"]],
        "by_kind": {kind: {cell: sum(r["agreement"] == cell for r in built if r["kind"] == kind)
                           for cell in AGREEMENT_CELLS} for kind in SCENE_KINDS},
        "check_reasons": _count(r["check_kind"] for r in built),
        "real_reasons": _count(r["real_kind"] for r in built),
        "check_no_succeeded_by_reason": _count(r["check_kind"] for r in built
                                               if r["agreement"] == "check no, succeeded"),
        "check_yes_failed_by_real_reason": _count(r["real_kind"] for r in built
                                                  if r["agreement"] == "check yes, failed"),
        "arm_touched_gray_but_succeeded": sum(bool(r["arm_touched_gray"]) and r["real_success"] for r in built),
        "check_ms": {"mean": round(float(times.mean()), 1), "median": round(float(np.median(times)), 1),
                     "p90": round(float(np.percentile(times, 90)), 1), "max": round(float(times.max()), 1)},
        "check_ms_mean_by_reason": {k: round(float(np.mean([r["check_ms"] for r in built if r["check_kind"] == k])), 1)
                                    for k in _count(r["check_kind"] for r in built)},
        "physics_s_median": round(float(np.median([r["physics_s"] for r in built])), 2),
    }


def plot_agreement(path: Path, rows: list[dict]) -> None:
    """Scenes per kind, stacked by the four agreement cells."""
    built = [r for r in rows if r["built"]]
    colours = dict(zip(AGREEMENT_CELLS, (plotting.OUTCOME_STYLE["slid"][0], plotting.OUTCOME_STYLE["toppled"][0],
                                         plotting.OUTCOME_STYLE["leaned"][0], plotting.BLUE_RAMP[-1])))
    kinds = list(SCENE_KINDS)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    left = np.zeros(len(kinds))
    for cell in AGREEMENT_CELLS:
        counts = np.array([sum(r["agreement"] == cell for r in built if r["kind"] == k) for k in kinds])
        ax.barh(kinds, counts, left=left, color=colours[cell], edgecolor="white", linewidth=2, label=cell)
        for i, c in enumerate(counts):
            if c:
                ax.text(left[i] + c / 2, i, str(c), ha="center", va="center", fontsize=8, color="white")
        left += counts
    ax.set_xlabel("scenes")
    ax.set_title("Retrieval check vs pick + pull in physics, by where the first gray stands", fontsize=10)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    ax.invert_yaxis()
    plotting.tidy(ax)
    fig.tight_layout()
    plotting.save(fig, path)


def picture_example(path: Path, record: dict) -> None:
    """Rebuild the scene, rerun pick + pull with video, and draw: the scene from above with the red's approach ray,
    and the front | wrist cameras at the frame the check flagged (check said no) or the last frame (check said yes)."""
    env = RedAndGraysShelf(record_video=True)
    try:
        _, scene, report = build_scene(record["scene"], env, record["study_seed"])
        run_pick_and_pull(env, report["xy"][TARGET])
        video = env.log.video
        if record["check_frame"] is not None:
            offset = 0 if record["check_primitive"] == "pick" else record["pick_frames"]
            index = min(offset + record["check_frame"], len(video) - 1)
            caption = f"real run at the frame the check flagged ({record['check_primitive']} frame " \
                      f"{record['check_frame']})"
        else:
            index, caption = len(video) - 1, "real run, last frame"
        fig = plt.figure(figsize=(14, 5))
        top = fig.add_axes([0.03, 0.12, 0.26, 0.76])
        plotting.draw_shelf(top, env.env.layout, env.env.region)
        red = report["xy"][TARGET]
        ray_end = env.geo.on_ray(red, env.geo.standoff_x)
        top.plot([ray_end[1], red[1]], [ray_end[0], red[0]], color="#8a8a86", ls="--", lw=1, label="approach ray")
        for name, xy in report["xy"].items():
            colour = plotting.OUTCOME_STYLE["toppled"][0] if name == TARGET else "#8a8a86"
            top.add_patch(Circle((xy[1], xy[0]), CYL_RADIUS, color=colour, zorder=3))
            top.annotate(name.replace("cyl_", ""), (xy[1], xy[0]), xytext=(0, 9), textcoords="offset points",
                         ha="center", fontsize=7)
        top.set_title(f"scene {record['scene']} ({record['kind']})", fontsize=9)
        image = fig.add_axes([0.31, 0.12, 0.68, 0.76])
        image.imshow(video[index])
        image.set_axis_off()
        image.set_title(caption, fontsize=9)
        fig.suptitle(f"{record['agreement']}.  check: {record['check_reason']}\nreal: {record['real_reason']}"
                     f"{'; arm touched ' + record['arm_touched_gray'] if record['arm_touched_gray'] else ''}",
                     fontsize=9)
        plotting.save(fig, path, dpi=110)
        plt.close(fig)
    finally:
        env.close()


def choose_examples(rows: list[dict]) -> list[dict]:
    """One per scene kind: the scene whose plan overlaps a gray most (the most surprising disagreement)."""
    chosen = []
    for kind in SCENE_KINDS:
        of_kind = [r for r in rows if r["kind"] == kind]
        if of_kind:
            chosen.append(min(of_kind, key=lambda r: r["plan_min_gray_clearance_mm"]
                              if r["plan_min_gray_clearance_mm"] is not None else float("inf")))
    return chosen


def time_checks(n_scenes: int, mode: str, study_seed: int) -> dict:
    """can_retrieve() timed one scene at a time in this process, nothing else running in it."""
    env = RedAndGraysShelf()
    times = []
    try:
        for index in range(n_scenes):
            built = build_scene(index, env, study_seed)
            if built is None:
                continue
            started = time.perf_counter()
            retrieval_check.can_retrieve(env, mode=mode)
            times.append((time.perf_counter() - started) * 1000)
    finally:
        env.close()
    return {"scenes": len(times), "mean": round(float(np.mean(times)), 1), "median": round(float(np.median(times)), 1),
            "max": round(float(max(times)), 1)}


def clearance_groups(rows: list[dict]) -> dict[str, list[float]]:
    """Planned smallest arm/held-red to gray distance (mm), by real outcome: succeeded, a gray toppled or left the
    shelf, failed for another reason (refused by the path check, nothing held)."""
    built = [r for r in rows if r["built"] and r["plan_min_gray_clearance_mm"] is not None]
    return {"succeeded": [r["plan_min_gray_clearance_mm"] for r in built if r["real_success"]],
            "a gray toppled or left the shelf": [r["plan_min_gray_clearance_mm"] for r in built
                                                 if r["real_kind"] in GRAY_FAILURES],
            "failed otherwise": [r["plan_min_gray_clearance_mm"] for r in built
                                 if not r["real_success"] and r["real_kind"] not in GRAY_FAILURES]}


def margin_table(rows: list[dict], thresholds_mm: list[int]) -> list[dict]:
    """For each gray-clearance threshold: answer yes when the plan's smallest gray clearance is at least it, no when
    below. Scenes the env refused before moving (path check, IK: the check's own exact refusals) are always no and
    are left out of the counts. false_yes: yes but really failed; missed: no but really succeeded."""
    judged = [r for r in rows if _true(r["built"]) and not r["real_reason"].startswith("pick refused")]

    def clearance(r: dict) -> float:
        return float(r["plan_min_gray_clearance_mm"]) if r["plan_min_gray_clearance_mm"] not in ("", None) \
            else retrieval_check.DISTANCE_CUTOFF_M * 1000

    return [{"threshold_mm": t, "judged_scenes": len(judged),
             "yes": sum(clearance(r) >= t for r in judged),
             "false_yes": sum(clearance(r) >= t and not _true(r["real_success"]) for r in judged),
             "no": sum(clearance(r) < t for r in judged),
             "missed_successes": sum(clearance(r) < t and _true(r["real_success"]) for r in judged)}
            for t in thresholds_mm]


def _true(value) -> bool:
    return value is True or value == "True"


def plot_clearance(path: Path, rows: list[dict]) -> None:
    """Histogram of the plan's smallest distance to a gray, per real outcome, with the check's margin."""
    groups = clearance_groups(rows)
    colours = (plotting.OUTCOME_STYLE["slid"][0], plotting.OUTCOME_STYLE["toppled"][0], "#8a8a86")
    values = [v for vs in groups.values() for v in vs]
    bins = np.arange(np.floor(min(values) / CLEARANCE_BIN_MM) * CLEARANCE_BIN_MM,
                     max(values) + 2 * CLEARANCE_BIN_MM, CLEARANCE_BIN_MM)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.hist(list(groups.values()), bins=bins, stacked=True, color=colours, edgecolor="white", linewidth=1,
            label=[f"{k} ({len(v)})" for k, v in groups.items()])
    ax.axvline(retrieval_check.ARM_GRAY_MARGIN_M * 1000, color=plotting.REFERENCE_COLOURS["neutral"], ls="--", lw=1.2,
               label="check's margin (5 mm)")
    ax.axvline(0, color="#222", lw=0.8)
    ax.set(xlabel="planned smallest distance, arm or held red to a gray (mm; negative = the plan overlaps it; "
                  "capped at 50)", ylabel="scenes")
    ax.set_title("Pick + pull in physics against how close the plan comes to a gray", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    plotting.tidy(ax)
    fig.tight_layout()
    plotting.save(fig, path)


def failed_otherwise(env: RedAndGraysShelf, record: dict, picture: Path) -> dict:
    """Why a scene failed without a gray failure. Refused by the path check: where the red stands relative to the
    side walls. Nothing held after pick: rerun pick and look at the gripper, the red and what the jaws touch. Saves a
    picture (scene from above, cameras at the end of pick or at the start, for a refusal)."""
    _, _, report = build_scene(int(record["scene"]), env)
    layout, red = env.env.layout, report["xy"][TARGET]
    out = {"scene": int(record["scene"]), "kind": record["kind"], "real_reason": record["real_reason"],
           "check_reason": record["check_reason"], "plan_min_gray_clearance_mm": record["plan_min_gray_clearance_mm"],
           "red_x_m": round(float(red[0]), 4), "red_y_m": round(float(red[1]), 4),
           "red_to_left_wall_mm": round((layout.half_width - CYL_RADIUS - red[1]) * 1000, 1),
           "red_to_right_wall_mm": round((layout.half_width - CYL_RADIUS + red[1]) * 1000, 1)}
    if record["real_reason"].startswith("pull refused"):
        before = env.env.cylinder_pose(TARGET)
        env.step(("pick", red, 0.0))
        after = env.env.cylinder_pose(TARGET)
        jaw_touches = sorted({measures.geom_label(env.model, g) for c in env.data.contact[: env.data.ncon]
                              for a, g in ((c.geom1, c.geom2), (c.geom2, c.geom1)) if a in env.env._jaw_ids
                              and g not in env.env._arm_ids})
        out.update({"gripper_reading_after_pick": round(float(env.env.joint_state[5]), 2),
                    "red_moved_mm": round(float(np.linalg.norm(after["pos"][:2] - before["pos"][:2])) * 1000, 1),
                    "red_tilt_after_pick_deg": round(after["tilt"], 2), "jaws_touch_after_pick": ";".join(jaw_touches),
                    "both_jaws_on_red": bool(env.env.held(TARGET)), "arm_touched_gray": env.log.arm_gray_touch})
        caption = "end of pick"
    else:
        caption = "start (the env refused the pick before moving)"
    fig = plt.figure(figsize=(14, 5))
    top = fig.add_axes([0.03, 0.12, 0.26, 0.76])
    plotting.draw_shelf(top, layout, env.env.region)
    for name, xy in report["xy"].items():
        colour = plotting.OUTCOME_STYLE["toppled"][0] if name == TARGET else "#8a8a86"
        top.add_patch(Circle((xy[1], xy[0]), CYL_RADIUS, color=colour, zorder=3))
    ray_end = env.geo.on_ray(red, env.geo.standoff_x)
    top.plot([ray_end[1], red[1]], [ray_end[0], red[0]], color="#8a8a86", ls="--", lw=1)
    top.set_title(f"scene {out['scene']} ({out['kind']})", fontsize=9)
    image = fig.add_axes([0.31, 0.12, 0.68, 0.76])
    image.imshow(env.snapshot())
    image.set_axis_off()
    image.set_title(caption, fontsize=9)
    fig.suptitle(f"real: {out['real_reason']}\ncheck: {out['check_reason']}; plan's gray clearance "
                 f"{out['plan_min_gray_clearance_mm']} mm", fontsize=9)
    plotting.save(fig, picture, dpi=110)
    plt.close(fig)
    return out


def part_otherwise(out: Path) -> dict:
    """The kinematic run's scenes that failed but not by a gray toppling or leaving the shelf."""
    rows = [r for r in runner.read_csv(out / "scenes.csv")
            if _true(r["built"]) and not _true(r["real_success"]) and r["real_kind"] not in GRAY_FAILURES]
    env = RedAndGraysShelf()
    try:
        found = [failed_otherwise(env, r, out / "otherwise" / f"scene{int(r['scene']):03d}.png") for r in rows]
    finally:
        env.close()
    runner.write_csv(out / "failed_otherwise.csv", found)
    refused = [f for f in found if not f["real_reason"].startswith("pull refused")]
    summary = {"scenes": len(found), "by_reason": _count(reason_kind(f["real_reason"]) for f in found),
               "refused_by_wall": _count(re.search(r"would hit (\w+)", f["real_reason"]).group(1) for f in refused),
               "refused_red_to_nearest_side_wall_mm": [min(min(f["red_to_left_wall_mm"], f["red_to_right_wall_mm"])
                                                           for f in refused),
                                                       max(min(f["red_to_left_wall_mm"], f["red_to_right_wall_mm"])
                                                           for f in refused)] if refused else None,
               "refused_red_y_m": [min(f["red_y_m"] for f in refused), max(f["red_y_m"] for f in refused)]
               if refused else None,
               "positive_gap": [{k: f[k] for k in ("scene", "real_reason", "check_reason", "plan_min_gray_clearance_mm",
                                                   "red_x_m", "red_y_m")}
                                for f in found if f["plan_min_gray_clearance_mm"] not in ("", None)
                                and float(f["plan_min_gray_clearance_mm"]) > 0],
               "nothing_held": [{k: f[k] for k in f if k not in ("check_reason",)}
                                for f in found if f["real_reason"].startswith("pull refused")]}
    print("failed otherwise:", json.dumps(summary, indent=1))
    return summary


def part_scenes(out: Path, quick: bool, workers: int, mode: str, study_seed: int) -> dict:
    """Every scene with the check in `mode`, then pick + pull in physics; tables, plots and pictures."""
    rows = runner.run_parallel([(i, mode, study_seed) for i in range(QUICK_N_SCENES if quick else N_SCENES)], scene_job, _init,
                               workers, "scenes: ")
    runner.write_csv(out / "scenes.csv", rows)
    summary = summarise(rows)
    groups = clearance_groups(rows)
    summary["plan_min_gray_clearance_mm_by_outcome"] = {
        k: {"scenes": len(v), "min": min(v), "median": round(float(np.median(v)), 1), "max": max(v)}
        for k, v in groups.items() if v}
    examples = {}
    for cell in ("check yes, failed", "check no, succeeded"):
        chosen = choose_examples([r for r in rows if r["built"] and r["agreement"] == cell])
        examples[cell] = []
        for r in chosen:
            name = f"example_{'yes_failed' if 'yes' in cell else 'no_succeeded'}_scene{r['scene']:03d}.png"
            picture_example(out / "examples" / name, r)
            examples[cell].append({"scene": r["scene"], "kind": r["kind"], "picture": f"examples/{name}",
                                   "plan_min_gray_clearance_mm": r["plan_min_gray_clearance_mm"],
                                   "check": r["check_reason"],
                                   "real": r["real_reason"], "arm_touched_gray": r["arm_touched_gray"]})
    summary["examples"] = examples
    summary["check_ms_single_process"] = time_checks(QUICK_TIMED_SCENES if quick else TIMED_SCENES, mode, study_seed)
    plot_agreement(out / "agreement.png", rows)
    if groups["succeeded"] or groups["a gray toppled or left the shelf"]:
        plot_clearance(out / "clearance_vs_outcome.png", rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="a few scenes into outputs/quick/")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--mode", choices=retrieval_check.CHECK_MODES, default="kinematic",
                        help="the check's mode for part scenes")
    parser.add_argument("--parts", nargs="+", default=["scenes"], choices=["scenes", "margins", "otherwise"])
    parser.add_argument("--seeds", choices=list(SEED_SETS), default="tuning", help="which set of scene seeds")
    args = parser.parse_args()
    study_seed = SEED_SETS[args.seeds]
    if args.seeds != "tuning" and set(args.parts) - {"scenes"}:
        parser.error("held-out seeds run part scenes only (margins and otherwise read the kinematic run)")
    if args.seeds != "tuning":
        base = OUT_DIR / f"{args.mode}_{args.seeds}"
    else:
        base = OUT_DIR if args.mode == "kinematic" else OUT_DIR / args.mode
    out = base / "quick" if args.quick else base
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    if "scenes" in args.parts:
        summary.update(part_scenes(out, args.quick, args.workers, args.mode, study_seed))
        rows = runner.read_csv(out / "scenes.csv")
        summary["margin_table_at_hybrid_thresholds"] = margin_table(
            rows, [round(retrieval_check.HYBRID_CLEAR_M * 1000), round(retrieval_check.HYBRID_DEEP_M * 1000)])
    kinematic = OUT_DIR / "quick" if args.quick else OUT_DIR
    if "margins" in args.parts:
        table = margin_table(runner.read_csv(kinematic / "scenes.csv"), MARGIN_TABLE_MM)
        runner.write_csv(kinematic / "margin_table.csv", table)
        print("margin table:", json.dumps(table))
    if "otherwise" in args.parts:
        otherwise = part_otherwise(kinematic)
        (kinematic / "failed_otherwise.json").write_text(json.dumps(otherwise, indent=1) + "\n")
    summary_path.write_text(json.dumps(summary, indent=1) + "\n")
    runner.save_config(out, {
        "quick": args.quick, "mode": args.mode, "parts": args.parts,
        "scenes": QUICK_N_SCENES if args.quick else N_SCENES, "study_seed": study_seed,
        "reset_seed": RESET_SEED, "physics": PHYSICS, "grays": GRAYS, "scene_kinds": SCENE_KINDS,
        "corridor_half_width_m": CORRIDOR_HALF_WIDTH_M, "beside_max_gap_m": BESIDE_MAX_GAP_M,
        "min_spacing_m": MIN_SPACING, "max_start_tilt_deg": MAX_START_TILT_DEG,
        "arm_gray_margin_m": retrieval_check.ARM_GRAY_MARGIN_M,
        "held_red_gray_margin_m": retrieval_check.HELD_RED_GRAY_MARGIN_M,
        "held_red_shelf_margin_m": retrieval_check.HELD_RED_SHELF_MARGIN_M,
        "hybrid_clear_m": retrieval_check.HYBRID_CLEAR_M, "hybrid_deep_m": retrieval_check.HYBRID_DEEP_M,
        "margin_table_mm": MARGIN_TABLE_MM})


if __name__ == "__main__":
    main()
