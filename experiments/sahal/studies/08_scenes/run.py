"""Study 08: hand-made scenes from easy to hard, each with a hand-written solution (sweeps, then retrieve).

    experiments/sahal/.venv/bin/python experiments/sahal/studies/08_scenes/run.py [--quick] [--workers 10]
        [--scenes NAME ...]

Writes each scene to experiments/sahal/scenes/NAME.json (common/scenes.py). For each scene and each friction:
place it, ask can_retrieve(mode="exact") at the start (it must say no, except for L0), then run the solution in
physics: every sweep (skills.sweep.sweep_cylinder, the cylinder's current pose decides the start), then the repo's
pick + pull. Records success, sweeps run, failures (refused, toppled, jammed against a wall, off the board),
glitches, and front-camera images before and after. --quick runs friction 0.55 only, into outputs/quick/.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # experiments/sahal, for `common` and `skills`

import common  # noqa: E402,F401
import imageio.v3 as iio  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from common import measures, plotting, retrieval_check, runner, scenes  # noqa: E402
from common.shelf_env import RedAndGraysShelf, RunLog  # noqa: E402
from skills import sweep as sweep_skill  # noqa: E402
from so101sim.shelf import TARGET  # noqa: E402
from so101sim.shelf_layout import CYL_RADIUS  # noqa: E402

# ---- settings ------------------------------------------------------------------------------------------------
OUT_DIR = Path(__file__).resolve().parent / "outputs"
FRICTIONS = [0.55, 0.4, 0.7]
QUICK_FRICTIONS = [0.55]
SCENE_FRICTION = 0.55  # the friction written into each scene file
IMAGE_FRICTION = 0.55  # the before/after pictures in the gallery

# Each scene: positions (x, y metres; x away from the robot, y to the robot's left), what it needs, and the
# hand-written solution: sweeps as (cylinder, direction in degrees, distance in metres), then retrieve. Scenes with
# "order_matters" also run their sweeps in reverse order, which should fail.
SCENE_SPECS = [
    {"name": "L0_open", "level": "L0", "needs": "retrieve",
     "description": "red in the open, both grays well out of the way",
     "positions": {"cyl_red": (0.42, 0.0), "cyl_gray_0": (0.44, 0.10), "cyl_gray_1": (0.37, -0.09)},
     "sweeps": []},
    {"name": "L1_front", "level": "L1", "needs": "1 sweep + retrieve",
     "description": "one gray straight in front of the red, on its approach ray",
     "positions": {"cyl_red": (0.43, 0.0), "cyl_gray_0": (0.37, 0.0), "cyl_gray_1": (0.44, 0.11)},
     "sweeps": [("cyl_gray_0", 90, 0.07)]},
    {"name": "L1_front_offset", "level": "L1", "needs": "1 sweep + retrieve",
     "description": "one gray in front of the red and a little to its right; sweep it right",
     "positions": {"cyl_red": (0.43, 0.03), "cyl_gray_0": (0.375, 0.015), "cyl_gray_1": (0.44, 0.12)},
     "sweeps": [("cyl_gray_0", 270, 0.07)]},
    {"name": "L1b_right_wall", "level": "L1b", "needs": "sweep the red off the right wall (not possible)",
     "description": "red 1 cm from the right wall, in the band where pick is refused; nothing else in the way",
     "positions": {"cyl_red": (0.40, -0.125), "cyl_gray_0": (0.44, 0.08), "cyl_gray_1": (0.37, 0.10)},
     "sweeps": [("cyl_red", 90, 0.03)]},
    {"name": "L2_two_in_line", "level": "L2", "needs": "2 sweeps + retrieve",
     "description": "two grays one behind the other on the red's approach ray",
     "positions": {"cyl_red": (0.45, 0.0), "cyl_gray_0": (0.395, 0.0), "cyl_gray_1": (0.345, 0.0)},
     "sweeps": [("cyl_gray_1", 90, 0.07), ("cyl_gray_0", 270, 0.07)]},
    {"name": "L2_side_by_side", "level": "L2", "needs": "1 long sweep + retrieve",
     "description": "two grays side by side across the approach corridor; one long sweep pushes both aside",
     "positions": {"cyl_red": (0.44, 0.0), "cyl_gray_0": (0.375, -0.019), "cyl_gray_1": (0.375, 0.019)},
     "sweeps": [("cyl_gray_0", 90, 0.10)]},
    {"name": "L2b_order", "level": "L2b", "needs": "2 sweeps in order + retrieve",
     "description": "gray A in front of the red; gray B at the front edge on the gripper's way to A's sweep: B first",
     "positions": {"cyl_red": (0.43, 0.0), "cyl_gray_0": (0.37, 0.0), "cyl_gray_1": (0.33, -0.035)},
     "sweeps": [("cyl_gray_1", 270, 0.06), ("cyl_gray_0", 90, 0.07)], "order_matters": True},
    {"name": "L2b_wall_order", "level": "L2b", "needs": "2 sweeps in order + retrieve",
     "description": "gray A in front of the red near the left wall (little room left); gray B on A's right: B first",
     "positions": {"cyl_red": (0.43, 0.105), "cyl_gray_0": (0.37, 0.085), "cyl_gray_1": (0.37, 0.035)},
     "sweeps": [("cyl_gray_1", 270, 0.07), ("cyl_gray_0", 270, 0.07)], "order_matters": True},
    {"name": "L3_right_wall", "level": "L3", "needs": "1 sweep + retrieve",
     "description": "red near the right wall (still graspable) behind a gray; the wall leaves only a sweep left",
     "positions": {"cyl_red": (0.42, -0.105), "cyl_gray_0": (0.36, -0.088), "cyl_gray_1": (0.44, 0.06)},
     "sweeps": [("cyl_gray_0", 90, 0.07)]},
    {"name": "L3_left_wall_two", "level": "L3", "needs": "2 sweeps + retrieve",
     "description": "red near the left wall behind a gray, a second gray just right of the red where the jaw opens",
     "positions": {"cyl_red": (0.44, 0.105), "cyl_gray_0": (0.37, 0.088), "cyl_gray_1": (0.425, 0.06)},
     "sweeps": [("cyl_gray_1", 315, 0.05), ("cyl_gray_0", 270, 0.07)]},
]

_ENV: RedAndGraysShelf | None = None


def _init() -> None:
    global _ENV
    _ENV = RedAndGraysShelf()


def scene_of(spec: dict) -> scenes.Scene:
    return scenes.Scene(name=spec["name"], level=spec["level"], description=spec["description"],
                        friction=SCENE_FRICTION, positions=spec["positions"])


def _front(env: RedAndGraysShelf) -> np.ndarray:
    return env.env._render("front").copy()


def run_sweep(env: RedAndGraysShelf, cylinder: str, direction_deg: float, distance_m: float) -> dict:
    """One sweep of the solution: plan from the cylinder's current pose, run it, measure."""
    before = env.env.cylinder_pose(cylinder)["pos"][:2].copy()
    plan = sweep_skill.sweep_cylinder(env, cylinder, math.radians(direction_deg), distance_m)
    record = {"cylinder": cylinder, "direction_deg": direction_deg, "distance_cm": round(distance_m * 100, 1)}
    if plan.refused is not None:
        return {**record, "ok": False, "failure": f"sweep refused: {plan.refused}"}
    env.log = RunLog()
    info = sweep_skill.execute(env, plan)
    log = env.log
    moved = env.env.cylinder_pose(cylinder)["pos"][:2] - before
    jammed = measures.is_wall_jam(max(log.max_contact_force_on_wall_n, log.max_gray_force_on_wall_n),
                                  log.arm_shelf_touch)
    glitch = measures.is_glitch(max(log.max_contact_force_n, log.max_gray_contact_force_n),
                                max(log.max_red_speed_m_s, log.max_gray_speed_m_s)) and not jammed
    failure = info["refused"] and f"sweep refused: {info['refused']}" or info["failure"] or (
        "jammed against a wall" if jammed else None)
    return {**record, "ok": failure is None, "failure": failure, "glitch": glitch, "jammed": jammed,
            "height_mm": round(plan.height_m * 1000), "moved_mm": [round(float(v) * 1000, 1) for v in moved]}


def solution_job(job: tuple[int, float, str, bool]) -> dict:
    """Place one scene at one friction, check it, run its solution (its sweeps reversed if `reverse`), save
    before/after front images (not for a reversed run)."""
    index, friction, image_dir, reverse = job
    spec, env = SCENE_SPECS[index], _ENV
    sweeps_to_run = list(reversed(spec["sweeps"])) if reverse else spec["sweeps"]
    scene = scene_of(spec)
    record = {"scene": spec["name"], "level": spec["level"], "needs": spec["needs"], "friction": friction,
              "reversed": reverse, "sweeps_planned": len(spec["sweeps"])}
    try:
        scenes.place_scene(env, scene, friction=friction)
    except scenes.SceneError as exc:
        return {**record, "placed": False, "success": False, "failure": str(exc)}
    start_ok, start_reason = retrieval_check.can_retrieve(env, mode="exact")
    record.update({"placed": True, "start_retrievable": start_ok, "start_reason": start_reason,
                   "start_as_expected": start_ok == (spec["level"] == "L0")})
    images = {"before": _front(env)}
    sweeps, failure = [], None
    for cylinder, direction_deg, distance_m in sweeps_to_run:
        result = run_sweep(env, cylinder, direction_deg, distance_m)
        sweeps.append(result)
        if not result["ok"]:
            failure = f"sweep {len(sweeps)} ({cylinder} {direction_deg} deg): {result['failure']}"
            break
    if failure is None:
        record["retrievable_after_sweeps_hybrid"] = retrieval_check.can_retrieve(env, mode="hybrid")[0]
        env.log = RunLog()
        result = retrieval_check.pick_and_pull(env, env.env.cylinder_pose(TARGET)["pos"][:2].copy())
        log = env.log
        record["retrieve_glitch"] = measures.is_glitch(max(log.max_contact_force_n, log.max_gray_contact_force_n),
                                                       max(log.max_red_speed_m_s, log.max_gray_speed_m_s),
                                                       grasping=True)
        if not result["success"]:
            failure = f"retrieve: {result['reason']}"
    images["after"] = _front(env)
    if image_dir and not reverse:
        for when, image in images.items():
            path = Path(image_dir) / f"{spec['name']}_mu{friction}_{when}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            iio.imwrite(path, image)
    return {**record, "sweeps_run": len(sweeps), "success": failure is None, "failure": failure,
            "sweep_glitches": sum(s.get("glitch", False) for s in sweeps),
            "sweeps": json.dumps(sweeps)}


def failure_kind(failure: str | None) -> str:
    if failure is None:
        return "ok"
    for key, label in (("refused", "sweep refused"), ("jammed", "jammed against a wall"),
                       ("toppled: cyl_gray", "gray toppled"), ("toppled: cyl_red", "red toppled"),
                       ("left the shelf: cyl_gray", "gray off the board"), ("left the shelf: cyl_red", "red off the board"),
                       ("hit the shelf", "arm hit the shelf"), ("retrieve", "retrieve failed")):
        if key in failure:
            return label
    return failure[:40]


def plot_gallery(path: Path, rows: list[dict], image_dir: Path, frictions: list[float]) -> None:
    """One row per scene: the scene from above with its sweeps, the front camera before and after (at
    IMAGE_FRICTION), and the result at each friction."""
    probe = RedAndGraysShelf()
    layout, region = probe.env.layout, probe.env.region
    probe.close()
    n = len(SCENE_SPECS)
    fig, axes = plt.subplots(n, 3, figsize=(13, 3.1 * n), gridspec_kw={"width_ratios": [1, 1.33, 1.33]})
    axes = np.atleast_2d(axes)
    for i, spec in enumerate(SCENE_SPECS):
        top = axes[i, 0]
        plotting.draw_shelf(top, layout, region)
        for name, (x, y) in spec["positions"].items():
            colour = plotting.OUTCOME_STYLE["toppled"][0] if name == TARGET else "#8a8a86"
            top.add_patch(Circle((y, x), CYL_RADIUS, color=colour, zorder=3))
        for k, (cylinder, direction_deg, distance_m) in enumerate(spec["sweeps"], start=1):
            x, y = spec["positions"][cylinder]
            dx, dy = distance_m * math.cos(math.radians(direction_deg)), distance_m * math.sin(math.radians(direction_deg))
            top.annotate("", xy=(y + dy, x + dx), xytext=(y, x), zorder=4,
                         arrowprops=dict(arrowstyle="-|>", color=plotting.BLUE_RAMP[-1], lw=1.6, mutation_scale=10))
            top.annotate(str(k), (y + dy, x + dx), xytext=(3, 3), textcoords="offset points", fontsize=8,
                         color=plotting.BLUE_RAMP[-1], weight="bold")
        top.set_xlabel("")
        results = []
        for mu in frictions:
            r = next((r for r in rows if r["scene"] == spec["name"] and r["friction"] == mu), None)
            results.append(f"μ {mu}: {'works' if r and r['success'] else 'fails (' + failure_kind(r and r['failure']) + ')'}")
        top.set_title(f"{spec['name']} ({spec['needs']})", fontsize=8.5)
        for j, when in enumerate(("before", "after"), start=1):
            image_path = image_dir / f"{spec['name']}_mu{IMAGE_FRICTION}_{when}.png"
            if image_path.exists():
                axes[i, j].imshow(iio.imread(image_path))
            axes[i, j].set_axis_off()
            axes[i, j].set_title(f"{when}, μ {IMAGE_FRICTION}" if j == 1 else "after the solution\n"
                                 + ";  ".join(results), fontsize=8)
    fig.tight_layout()
    plotting.save(fig, path, dpi=90)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="friction 0.55 only, into outputs/quick/")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    out = OUT_DIR / "quick" if args.quick else OUT_DIR
    frictions = QUICK_FRICTIONS if args.quick else FRICTIONS
    probe = RedAndGraysShelf()
    try:
        for spec in SCENE_SPECS:
            problems = scenes.layout_problems(scene_of(spec), probe.env.layout)
            if problems:
                raise scenes.SceneError(f"{spec['name']}: " + "; ".join(problems))
            scenes.save_scene(scene_of(spec))
    finally:
        probe.close()
    image_dir = out / "images"
    jobs = [(i, mu, str(image_dir), False) for i in range(len(SCENE_SPECS)) for mu in frictions]
    jobs += [(i, mu, "", True) for i, spec in enumerate(SCENE_SPECS) if spec.get("order_matters") for mu in frictions]
    all_rows = runner.run_parallel(jobs, solution_job, _init, args.workers, "scenes: ")
    rows = [r for r in all_rows if not r["reversed"]]
    reversed_rows = [r for r in all_rows if r["reversed"]]
    for r in all_rows:
        r["failure_kind"] = failure_kind(r["failure"])
    runner.write_csv(out / "solutions.csv", all_rows)
    table = [{"scene": spec["name"], "level": spec["level"], "needs": spec["needs"],
              **{f"works_mu_{mu}": next(r["success"] for r in rows if r["scene"] == spec["name"] and r["friction"] == mu)
                 for mu in frictions},
              "start_as_expected": all(r.get("start_as_expected") for r in rows if r["scene"] == spec["name"]),
              "reversed_order_works": ";".join(f"μ {r['friction']}: {'works' if r['success'] else r['failure_kind']}"
                                               for r in reversed_rows if r["scene"] == spec["name"]) or "-",
              "failures": "; ".join(f"μ {r['friction']}: {r['failure_kind']}" for r in rows
                                    if r["scene"] == spec["name"] and not r["success"])}
             for spec in SCENE_SPECS]
    runner.write_csv(out / "table.csv", table)
    (out / "summary.json").write_text(json.dumps({"table": table, "solutions": {
        s["name"]: {"needs": s["needs"], "sweeps": s["sweeps"]} for s in SCENE_SPECS}}, indent=1) + "\n")
    print(json.dumps(table, indent=1))
    plot_gallery(out / "gallery.png", rows, image_dir, frictions)
    runner.save_config(out, {"quick": args.quick, "frictions": frictions, "scene_friction": SCENE_FRICTION,
                             "image_friction": IMAGE_FRICTION, "scenes_dir": str(scenes.SCENES_DIR),
                             "density_kg_m3": scenes.DENSITY_KG_M3, "scene_specs": SCENE_SPECS})


if __name__ == "__main__":
    main()
