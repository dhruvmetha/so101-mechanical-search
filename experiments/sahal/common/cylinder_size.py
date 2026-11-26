"""Shelf cylinders of another size, without touching repo files.

The repo's cylinders are 3 cm wide and 8 cm tall (`so101sim.shelf_layout`), and its grasp and carry heights and
gripper values are set for them. `use(size)` makes this process build every shelf model with cylinders of `size` and
sets the matching constants in the repo's modules, so the env, its primitives and its checks all see the new size:

    from common import cylinder_size
    cylinder_size.use(cylinder_size.CylinderSize(radius_m=0.02, height_m=0.05))   # before building an env
    env = RedAndGraysShelf()

What changes, for this process only:
- the model: `so101sim.env.load_model` is wrapped to resize every `cyl_*` geom in the MjSpec before compiling, so
  mass, inertia and the collision bounding boxes follow (resizing a compiled model's geom_size misses the midphase);
- `shelf_layout.CYL_RADIUS`, `CYL_HALF_HEIGHT`, `GRASP_UP` (grasp height, default half the height), `CARRY_UP`
  (grasp height + PICK_LIFT_M, as in the repo) and `shelf.CYL_RADIUS`, `CYL_HALF_HEIGHT`, `MIN_SPACING`;
- `side_primitives.GRIP_PLAN` and `GRIP_HOLDING`: the gripper values whose jaw gap is the diameter plus the repo's
  planning slack, and the diameter itself (the repo's 22 and 19.4 give 34.3 and 30.4 mm for 3 cm);
- the side opening (`open_gap_m()`): the diameter plus the repo's 13 mm (43 mm for 3 cm). `SingleRedShelf` applies it.

`use(REPO)` puts every value back. Spawned workers start from the repo's values, so call `use` in each worker.
"""

from __future__ import annotations

from dataclasses import dataclass

import common  # noqa: F401  sets sys.path and MUJOCO_GL before mujoco loads
import mujoco

from so101sim import env as so101_env
from so101sim import primitives, shelf, shelf_layout, side_primitives

REPO_RADIUS_M = 0.015
REPO_HEIGHT_M = 0.08
PICK_LIFT_M = shelf_layout.CARRY_UP - shelf_layout.GRASP_UP  # pick lifts the held cylinder this far (1 cm)
SPACING_GAP_M = shelf.MIN_SPACING - 2 * shelf_layout.CYL_RADIUS  # the env's surface-to-surface spacing (5 mm)
OPEN_GAP_SLACK_M = side_primitives.SIDE_OPEN_GAP - 2 * REPO_RADIUS_M  # 13 mm wider than the cylinder going in
_REPO_LOAD_MODEL = so101_env.load_model
_REPO = {"GRIP_PLAN": primitives.GRIP_PLAN, "GRIP_HOLDING": primitives.GRIP_HOLDING,
         "GRASP_UP": shelf_layout.GRASP_UP, "CARRY_UP": shelf_layout.CARRY_UP}
GRIP_SEARCH_STEPS = 40  # bisection steps for a gripper value with a given jaw gap


@dataclass(frozen=True)
class CylinderSize:
    """Cylinder radius and height, and the grasp height (jaw midpoint above the board; None: half the height)."""

    radius_m: float
    height_m: float
    grasp_up_m: float | None = None

    @property
    def grasp_height_m(self) -> float:
        return self.height_m / 2 if self.grasp_up_m is None else self.grasp_up_m

    @property
    def name(self) -> str:
        return f"r{self.radius_m * 100:g}_h{self.height_m * 100:g}_g{self.grasp_height_m * 100:g}"

    @property
    def is_repo(self) -> bool:
        return (self.radius_m, self.height_m, self.grasp_height_m) == (REPO_RADIUS_M, REPO_HEIGHT_M, _REPO["GRASP_UP"])


REPO = CylinderSize(REPO_RADIUS_M, REPO_HEIGHT_M, grasp_up_m=_REPO["GRASP_UP"])
_current = REPO


def current() -> CylinderSize:
    return _current


def open_gap_m(size: CylinderSize | None = None) -> float:
    """The jaw gap the side grasp opens to: the diameter plus the repo's slack."""
    size = size or _current
    return side_primitives.SIDE_OPEN_GAP if size.is_repo else 2 * size.radius_m + OPEN_GAP_SLACK_M


def use(size: CylinderSize) -> None:
    """Make this process build shelf models with cylinders of `size` and set the matching repo constants."""
    global _current
    _current = size
    half = size.height_m / 2
    shelf_layout.CYL_RADIUS = shelf.CYL_RADIUS = size.radius_m
    shelf_layout.CYL_HALF_HEIGHT = shelf.CYL_HALF_HEIGHT = half
    shelf.MIN_SPACING = 2 * size.radius_m + SPACING_GAP_M
    shelf_layout.GRASP_UP = size.grasp_height_m
    shelf_layout.CARRY_UP = size.grasp_height_m + PICK_LIFT_M
    if size.is_repo:
        so101_env.load_model = _REPO_LOAD_MODEL
        side_primitives.GRIP_PLAN, side_primitives.GRIP_HOLDING = _REPO["GRIP_PLAN"], _REPO["GRIP_HOLDING"]
        return
    so101_env.load_model = _resized_loader(size)
    model = _REPO_LOAD_MODEL(shelf.SHELF_SCENE_PATH)
    planning_slack_m = side_primitives.jaw_gap(model, _REPO["GRIP_PLAN"]) - 2 * REPO_RADIUS_M
    side_primitives.GRIP_PLAN = gripper_for_gap(model, 2 * size.radius_m + planning_slack_m)
    side_primitives.GRIP_HOLDING = gripper_for_gap(model, 2 * size.radius_m)


def gripper_for_gap(model: mujoco.MjModel, gap_m: float) -> float:
    """The lerobot gripper value (0 to 100) whose jaw gap is `gap_m`; the gap grows with the value."""
    low, high = 0.0, 100.0
    for _ in range(GRIP_SEARCH_STEPS):
        middle = (low + high) / 2
        low, high = (middle, high) if side_primitives.jaw_gap(model, middle) < gap_m else (low, middle)
    return round((low + high) / 2, 3)


def holding_gripper() -> float:
    """The gripper value the jaws stop at on the current cylinder (planning value for a held cylinder)."""
    return side_primitives.GRIP_HOLDING


def _resized_loader(size: CylinderSize):
    """so101sim.env.load_model, with every cyl_* geom resized before compiling."""

    def load_model(scene_path=so101_env.SCENE_PATH):
        spec = mujoco.MjSpec.from_file(str(scene_path))
        solref, solimp = list(spec.default.geom.solref), list(spec.default.geom.solimp)
        for geom in spec.geoms:
            if geom.classname.name in so101_env.GRIPPER_CLASSES:
                geom.solref, geom.solimp = solref, solimp
        for geom in spec.geoms:
            if geom.name.startswith("cyl_"):
                old_half = float(geom.size[1])
                geom.size = [size.radius_m, size.height_m / 2, 0.0]
                body = geom.parent
                body.pos = [body.pos[0], body.pos[1], body.pos[2] - old_half + size.height_m / 2]
        return spec.compile()

    return load_model
