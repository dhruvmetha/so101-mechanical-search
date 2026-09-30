"""The checks that choose the shelf layout.

scripts/size_shelf.py runs all of them. tests/test_shelf_checks.py reruns the cheap ones against the committed
scene. Rules and thresholds: the sizing rule in docs/superpowers/specs/2026-09-26-shelf-search-design.md.
"""

import mujoco
import numpy as np

from so101sim import grasp, ik, primitives
from so101sim.search_checks import marker_coverage, set_arm, set_pose  # noqa: F401  the bin's probe helpers
from so101sim.shelf_layout import MARKER_PREFIX

GRID = 0.005
# The moving jaw opens toward the right wall, so centres this close to it need a push before a pick.
RIGHT_STRIP = 0.06
# The closed gripper is wider than a cylinder. At centres this close to a side wall the fixed jaw and gripper body
# hit the left wall and the moving jaw hits the right one (first sizing run, 2026-09-26, the spec's amendment).
# Every collision check skips them; a cylinder there needs a push first, and the path check refuses the rest.
WALL_STRIP = 0.03
# A carried arm bounces about 5 mm around its plan, so a pose that misses the shelf by less than this counts as a hit.
# Found 2026-09-26: with a 13 cm opening the camera mount scraped the top board during a place. dm1487 chose 6 mm
# over 8: at 8 no shelf passes, because pushing deep in the bay takes the forearm 6 to 8 mm over the board's front
# edge. Checked on a model used only for checks, never for physics.
CLEARANCE = 0.006
FRONT_MIN = 0.60
WRIST_MIN = 0.40
MIN_AREA = 0.12 * 0.18  # the spec's stop rule, read as an area
IMAGE_SIZE = (480, 640)  # height, width: the env's default observation size


def grid(lo, hi, step=GRID):
    n = int(np.floor((hi - lo) / step + 1e-9))
    return [round(lo + i * step, 4) for i in range(n + 1)]


def centre_points(layout):
    x_lo, x_hi, y_lo, y_hi = layout.region()
    return [(x, y) for x in grid(x_lo, x_hi) for y in grid(y_lo, y_hi)]


def aisle_points(layout):
    _, _, y_lo, y_hi = layout.region()
    return [(round(layout.aisle_x, 4), y) for y in grid(y_lo, y_hi)]


def ray_points(layout, geo, step=GRID):
    """Points on every pull path: from each aisle point out along its ray to the standoff line."""
    out = []
    for p in aisle_points(layout):
        a, b = geo.on_ray(p, layout.aisle_x), geo.on_ray(p, layout.standoff_x)
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        out += [tuple(np.round(a + (i / n) * (b - a), 4)) for i in range(n + 1)]
    return out


def near_right_wall(layout, xy, strip=RIGHT_STRIP):
    return xy[1] + layout.half_width <= strip + 1e-9


def in_wall_strip(layout, xy, strip=WALL_STRIP):
    return layout.half_width - abs(xy[1]) <= strip + 1e-9


def region_area(layout):
    x_lo, x_hi, y_lo, y_hi = layout.region()
    return max(0.0, x_hi - x_lo) * max(0.0, y_hi - y_lo)


def sizing_checks(L, geo, open_value):
    """The sizing rule's checks for layout `L`: (name, points, height, IK gripper, gripper for the collision check,
    exempt), likeliest to fail first. Reach is checked everywhere. Collisions skip the strip along each side wall,
    where the gripper is too wide to fit beside a cylinder (the spec's first amendment), and the open jaw also skips
    the right-wall strip. `carry` covers place carrying a cylinder in to its spot, which the aisle and pull lines
    don't."""
    G = primitives
    # Farthest first, where reach runs out.
    centres = sorted(centre_points(L), key=lambda p: -np.hypot(p[0] - geo.pan_xy[0], p[1] - geo.pan_xy[1]))
    aisle = aisle_points(L)
    standoff = [tuple(np.round(geo.on_ray(p, L.standoff_x), 4)) for p in aisle]
    edge = [tuple(np.round(geo.on_ray(p, L.front_x), 4)) for p in aisle]
    strip = lambda p: in_wall_strip(L, p)  # noqa: E731
    checks = (  # name, points, height, IK gripper, gripper for the collision check, exempt
        ("grasp", centres, L.grasp_z, G.GRIP_PLAN, G.GRIP_PLAN, strip),
        ("open jaw", centres, L.grasp_z, G.GRIP_PLAN, open_value, lambda p: strip(p) or near_right_wall(L, p)),
        ("aisle", aisle, L.carry_z, G.GRIP_HOLDING, G.GRIP_HOLDING, strip),
        ("carry", centres, L.carry_z, G.GRIP_HOLDING, G.GRIP_HOLDING, strip),
        ("standoff at grasp height", standoff, L.grasp_z, G.GRIP_PLAN, G.GRIP_PLAN, None),
        ("standoff at carry height", standoff, L.carry_z, G.GRIP_HOLDING, G.GRIP_HOLDING, None),
        ("pull path", ray_points(L, geo), L.carry_z, G.GRIP_HOLDING, G.GRIP_HOLDING, strip),
        ("push", edge + aisle + centres, L.push_z, G.GRIP_CLOSE, G.GRIP_CLOSE, strip),
    )
    return checks


def level_pose(model, xyz, grip):
    """Arm angles in radians for a level, mount-up gripper with its jaw midpoint within 1 mm of `xyz`, or None.
    `grip` is the lerobot gripper value IK plans against."""
    q, info = ik.solve_side(model, xyz, gripper=primitives._grip_rad(model, grip), restarts=4)
    return q if info["pos_err"] < 1e-3 and info["tilt"] < 2.0 else None


def shelf_geom_ids(model):
    """Every shelf panel and leg, plus the table: everything fixed the arm could hit. Probe markers excluded."""
    ids = {g for g in range(model.ngeom) if model.geom(g).name.startswith("shelf_")}
    return frozenset(ids | {model.geom("table").id})


def lowest_camera(options, minimum=FRONT_MIN):
    """The (camera position, share of board seen) to use: the lowest spot that sees `minimum` of the board, the
    most board among spots at that height. None if none does. Lower is better, because a camera near shelf height
    sees cylinders from the side, so the red can hide behind grays (spec, third amendment)."""
    passing = [o for o in options if o[1] >= minimum]
    return min(passing, key=lambda o: (o[0][2], -o[1])) if passing else None


def add_clearance(model, margin=CLEARANCE):
    """Grow every shelf box and the table by `margin` on each side, so `clear` and `grasp.touching` report near misses.

    The midphase goes off too. Its bounding-box tree is built at compile time, so with the old sizes it skipped geom
    pairs inside the grown band before testing them. With it off, this model agreed with a shelf recompiled 8 mm
    bigger on 2719 of 2719 sampled frames (final review, 2026-09-26). Only for a model used for checks, never for
    physics.
    """
    for g in shelf_geom_ids(model):
        model.geom_size[g] += margin
        model.geom_aabb[g, 3:] += margin
        model.geom_rbound[g] = np.linalg.norm(model.geom_size[g])
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MIDPHASE)
    return model


def clear(model, data, q, grip, arm_ids, scene_ids):
    """True if the arm at `q`, gripper at `grip` in lerobot units, touches nothing in `scene_ids`."""
    set_arm(model, data, q, grip)
    return not grasp.touching(data, arm_ids, scene_ids)


__all__ = ["GRID", "RIGHT_STRIP", "WALL_STRIP", "CLEARANCE", "add_clearance", "lowest_camera", "sizing_checks", "in_wall_strip", "FRONT_MIN", "WRIST_MIN", "MIN_AREA", "IMAGE_SIZE", "MARKER_PREFIX", "grid",
           "centre_points", "aisle_points", "ray_points", "near_right_wall", "region_area", "level_pose",
           "shelf_geom_ids", "clear", "marker_coverage", "set_arm", "set_pose"]
