"""The checks that choose the mechanical-search layout.

scripts/size_search_bin.py runs all of them to pick a layout. tests/test_search_checks.py reruns the cheap
ones against the committed scene. The rules and thresholds are the spec's sizing rule and its cameras and
start pose section: docs/superpowers/specs/2026-09-25-mechanical-search-design.md.
"""

import mujoco
import numpy as np

from so101sim import grasp, ik, primitives, units
from so101sim.search_layout import MARKER_PREFIX

GRID = 0.005
YAWS = tuple(np.radians([0.0, 45.0, 90.0, 135.0]))
# A cube whose face is this close to a wall needs a push before a straight-down grasp fits: the gripper is
# wider than the cube. Measured in the spec's amendment.
WALL_STRIP = 0.02
FRONT_MIN = 0.70
WRIST_MIN = 0.50
IMAGE_SIZE = (480, 640)  # height, width: the env's default observation size
GEOM = int(mujoco.mjtObj.mjOBJ_GEOM)


def grid_points(layout, step=GRID):
    """Cube-centre positions on a `step` grid, out to where a cube touches the walls."""
    reach = layout.width / 2 - primitives.CUBE_HALF
    n = int(np.floor(reach / step + 1e-9))
    offsets = np.arange(-n, n + 1) * step
    return [(round(layout.center_x + dx, 4), round(float(dy), 4)) for dx in offsets for dy in offsets]


def near_wall(layout, xy, strip=WALL_STRIP):
    """True when the cube's face is within `strip` of any wall, the edge included."""
    face_x = layout.width / 2 - abs(xy[0] - layout.center_x) - primitives.CUBE_HALF
    face_y = layout.width / 2 - abs(xy[1]) - primitives.CUBE_HALF
    return min(face_x, face_y) <= strip + 1e-9


def _ok(info):
    return info["pos_err"] < 1e-3 and info["tilt"] < 1.0


def reachable(model, xyz, grip):
    """True if top-down IK puts the jaw midpoint within 1 mm of `xyz`, pointing down to 1 degree, for at
    least one of YAWS. `grip` is the gripper value in lerobot units that IK plans against."""
    grip_rad = primitives._grip_rad(model, grip)
    return any(_ok(ik.solve_topdown(model, xyz, yaw=yaw, gripper=grip_rad, restarts=2)[1]) for yaw in YAWS)


def grasp_poses(model, xyz):
    """Arm angles in radians, one per yaw that IK reaches at `xyz`, planned at GRIP_PLAN like `pick`."""
    grip_rad = primitives._grip_rad(model, primitives.GRIP_PLAN)
    poses = []
    for yaw in YAWS:
        q, info = ik.solve_topdown(model, xyz, yaw=yaw, gripper=grip_rad, restarts=2)
        if _ok(info):
            poses.append(q)
    return poses


def scene_geom_ids(model):
    """The bin, the riser and the table: everything fixed the arm could hit. Probe geoms excluded."""
    bin_id = model.body("bin").id
    ids = {g for g in range(model.ngeom) if model.geom_bodyid[g] == bin_id
           and not model.geom(g).name.startswith(MARKER_PREFIX) and model.geom(g).name != "drop_column"}
    return frozenset(ids | {model.geom("riser").id, model.geom("table").id})


def _joint_adr(model):
    return model.jnt_qposadr[[model.joint(n).id for n in units.JOINT_NAMES]]


def set_arm(model, data, arm_rad, gripper):
    """Five arm joints in radians plus the gripper in lerobot units, then forward, which finds contacts."""
    adr = _joint_adr(model)
    data.qpos[adr[:5]] = arm_rad
    data.qpos[adr[5]] = primitives._grip_rad(model, gripper)
    mujoco.mj_forward(model, data)


def set_pose(model, data, pose):
    """All six joints from lerobot units, then forward."""
    grip_range = tuple(model.jnt_range[model.joint("gripper").id])
    data.qpos[_joint_adr(model)] = units.to_rad(np.asarray(pose, dtype=np.float64), grip_range)
    mujoco.mj_forward(model, data)


def clearance_ok(model, data, poses, arm_ids, scene_ids):
    """True if at least one of `poses`, with the jaws opened to GRIP_OPEN, touches nothing in the scene."""
    for q in poses:
        set_arm(model, data, q, primitives.GRIP_OPEN)
        if not grasp.touching(data, arm_ids, scene_ids):
            return True
    return False


def drop_column_clear(model, data, home):
    """True if no arm geom touches the probe scene's drop column with the arm at `home`."""
    set_pose(model, data, home)
    return not grasp.touching(data, grasp.arm_geom_ids(model), {model.geom("drop_column").id})


def marker_coverage(model, data, renderer, camera, home):
    """Share of the probe scene's floor markers that `camera` sees, with the arm at `home`."""
    set_pose(model, data, home)
    markers = {g for g in range(model.ngeom) if model.geom(g).name.startswith(MARKER_PREFIX)}
    renderer.enable_segmentation_rendering()
    try:
        renderer.update_scene(data, camera=camera)
        seg = renderer.render()
    finally:
        renderer.disable_segmentation_rendering()
    seen = set(np.unique(seg[..., 0][seg[..., 1] == GEOM]).tolist())
    return len(markers & seen) / len(markers)
