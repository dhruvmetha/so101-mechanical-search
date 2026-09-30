"""Pick, place, pull, push and look on a shelf, with the gripper level and the camera mount on top.

Each primitive returns an (N, 6) array of joint targets in lerobot units at 30 Hz, like `primitives`, and touches
no physics. With the tool level, this arm can only point along a ray from the pan axis (see `ik.solve_side`), so
every move goes in and out along rays. Primitives start and end on the standoff line outside the shelf, except
that pick stops with the held cylinder on the aisle line, still over the board, and place and pull start there.
Held cylinders only move sideways along the aisle line, so a gray never leaves the board unless a policy pulls it.
Spec: docs/superpowers/specs/2026-09-26-shelf-search-design.md.
"""

import dataclasses

import mujoco
import numpy as np

from so101sim import ik, units
from so101sim.primitives import (GRIP_CLOSE, GRIP_HOLDING, GRIP_PLAN, MAX_JOINT_STEP_DEG, Unreachable, _Waypoint,
                                 _grip_rad, _retime, lead_in)

# How far the jaws open to go in and to let go. GRIP_OPEN swings the moving jaw 55 mm out, which knocked over the
# right-hand neighbour 2 cm away in the 2026-09-26 spike. A 50 mm gap (lerobot 33) dragged a released cylinder 14 mm
# to the right as the jaw swung out; 43 mm (28) drifts it 1.4 mm and still clears a 3 cm cylinder going in. With
# noslip on (2026-09-29) the drags are 2.5 and 2.2 mm, and 28 stays for its narrower swing.
SIDE_OPEN_GAP = 0.043
PUSH_STROKE = 0.05
PUSH_DIRECTIONS = 16
LIN_STEP = 0.004
POS_TOL = 2e-3
TILT_TOL = 2.0
# A seeded solve that fails falls back to a fresh one. If that lands this far from the seed on any joint, it found
# another branch, and ramping between branches in joint space would sweep the gripper through the shelf.
BRANCH_JUMP = np.radians(20.0)
LEVEL_DEG = 5.0  # bridge treats a gripper this close to level as level
ON_LINE = 0.005  # and a jaw midpoint this close to the standoff line as on it


@dataclasses.dataclass(frozen=True)
class Geometry:
    """The lines and heights the primitives move on."""

    pan_xy: np.ndarray
    front_x: float
    aisle_x: float
    standoff_x: float
    grasp_z: float
    carry_z: float
    push_z: float

    @classmethod
    def from_layout(cls, model, layout):
        return cls(ik.pan_axis_xy(model), layout.front_x, layout.aisle_x, layout.standoff_x,
                   layout.grasp_z, layout.carry_z, layout.push_z)

    def on_ray(self, through_xy, x):
        """Where the ray from the pan axis through `through_xy` crosses the line at `x`."""
        d = np.asarray(through_xy, dtype=np.float64)[:2] - self.pan_xy
        return self.pan_xy + d * (x - self.pan_xy[0]) / d[0]


def jaw_gap(model, lerobot_value):
    """Clear gap between the jaw tips at a gripper value, in metres."""
    data = mujoco.MjData(model)
    data.qpos[model.jnt_qposadr[model.joint("gripper").id]] = _grip_rad(model, lerobot_value)
    mujoco.mj_kinematics(model, data)
    ids = [model.geom(g).id for g in ik.JAW_TIP_GEOMS]
    return float(np.linalg.norm(data.geom_xpos[ids[0]] - data.geom_xpos[ids[1]]) - model.geom_size[ids, 0].sum())


def side_open_value(model, gap=SIDE_OPEN_GAP):
    """The smallest whole lerobot gripper value with a jaw gap of at least `gap`."""
    for value in range(101):
        if jaw_gap(model, value) >= gap:
            return float(value)
    raise ValueError(f"the jaws never open {gap * 1000:.0f} mm")


def push_directions(n=PUSH_DIRECTIONS):
    return np.arange(n) * (2 * np.pi / n)


def _ok(info):
    return info["pos_err"] < POS_TOL and info["tilt"] < TILT_TOL


def _wp(xy, z, grip, hold=0):
    return _Waypoint([float(xy[0]), float(xy[1]), float(z)], 0.0, grip, hold)


def _trace(model, waypoints, ik_grip, q_start=None, lin_step=LIN_STEP):
    """Solve level IK along the straight tool path through `waypoints`, each solve seeded by the last.

    `ik_grip` is the gripper value in lerobot units that IK plans against, held fixed and independent of the
    commanded gripper column. `q_start` seeds the first solve, so a bridge stays on the branch the arm is on.
    """
    grip_rad = _grip_rad(model, ik_grip)
    arms, grips = [], []

    def solve(pos, seed):
        q, info = ik.solve_side(model, pos, gripper=grip_rad, q_init=seed, restarts=0 if seed is not None else 6)
        if not _ok(info):
            q, info = ik.solve_side(model, pos, gripper=grip_rad, restarts=6)
            if _ok(info) and seed is not None and np.abs(q - seed).max() > BRANCH_JUMP:
                raise Unreachable(f"level IK jumped branches at {np.round(pos, 4).tolist()}")
        if not _ok(info):
            raise Unreachable(f"no level solution at {np.round(pos, 4).tolist()}: "
                              f"{info['pos_err'] * 1000:.1f} mm off, {info['tilt']:.1f} deg of tilt")
        return q

    def add(q, grip, n=1):
        for _ in range(n):
            arms.append(q)
            grips.append(grip)

    seed = solve(waypoints[0].pos, q_start)
    add(seed, waypoints[0].grip, 1 + waypoints[0].hold)
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        steps = max(1, int(np.ceil(np.linalg.norm(b.pos - a.pos) / lin_step)))
        for i in range(1, steps + 1):
            t = i / steps
            seed = solve(a.pos + t * (b.pos - a.pos), seed)
            add(seed, a.grip + t * (b.grip - a.grip))
        add(seed, b.grip, b.hold)
    return np.array(arms), np.array(grips)


def pick(model, geo, xy, open_value):
    """Grasp the cylinder standing at `xy` and draw it back to the aisle, still over the board.

    In along the ray at grasp height with the jaws open, close and hold while the moving jaw sweeps the cylinder
    against the fixed one, rise 1 cm, then back out along the same ray until the jaw midpoint is on the aisle line.
    IK plans at GRIP_PLAN so the closed jaws land centred.
    """
    so, aisle = geo.on_ray(xy, geo.standoff_x), geo.on_ray(xy, geo.aisle_x)
    path = [
        _wp(so, geo.grasp_z, open_value),
        _wp(xy, geo.grasp_z, open_value, hold=3),
        _wp(xy, geo.grasp_z, GRIP_CLOSE, hold=8),
        _wp(xy, geo.carry_z, GRIP_CLOSE),
        _wp(aisle, geo.carry_z, GRIP_CLOSE, hold=3),
    ]
    return _retime(*_trace(model, path, GRIP_PLAN))


def place(model, geo, here_xy, xy, open_value):
    """Carry the held cylinder from the aisle point `here_xy` along the aisle line to the ray through `xy`, in to
    `xy`, set it down, open, and back out to the standoff line."""
    path = [
        _wp(geo.on_ray(here_xy, geo.aisle_x), geo.carry_z, GRIP_CLOSE),
        _wp(geo.on_ray(xy, geo.aisle_x), geo.carry_z, GRIP_CLOSE),
        _wp(xy, geo.carry_z, GRIP_CLOSE, hold=2),
        _wp(xy, geo.grasp_z, GRIP_CLOSE, hold=3),
        _wp(xy, geo.grasp_z, open_value, hold=8),
        _wp(geo.on_ray(xy, geo.standoff_x), geo.grasp_z, open_value, hold=3),
    ]
    return _retime(*_trace(model, path, GRIP_HOLDING))


def pull(model, geo, here_xy):
    """Take the held cylinder straight out along its ray from the aisle to the standoff line, off the board.

    That puts its centre 3 cm past the front edge. It's the success move for the red cylinder and a failure for
    any gray.
    """
    path = [
        _wp(geo.on_ray(here_xy, geo.aisle_x), geo.carry_z, GRIP_CLOSE),
        _wp(geo.on_ray(here_xy, geo.standoff_x), geo.carry_z, GRIP_CLOSE, hold=3),
    ]
    return _retime(*_trace(model, path, GRIP_HOLDING))


def push(model, geo, xy, heading, stroke=PUSH_STROKE):
    """Jaws closed, in along the ray to `xy`, `stroke` metres along `heading`, then back the way it came.

    Goes in at grasp height and drops to push height only past the front edge, because level reach can't get as
    close to the base at push height: 0.279 m straight ahead against 0.259 m at grasp height (2026-09-26).
    """
    xy = np.asarray(xy, dtype=np.float64)
    end = xy + stroke * np.array([np.cos(heading), np.sin(heading)])
    so, edge = geo.on_ray(xy, geo.standoff_x), geo.on_ray(xy, geo.front_x)
    path = [
        _wp(so, geo.grasp_z, GRIP_CLOSE),
        _wp(edge, geo.grasp_z, GRIP_CLOSE),
        _wp(edge, geo.push_z, GRIP_CLOSE),
        _wp(xy, geo.push_z, GRIP_CLOSE),
        _wp(end, geo.push_z, GRIP_CLOSE, hold=2),
        _wp(xy, geo.push_z, GRIP_CLOSE),
        _wp(edge, geo.push_z, GRIP_CLOSE),
        _wp(edge, geo.grasp_z, GRIP_CLOSE),
        _wp(so, geo.grasp_z, GRIP_CLOSE, hold=2),
    ]
    return _retime(*_trace(model, path, GRIP_CLOSE))


def look(model, geo, xy, hold=5):
    """Stop on the standoff line on the ray through `xy`, so the wrist camera looks down that ray."""
    return _retime(*_trace(model, [_wp(geo.on_ray(xy, geo.standoff_x), geo.grasp_z, GRIP_PLAN, hold=hold)],
                           GRIP_PLAN))


def tool_point(model, frame, data=None):
    """Jaw midpoint for a lerobot frame with the gripper taken at GRIP_PLAN, and whether the gripper is level.

    The midpoint moves 65 mm across the gripper's range, so both ends of a bridge are measured at one value.
    """
    data = mujoco.MjData(model) if data is None else data
    frame = np.array(frame, dtype=np.float64)
    frame[5] = GRIP_PLAN
    adr = model.jnt_qposadr[[model.joint(n).id for n in units.JOINT_NAMES]]
    data.qpos[adr] = units.to_rad(frame, tuple(model.jnt_range[model.joint("gripper").id]))
    mujoco.mj_kinematics(model, data)
    up = data.site_xmat[model.site("gripperframe").id].reshape(3, 3)[2, ik.SIDE_UP_AXIS]
    return ik.grasp_point(model, data).copy(), bool(up >= np.cos(np.radians(LEVEL_DEG)))


def bridge(model, geo, current, traj, max_step=MAX_JOINT_STEP_DEG):
    """Get from wherever the arm is to the start of `traj` without sweeping through the shelf.

    A level arm inside the bay first backs straight out along its own ray to the standoff line, then slides along
    that line and onto the start. A joint ramp is an arbitrary curve in tool space, which inside the bay knocks
    cylinders over, so it only covers small final corrections, a start that isn't on the standoff line (place and
    pull start where pick left the arm), and poses that aren't level.
    """
    current = np.asarray(current, dtype=np.float64)
    here, here_level = tool_point(model, current)
    there, there_level = tool_point(model, traj[0])
    if (np.linalg.norm(here - there) < 0.003 or not (here_level and there_level)
            or there[0] > geo.standoff_x + ON_LINE):
        return lead_in(current, traj, max_step)
    path = [here]
    if here[0] > geo.standoff_x + ON_LINE:
        path.append(np.r_[geo.on_ray(here, geo.standoff_x), here[2]])
    path += [np.r_[there[:2], path[-1][2]], there]
    waypoints = [_Waypoint(p, 0.0, float(current[5])) for p in path]
    try:
        cross = _retime(*_trace(model, waypoints, GRIP_PLAN, q_start=np.radians(current[:5])), max_step)
    except Unreachable:
        return lead_in(current, traj, max_step)
    return np.vstack([lead_in(current, cross, max_step), lead_in(cross[-1], traj, max_step)])
