"""Push, pick and place as parameterized joint trajectories.

Each primitive returns an (N, 6) array of joint targets in lerobot units at 30 Hz, ready to feed to
`SO101Env.step` one row at a time. Nothing here touches physics: a primitive is a plan, and whether it
worked is a question for the env afterwards.

The parameterization follows VPG (Zeng et al., IROS 2018), which ran it on a real UR5. A push is a start
point, a direction and a fixed stroke rather than two endpoints, because on this arm shoulder_pan and the
push direction are coupled, so most endpoint pairs you can write down are pushes the arm cannot perform.
A start plus a direction is one stroke to check for reachability, and the discrete direction set means
the reachable directions at a point can be precomputed.

Two measured facts drive the constants. The jaw gap runs 4.1 mm at lerobot gripper 0 to 133.4 mm at 100,
crossing the 30 mm cube width at about 18. And because the moving jaw pivots instead of sliding, the
midpoint between the jaw tips travels 64.7 mm over that same range. So the arm pose gets planned at the
gripper value where the jaws will close on the cube, not the wider value used to approach. Planning it
the other way centres the cube between open jaws, and closing then drags it 12.6 mm into the fixed jaw
before both jaws touch.
"""

import mujoco
import numpy as np

from so101sim import ik, units

CUBE_HALF = 0.015

# Gripper commands, lerobot units. GRIP_PLAN is what IK solves against during a pick, so the closed jaws
# land centred on the cube with about 3 mm of standoff from the fixed jaw. GRIP_OPEN only ever appears in
# the command column. GRIP_CLOSE sits well under the cube width so the servo keeps pressing once the jaws
# touch.
GRIP_OPEN = 35.0
GRIP_PLAN = 22.0
GRIP_CLOSE = 10.0

# The angle the gripper joint actually sits at while holding a cube. Commanding GRIP_CLOSE does not get
# there: the cube blocks the jaws at a 30 mm gap, which the gap curve puts at 18. IK for a loaded arm has
# to use this, not the command, because 8 lerobot units of gripper is 13 mm of jaw midpoint.
GRIP_HOLDING = 18.0

# Transit height for the jaw midpoint. Has to clear the 29 mm bin rim carrying a cube, and stay low
# enough that top-down IK still solves, which gets harder with height and gives out near z=0.10.
HOVER_Z = 0.055
PUSH_Z = 0.012
RELEASE_Z = 0.050

PUSH_DIRECTIONS = 16
PUSH_STROKE = 0.08

# lerobot runs the follower with max_relative_target=4.0 and silently clips anything larger, so the arm
# lags a plan that violates it. Half that leaves margin and gives smoother motion. 30 Hz * 2 deg = 60 deg/s.
MAX_JOINT_STEP_DEG = 2.0
LEROBOT_STEP_LIMIT_DEG = 4.0

# bridge treats a gripper this far below the hover height as not yet hovering, and ramps in joint space
# instead. Relative to hover_z because a sunken bin puts travel height well under the table-level 0.055,
# and a fixed threshold there would send every crossing through the joint ramp and the pile.
BRIDGE_LOW_MARGIN = 0.025


class Unreachable(RuntimeError):
    """Raised when IK cannot follow the requested tool path."""


class _Waypoint:
    __slots__ = ("pos", "yaw", "grip", "hold")

    def __init__(self, pos, yaw, grip, hold=0):
        self.pos = np.asarray(pos, dtype=np.float64)
        self.yaw = float(yaw)
        self.grip = float(grip)
        self.hold = int(hold)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _grip_rad(model, lerobot_value):
    grip_range = tuple(model.jnt_range[model.joint("gripper").id])
    return float(units.to_rad(np.array([0, 0, 0, 0, 0, lerobot_value], dtype=float), grip_range)[5])


def _trace(model, waypoints, ik_grip, lin_step=0.004, yaw_step=np.radians(12), exact_yaw=False):
    """Solve IK along the piecewise-linear tool path through `waypoints`.

    `ik_grip` is the gripper value in lerobot units that IK solves against, held fixed for the whole
    path. It is deliberately independent of the commanded gripper column. By default IK may turn any
    waypoint's yaw half a turn; `exact_yaw` holds it to the yaw given.
    """
    grip_rad = _grip_rad(model, ik_grip)
    flips = (0.0,) if exact_yaw else (0.0, np.pi)
    arms, grips = [], []

    def solve(pos, yaw, seed):
        q, info = ik.solve_topdown(model, pos, yaw=yaw, gripper=grip_rad, q_init=seed,
                                   restarts=0 if seed is not None else 5, flips=flips)
        if info["pos_err"] > 2e-3 or info["tilt"] > 2.0:
            q, info = ik.solve_topdown(model, pos, yaw=yaw, gripper=grip_rad, restarts=6, flips=flips)
        if info["pos_err"] > 2e-3 or info["tilt"] > 2.0:
            raise Unreachable(
                f"no top-down solution at {np.round(pos, 4).tolist()} yaw {np.degrees(yaw):.0f} deg: "
                f"{info['pos_err'] * 1000:.1f} mm off, {info['tilt']:.1f} deg of tilt"
            )
        return q

    seed = solve(waypoints[0].pos, waypoints[0].yaw, None)
    arms.append(seed)
    grips.append(waypoints[0].grip)
    for _ in range(waypoints[0].hold):
        arms.append(seed)
        grips.append(waypoints[0].grip)

    for a, b in zip(waypoints[:-1], waypoints[1:]):
        dyaw = _wrap(b.yaw - a.yaw)
        steps = max(1, int(np.ceil(max(np.linalg.norm(b.pos - a.pos) / lin_step, abs(dyaw) / yaw_step))))
        for i in range(1, steps + 1):
            t = i / steps
            seed = solve(a.pos + t * (b.pos - a.pos), a.yaw + t * dyaw, seed)
            arms.append(seed)
            grips.append(a.grip + t * (b.grip - a.grip))
        for _ in range(b.hold):
            arms.append(seed)
            grips.append(b.grip)

    return np.array(arms), np.array(grips)


def _retime(arms_rad, grips, max_step=MAX_JOINT_STEP_DEG):
    """Subdivide so no joint and no gripper column moves more than `max_step` between frames."""
    frames = np.empty((len(arms_rad), 6))
    frames[:, :5] = np.degrees(arms_rad)
    frames[:, 5] = grips

    out = [frames[0]]
    for a, b in zip(frames[:-1], frames[1:]):
        n = max(1, int(np.ceil(np.abs(b - a).max() / max_step)))
        for i in range(1, n + 1):
            out.append(a + (i / n) * (b - a))
    return np.asarray(out, dtype=np.float32)


def _flipped(yaw, flip):
    """Returns (yaw to plan with, whether IK must hold that yaw exactly).

    `flip` chooses between the two wrist rolls that close the jaws on the same faces: 0 plans at the yaw
    given and 1 at yaw + pi. None, the default, lets IK take whichever it finds first. Near the base the
    choice decides whether the wrist camera mount hits the shoulder.
    """
    return (yaw, False) if flip is None else (yaw + np.pi * int(flip), True)


def pick(model, xy, yaw, hover_z=HOVER_Z, z=CUBE_HALF, flip=None, lift_z=None):
    """Grasp a cube at `xy` with its faces aligned to `yaw` and its centre at height `z`.

    The default `z` is a cube standing on the table. A cube in a pile sits higher, and a cube in a sunken
    bin sits lower. Hovers, descends with the jaws open, closes on the cube, lifts. The arm holds still
    while the jaws close, which is what the real servo does, so the moving jaw sweeps the cube against the
    fixed one. `_flipped` explains `flip`. `lift_z` ends the lift higher than `hover_z`, for a target that
    has to clear a success line.
    """
    yaw, exact = _flipped(yaw, flip)
    x, y = float(xy[0]), float(xy[1])
    top = np.array([x, y, hover_z])
    grasp = np.array([x, y, z])
    path = [
        _Waypoint(top, yaw, GRIP_OPEN),
        _Waypoint(grasp, yaw, GRIP_OPEN, hold=3),
        _Waypoint(grasp, yaw, GRIP_CLOSE, hold=8),
        _Waypoint(top if lift_z is None else np.array([x, y, lift_z]), yaw, GRIP_CLOSE, hold=3),
    ]
    return _retime(*_trace(model, path, GRIP_PLAN, exact_yaw=exact))


def place(model, xy, yaw=0.0, hover_z=HOVER_Z, release_z=RELEASE_Z, flip=None):
    """Carry whatever is in the jaws to `xy` and drop it.

    Releases above the rim rather than lowering into the bin, because the cubes already in there are in
    the way and top-down reach runs out before the arm could get inside anyway. `_flipped` explains
    `flip`.
    """
    yaw, exact = _flipped(yaw, flip)
    x, y = float(xy[0]), float(xy[1])
    over = np.array([x, y, hover_z])
    drop = np.array([x, y, release_z])
    path = [
        _Waypoint(over, yaw, GRIP_CLOSE),
        _Waypoint(drop, yaw, GRIP_CLOSE, hold=3),
        _Waypoint(drop, yaw, GRIP_OPEN, hold=8),
        _Waypoint(over, yaw, GRIP_OPEN, hold=3),
    ]
    return _retime(*_trace(model, path, GRIP_HOLDING, exact_yaw=exact))


def push(model, xy, direction, stroke=PUSH_STROKE, hover_z=HOVER_Z, push_z=PUSH_Z, flip=None):
    """Drag the closed jaws `stroke` metres from `xy` along `direction` radians, at table height.

    The jaws stay shut throughout and sit broadside to the travel, so the pair presents its widest face
    to whatever it hits. `_flipped` explains `flip`.
    """
    start = np.array([float(xy[0]), float(xy[1]), push_z])
    end = start + stroke * np.array([np.cos(direction), np.sin(direction), 0.0])
    yaw, exact = _flipped(direction + np.pi / 2, flip)
    path = [
        _Waypoint(start[:2].tolist() + [hover_z], yaw, GRIP_CLOSE),
        _Waypoint(start, yaw, GRIP_CLOSE, hold=2),
        _Waypoint(end, yaw, GRIP_CLOSE, hold=2),
        _Waypoint(end[:2].tolist() + [hover_z], yaw, GRIP_CLOSE),
    ]
    return _retime(*_trace(model, path, GRIP_CLOSE, exact_yaw=exact))


def push_directions(n=PUSH_DIRECTIONS):
    """The discrete push headings, VPG-style."""
    return np.arange(n) * (2 * np.pi / n)


def lead_in(start, traj, max_step=MAX_JOINT_STEP_DEG):
    """Bridge the arm's current joint state to the first frame of `traj`, in joint space.

    Without this the first commanded frame is wherever the plan happens to begin, the arm lunges for it,
    and it sweeps whatever is in the way. A primitive cannot do this itself because it does not know
    where the arm is, so the env prepends it. lerobot's replay does the same thing on hardware.
    """
    start = np.asarray(start, dtype=np.float32).reshape(6)
    steps = int(np.ceil(np.abs(traj[0] - start).max() / max_step))
    if steps < 1:
        return traj
    ramp = start + np.linspace(0.0, 1.0, steps + 1)[1:, None] * (traj[0] - start)
    return np.vstack([ramp, traj]).astype(np.float32)


def tool_pose(model, frame, data=None):
    """Where a single lerobot-unit frame puts the jaws: (grasp point xyz, jaw yaw, tilt from vertical)."""
    data = mujoco.MjData(model) if data is None else data
    joint_ids = [model.joint(n).id for n in units.JOINT_NAMES]
    grip_range = tuple(model.jnt_range[model.joint("gripper").id])
    data.qpos[model.jnt_qposadr[joint_ids]] = units.to_rad(np.asarray(frame, dtype=np.float64), grip_range)
    mujoco.mj_kinematics(model, data)
    rot = data.site_xmat[model.site("gripperframe").id].reshape(3, 3)
    jaw = rot[:, ik.JAW_AXIS]
    return (ik.grasp_point(model, data).copy(),
            float(np.arctan2(jaw[1], jaw[0])),
            float(np.arccos(np.clip(rot[:, ik.TOOL_AXIS] @ ik.DOWN, -1.0, 1.0))))


def bridge(model, current, traj, hover_z=HOVER_Z, max_step=MAX_JOINT_STEP_DEG):
    """Get from wherever the arm is to the start of `traj` without dragging through the scene.

    A joint-space ramp is an arbitrary curve in tool space, so crossing from the bin back to the table
    that way sweeps the gripper through whatever is already in the bin. Every primitive begins and ends
    hovering top-down, so once the arm is in that state the crossing happens in a straight tool-space
    line at hover height instead. The ramp is still the right answer at the start of an episode, where
    the arm sits at HOME with the gripper nowhere near vertical.
    """
    here, here_yaw, tilt = tool_pose(model, current)
    if tilt > np.radians(20.0) or here[2] < hover_z - BRIDGE_LOW_MARGIN:
        return lead_in(current, traj, max_step)

    there, there_yaw, _ = tool_pose(model, traj[0])
    grip = float(current[5])
    path = [
        _Waypoint([here[0], here[1], hover_z], here_yaw, grip),
        _Waypoint([there[0], there[1], hover_z], there_yaw, grip),
    ]
    try:
        cross = _retime(*_trace(model, path, grip))
    except Unreachable:
        return lead_in(current, traj, max_step)
    return np.vstack([lead_in(current, cross, max_step), lead_in(cross[-1], traj, max_step)])
