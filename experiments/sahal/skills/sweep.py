"""Sweep: the shelf push's route with the jaws closed and the gripper level, slowed down while it is low.

Route (the repo's `side_primitives.push`): from the standoff line in along the ray through the start point at grasp
height to the board's front edge, drop to sweep height, go to the start point, stroke to the end point, retrace to
the start point and the front edge, rise to grasp height and go back out to the standoff line. The arm moves at
CONTACT_SPEED_FRACTION of the repo's normal speed only over the stroke and the last SLOW_LEAD_M of the approach to
the start point; everything else (in, drop, approach, retrace, out) runs at normal speed. Sweep height is the lowest
of SWEEP_HEIGHTS_M that passes the env's path check.

Not supported for now: sweeps toward the robot (within TOWARD_ROBOT_HALF_ANGLE_DEG of 180 deg). The route enters
low at the front edge, so to get behind a cylinder it passes through it; in study 05, 15 of 18 such sweeps toppled.
They are refused unless `allow_toward_robot=True` (for studies that measure it). DEFAULT_DIRECTIONS_DEG leaves 180 deg
out.

Reused from the repo, not copied: level IK along straight tool paths (`side_primitives._trace`, a solve every 4 mm),
the 2 deg/frame joint cap (`primitives._retime`), the bridge from the current pose (`side_primitives.bridge`) and
the env's path check (`ShelfPrimitiveEnv._path_hit`: arm against shelf and table, 6 mm margin).

    plan = sweep(env, x_start_m, y_start_m, direction_rad, duration_s)    # or sweep_points(...)
    if plan.refused is None:
        execute(env, plan)
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import skills  # noqa: F401  sets sys.path and MUJOCO_GL
import mujoco
import numpy as np

from common.shelf_env import PUSH_START_GAP_M, SingleRedShelf, push_start_back_distance
from so101sim import ik, side_primitives, units
from so101sim.env import CONTROL_HZ
from so101sim.primitives import GRIP_CLOSE, Unreachable, _grip_rad, _retime

# Median jaw-midpoint speed along the repo's push at push height, normal speed: 0.059 to 0.060 m/s at five starts
# and headings (measured 2026-11-21 by forward kinematics of side_primitives.push).
NORMAL_TOOL_SPEED_M_S = 0.060
CONTACT_SPEED_FRACTION = 1 / 3  # study 04: no topples at half speed or slower, mean tilt <= 5.15 deg at 1/3
CONTACT_SPEED_M_S = NORMAL_TOOL_SPEED_M_S * CONTACT_SPEED_FRACTION
SWEEP_HEIGHTS_M = (0.023, 0.024, 0.025, 0.026)  # tried lowest first; jaw midpoint above the board
SLOW_LEAD_M = 0.01  # slow down this far before the start point, so the arm is slow before it can touch anything
END_HOLD_FRAMES = 2  # pause at the end of the stroke, as push does
OUT_HOLD_FRAMES = 2  # pause back on the standoff line, as push does
DEFAULT_DIRECTIONS_DEG = (0, 45, 90, 135, 225, 270, 315)  # 0 = away from the robot, 90 = robot's left; no 180
TOWARD_ROBOT_HALF_ANGLE_DEG = 22.5  # headings closer than this to 180 deg are refused (halfway to 135 and 225)


@dataclass(frozen=True)
class SweepPlan:
    """A planned sweep. Positions in metres (world x, y), angles in radians, times in seconds.

    `trajectory` is (frames, 6) joint targets in lerobot units at 30 Hz, starting on the standoff line; the env adds
    the bridge from wherever the arm is. `refused` says why there is no trajectory, or is None.
    """

    start_xy_m: tuple[float, float]
    end_xy_m: tuple[float, float]
    direction_rad: float
    length_m: float
    duration_s: float
    height_m: float | None = None
    trajectory: np.ndarray | None = None
    slow_frames: int = 0
    refused: str | None = None


def sweep(env: SingleRedShelf, x_start_m: float, y_start_m: float, direction_rad: float,
          duration_s: float, allow_toward_robot: bool = False) -> SweepPlan:
    """Plan a sweep from a start point along a direction for a duration.

    x_start_m, y_start_m: where the jaw midpoint starts the stroke, world metres.
    direction_rad: stroke heading in the board plane, radians; 0 = +x (away from the robot), pi/2 = +y (robot's left).
    duration_s: stroke time in seconds at CONTACT_SPEED_M_S; the stroke length is CONTACT_SPEED_M_S * duration_s.
    allow_toward_robot: plan sweeps toward the robot instead of refusing them (not supported; for studies only).
    """
    start = np.array([x_start_m, y_start_m], dtype=np.float64)
    length = CONTACT_SPEED_M_S * float(duration_s)
    end = start + length * np.array([math.cos(direction_rad), math.sin(direction_rad)])
    return _plan(env, start, end, float(direction_rad), length, float(duration_s), allow_toward_robot)


def sweep_points(env: SingleRedShelf, x_start_m: float, y_start_m: float, x_end_m: float,
                 y_end_m: float, allow_toward_robot: bool = False) -> SweepPlan:
    """Plan the same motion from a start and an end point.

    x_start_m, y_start_m: where the jaw midpoint starts the stroke, world metres.
    x_end_m, y_end_m: where the stroke ends, world metres. Direction and duration follow from the two points.
    allow_toward_robot: as for sweep().
    """
    start = np.array([x_start_m, y_start_m], dtype=np.float64)
    end = np.array([x_end_m, y_end_m], dtype=np.float64)
    delta = end - start
    length = float(np.linalg.norm(delta))
    return _plan(env, start, end, math.atan2(delta[1], delta[0]), length, length / CONTACT_SPEED_M_S,
                 allow_toward_robot)


def sweep_cylinder(env: SingleRedShelf, cylinder: str, direction_rad: float, distance_m: float) -> SweepPlan:
    """Plan a sweep that moves the named cylinder about `distance_m` along `direction_rad`, from its current pose.

    The stroke starts where the arm comes no nearer than PUSH_START_GAP_M to that cylinder on the way in (found
    kinematically) and is distance_m + PUSH_START_GAP_M long, so the jaws travel distance_m after contact. Other
    cylinders are not checked on the way in. Study 05: real sweeps move a cylinder less than asked (distance ratio
    about 0.94 away from the robot, 0.82 to 0.85 sideways) and bend it away from the robot.
    """
    inner = env.env
    xy = inner.cylinder_pose(cylinder)["pos"][:2].copy()
    unit = np.array([math.cos(direction_rad), math.sin(direction_rad)])
    try:
        back = push_start_back_distance(env, xy, direction_rad, SWEEP_HEIGHTS_M[0], cylinder=cylinder)
    except Unreachable as exc:
        start = tuple(xy.tolist())
        return SweepPlan(start, start, direction_rad, distance_m, 0.0, refused=f"IK, start behind {cylinder}: {exc}")
    start = xy - back * unit
    return sweep(env, float(start[0]), float(start[1]), direction_rad,
                 (distance_m + PUSH_START_GAP_M) / CONTACT_SPEED_M_S)


def execute(env: SingleRedShelf, plan: SweepPlan) -> dict:
    """Run a planned sweep through the env's own step (holding check, bridge, path check, physics checks, settle).
    Returns the env's info. Raises ValueError for a refused plan."""
    if plan.refused is not None:
        raise ValueError(f"cannot execute a refused sweep: {plan.refused}")
    _, _, _, _, info = env.run_planned(plan.trajectory, plan.start_xy_m, plan.direction_rad)
    return info


def _plan(env: SingleRedShelf, start: np.ndarray, end: np.ndarray, direction_rad: float, length_m: float,
          duration_s: float, allow_toward_robot: bool) -> SweepPlan:
    """Check the board, then try each height lowest first until one passes IK and the env's path check."""
    plan = SweepPlan(tuple(start.tolist()), tuple(end.tolist()), direction_rad, length_m, duration_s)
    layout = env.env.layout
    if length_m <= 0:
        return dataclasses.replace(plan, refused="zero-length sweep")
    if toward_robot(direction_rad) and not allow_toward_robot:
        return dataclasses.replace(plan, refused=f"direction {math.degrees(direction_rad):.0f} deg is toward the "
                                                 "robot: not supported (the low entry passes through the cylinder)")
    if not layout.on_board(start):
        return dataclasses.replace(plan, refused=f"start {np.round(start, 4).tolist()} is off the board")
    if not layout.on_board(end):
        return dataclasses.replace(plan, refused=f"end {np.round(end, 4).tolist()} is off the board")
    reasons = []
    for height in SWEEP_HEIGHTS_M:
        geometry = dataclasses.replace(env.base_geometry, push_z=layout.board_z + height)
        try:
            trajectory, slow_frames = build_trajectory(env.model, geometry, start, end)
        except Unreachable as exc:
            reasons.append(f"{height * 1000:.0f} mm: IK, {exc}")
            continue
        bridged = side_primitives.bridge(env.model, env.geo, env.env.joint_state, trajectory)
        hit = env._path_hit(bridged)
        if hit is None:
            return dataclasses.replace(plan, height_m=height, trajectory=trajectory, slow_frames=slow_frames)
        reasons.append(f"{height * 1000:.0f} mm: path check, would hit {hit[1]}")
    return dataclasses.replace(plan, refused="; ".join(reasons))


def toward_robot(direction_rad: float) -> bool:
    """True if a heading (radians, 0 = away from the robot) is within TOWARD_ROBOT_HALF_ANGLE_DEG of 180 deg."""
    off_away = abs(math.degrees(math.atan2(math.sin(direction_rad), math.cos(direction_rad))))
    return off_away > 180 - TOWARD_ROBOT_HALF_ANGLE_DEG


def build_trajectory(model: mujoco.MjModel, geometry: side_primitives.Geometry, start: np.ndarray,
                     end: np.ndarray) -> tuple[np.ndarray, int]:
    """The sweep's joint trajectory (lerobot units, 30 Hz) at `geometry.push_z`, and how many frames run slow.

    Raises primitives.Unreachable if level IK fails anywhere on the route.
    """
    wp = side_primitives._wp
    standoff, edge = geometry.on_ray(start, geometry.standoff_x), geometry.on_ray(start, geometry.front_x)
    high, low = geometry.grasp_z, geometry.push_z
    approach = np.asarray(edge, dtype=np.float64) - start
    gap = float(np.linalg.norm(approach))
    slow_from = start + approach * (min(SLOW_LEAD_M, gap) / gap) if gap > 0 else start
    free_in = [wp(standoff, high, GRIP_CLOSE), wp(edge, high, GRIP_CLOSE), wp(edge, low, GRIP_CLOSE),
               wp(slow_from, low, GRIP_CLOSE)]
    slow_path = [wp(slow_from, low, GRIP_CLOSE), wp(start, low, GRIP_CLOSE),
                 wp(end, low, GRIP_CLOSE, hold=END_HOLD_FRAMES)]
    free_out = [wp(end, low, GRIP_CLOSE), wp(start, low, GRIP_CLOSE), wp(edge, low, GRIP_CLOSE),
                wp(edge, high, GRIP_CLOSE), wp(standoff, high, GRIP_CLOSE, hold=OUT_HOLD_FRAMES)]
    arms_in, grips_in = side_primitives._trace(model, free_in, GRIP_CLOSE)
    arms_slow, grips_slow = side_primitives._trace(model, slow_path, GRIP_CLOSE, q_start=arms_in[-1])
    arms_out, grips_out = side_primitives._trace(model, free_out, GRIP_CLOSE, q_start=arms_slow[-1])
    frames_in = _retime(arms_in, grips_in)
    frames_slow = _retime(*_at_tool_speed(model, arms_slow, grips_slow, CONTACT_SPEED_M_S))
    frames_out = _retime(arms_out, grips_out)
    trajectory = np.vstack([frames_in, frames_slow[1:], frames_out[1:]])
    return trajectory, len(frames_slow) - 1


def _at_tool_speed(model: mujoco.MjModel, arm_angles: np.ndarray, grips: np.ndarray,
                   speed_m_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Subdivide consecutive IK solutions so the jaw midpoint moves at most `speed_m_s` per 30 Hz frame.

    `arm_angles` are (n, 5) radians along a straight tool path (side_primitives._trace), `grips` lerobot units.
    Repeated solutions (holds) stay single frames. Joint angles are interpolated linearly between neighbouring
    solutions, which are at most 4 mm apart, as primitives._retime does.
    """
    step_m = speed_m_s / CONTROL_HZ
    points = _jaw_midpoints(model, arm_angles)
    arms, out_grips = [arm_angles[0]], [grips[0]]
    for i in range(1, len(arm_angles)):
        pieces = max(1, math.ceil(float(np.linalg.norm(points[i] - points[i - 1])) / step_m))
        for k in range(1, pieces + 1):
            t = k / pieces
            arms.append(arm_angles[i - 1] + t * (arm_angles[i] - arm_angles[i - 1]))
            out_grips.append(grips[i - 1] + t * (grips[i] - grips[i - 1]))
    return np.array(arms), np.array(out_grips)


def _jaw_midpoints(model: mujoco.MjModel, arm_angles: np.ndarray) -> np.ndarray:
    """Jaw midpoint (world metres) for each row of arm angles, with the gripper at GRIP_CLOSE."""
    data = mujoco.MjData(model)
    joints = model.jnt_qposadr[[model.joint(n).id for n in units.JOINT_NAMES]]
    data.qpos[joints[5]] = _grip_rad(model, GRIP_CLOSE)
    out = []
    for q in arm_angles:
        data.qpos[joints[:5]] = q
        mujoco.mj_kinematics(model, data)
        out.append(ik.grasp_point(model, data).copy())
    return np.array(out)
