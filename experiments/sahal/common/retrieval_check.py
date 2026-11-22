"""Can the red be retrieved now? A kinematic check of the repo's pick + pull on the red, from full state.

    ok, reason = can_retrieve(env)                    # hybrid (default): fast when clear, else simulated
    ok, reason = can_retrieve(env, mode="kinematic")  # no physics at all, 5 mm from the grays (too strict)
    ok, reason = can_retrieve(env, mode="exact")      # always simulate pick + pull; for verifying scenes
    gap_m = min_gray_clearance(env)                # the plan's smallest arm/held-red to gray distance

No physics is stepped: the env's model and state are only read, and the joint trajectories come from the repo's own
planners (`side_primitives.pick`, `pull`, `bridge`). In order, the first failure is returned:

1. the red is inside the env's action box (pick refuses otherwise);
2. level IK solves pick and pull (`primitives.Unreachable` otherwise);
3. for pick, then pull: the env's path check (`ShelfPrimitiveEnv._path_hit`: arm against shelf and table with 6 mm
   of margin), and, frame by frame, every arm geom at least ARM_GRAY_MARGIN_M from every gray cylinder on the board;
   once the jaws have closed, the held red at least HELD_RED_GRAY_MARGIN_M from every gray and HELD_RED_SHELF_MARGIN_M
   from the shelf's walls and top board. Within a primitive the earliest frame wins. The board is left out for the
   red: it stands on it when grasped, and pick lifts it 1 cm before moving it.

The held red is placed upright at the jaw midpoint computed with the gripper at GRIP_HOLDING, from the first frame
where the gripper command reaches GRIP_CLOSE. In physics the jaws stop on the red near GRIP_HOLDING (a hold at the
centre read 18.97), and there the jaw midpoint is within 1.4 mm of the red's centre (measured 2026-11-21). The arm is
also checked with the gripper at GRIP_HOLDING while it holds, because the jaws can't close further than the red.
Before the close the red stands where it is and the arm is not checked against it (the grasp has to touch it).

Hybrid mode. Study 07 found the 5 mm gray margin far too strict: pick + pull often shove a gray aside and still
succeed (planned overlaps down to -27 mm), while every scene where a gray toppled or left the shelf had an overlap of
at least 7.9 mm. So: the exact refusals first (action box, IK, the env's path check against the shelf, the held red
near a wall), then the plan's smallest gray clearance decides. At HYBRID_CLEAR_M or more: yes. Below HYBRID_DEEP_M:
no. In between: run pick + pull in physics from the current state, put the state back, and return what happened.
HYBRID_CLEAR_M comes from study 07's tuning scenes (seed 7, outputs/margin_table.csv: no wrong fast yes down to
-7 mm). HYBRID_DEEP_M was -25 mm from the same table, then moved to -33 mm after the held-out scenes (seed 70) had
successes with overlaps down to 35.2 mm; a third set (seed 700) tests it.

Exact mode always runs pick + pull in physics from the current state and puts the state back: the answer is what
the env would do, at about 1.4 s per check.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import common  # noqa: F401  sets sys.path and MUJOCO_GL before mujoco loads
import mujoco
import numpy as np

from common.measures import geom_label
from so101sim import ik, side_primitives, units
from so101sim.primitives import GRIP_CLOSE, GRIP_HOLDING, Unreachable, _grip_rad
from so101sim.shelf import ACTION_TOLERANCE, TARGET, ShelfPrimitiveEnv

ARM_GRAY_MARGIN_M = 0.005  # no arm geom nearer than this to a gray; the executed arm strays a few mm from its plan
HELD_RED_GRAY_MARGIN_M = 0.005  # the held red no nearer than this to a gray
HELD_RED_SHELF_MARGIN_M = 0.003  # the held red no nearer than this to the shelf's walls and top board
RED_SHELF_GEOMS = ("shelf_left", "shelf_right", "shelf_back", "shelf_top")
CLOSED_GRIP_SLACK = 0.5  # lerobot units: a gripper command within this of GRIP_CLOSE counts as closed
DISTANCE_CUTOFF_M = 0.05  # mj_geomDistance search radius; anything farther counts as clear
HYBRID_CLEAR_M = -0.005  # hybrid: plan's gray clearance at least this -> yes (study 07: 0 of 89 such answers wrong)
HYBRID_DEEP_M = -0.033  # hybrid: below this -> no (study 07 held-out: 1 of 31 such answers wrong); between -> simulate
CHECK_MODES = ("hybrid", "kinematic", "exact")
_SAVED_ENV_ATTRIBUTES = ("_n", "_last_grip", "_aisle_xy", "log", "would_refuse", "phase")
_SAVED_INNER_ATTRIBUTES = ("_steps", "_streak", "_hit")


@dataclass(frozen=True)
class _Ids:
    arm: tuple[int, ...]
    red_shelf: tuple[int, ...]
    red: int
    red_qpos: int
    arm_qpos: np.ndarray
    gripper_range: tuple[float, float]
    probe: mujoco.MjData


def can_retrieve(env: ShelfPrimitiveEnv, mode: str = "hybrid") -> tuple[bool, str]:
    """(True, reason) if pick then pull on the red would work, else (False, the first reason it fails).

    mode "hybrid" (default) and "exact": see the module docstring; the reason starts with "simulated: " when physics
    decided. mode "kinematic": every check above, no physics.
    """
    if mode == "hybrid":
        return _hybrid(env)
    if mode == "exact":
        return _simulate(env, env.env.cylinder_pose(TARGET)["pos"][:2].copy())
    if mode != "kinematic":
        raise ValueError(f"unknown mode {mode!r}, expected one of {CHECK_MODES}")
    inner, model = env.env, env.model
    ids = _ids(env)
    red_xy = inner.cylinder_pose(TARGET)["pos"][:2].copy()
    box = getattr(env, "strict_action_box", env.action_space["xy"])
    if np.abs(red_xy - np.clip(red_xy, box.low, box.high)).max() > ACTION_TOLERANCE:
        return False, f"red at {np.round(red_xy, 4).tolist()} is outside the action box"
    try:
        pick, pull, held_from = _plans(env, red_xy)
    except Unreachable as exc:
        return False, f"IK: {exc}"
    grays = _grays_on_board(env)
    for name, trajectory, first_held in (("pick", pick, held_from), ("pull", pull, 0)):
        shelf_hit = ShelfPrimitiveEnv._path_hit(env, trajectory)  # the repo's own check, not a subclass's bypass
        clearance_hit = _clearance_hit(env, ids, grays, trajectory, first_held)
        hits = [h for h in (shelf_hit and (shelf_hit[0], f"arm within 6 mm of {shelf_hit[1]}"), clearance_hit) if h]
        if hits:
            frame, what = min(hits)
            return False, f"{name}: {what} at frame {frame}"
    return True, "ok"


def min_gray_clearance(env: ShelfPrimitiveEnv) -> float | None:
    """Smallest distance in metres between any arm geom or the held red and any gray, over the whole planned pick
    and pull (negative: the plan overlaps a gray; capped at DISTANCE_CUTOFF_M). None if IK fails or there are no
    grays."""
    try:
        plans = _plans(env, env.env.cylinder_pose(TARGET)["pos"][:2].copy())
    except Unreachable:
        return None
    return _plan_gray_clearance(env, *plans)


def _plan_gray_clearance(env: ShelfPrimitiveEnv, pick: np.ndarray, pull: np.ndarray, held_from: int,
                         stop_below_m: float = -np.inf) -> float | None:
    """min_gray_clearance for given plans; stops scanning once the clearance is below `stop_below_m`."""
    ids, grays = _ids(env), _grays_on_board(env)
    if not grays:
        return None
    model, probe = env.model, ids.probe
    smallest = DISTANCE_CUTOFF_M
    for trajectory, first_held in ((pick, held_from), (pull, 0)):
        for k in _posed_frames(env, ids, trajectory, first_held):
            movers = ids.arm + ((ids.red,) if k >= first_held else ())
            for mover in movers:
                for gray in grays:
                    smallest = min(smallest, mujoco.mj_geomDistance(model, probe, mover, gray, DISTANCE_CUTOFF_M, None))
            if smallest < stop_below_m:
                return smallest
    return smallest


def _hybrid(env: ShelfPrimitiveEnv) -> tuple[bool, str]:
    """Exact refusals, then the plan's gray clearance, then physics for the unclear middle."""
    ids = _ids(env)
    red_xy = env.env.cylinder_pose(TARGET)["pos"][:2].copy()
    box = getattr(env, "strict_action_box", env.action_space["xy"])
    if np.abs(red_xy - np.clip(red_xy, box.low, box.high)).max() > ACTION_TOLERANCE:
        return False, f"red at {np.round(red_xy, 4).tolist()} is outside the action box"
    try:
        pick, pull, held_from = _plans(env, red_xy)
    except Unreachable as exc:
        return False, f"IK: {exc}"
    for name, trajectory, first_held in (("pick", pick, held_from), ("pull", pull, 0)):
        hit = ShelfPrimitiveEnv._path_hit(env, trajectory)
        if hit is not None:
            return False, f"{name}: arm within 6 mm of {hit[1]} at frame {hit[0]}"
        for k in _posed_frames(env, ids, trajectory, first_held):
            if k >= first_held and any(_nearer(env.model, ids.probe, ids.red, shelf, HELD_RED_SHELF_MARGIN_M)
                                       for shelf in ids.red_shelf):
                return False, f"{name}: held red within {HELD_RED_SHELF_MARGIN_M * 1000:g} mm of a wall at frame {k}"
    clearance = _plan_gray_clearance(env, pick, pull, held_from, stop_below_m=HYBRID_DEEP_M)
    if clearance is None or clearance >= HYBRID_CLEAR_M:
        gap = "no grays" if clearance is None else f"{clearance * 1000:.1f} mm"
        return True, f"clear of the grays (plan's smallest gap {gap})"
    if clearance < HYBRID_DEEP_M:
        return False, f"plan overlaps a gray by {-clearance * 1000:.1f} mm or more"
    return _simulate(env, red_xy)


def _simulate(env: ShelfPrimitiveEnv, red_xy: np.ndarray) -> tuple[bool, str]:
    """Pick + pull in physics from the current state, the state put back afterwards."""
    with simulation_copy(env):
        result = pick_and_pull(env, red_xy)
    return result["success"], f"simulated: {result['reason']}"


def pick_and_pull(env: ShelfPrimitiveEnv, red_xy: np.ndarray) -> dict:
    """Run the repo's pick on the red and, if pick ran cleanly, pull, in physics from the current state.
    Returns success (the env's is_success), a reason, and how many frames pick took."""
    _, _, terminated, _, pick = env.step(("pick", np.asarray(red_xy, dtype=np.float64), 0.0))
    out = {"pick_frames": pick["frames"], "success": False}
    if pick["refused"] is not None:
        return {**out, "reason": f"pick refused: {pick['refused']}"}
    if terminated:
        return {**out, "reason": f"pick: {pick['failure']}"}
    _, _, _, _, pull = env.step(("pull", np.zeros(2), 0.0))
    if pull["refused"] is not None:
        return {**out, "reason": f"pull refused: {pull['refused']}"}
    if pull["failure"] is not None:
        return {**out, "reason": f"pull: {pull['failure']}"}
    if not pull["is_success"]:
        return {**out, "reason": "pull ended without the red out and held"}
    return {**out, "success": True, "reason": "retrieved"}


@contextmanager
def simulation_copy(env: ShelfPrimitiveEnv) -> Iterator[None]:
    """Run physics inside the block, then put the env back exactly: the whole MjData (warm starts included) and the
    env's own bookkeeping. A RunLog, if the env keeps one, is swapped for an empty one while inside."""
    saved = mujoco.MjData(env.model)
    mujoco.mj_copyData(saved, env.model, env.data)
    outer = {k: getattr(env, k) for k in _SAVED_ENV_ATTRIBUTES if hasattr(env, k)}
    inner = {k: getattr(env.env, k) for k in _SAVED_INNER_ATTRIBUTES if hasattr(env.env, k)}
    if "log" in outer:
        env.log = type(outer["log"])()
    try:
        yield
    finally:
        mujoco.mj_copyData(env.data, env.model, saved)
        for k, v in outer.items():
            setattr(env, k, v)
        for k, v in inner.items():
            setattr(env.env, k, v)


def _plans(env: ShelfPrimitiveEnv, red_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """The repo's pick (bridged from the arm's current joints) and pull (bridged from pick's end) for the red, and
    the first pick frame where the gripper command has closed. Raises Unreachable."""
    model, geo = env.model, env.geo
    pick = side_primitives.bridge(model, geo, env.env.joint_state, side_primitives.pick(model, geo, red_xy,
                                                                                       env.open_value))
    pull = side_primitives.bridge(model, geo, pick[-1], side_primitives.pull(model, geo, geo.on_ray(red_xy, geo.aisle_x)))
    closed = np.flatnonzero(pick[:, 5] <= GRIP_CLOSE + CLOSED_GRIP_SLACK)
    return pick, pull, int(closed[0]) if len(closed) else len(pick)


def _posed_frames(env: ShelfPrimitiveEnv, ids: _Ids, trajectory: np.ndarray, first_held: int):
    """Pose the probe at each frame of `trajectory` (the red moved to the jaws from `first_held` on); yields the
    frame index with the probe posed."""
    model, probe = env.model, ids.probe
    probe.qpos[:] = env.data.qpos
    holding_rad = _grip_rad(model, GRIP_HOLDING)
    for k, row in enumerate(trajectory):
        held = k >= first_held
        probe.qpos[ids.arm_qpos] = units.to_rad(np.asarray(row, dtype=np.float64), ids.gripper_range)
        if held:
            probe.qpos[ids.arm_qpos[5]] = holding_rad
        mujoco.mj_kinematics(model, probe)
        if held:
            probe.qpos[ids.red_qpos:ids.red_qpos + 7] = [*ik.grasp_point(model, probe), 1.0, 0.0, 0.0, 0.0]
            mujoco.mj_kinematics(model, probe)
        yield k


def _clearance_hit(env: ShelfPrimitiveEnv, ids: _Ids, grays: list[int], trajectory: np.ndarray,
                   first_held: int) -> tuple[int, str] | None:
    """(frame, what) for the first frame where an arm geom is too near a gray, or the held red too near a gray or
    a wall or the top board; None if the whole trajectory is clear."""
    model, probe = env.model, ids.probe
    for k in _posed_frames(env, ids, trajectory, first_held):
        held = k >= first_held
        for arm in ids.arm:
            for gray in grays:
                if _nearer(model, probe, arm, gray, ARM_GRAY_MARGIN_M):
                    return k, f"arm ({geom_label(model, arm)}) within {ARM_GRAY_MARGIN_M * 1000:g} mm of " \
                              f"{geom_label(model, gray)}"
        if not held:
            continue
        for gray in grays:
            if _nearer(model, probe, ids.red, gray, HELD_RED_GRAY_MARGIN_M):
                return k, f"held red within {HELD_RED_GRAY_MARGIN_M * 1000:g} mm of {geom_label(model, gray)}"
        for shelf in ids.red_shelf:
            if _nearer(model, probe, ids.red, shelf, HELD_RED_SHELF_MARGIN_M):
                return k, f"held red within {HELD_RED_SHELF_MARGIN_M * 1000:g} mm of {geom_label(model, shelf)}"
    return None


def _nearer(model: mujoco.MjModel, data: mujoco.MjData, geom_a: int, geom_b: int, margin_m: float) -> bool:
    """True if two geoms are closer than `margin_m` (negative distance: overlapping). A bounding-sphere test skips
    far pairs before the exact distance."""
    gap = np.linalg.norm(data.geom_xpos[geom_a] - data.geom_xpos[geom_b]) - model.geom_rbound[geom_a] \
        - model.geom_rbound[geom_b]
    if gap >= margin_m:
        return False
    return mujoco.mj_geomDistance(model, data, geom_a, geom_b, DISTANCE_CUTOFF_M, None) < margin_m


def _grays_on_board(env: ShelfPrimitiveEnv) -> list[int]:
    """Geom ids of every cylinder other than the red standing over the board (parked ones are off stage)."""
    inner, model, data = env.env, env.model, env.data
    out = []
    for b in range(model.nbody):
        name = model.body(b).name
        if not name.startswith("cyl_") or name == TARGET:
            continue
        address = model.jnt_qposadr[model.body(b).jntadr[0]]
        if inner.layout.on_board(data.qpos[address:address + 2]) and data.qpos[address + 2] > 0:
            out.append(model.geom(name).id)
    return out


def _ids(env: ShelfPrimitiveEnv) -> _Ids:
    """Geom and joint addresses the check needs, cached on the env."""
    cached = getattr(env, "_retrieval_ids", None)
    if cached is None:
        inner, model = env.env, env.model
        red_body = model.body(TARGET)
        cached = _Ids(arm=tuple(sorted(inner._arm_ids)),
                      red_shelf=tuple(model.geom(name).id for name in RED_SHELF_GEOMS),
                      red=model.geom(TARGET).id, red_qpos=int(model.jnt_qposadr[red_body.jntadr[0]]),
                      arm_qpos=np.asarray(inner._qpos_adr), gripper_range=inner._gripper_range,
                      probe=mujoco.MjData(model))
        env._retrieval_ids = cached
    return cached
