"""The shelf with only the red cylinder on it, the settings the studies change, and the shared push run.

How the gray cylinders go away without editing the repo: mj_resetData parks every cylinder on the floor at
x = 1.2 m, z = -0.75 m. The normal reset then stands them on the board; ours never does, and it shrinks the inner
env's list of task cylinders to the red, so the topple, off-board and at-rest checks only look at the red. Moving the
grays off the board would fail "left the shelf"; turning their collisions off drops them forever, which keeps
settled() from ever passing.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import common  # noqa: F401  sets sys.path and MUJOCO_GL before mujoco loads
import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from common import measures
from so101sim import ik, rest, side_primitives, units
from so101sim.primitives import GRIP_CLOSE, _grip_rad
from so101sim.shelf import TARGET, TOPPLE_DEG, ShelfPrimitiveEnv, _Refused
from so101sim.shelf_layout import CYL_RADIUS

MID_RANGE_PHYSICS = {"friction": 0.55, "density": 900.0}  # middle of the env's friction and density ranges
SAMPLE_PHYSICS = "sample"  # draw friction and density from the env's ranges for the seed, as ShelfEnv does
PUSH_DENSITY_KG_M3 = 900.0
PUSH_RESET_SEED = 0
PUSH_START_GAP_M = 0.005  # no arm geom within this of the cylinder on the way in or at the push start
PUSH_START_SEARCH_M = 0.06
PUSH_START_TOLERANCE_M = 5e-4
JITTER_RADIUS_M = 0.001
# The region's centre, written out: computing it as (0.355 + 0.46) / 2 differs in the last bit, and the earlier
# studies used this literal (piles and pushes are chaotic, so the last bit can matter).
REGION_CENTRE_XY_M = (0.4075, 0.0)
PUSH_DIRECTIONS_RAD = {"away": 0.0, "left": np.pi / 2, "right": -np.pi / 2}  # away = +x, left = +y (robot's left)


@dataclass
class RunLog:
    """What one run recorded, frame by frame and substep by substep."""

    frames: list[dict] = field(default_factory=list)
    arm_contacts: list[measures.Contact] = field(default_factory=list)
    max_contact_force_n: float = 0.0
    max_red_speed_m_s: float = 0.0
    max_contact_force_on_wall_n: float = 0.0  # peak arm-red force over substeps where the red touches a wall or top
    red_wall_touch: str | None = None  # first wall or top board the red touched
    arm_gray_touch: str | None = None  # first gray cylinder an arm geom touched
    max_gray_contact_force_n: float = 0.0  # peak arm force on any one gray
    max_gray_speed_m_s: float = 0.0
    max_gray_force_on_wall_n: float = 0.0  # peak arm force on a gray while that gray touches a wall or the top board
    arm_shelf_touch: str | None = None
    substeps: list[dict] | None = None
    video: list[np.ndarray] = field(default_factory=list)


class SingleRedShelf(ShelfPrimitiveEnv):
    """ShelfPrimitiveEnv with only the red cylinder in the task, measured at every physics substep.

    widen_action_box: accept xy anywhere a cylinder fits on the board, not only the env's reset region.
    bypass_path_check: run primitives the path check would refuse; its verdict goes to `would_refuse`.
    arm_touch_ends_episode: False keeps going when the arm touches the shelf, recording it in `log.arm_shelf_touch`.
    record_video: keep a front | wrist image per 30 Hz frame in `log.video`.
    """

    def __init__(self, *, widen_action_box: bool = False, bypass_path_check: bool = False,
                 arm_touch_ends_episode: bool = True, record_video: bool = False) -> None:
        super().__init__(on_frame=self._on_frame)
        inner = self.env
        inner.cyls, inner.grays = (TARGET,), ()
        inner._rest_dofs = rest.free_dofs(self.model, inner.cyls)
        inner._cyl_geoms = frozenset({inner._geom[TARGET]})
        inner._apply = self._apply_and_measure  # instance attributes shadow ShelfEnv's methods for this env only
        if not arm_touch_ends_episode:
            inner._failure = self._failure_ignoring_arm_touch
        self.strict_action_box = self.action_space["xy"]
        if widen_action_box:
            layout = inner.layout
            low = np.array([layout.front_x + CYL_RADIUS, -layout.half_width + CYL_RADIUS], dtype=np.float32)
            high = np.array([layout.back_x - CYL_RADIUS, layout.half_width - CYL_RADIUS], dtype=np.float32)
            self.action_space = spaces.Dict({**self.action_space.spaces, "xy": spaces.Box(low, high)})
        self.bypass_path_check = bypass_path_check
        self.record_video = record_video
        self.record_substeps = False
        self.base_geometry = self.geo
        self.push_speed_fraction = 1.0
        self.planned_trajectory: np.ndarray | None = None
        self.would_refuse = None
        self.phase = None
        self.log = RunLog()
        self._all_cylinders = [self.model.body(b).name for b in range(self.model.nbody)
                               if self.model.body(b).name.startswith("cyl_")]
        self._push_start_cache: dict[tuple[str, float], float] = {}

    # ---- setup -----------------------------------------------------------------------------------------------

    def reset(self, *, seed: int = 0, red_xy=None, physics: dict | str | None = None) -> dict:
        """Start pose, physics, and the red standing at `red_xy` (default: the region's centre). Returns place_red's
        report. `physics` is a {"friction", "density"} dict, SAMPLE_PHYSICS, or None for MID_RANGE_PHYSICS."""
        self._start(seed, physics)
        return self.place_red(np.array(REGION_CENTRE_XY_M) if red_xy is None else red_xy)

    def _start(self, seed: int, physics: dict | str | None) -> None:
        """Seed, physics, the arm at its start pose and every cylinder parked off stage (reset without placing)."""
        gym.Env.reset(self, seed=seed)
        inner = self.env
        gym.Env.reset(inner, seed=seed)
        inner.physics = inner._set_physics(seed, None if physics == SAMPLE_PHYSICS else (physics or MID_RANGE_PHYSICS))
        inner._start_pose()
        self._n, self._aisle_xy = 0, None
        self._last_grip = float(inner.layout.home[5])
        inner._steps, inner._streak = 0, 0
        inner._reset_info = {"hidden": False, "physics": dict(inner.physics)}
        self.log = RunLog()

    def place_red(self, xy) -> dict:
        """Stand the red upright at `xy` (anywhere on the board) and let it come to rest. Reports where it ended."""
        inner = self.env
        inner._place(TARGET, np.asarray(xy, dtype=np.float64))
        touched = inner._settle()

        def frame() -> None:
            mujoco.mj_step(self.model, self.data, nstep=inner._n_substeps)

        resting = rest.come_to_rest(frame, self.data, inner._rest_dofs)
        mujoco.mj_forward(self.model, self.data)
        pose = inner.cylinder_pose(TARGET)
        return {"pos": pose["pos"], "tilt": pose["tilt"], "touched_arm": bool(touched), "resting": bool(resting),
                "on_board": inner.layout.on_board(pose["pos"][:2]), "grays_on_board": self.grays_on_board()}

    def grays_on_board(self) -> int:
        """Gray cylinders standing over the board footprint, from the true state. Always 0 here."""
        count = 0
        for name in self._all_cylinders:
            if name == TARGET:
                continue
            address = self.model.jnt_qposadr[self.model.body(name).jntadr[0]]
            xyz = self.data.qpos[address:address + 3]
            count += self.env.layout.on_board(xyz[:2]) and xyz[2] > 0
        return int(count)

    def set_push_height(self, height_above_board_m: float) -> None:
        self.geo = dataclasses.replace(self.base_geometry, push_z=self.env.layout.board_z + height_above_board_m)

    def in_strict_action_box(self, xy) -> bool:
        """Whether the unmodified env would accept `xy` (its box, plus its 5 mm tolerance)."""
        box, xy = self.strict_action_box, np.asarray(xy)
        return bool(np.all(xy >= box.low - 0.005) and np.all(xy <= box.high + 0.005))

    def snapshot(self) -> np.ndarray:
        return np.hstack([self.env._render("front").copy(), self.env._render("wrist").copy()])

    # ---- overrides -------------------------------------------------------------------------------------------

    def run_planned(self, trajectory: np.ndarray, xy, heading_rad: float):
        """Run a trajectory planned outside the env (lerobot units, 30 Hz) through the env's own `step` as a push:
        the same holding check, bridge from the current pose, path check, frame-by-frame checks and settle."""
        self.planned_trajectory = np.asarray(trajectory, dtype=np.float32)
        try:
            return self.step(("push", np.asarray(xy, dtype=np.float64), float(heading_rad)))
        finally:
            self.planned_trajectory = None

    def _plan(self, kind, xy, angle):
        if kind == "push" and self.planned_trajectory is not None:
            if self.holding():
                raise _Refused("push with a cylinder still in the gripper; place or pull it first (holding)")
            return self.planned_trajectory
        trajectory = super()._plan(kind, xy, angle)
        if kind == "push":
            trajectory = resample_trajectory(trajectory, self.push_speed_fraction)
        return trajectory

    def _path_hit(self, trajectory):
        hit = super()._path_hit(trajectory)
        if self.bypass_path_check:
            self.would_refuse = hit
            return None
        return hit

    def _on_frame(self, _env) -> None:
        inner = self.env
        pose = inner.cylinder_pose(TARGET)
        self.log.frames.append({"phase": self.phase, "frame": inner._steps, "x": pose["pos"][0], "y": pose["pos"][1],
                                "z": pose["pos"][2], "tilt": pose["tilt"], "held": inner.held(TARGET),
                                "gripper": float(inner.joint_state[5])})
        if self.record_video:
            self.log.video.append(self.snapshot())

    def _apply_and_measure(self, action) -> None:
        """ShelfEnv._apply (one 30 Hz control step, arm-shelf check every substep), plus arm-cylinder contacts,
        their peak force, red-wall contacts and the cylinder's peak speed every substep. Reads state only, so physics
        is unchanged."""
        inner, model, data, log = self.env, self.model, self.data, self.log
        action = np.clip(action, inner.action_space.low, inner.action_space.high)
        data.ctrl[:] = units.to_rad(action, inner._gripper_range)
        inner._hit = None
        red, board_z = inner._geom[TARGET], inner.layout.board_z
        dof = inner._dofadr[TARGET]
        board = frozenset({model.geom("shelf_board").id})
        grays = inner._cyl_geoms - {red}
        walls = frozenset(model.geom(name).id for name in measures.WALL_GEOMS)
        for _ in range(inner._n_substeps):
            mujoco.mj_step(model, data)
            if inner._hit is None:
                inner._hit = inner.arm_hit()
            if inner._hit is not None and log.arm_shelf_touch is None:
                log.arm_shelf_touch = inner._hit
            contacts = measures.red_contacts(model, data, red, inner._arm_ids, board_z, self.record_substeps)
            log.arm_contacts += contacts
            speed = float(np.linalg.norm(data.qvel[dof:dof + 3]))
            log.max_red_speed_m_s = max(log.max_red_speed_m_s, speed)
            if grays:
                if log.arm_gray_touch is None:
                    log.arm_gray_touch = _first_touch(model, data, inner._arm_ids, grays)
                self._measure_grays(grays, walls)
            on_wall = measures.red_contacts(model, data, red, walls, board_z)
            if on_wall and log.red_wall_touch is None:
                log.red_wall_touch = on_wall[0].geom
            if contacts:
                force = sum(c.normal_force_n for c in contacts)
                log.max_contact_force_n = max(log.max_contact_force_n, force)
                if on_wall:
                    log.max_contact_force_on_wall_n = max(log.max_contact_force_on_wall_n, force)
            if self.record_substeps:
                log.substeps = [] if log.substeps is None else log.substeps
                base = measures.red_contacts(model, data, red, board, board_z, True)
                log.substeps.append({
                    "time_s": data.time, "tilt_deg": inner.tilt(TARGET), "red_speed_m_s": speed,
                    "jaw_point_xyz": ik.grasp_point(model, data).copy(),
                    "arm_normal_n": sum(c.normal_force_n for c in contacts),
                    "arm_tangential_n": sum(c.tangential_force_n for c in contacts),
                    "arm_slip_m_s": max((c.slip_speed_m_s for c in contacts), default=0.0),
                    "board_normal_n": sum(c.normal_force_n for c in base),
                    "base_slip_m_s": max((c.slip_speed_m_s for c in base), default=float("nan")),
                })

    def _measure_grays(self, grays: frozenset[int], walls: frozenset[int]) -> None:
        """Peak arm force on each gray, the grays' peak speed, and the arm force on a gray pinned to a wall."""
        model, data, log, inner = self.model, self.data, self.log, self.env
        for gray in grays:
            arm = measures.red_contacts(model, data, gray, inner._arm_ids, inner.layout.board_z)
            force = sum(c.normal_force_n for c in arm)
            log.max_gray_contact_force_n = max(log.max_gray_contact_force_n, force)
            if arm and measures.red_contacts(model, data, gray, walls, inner.layout.board_z):
                log.max_gray_force_on_wall_n = max(log.max_gray_force_on_wall_n, force)
            name = model.body(model.geom_bodyid[gray]).name
            dof = inner._dofadr[name]
            log.max_gray_speed_m_s = max(log.max_gray_speed_m_s, float(np.linalg.norm(data.qvel[dof:dof + 3])))

    def _failure_ignoring_arm_touch(self):
        """ShelfEnv._failure without its first check: an arm-shelf touch is recorded, not an episode end."""
        inner = self.env
        for name in inner.cyls:
            if inner.tilt(name) > TOPPLE_DEG:
                return f"toppled: {name}"
            if not inner.layout.on_board(inner._xy(name)) and (name != TARGET or not inner.held(name)):
                return f"left the shelf: {name}"
        return None


class RedAndGraysShelf(SingleRedShelf):
    """SingleRedShelf with some gray cylinders standing on the board too, placed by the caller. The grays placed join
    the task's checks (topple, off the board, at rest); the rest stay parked off stage."""

    def reset_scene(self, *, seed: int = 0, red_xy, gray_xys: dict[str, tuple[float, float]],
                    physics: dict | str | None = None) -> dict:
        """Stand the red at `red_xy` and each named gray at its xy, then let them come to rest. Returns where each
        cylinder ended, its tilt, whether any touched the arm and whether they came to rest."""
        self._start(seed, physics)
        inner = self.env
        names = (TARGET, *gray_xys)
        inner.cyls, inner.grays = names, tuple(gray_xys)
        inner._rest_dofs = rest.free_dofs(self.model, names)
        inner._cyl_geoms = frozenset(inner._geom[n] for n in names)
        inner._place(TARGET, np.asarray(red_xy, dtype=np.float64))
        for name, xy in gray_xys.items():
            inner._place(name, np.asarray(xy, dtype=np.float64))
        touched = inner._settle()

        def frame() -> None:
            mujoco.mj_step(self.model, self.data, nstep=inner._n_substeps)

        resting = rest.come_to_rest(frame, self.data, inner._rest_dofs)
        mujoco.mj_forward(self.model, self.data)
        poses = {n: inner.cylinder_pose(n) for n in names}
        return {"xy": {n: poses[n]["pos"][:2].copy() for n in names}, "tilt": {n: poses[n]["tilt"] for n in names},
                "touched_arm": bool(touched), "resting": bool(resting)}


def _first_touch(model: mujoco.MjModel, data: mujoco.MjData, arm: frozenset[int], others: frozenset[int]) -> str | None:
    """Name of the first geom in `others` that an arm geom is touching now, or None."""
    for c in data.contact[: data.ncon]:
        if c.geom1 in arm and c.geom2 in others:
            return model.geom(c.geom2).name
        if c.geom2 in arm and c.geom1 in others:
            return model.geom(c.geom1).name
    return None


def resample_trajectory(trajectory: np.ndarray, speed_fraction: float) -> np.ndarray:
    """The same path at `speed_fraction` of its speed: linear interpolation at frame spacing `speed_fraction`."""
    if speed_fraction >= 1.0:
        return trajectory
    n = len(trajectory)
    times = np.arange(0.0, n - 1 + 1e-9, speed_fraction)
    if times[-1] < n - 1:
        times = np.r_[times, n - 1]
    index = np.arange(n)
    columns = [np.interp(times, index, trajectory[:, j]) for j in range(trajectory.shape[1])]
    return np.stack(columns, axis=1).astype(np.float32)


def push_start_back_distance(env: SingleRedShelf, cylinder_xy: np.ndarray, heading_rad: float,
                             height_above_board_m: float, cylinder: str = TARGET) -> float:
    """How far behind the cylinder centre, against `heading_rad`, the push starts so that no arm geom comes within
    PUSH_START_GAP_M of `cylinder` on the way in or at the start. Kinematic, by bisection. Other cylinders are not
    checked."""
    model = env.model
    probe = mujoco.MjData(model)
    probe.qpos[:] = env.data.qpos
    inner = env.env
    red = inner._geom[cylinder]
    saved_margin = model.geom_margin[red]
    model.geom_margin[red] = PUSH_START_GAP_M
    direction = np.array([np.cos(heading_rad), np.sin(heading_rad)])
    env.set_push_height(height_above_board_m)
    geometry = env.geo

    def clear(back_m: float) -> bool:
        xy = cylinder_xy - back_m * direction
        path = [side_primitives._wp(geometry.on_ray(xy, geometry.standoff_x), geometry.grasp_z, GRIP_CLOSE),
                side_primitives._wp(geometry.on_ray(xy, geometry.front_x), geometry.grasp_z, GRIP_CLOSE),
                side_primitives._wp(geometry.on_ray(xy, geometry.front_x), geometry.push_z, GRIP_CLOSE),
                side_primitives._wp(xy, geometry.push_z, GRIP_CLOSE)]
        arm_angles, _ = side_primitives._trace(model, path, GRIP_CLOSE)
        for q in arm_angles:
            probe.qpos[inner._qpos_adr[:5]] = q
            probe.qpos[inner._qpos_adr[5]] = _grip_rad(model, GRIP_CLOSE)
            mujoco.mj_forward(model, probe)
            for c in probe.contact[: probe.ncon]:
                if red in (c.geom1, c.geom2) and (c.geom1 in inner._arm_ids or c.geom2 in inner._arm_ids):
                    return False
        return True

    try:
        low, high = CYL_RADIUS, CYL_RADIUS + PUSH_START_SEARCH_M
        while high - low > PUSH_START_TOLERANCE_M:
            middle = 0.5 * (low + high)
            low, high = (low, middle) if clear(middle) else (middle, high)
        return high
    finally:
        model.geom_margin[red] = saved_margin


def seeded_jitter(seed: list[int]) -> np.ndarray:
    """An xy offset uniform over a disc of radius JITTER_RADIUS_M (metres), fixed by `seed`, not by the worker."""
    rng = np.random.default_rng(seed)
    angle, radius = rng.uniform(0, 2 * np.pi), JITTER_RADIUS_M * np.sqrt(rng.uniform())
    return radius * np.array([np.cos(angle), np.sin(angle)])


def jitter_offset(repetition: int, height_cm: float, friction: float, direction: str) -> np.ndarray:
    """The push-start jitter of studies 03 and 04: the seed is (repetition, height in cm x 100, friction x 100,
    direction index), the rule those studies used, so the same cell and repetition get the same jitter."""
    direction_index = list(PUSH_DIRECTIONS_RAD).index(direction)
    return seeded_jitter([repetition, int(round(height_cm * 100)), int(round(friction * 100)), direction_index])


def run_push(env: SingleRedShelf, *, direction: str, height_cm: float, friction: float,
             jitter_xy=(0.0, 0.0), speed_fraction: float = 1.0, arm_friction: float | None = None,
             record_substeps: bool = False) -> dict:
    """The red alone at the region's centre, one 5 cm push from just behind it. Returns one flat record.

    `arm_friction` sets the sliding friction of every arm geom for this run only (normally 1.0; MuJoCo uses the larger
    of a contact's two frictions).
    """
    heading = PUSH_DIRECTIONS_RAD[direction]
    height_m = height_cm / 100
    placed = env.reset(seed=PUSH_RESET_SEED, physics={"friction": friction, "density": PUSH_DENSITY_KG_M3})
    start = placed["pos"][:2].copy()
    key = (direction, height_cm)
    if key not in env._push_start_cache:
        env._push_start_cache[key] = push_start_back_distance(env, start, heading, height_m)
    back = env._push_start_cache[key]
    unit = np.array([np.cos(heading), np.sin(heading)])
    env.set_push_height(height_m)
    env.would_refuse = None
    env.push_speed_fraction = speed_fraction
    env.record_substeps = record_substeps
    env.phase = "push"
    arm_geoms = sorted(env.env._arm_ids)
    saved_friction = env.model.geom_friction[arm_geoms, 0].copy()
    if arm_friction is not None:
        env.model.geom_friction[arm_geoms, 0] = arm_friction
    try:
        _, _, _, _, info = env.step(("push", start - back * unit + np.asarray(jitter_xy), heading))
    finally:
        env.model.geom_friction[arm_geoms, 0] = saved_friction
        env.push_speed_fraction = 1.0
        env.record_substeps = False
    log = env.log
    pose = env.env.cylinder_pose(TARGET)
    tilts = [f["tilt"] for f in log.frames] or [0.0]
    moved = pose["pos"][:2] - start
    contacts = log.arm_contacts
    parts = {"jaw tip sphere": 0, "jaw (box or mesh)": 0, "gripper body": 0}
    for c in contacts:
        part = measures.contact_part(c.geom)
        parts[part] = parts.get(part, 0) + 1
    total = max(1, len(contacts))
    heights = [c.height_above_board_m for c in contacts]
    glitch = measures.is_glitch(log.max_contact_force_n, log.max_red_speed_m_s)
    max_tilt = max(tilts)
    return {
        "direction": direction, "height_cm": height_cm, "friction": friction,
        "speed_fraction": speed_fraction, "arm_friction": arm_friction if arm_friction is not None else "default",
        "r_over_mu_cm": round(100 * CYL_RADIUS / friction, 2),
        "jitter_x_mm": round(float(jitter_xy[0]) * 1000, 2), "jitter_y_mm": round(float(jitter_xy[1]) * 1000, 2),
        "start_back_mm": round(back * 1000, 1),
        "refused": info["refused"], "would_refuse": env.would_refuse[1] if env.would_refuse else None,
        "arm_touched_shelf": log.arm_shelf_touch,
        "max_tilt_deg": round(max_tilt, 2), "final_tilt_deg": round(pose["tilt"], 2),
        "max_tilt_jump_per_frame_deg": round(float(np.abs(np.diff(tilts)).max()) if len(tilts) > 1 else 0.0, 2),
        "glitch": glitch, "toppled": bool(max_tilt > TOPPLE_DEG and not glitch),
        "outcome": measures.tilt_outcome(max_tilt, glitch),
        "moved_mm": round(float(np.linalg.norm(moved)) * 1000, 1),
        "moved_along_mm": round(float(moved @ unit) * 1000, 1),
        "max_contact_force_n": round(log.max_contact_force_n, 2),
        "max_red_speed_m_s": round(log.max_red_speed_m_s, 3),
        "first_contact_geom": contacts[0].geom if contacts else None,
        "first_contact_part": measures.contact_part(contacts[0].geom) if contacts else None,
        "first_contact_height_mm": round(contacts[0].height_above_board_m * 1000, 1) if contacts else None,
        "contact_height_min_mm": round(min(heights) * 1000, 1) if heights else None,
        "contact_height_max_mm": round(max(heights) * 1000, 1) if heights else None,
        "contact_substeps": len(contacts),
        "share_jaw_tip": round(parts["jaw tip sphere"] / total, 3),
        "share_jaw_box": round(parts["jaw (box or mesh)"] / total, 3),
        "share_gripper_body": round(parts["gripper body"] / total, 3),
        "failure": info["failure"], "frames": info["frames"],
    }
