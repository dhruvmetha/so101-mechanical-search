"""Mechanical search on a shelf: find the one red cylinder among gray ones and pull it out, toppling nothing.

Spec: docs/superpowers/specs/2026-09-26-shelf-search-design.md.

`ShelfEnv` is the 30 Hz env, with `SO101Env`'s actions and observations, so a lerobot policy runs on it unchanged.
`ShelfPrimitiveEnv` wraps it so one step is a whole pick, place, push, pull or look, with the 30 Hz frames still
running underneath and every check on each one.

Four rules shape this file. The arm touching the shelf or the table ends the episode. A cylinder tilted past 20
degrees has toppled, held or not, and ends it too. Every gray stays on the board the whole time, held or not, so the
only way off the shelf is `pull`, and only the red cylinder should take it. And the observation holds images and joint angles only. Cylinder poses and the hidden label
go in `info`, because a policy on hardware gets neither.
"""

import copy

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from so101sim import SHELF_SCENE_PATH, grasp, primitives, rest, side_primitives as sp, units
from so101sim.env import CAMERAS, CONTROL_HZ, SO101Env
from so101sim.mechsearch import PARTIAL_MAX_SHARE, VISIBILITY_SIZE, VISIBLE_MIN_PIXELS, face_pixels
from so101sim.shelf_checks import WALL_STRIP, add_clearance
from so101sim.shelf_layout import CYL_HALF_HEIGHT, CYL_RADIUS, GRAY_GROUP, TARGET_NAME, ShelfLayout

TARGET = TARGET_NAME
TASK = "retrieve the red cylinder from the shelf"

MIN_SPACING = 2 * CYL_RADIUS + 0.005
# A hidden red cylinder goes in the back two thirds of the region, with a gray on each camera's sight line to it,
# this far in front of it.
HIDDEN_BACK = 1 / 3
SIGHT_GAP = (0.035, 0.05)
# A hidden red gets grays where a camera still sees it: on that camera's sight line, at the first of these distances
# that's free, up to HIDDEN_OCCLUDERS of them. From the red, the two cameras' sight lines can run within 15 degrees of
# each other, so one gray per camera at a fixed distance often lost the wrist camera's gray to the front camera's,
# and the wrist camera saw the whole red (37 of 60 hidden attempts, 2026-09-26).
OCCLUDE_GAPS = (0.035, 0.045, 0.055, 0.065, 0.075)
HIDDEN_OCCLUDERS = 3
# A partial red gets the same sight-line grays shifted sideways by this much to a random side, so they cover part
# of it. Placed at random, the red was more than half visible in 79 of 150 attempts (spec, second amendment).
# Swept 2026-09-26 over 200 seeds at 12 and 14 cylinders: 0.4-1.2 cm kept 62% and 48% of partial attempts, 0.8-1.6
# kept 64% and 57%, 1.2-2.2 kept 51% and 49%. Less shift hides the red, more shows most of it.
PARTIAL_SHIFT = (0.008, 0.016)
PLACE_LIFT = 0.0005  # cylinders are placed this far over the board, so they settle onto it
SETTLE_QVEL = 0.2
SETTLE_SECONDS = 0.5
UPRIGHT_DEG = 2.0
REGION_SLACK = 0.002
FREE_SPOT_CLEARANCE = 0.045
FREE_SPOT_GRID = 0.005
MAX_RESET_ATTEMPTS = 50
GEOM = int(mujoco.mjtObj.mjOBJ_GEOM)

MAX_PRIMITIVES = 20
PUSH_STROKE = sp.PUSH_STROKE
SETTLE_MAX_FRAMES = 60
# Positions this close outside the action box get clipped into it instead of refused.
ACTION_TOLERANCE = 0.005
# Holding means the gripper was last told to close and the joint stopped short of it. The real arm reads the same
# joint. Measured 2026-09-26: an empty close reads 10.0; held cylinders read 15.8 to 17.8 for grasps up to 8 mm off
# centre, and 12.7 for one taken off-centre in a planner run, with both jaws on it. 14 wrongly refused that pull.
HOLDING_MIN = 11.0

TOPPLE_DEG = 20.0  # just under atan(3 / 8) = 20.6, past which a 3 x 8 cm cylinder can't stand back up
SUCCESS_PAST_EDGE = 0.02
SUCCESS_FRAMES = 5


def tilt_deg(xmat):
    """Degrees between a body's z axis, which is a cylinder's axis, and vertical."""
    return float(np.degrees(np.arccos(np.clip(np.asarray(xmat).reshape(3, 3)[2, 2], -1.0, 1.0))))


class ShelfEnv(SO101Env):
    def __init__(self, scene_path=SHELF_SCENE_PATH, max_steps=2700, hidden_prob=0.5, **env_kwargs):
        super().__init__(scene_path=scene_path, **env_kwargs)
        self.max_steps = max_steps
        self.hidden_prob = float(hidden_prob)
        self.layout = ShelfLayout.from_model(self.model)
        self.region = self.layout.region()
        names = [self.model.body(b).name for b in range(self.model.nbody)]
        self.cyls = tuple(sorted((n for n in names if n.startswith("cyl_")), key=lambda n: (n != TARGET, len(n), n)))
        self.grays = tuple(n for n in self.cyls if n != TARGET)
        joints = {n: self.model.body(n).jntadr[0] for n in self.cyls}
        self._qadr = {n: self.model.jnt_qposadr[j] for n, j in joints.items()}
        self._dofadr = {n: self.model.jnt_dofadr[j] for n, j in joints.items()}
        self._rest_dofs = rest.free_dofs(self.model, self.cyls)
        self._body = {n: self.model.body(n).id for n in self.cyls}
        self._geom = {n: self.model.geom(n).id for n in self.cyls}
        self._cyl_geoms = frozenset(self._geom.values())
        self._jaw_ids = grasp.jaw_geom_ids(self.model)
        self._arm_ids = grasp.arm_geom_ids(self.model)
        # The arm touching any of these is a failure (spec, first amendment).
        self._shelf_ids = frozenset({g for g in range(self.model.ngeom) if self.model.geom(g).name.startswith("shelf_")}
                                    | {self.model.geom("table").id})
        self._home = np.asarray(self.layout.home, dtype=np.float64)
        self._seg = mujoco.Renderer(self.model, *VISIBILITY_SIZE)
        self._seg.enable_segmentation_rendering()
        self._hide_grays = mujoco.MjvOption()
        self._hide_grays.geomgroup[GRAY_GROUP] = 0
        self._steps = 0
        self._streak = 0
        self._reset_info = {}

    # ---- gym api -------------------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        # Seed np_random only. SO101Env.reset would render an observation nobody uses.
        gym.Env.reset(self, seed=seed)
        options = options or {}
        hidden = options.get("hidden")
        if hidden is None:
            hidden = bool(self.np_random.random() < self.hidden_prob)
        reason = None
        for attempt in range(1, MAX_RESET_ATTEMPTS + 1):
            self._start_pose()
            reason, labels = self._build(bool(hidden))
            if reason is None:
                break
        else:
            raise RuntimeError(f"no usable shelf in {MAX_RESET_ATTEMPTS} attempts, the last because {reason}")
        self._steps, self._streak = 0, 0
        self._reset_info = {"hidden": bool(hidden), **labels, "reset_attempts": attempt}
        return self._obs(), self._info(False, None)

    def step(self, action):
        reward, terminated, truncated, info = self.advance(action)
        return self._obs(), reward, terminated, truncated, info

    def advance(self, action):
        """One 30 Hz frame of physics and checks, without rendering.

        The primitive env calls this every frame and renders once at the end. Checks run in the spec's order: the
        arm touching the shelf, a topple, a cylinder off the board, then success, then the time limit.
        """
        self._apply(action)
        self._steps += 1
        failure = self._failure()
        success = False
        if failure is None:
            self._streak = self._streak + 1 if self._target_retrieved() else 0
            success = self._streak >= SUCCESS_FRAMES
        terminated = success or failure is not None
        truncated = not terminated and self.max_steps is not None and self._steps >= self.max_steps
        return float(success), terminated, truncated, self._info(success, failure)

    def close(self):
        self._seg.close()
        super().close()

    # ---- task state ----------------------------------------------------------------------------

    def cylinder_pose(self, name):
        adr = self._qadr[name]
        return {"pos": self.data.qpos[adr:adr + 3].copy(), "tilt": self.tilt(name)}

    def tilt(self, name):
        return tilt_deg(self.data.xmat[self._body[name]])

    def held(self, name):
        """Both jaws on the cylinder."""
        return grasp.is_grasped(self.model, self.data, self._geom[name], self._jaw_ids)

    def arm_hit(self):
        """Name of the shelf panel, leg or table an arm geom is touching, or None."""
        for c in self.data.contact[: self.data.ncon]:
            if c.geom1 in self._arm_ids and c.geom2 in self._shelf_ids:
                return self.model.geom(c.geom2).name
            if c.geom2 in self._arm_ids and c.geom1 in self._shelf_ids:
                return self.model.geom(c.geom1).name
        return None

    def settled(self):
        return bool(np.abs(self.data.qvel).max() <= SETTLE_QVEL)

    def at_rest(self):
        """Every cylinder still by the rule both task envs share (so101sim.rest). Stricter than settled()."""
        return rest.still(self.data, self._rest_dofs)

    def free_spot(self):
        """The first point on a 5 mm grid over the region at least FREE_SPOT_CLEARANCE from every cylinder
        standing on the board, or None."""
        x_lo, x_hi, y_lo, y_hi = self.region
        xs = np.arange(x_lo, x_hi + 1e-9, FREE_SPOT_GRID)
        ys = np.arange(y_lo, y_hi + 1e-9, FREE_SPOT_GRID)
        grid = np.stack(np.meshgrid(xs, ys, indexing="ij"), -1).reshape(-1, 2)
        centres = np.array([self._xy(n) for n in self.cyls if self.layout.on_board(self._xy(n))]).reshape(-1, 2)
        if len(centres) == 0:
            return grid[len(grid) // 2].copy()
        gaps = np.linalg.norm(grid[:, None, :] - centres[None, :, :], axis=-1).min(axis=1)
        free = np.flatnonzero(gaps >= FREE_SPOT_CLEARANCE)
        return grid[free[0]].copy() if len(free) else None

    def target_pixels(self, hide_grays=False):
        """Face pixels of the red cylinder in each camera, at VISIBILITY_SIZE. With `hide_grays`, what each camera
        would see of it if nothing covered it."""
        counts = {}
        for name, camera in CAMERAS.items():
            self._seg.update_scene(self.data, camera=camera, scene_option=self._hide_grays if hide_grays else None)
            seg = self._seg.render()
            counts[name] = face_pixels((seg[..., 0] == self._geom[TARGET]) & (seg[..., 1] == GEOM))
        return counts

    def task_info(self):
        """`info` as it stands, for a primitive that was refused before any frame ran."""
        return self._info(False, None)

    # ---- internals -----------------------------------------------------------------------------

    def _xy(self, name):
        adr = self._qadr[name]
        return self.data.qpos[adr:adr + 2].copy()

    def _start_pose(self):
        mujoco.mj_resetData(self.model, self.data)  # also parks every cylinder back off stage
        q = units.to_rad(self._home, self._gripper_range)
        self.data.qpos[self._qpos_adr] = q
        self.data.ctrl[:] = q
        mujoco.mj_forward(self.model, self.data)

    def _in_region(self, xy, slack=0.0):
        x_lo, x_hi, y_lo, y_hi = self.region
        return bool(x_lo - slack <= xy[0] <= x_hi + slack and y_lo - slack <= xy[1] <= y_hi + slack)

    def _free(self, xy, placed):
        return self._in_region(xy) and all(np.linalg.norm(xy - p) >= MIN_SPACING for p in placed)

    def _sample_xy(self, placed, x_lo=None, wall=None, tries=200):
        """A random free xy in the region, optionally behind `x_lo` and `wall` or more from both side walls."""
        lo_x, hi_x, lo_y, hi_y = self.region
        lo_x = lo_x if x_lo is None else x_lo
        if wall is not None:
            lo_y, hi_y = max(lo_y, -self.layout.half_width + wall), min(hi_y, self.layout.half_width - wall)
        for _ in range(tries):
            xy = np.array([self.np_random.uniform(lo_x, hi_x), self.np_random.uniform(lo_y, hi_y)])
            if self._free(xy, placed):
                return xy
        return None

    def _place(self, name, xy):
        adr, dof = self._qadr[name], self._dofadr[name]
        self.data.qpos[adr:adr + 3] = [xy[0], xy[1], self.layout.board_z + CYL_HALF_HEIGHT + PLACE_LIFT]
        self.data.qpos[adr + 3:adr + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[dof:dof + 6] = 0.0

    def _toward(self, camera, xy):
        """Unit vector in the board plane from `xy` toward `camera`."""
        eye = self.data.cam_xpos[self.model.camera(camera).id][:2]
        return (eye - xy) / np.linalg.norm(eye - xy)

    def _settle(self):
        """Step until still or SETTLE_SECONDS pass. Returns True if a cylinder touched the arm on the way."""
        mujoco.mj_forward(self.model, self.data)
        touched = False
        for _ in range(int(round(SETTLE_SECONDS * CONTROL_HZ))):
            mujoco.mj_step(self.model, self.data, nstep=self._n_substeps)
            touched = touched or grasp.touching(self.data, self._cyl_geoms, self._arm_ids)
            if self.settled():
                break
        return touched

    def _build(self, hidden):
        """Stand every cylinder on the board. Returns (why the shelf is unusable or None, labels for info)."""
        grays = list(self.grays)
        self.np_random.shuffle(grays)
        placed = []

        def put(name, xy):
            self._place(name, xy)
            placed.append(np.asarray(xy, dtype=np.float64))

        # The red goes first, in the back two thirds and never in the wall strip, where the gripper can't fit beside
        # it (spec, first amendment). Then grays in front of it: for a hidden start, wherever a camera still sees it;
        # for a partial one, one per camera shifted sideways so it covers part of the red (spec, second amendment).
        x_lo, x_hi, _, _ = self.region
        red = self._sample_xy(placed, x_lo=x_lo + HIDDEN_BACK * (x_hi - x_lo), wall=WALL_STRIP)
        if red is None:
            return "no room for the red cylinder", {}
        put(TARGET, red)
        if hidden:
            for _ in range(HIDDEN_OCCLUDERS):
                mujoco.mj_forward(self.model, self.data)
                pixels = self.target_pixels()
                seen = [c for c in ("wrist", "front") if pixels[c]]  # the wrist camera is nearer and sees more
                if not seen or not grays:
                    break
                toward = self._toward(CAMERAS[seen[0]], red)
                spot = next((red + gap * toward for gap in OCCLUDE_GAPS if self._free(red + gap * toward, placed)),
                            None)
                if spot is None:
                    break
                put(grays.pop(), spot)
        else:
            for camera in CAMERAS.values():
                toward = self._toward(camera, red)
                side = np.array([-toward[1], toward[0]])
                shift = self.np_random.choice([-1.0, 1.0]) * self.np_random.uniform(*PARTIAL_SHIFT)
                xy = red + self.np_random.uniform(*SIGHT_GAP) * toward + shift * side
                if grays and self._free(xy, placed):
                    put(grays.pop(), xy)
        for name in grays:
            xy = self._sample_xy(placed)
            if xy is None:
                return "no room left on the shelf", {}
            put(name, xy)
        touched = [self._settle()]

        # The episode starts only on a shelf that has stopped. _settle's check is looser on purpose.
        def frame():
            mujoco.mj_step(self.model, self.data, nstep=self._n_substeps)
            touched[0] = touched[0] or grasp.touching(self.data, self._cyl_geoms, self._arm_ids)

        resting = rest.come_to_rest(frame, self.data, self._rest_dofs)
        mujoco.mj_forward(self.model, self.data)
        return self._rejection(hidden, touched[0], resting)

    def _rejection(self, hidden, touched, resting):
        if touched:
            return "a cylinder touched the arm while settling", {}
        for name in self.cyls:
            if self.tilt(name) > UPRIGHT_DEG:
                return f"{name} tilted {self.tilt(name):.1f} deg while settling", {}
            if not self._in_region(self._xy(name), REGION_SLACK):
                return f"{name} slid out of the region", {}
        if not resting:
            return "the shelf was still moving after settling", {}
        if self.free_spot() is None:
            return "no free spot left on the board", {}
        pixels = self.target_pixels()
        if hidden:
            if any(pixels.values()):
                return f"the hidden red cylinder shows {pixels}", {}
            return None, {"target_pixels": pixels, "target_visible_share": {c: 0.0 for c in pixels}}
        if max(pixels.values()) < VISIBLE_MIN_PIXELS:
            return f"the red cylinder shows only {pixels}", {}
        unoccluded = self.target_pixels(hide_grays=True)
        shares = {c: pixels[c] / unoccluded[c] if unoccluded[c] else 0.0 for c in pixels}
        if max(shares.values()) > PARTIAL_MAX_SHARE:
            return f"the red cylinder shows too much of itself {shares}", {}
        return None, {"target_pixels": pixels, "target_visible_share": shares}

    def _failure(self):
        hit = self.arm_hit()
        if hit is not None:
            return f"hit the shelf: {hit}"
        for name in self.cyls:
            if self.tilt(name) > TOPPLE_DEG:
                return f"toppled: {name}"
        for name in self.cyls:
            if not self.layout.on_board(self._xy(name)) and (name != TARGET or not self.held(name)):
                return f"left the shelf: {name}"
        return None

    def _target_retrieved(self):
        """Both jaws on the red cylinder, its centre SUCCESS_PAST_EDGE in front of the front edge, and upright."""
        if not self.held(TARGET):
            return False
        return bool(self._xy(TARGET)[0] <= self.layout.front_x - SUCCESS_PAST_EDGE
                    and self.tilt(TARGET) < TOPPLE_DEG)

    def _info(self, success, failure):
        adr = self._qadr[TARGET]
        return {
            "target": {"pos": self.data.qpos[adr:adr + 3].copy(), "quat": self.data.qpos[adr + 3:adr + 7].copy()},
            "cylinders": {n: self.cylinder_pose(n) for n in self.cyls},
            **self._reset_info,
            "n_on_board": sum(self.layout.on_board(self._xy(n)) for n in self.cyls),
            "is_success": bool(success),
            "failure": failure,
            "steps": self._steps,
            "task": TASK,
        }


class _Refused(Exception):
    """A primitive the env won't run."""


class ShelfPrimitiveEnv(gym.Env):
    """One step is one whole pick, place, push, pull or look on the shelf.

    The action is {"kind", "xy", "angle"}, or the tuple (kind, xy, angle). `xy` is a point on the board: the
    cylinder to pick, the spot to place on, where a push starts, or the ray to look down. `angle` is the push
    heading and is ignored otherwise. `pull` ignores `xy` and takes the held cylinder out along the ray it came in
    on. There's no z: every cylinder stands on the same board.

    Refused, with the arm left where it is and the step still counted: `xy` outside the box, a push ending outside
    it, IK failing, pick, push or look while holding, place or pull while not holding, and any path that would put
    the arm into the shelf or the table.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": CONTROL_HZ}
    PRIMITIVES = ("pick", "place", "push", "pull", "look")

    def __init__(self, max_primitives=MAX_PRIMITIVES, push_stroke=PUSH_STROKE, on_frame=None, **env_kwargs):
        # A primitive runs to completion inside one step, so a recorder needs a hook to see the frames under it.
        self.on_frame = on_frame
        self.env = ShelfEnv(max_steps=None, **env_kwargs)
        self.model, self.data = self.env.model, self.env.data
        self.geo = sp.Geometry.from_layout(self.model, self.env.layout)
        self.open_value = sp.side_open_value(self.model)
        self.max_primitives = int(max_primitives)
        self.push_stroke = float(push_stroke)
        x_lo, x_hi, y_lo, y_hi = self.env.region
        self.observation_space = self.env.observation_space
        self.action_space = spaces.Dict({
            "kind": spaces.Discrete(len(self.PRIMITIVES)),
            "xy": spaces.Box(np.array([x_lo, y_lo], dtype=np.float32), np.array([x_hi, y_hi], dtype=np.float32)),
            "angle": spaces.Box(-np.pi, np.pi, (), dtype=np.float32),
        })
        self._n = 0
        self._last_grip = float(self.env.layout.home[5])
        self._aisle_xy = None
        # The path check runs on a copy with a clearance margin on the shelf, so a plan that only grazes it is refused
        # too. The copy never steps physics.
        self._probe_model = add_clearance(copy.deepcopy(self.model))
        self._probe = mujoco.MjData(self._probe_model)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._n = 0
        self._aisle_xy = None
        out = self.env.reset(seed=seed, options=options)
        self._last_grip = float(self.env.layout.home[5])
        return out

    def holding(self):
        return bool(self._last_grip <= primitives.GRIP_CLOSE + 1.0 and float(self.env.joint_state[5]) >= HOLDING_MIN)

    def step(self, action):
        kind, xy, angle = self._parse(action)
        self._n += 1
        refused, terminated, frames = None, False, 0
        info = self.env.task_info()
        try:
            traj = self._plan(kind, xy, angle)
        except _Refused as exc:
            refused, traj = str(exc), None
        except primitives.Unreachable as exc:
            refused, traj = f"unreachable: {exc}", None

        if traj is not None:
            traj = sp.bridge(self.model, self.geo, self.env.joint_state, traj)
            hit = self._path_hit(traj)
            if hit is not None:
                refused, traj = f"the path would hit {hit[1]} at frame {hit[0]}", None
        if traj is not None:
            for row in traj:
                terminated, info = self._frame(row)
                frames += 1
                if terminated:
                    break
            else:
                for _ in range(SETTLE_MAX_FRAMES):
                    terminated, info = self._frame(traj[-1])
                    frames += 1
                    if terminated or self.env.settled():
                        break

        reward = 1.0 if info["is_success"] else 0.0
        truncated = not terminated and self._n >= self.max_primitives
        info = {**info, "primitive": kind, "frames": frames, "refused": refused}
        return self.env._obs(), reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def _path_hit(self, traj):
        """(frame, shelf geom) where the planned path first puts an arm geom into the shelf or the table, or None.

        Kinematic only, with the cylinders where they stand, and with `shelf_checks.CLEARANCE` of margin, because the
        executed arm bounces a few millimetres around its plan. The arm touching the shelf is very bad behaviour (spec,
        first amendment), so the env refuses such a path rather than run it. The inner env still ends the episode if
        the executed motion touches anyway.
        """
        probe = self._probe
        probe.qpos[:] = self.data.qpos
        arm, shelf = self.env._arm_ids, self.env._shelf_ids
        for k, row in enumerate(traj):
            probe.qpos[self.env._qpos_adr] = units.to_rad(np.asarray(row, dtype=np.float64), self.env._gripper_range)
            mujoco.mj_forward(self._probe_model, probe)
            for c in probe.contact[: probe.ncon]:
                if c.geom1 in arm and c.geom2 in shelf:
                    return k, self.model.geom(c.geom2).name
                if c.geom2 in arm and c.geom1 in shelf:
                    return k, self.model.geom(c.geom1).name
        return None

    def close(self):
        self.env.close()

    def _frame(self, row):
        self._last_grip = float(row[5])
        _, terminated, _, info = self.env.advance(row)
        if self.on_frame is not None:
            self.on_frame(self)
        return terminated, info

    def _parse(self, action):
        if isinstance(action, dict):
            kind, xy, angle = action["kind"], action.get("xy", (0.0, 0.0)), action.get("angle", 0.0)
        else:
            kind, xy, angle = action
        if isinstance(kind, (int, np.integer)):
            kind = self.PRIMITIVES[int(kind)]
        if kind not in self.PRIMITIVES:
            raise ValueError(f"unknown primitive {kind!r}, expected one of {self.PRIMITIVES}")
        return kind, np.asarray(xy, dtype=np.float64).reshape(2), float(angle)

    def _inside(self, xy, what):
        box = self.action_space["xy"]
        clipped = np.clip(xy, box.low, box.high)
        if np.abs(xy - clipped).max() > ACTION_TOLERANCE:
            raise _Refused(f"{what} {np.round(xy, 4).tolist()} is outside the action box")
        return clipped

    def _plan(self, kind, xy, angle):
        holding = self.holding()
        if kind in ("place", "pull") and not holding:
            raise _Refused(f"{kind} needs a held cylinder and the gripper isn't holding one")
        if kind in ("pick", "push", "look") and holding:
            raise _Refused(f"{kind} with a cylinder still in the gripper; place or pull it first (holding)")
        if kind == "pull":
            return sp.pull(self.model, self.geo, self._aisle_xy)
        xy = self._inside(xy, kind)
        if kind == "pick":
            traj = sp.pick(self.model, self.geo, xy, self.open_value)
            self._aisle_xy = self.geo.on_ray(xy, self.geo.aisle_x)
            return traj
        if kind == "place":
            return sp.place(self.model, self.geo, self._aisle_xy, xy, self.open_value)
        if kind == "look":
            return sp.look(self.model, self.geo, xy)
        end = xy + self.push_stroke * np.array([np.cos(angle), np.sin(angle)])
        self._inside(end, f"a push heading {np.degrees(angle):.0f} deg ends at")
        return sp.push(self.model, self.geo, xy, angle, stroke=self.push_stroke)
