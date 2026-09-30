"""Mechanical search: find the one red cube in a bin of gray ones and lift it out.

Spec: docs/superpowers/specs/2026-09-25-mechanical-search-design.md.

`MechSearchEnv` is the 30 Hz env. Actions and observations are `SO101Env`'s, so a lerobot policy runs on
it unchanged. `MechSearchPrimitiveEnv` wraps it so one step is a whole pick, place or push, with the 30 Hz
frames still happening underneath and every task check running on each one.

Two rules shape this file. Every cube stays in the bin: once any cube's centre leaves the bin's footprint
the episode ends as a failure, so all rearranging happens inside the bin. And the observation holds images
and joint angles only. Cube poses, the buried label and the red pixel counts go in `info`, because a
policy on hardware gets none of them.
"""

import itertools

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from so101sim import SEARCH_SCENE_PATH, grasp, primitives, rest, search_layout as sl, units
from so101sim.binpick import bin_geometry, cube_names
from so101sim.env import CAMERAS, CONTROL_HZ, SO101Env

TARGET = "cube_red"
TASK = "retrieve the red cube from the bin"
CUBE_HALF = primitives.CUBE_HALF

# Reset drops cubes in waves of up to WAVE, at least WAVE_SPACING apart so none land on each other in
# mid-air, with each wave's bottoms DROP_GAP over the highest cube already in the bin.
WAVE = 4
WAVE_SPACING = 0.045
DROP_GAP = 0.01
DROP_INSET = CUBE_HALF + 0.003
# The free floor spot keeps this far from every wall, centre to wall.
WALL_INSET = 0.018
# A buried red cube is boxed in: four grays flush against its sides BOX_GAP away, and one on top. One gray
# dropped on top left the sides showing. BOX_INSET keeps the side grays inside the walls whatever the yaw.
BOX_GAP = 0.001
BOX_STEP = 2 * CUBE_HALF + BOX_GAP
BOX_INSET = BOX_STEP + CUBE_HALF * np.sqrt(2) + 0.002
# Two cubes' half-diagonals plus 2.6 mm, so a cube fits at the spot whatever its neighbours' yaw.
FREE_SPOT_CLEARANCE = 0.045
FREE_SPOT_GRID = 0.005
SETTLE_QVEL = 0.2
SETTLE_SECONDS = 1.0
# Partial starts keep only about 12% of dropped piles (measured 2026-09-26), so 50 attempts would give up about
# once in 600 resets. At 200 it is about once in 10^11, and a partial reset still averages 2.7 s.
MAX_RESET_ATTEMPTS = 200

# Visibility is counted at the default observation size whatever size the env renders, so asking for
# small images doesn't change which piles count as buried.
VISIBILITY_SIZE = (480, 640)
VISIBLE_MIN_PIXELS = 100
# A partial start may show at most this share of what each camera would see of the red cube with every gray cube
# hidden. A fully visible red cube is a plain pick, which tests nothing about search.
PARTIAL_MAX_SHARE = 0.5
# A red pixel counts only if the whole (2 * SEAM_RADIUS + 1) square around it is red. That drops the 1 to 4
# pixel seams between touching cubes, which by eye leave a cube buried, and keeps any visible face.
SEAM_RADIUS = 2
GEOM = int(mujoco.mjtObj.mjOBJ_GEOM)

SUCCESS_CLEARANCE = 0.01
SUCCESS_FRAMES = 5
# How far the red cube's bottom must rise above the resting height of a top-layer cube. A primitive pick holds
# it about 2.8 cm up (measured 2026-09-25: lowest corner at floor + 8.8 cm), so 2 cm leaves margin.
SUCCESS_LIFT = 0.02

MAX_PRIMITIVES = 15
PUSH_STROKE = 0.05
SETTLE_MAX_FRAMES = 60
# Positions this close outside the action box get clipped into it instead of refused. A cube resting on the
# floor sinks a fraction of a millimetre into it, so its reported centre sits just under the box's lowest z.
ACTION_TOLERANCE = 0.005


def face_pixels(mask, r=SEAM_RADIUS):
    """Pixels of `mask` whose whole (2r+1) square is in `mask`, with nothing off the image counting."""
    h, w = mask.shape
    padded = np.zeros((h + 2 * r, w + 2 * r), dtype=bool)
    padded[r:r + h, r:r + w] = mask
    core = np.ones((h, w), dtype=bool)
    for dy in range(2 * r + 1):
        for dx in range(2 * r + 1):
            core &= padded[dy:dy + h, dx:dx + w]
    return int(core.sum())


class MechSearchEnv(SO101Env):
    def __init__(self, scene_path=SEARCH_SCENE_PATH, max_steps=1800, buried_prob=0.5, **env_kwargs):
        super().__init__(scene_path=scene_path, **env_kwargs)
        self.max_steps = max_steps
        self.buried_prob = float(buried_prob)
        self.layout = sl.Layout.from_model(self.model)
        self.bin = bin_geometry(self.model)
        self.cubes = cube_names(self.model)
        self.grays = tuple(n for n in self.cubes if n != TARGET)
        self._qadr = {n: self.model.jnt_qposadr[self.model.body(n).jntadr[0]] for n in self.cubes}
        self._dofadr = {n: self.model.jnt_dofadr[self.model.body(n).jntadr[0]] for n in self.cubes}
        self._rest_dofs = rest.free_dofs(self.model, self.cubes)
        self._body = {n: self.model.body(n).id for n in self.cubes}
        self._geom = {n: self.model.geom(n).id for n in self.cubes}
        self._cube_geoms = frozenset(self._geom.values())
        self._jaw_ids = grasp.jaw_geom_ids(self.model)
        self._arm_ids = grasp.arm_geom_ids(self.model)
        self._home = np.asarray(self.layout.home, dtype=np.float64)
        # The red cube's lowest corner must clear the rim by SUCCESS_CLEARANCE and rise SUCCESS_LIFT above where a
        # top-layer cube rests. With 5 cm walls, rim + 1 cm is exactly that resting height, so the rim alone let a
        # grasp and a 2 mm twitch count as retrieved.
        self._success_z = max(self.bin["rim_z"] + SUCCESS_CLEARANCE,
                              self.bin["floor_z"] + (sl.PILE_LAYERS - 1) * sl.LAYER + SUCCESS_LIFT)
        self._seg = mujoco.Renderer(self.model, *VISIBILITY_SIZE)
        self._seg.enable_segmentation_rendering()
        self._hide_grays = mujoco.MjvOption()
        self._hide_grays.geomgroup[sl.GRAY_GROUP] = 0
        self._dropped = []
        self._steps = 0
        self._streak = 0
        self._reset_info = {}

    # ---- gym api -------------------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        # Seed np_random only. SO101Env.reset would also render an observation nobody uses.
        gym.Env.reset(self, seed=seed)
        options = options or {}
        buried = options.get("buried")
        if buried is None:
            buried = bool(self.np_random.random() < self.buried_prob)
        reason = None
        for attempt in range(1, MAX_RESET_ATTEMPTS + 1):
            self._start_pose()
            reason, pixels = self._build_pile(bool(buried))
            if reason is None:
                break
        else:
            raise RuntimeError(f"no usable pile in {MAX_RESET_ATTEMPTS} attempts, the last because {reason}")
        self._steps, self._streak = 0, 0
        self._reset_info = {"buried": bool(buried), **pixels, "reset_attempts": attempt}
        return self._obs(), self._info(False, None)

    def step(self, action):
        reward, terminated, truncated, info = self.advance(action)
        return self._obs(), reward, terminated, truncated, info

    def advance(self, action):
        """One 30 Hz frame of physics and task checks, without rendering.

        The primitive env calls this for every frame and renders once at the end. Checks run in the spec's
        order: a cube out of the bin, then success, then the time limit.
        """
        self._apply(action)
        self._steps += 1
        failure = self._left_bin()
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

    def cube_pose(self, name):
        """Centre position, and yaw of the cube's x axis about world z."""
        adr = self._qadr[name]
        mat = self.data.xmat[self._body[name]].reshape(3, 3)
        return {"pos": self.data.qpos[adr:adr + 3].copy(), "yaw": float(np.arctan2(mat[1, 0], mat[0, 0]))}

    def settled(self):
        return bool(np.abs(self.data.qvel).max() <= SETTLE_QVEL)

    def at_rest(self):
        """Every cube still by the rule both task envs share (so101sim.rest). Stricter than settled()."""
        return rest.still(self.data, self._rest_dofs)

    def free_spot(self):
        """The first grid point on the floor where a cube fits, or None.

        Conservative: WALL_INSET from every wall and FREE_SPOT_CLEARANCE from the centre of every
        bottom-layer cube. A cube on a higher layer can still overhang it.
        """
        reach = self.bin["half"] - WALL_INSET
        n = int(np.floor(reach / FREE_SPOT_GRID + 1e-9))
        offsets = np.arange(-n, n + 1) * FREE_SPOT_GRID
        grid = self.bin["center"] + np.stack(np.meshgrid(offsets, offsets, indexing="ij"), -1).reshape(-1, 2)
        layer_top = self.bin["floor_z"] + sl.LAYER
        bottom = np.array([self.data.qpos[self._qadr[c]:self._qadr[c] + 2] for c in self.cubes
                           if self.data.qpos[self._qadr[c] + 2] < layer_top]).reshape(-1, 2)
        if len(bottom) == 0:
            return grid[len(grid) // 2].copy()
        gaps = np.linalg.norm(grid[:, None, :] - bottom[None, :, :], axis=-1).min(axis=1)
        free = np.flatnonzero(gaps >= FREE_SPOT_CLEARANCE)
        return grid[free[0]].copy() if len(free) else None

    def target_pixels(self, hide_grays=False):
        """Face pixels of the red cube in each camera, from a segmentation render at VISIBILITY_SIZE.

        With `hide_grays`, the gray cubes are left out of the render, which gives what each camera would see of the
        red cube if nothing covered it.
        """
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

    def _start_pose(self):
        mujoco.mj_resetData(self.model, self.data)
        q = units.to_rad(self._home, self._gripper_range)
        self.data.qpos[self._qpos_adr] = q
        self.data.ctrl[:] = q
        mujoco.mj_forward(self.model, self.data)

    def _left_bin(self):
        for name in self.cubes:
            if self._outside(name):
                return f"cube left the bin: {name}"
        return None

    def _lowest_corner(self, name):
        """The z of a cube's lowest corner, whatever way up it sits."""
        mat = self.data.xmat[self._body[name]].reshape(3, 3)
        return float(self.data.qpos[self._qadr[name] + 2] - CUBE_HALF * np.abs(mat[2]).sum())

    def _target_retrieved(self):
        """Both jaws on the red cube, with its lowest corner over the rim and lifted clear of the top layer."""
        if not grasp.is_grasped(self.model, self.data, self._geom[TARGET], self._jaw_ids):
            return False
        return self._lowest_corner(TARGET) >= self._success_z

    def _outside(self, name):
        adr = self._qadr[name]
        return bool(np.abs(self.data.qpos[adr:adr + 2] - self.bin["center"]).max() > self.bin["half"])

    def _sample_xy(self, taken, inset, tries=200):
        """A random xy inside the bin, `inset` from the walls and WAVE_SPACING from every point in `taken`."""
        reach = self.bin["half"] - inset
        for _ in range(tries):
            xy = self.bin["center"] + self.np_random.uniform(-reach, reach, 2)
            if all(np.linalg.norm(xy - t) >= WAVE_SPACING for t in taken):
                return xy
        return None

    def _place(self, name, xy, z, yaw=None):
        if yaw is None:
            yaw = self.np_random.uniform(-np.pi, np.pi)
        adr, dof = self._qadr[name], self._dofadr[name]
        self.data.qpos[adr:adr + 3] = [xy[0], xy[1], z]
        self.data.qpos[adr + 3:adr + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        self.data.qvel[dof:dof + 6] = 0.0

    def _drop_z(self):
        """Centre height for the next wave: bottoms DROP_GAP over the highest cube in the bin so far."""
        tops = [self.data.qpos[self._qadr[n] + 2] + CUBE_HALF for n in self._dropped]
        return max(tops, default=self.bin["floor_z"]) + DROP_GAP + CUBE_HALF

    def _settle(self):
        """Step until still or SETTLE_SECONDS pass. Returns True if a cube touched the arm on the way."""
        touched = False
        for _ in range(int(round(SETTLE_SECONDS * CONTROL_HZ))):
            mujoco.mj_step(self.model, self.data, nstep=self._n_substeps)
            touched = touched or grasp.touching(self.data, self._cube_geoms, self._arm_ids)
            if self.settled():
                break
        return touched

    def _build_pile(self, buried):
        """Drop every cube into the bin in waves. Returns (why the pile is unusable or None, red pixels)."""
        grays = list(self.grays)
        self.np_random.shuffle(grays)
        self._dropped = []
        touched = False
        if buried:
            floor = self.bin["floor_z"] + CUBE_HALF
            red_xy = self._sample_xy([], BOX_INSET)
            yaw = self.np_random.uniform(-np.pi, np.pi)
            axes = (np.array([np.cos(yaw), np.sin(yaw)]), np.array([-np.sin(yaw), np.cos(yaw)]))
            self._place(TARGET, red_xy, floor, yaw)
            for gray, (axis, sign) in zip(grays, itertools.product(axes, (1, -1))):
                self._place(gray, red_xy + sign * BOX_STEP * axis, floor, yaw)
            self._place(grays[4], red_xy, floor + BOX_STEP, yaw)
            self._dropped += [TARGET, *grays[:5]]
            touched = self._settle()
            queue = grays[5:]
        else:
            # The red cube drops in a middle wave, never the first or the last, so later cubes land on and around it.
            queue = list(grays)
            queue.insert(int(self.np_random.integers(WAVE, max(WAVE + 1, len(grays) - WAVE + 1))), TARGET)
        while queue:
            names, queue = queue[:WAVE], queue[WAVE:]
            wave, spots = [], []
            for name in names:
                xy = self._sample_xy(spots, DROP_INSET)
                if xy is None:
                    return "no room left to drop a wave", None
                spots.append(xy)
                wave.append((name, xy))
            z = self._drop_z()
            for name, xy in wave:
                self._place(name, xy, z)
                self._dropped.append(name)
            touched = self._settle() or touched
        # The episode starts only on a pile that has stopped. The settle between waves is looser on purpose.
        touch = [touched]

        def frame():
            mujoco.mj_step(self.model, self.data, nstep=self._n_substeps)
            touch[0] = touch[0] or grasp.touching(self.data, self._cube_geoms, self._arm_ids)

        resting = rest.come_to_rest(frame, self.data, self._rest_dofs)
        mujoco.mj_forward(self.model, self.data)
        return self._rejection(buried, touch[0], resting)

    def _rejection(self, buried, touched, resting):
        if touched:
            return "a cube touched the arm while settling", None
        outside = [n for n in self.cubes if self._outside(n)]
        if outside:
            return f"{outside[0]} landed outside the bin", None
        ceiling = self.bin["floor_z"] + (sl.PILE_LAYERS - 1) * sl.LAYER + CUBE_HALF + 0.01
        high = [n for n in self.cubes if self.data.qpos[self._qadr[n] + 2] > ceiling]
        if high:
            return f"{high[0]} sits above the third layer", None
        if not resting:
            return "the pile was still moving after the last wave", None
        if self.free_spot() is None:
            return "no free spot left on the floor", None
        pixels = self.target_pixels()
        if buried and any(pixels.values()):
            return f"the buried red cube shows {pixels}", None
        shares = {cam: 0.0 for cam in pixels}
        if not buried:
            if max(pixels.values()) < VISIBLE_MIN_PIXELS:
                return f"the red cube shows only {pixels}", None
            unoccluded = self.target_pixels(hide_grays=True)
            shares = {cam: pixels[cam] / unoccluded[cam] if unoccluded[cam] else 0.0 for cam in pixels}
            if max(shares.values()) > PARTIAL_MAX_SHARE:
                return f"the red cube is too visible: {shares}", None
        return None, {"target_pixels": pixels, "target_visible_share": shares}

    def _info(self, success, failure):
        adr = self._qadr[TARGET]
        return {
            "target": {"pos": self.data.qpos[adr:adr + 3].copy(), "quat": self.data.qpos[adr + 3:adr + 7].copy()},
            "cubes": {n: self.cube_pose(n) for n in self.cubes},
            **self._reset_info,
            "n_in_bin": sum(not self._outside(n) for n in self.cubes),
            "is_success": bool(success),
            "failure": failure,
            "steps": self._steps,
            "task": TASK,
        }


class _Refused(Exception):
    """A primitive the env won't run: outside the action box, or a push that would leave the bin."""


class MechSearchPrimitiveEnv(gym.Env):
    """One step is one whole pick, place or push inside the bin.

    The action is {"kind", "xyz", "angle"}, or the tuple (kind, xyz, angle). The dict may also carry
    "flip", 0 or 1, to choose the wrist roll (see `primitives._flipped`). `xyz` is the jaw midpoint:
    the grasp point for pick, the held cube's centre at release for place, and the closed jaw tips' height
    along a push. z is in the action because one xy can hold a cube on any of three layers. The angle is
    jaw yaw for pick, the held cube's yaw for place, and the heading for push. Every move across the bin
    happens at `travel_z`.

    Anything outside the action box, and any push that would end outside the bin, is refused. So is
    anything IK can't reach. The arm doesn't move, the step still counts, and `info["refused"]` says why.
    `info["failure"]` is only for things that end the episode.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": CONTROL_HZ}
    PRIMITIVES = ("pick", "place", "push")

    def __init__(self, max_primitives=MAX_PRIMITIVES, push_stroke=PUSH_STROKE, on_frame=None, **env_kwargs):
        # A primitive runs to completion inside one step, so a recorder needs a hook to see the 30 Hz
        # frames underneath it.
        self.on_frame = on_frame
        self.env = MechSearchEnv(max_steps=None, **env_kwargs)
        self.model, self.data = self.env.model, self.env.data
        self.bin = self.env.bin
        self.max_primitives = int(max_primitives)
        self.push_stroke = float(push_stroke)
        self.travel_z = sl.travel_z(self.bin["floor_z"], self.bin["rim_z"])

        self._inset = self.bin["half"] - CUBE_HALF
        c, f = self.bin["center"], self.bin["floor_z"]
        low = np.array([c[0] - self._inset, c[1] - self._inset, f + CUBE_HALF], dtype=np.float32)
        high = np.array([c[0] + self._inset, c[1] + self._inset,
                         f + CUBE_HALF + (sl.PILE_LAYERS - 1) * sl.LAYER], dtype=np.float32)
        self.observation_space = self.env.observation_space
        self.action_space = spaces.Dict({
            "kind": spaces.Discrete(len(self.PRIMITIVES)),
            "xyz": spaces.Box(low, high, dtype=np.float32),
            "angle": spaces.Box(-np.pi, np.pi, (), dtype=np.float32),
        })
        self._n = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._n = 0
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        kind, xyz, angle, flip = self._parse(action)
        self._n += 1
        refused, terminated, frames = None, False, 0
        info = self.env.task_info()
        try:
            traj = self._plan(kind, xyz, angle, flip)
        except _Refused as exc:
            refused, traj = str(exc), None
        except primitives.Unreachable as exc:
            refused, traj = f"unreachable: {exc}", None

        if traj is not None:
            traj = primitives.bridge(self.model, self.env.joint_state, traj, hover_z=self.travel_z)
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

    def close(self):
        self.env.close()

    def _frame(self, row):
        _, terminated, _, info = self.env.advance(row)
        if self.on_frame is not None:
            self.on_frame(self)
        return terminated, info

    def _parse(self, action):
        flip = None
        if isinstance(action, dict):
            kind, xyz, angle = action["kind"], action["xyz"], action.get("angle", 0.0)
            flip = action.get("flip")
        else:
            kind, xyz, angle = action
        if isinstance(kind, (int, np.integer)):
            kind = self.PRIMITIVES[int(kind)]
        if kind not in self.PRIMITIVES:
            raise ValueError(f"unknown primitive {kind!r}, expected one of {self.PRIMITIVES}")
        return kind, np.asarray(xyz, dtype=np.float64).reshape(3), float(angle), flip

    def _plan(self, kind, xyz, angle, flip=None):
        box = self.action_space["xyz"]
        clipped = np.clip(xyz, box.low, box.high)
        if np.abs(xyz - clipped).max() > ACTION_TOLERANCE:
            raise _Refused(f"{np.round(xyz, 4).tolist()} is outside the action box")
        xy, z = clipped[:2], float(clipped[2])
        if kind == "pick":
            return primitives.pick(self.model, xy, angle, hover_z=self.travel_z, z=z, flip=flip)
        if kind == "place":
            return primitives.place(self.model, xy, angle, hover_z=self.travel_z, release_z=z, flip=flip)
        end = xy + self.push_stroke * np.array([np.cos(angle), np.sin(angle)])
        if np.abs(end - self.bin["center"]).max() > self._inset + 1e-9:
            raise _Refused(f"a push from {np.round(xy, 4).tolist()} heading {np.degrees(angle):.0f} deg "
                           "would end outside the bin")
        return primitives.push(self.model, xy, angle, stroke=self.push_stroke, hover_z=self.travel_z, push_z=z,
                               flip=flip)
