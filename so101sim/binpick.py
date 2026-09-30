"""Bin picking with push, pick and place primitives.

`BinPickEnv` wraps `SO101Env`. One `step` runs a whole primitive, so an episode is a handful of decisions
rather than hundreds of joint commands. The joint commands still happen underneath at 30 Hz, so anything
recorded here is also a demonstration in lerobot's units.

Object poses live in `info`, never in the observation. A policy sees images and proprioception, the same
as it would on hardware. The scripted policy in `scripts/binpick_demo.py` reads `info` because it is a
planner with access to the simulator, not a candidate for deployment.

Success thresholds follow RoboMME (arXiv:2603.04639) where they apply, so numbers from here are
comparable to something published: a lift counts at z > 0.05 with the object grasped, and the scene
counts as settled at max|qvel| <= 0.2.
"""

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from so101sim import SCENE_PATH, grasp, primitives
from so101sim.env import CONTROL_HZ, SO101Env

# Cubes spawn in an annulus sector, because that is the shape of the region where a pick actually works.
# Measured over every cube orientation, checking both the grasp at table height and the lift to hover:
# radius 0.126 to 0.286 from the base is reachable, and the corners of any enclosing rectangle are not.
# The angular window stops short of the bin.
SPAWN_RADIUS = (0.18, 0.26)
SPAWN_ANGLE = (np.radians(-50.0), np.radians(25.0))

# Two cubes with flush faces cannot be grasped, because the jaw has nowhere to go. That is the whole
# reason push exists here, so reset creates such a pair on purpose rather than hoping randomness does.
DEFAULT_CONTACT_PAIRS = 1

SETTLE_QVEL = 0.2
SETTLE_MAX_STEPS = 60
LIFT_Z = 0.05

# How high above the bin floor a cube still counts as in the bin. Generous on purpose, so a cube resting
# on two others still counts. Cubes carried above this are excluded by the grasp check, not by height.
STACK_CEILING = 0.10


def bin_geometry(model):
    """Bin extent read off the model, so these numbers cannot drift from objects.xml."""
    body = model.body("bin")
    wall = model.geom("bin_wall_px")
    floor = model.geom("bin_floor")
    return {
        "center": np.asarray(body.pos[:2], dtype=np.float64).copy(),
        "half": float(wall.pos[0] - wall.size[0]),
        "rim_z": float(body.pos[2] + wall.pos[2] + wall.size[2]),
        "floor_z": float(body.pos[2] + floor.pos[2] + floor.size[2]),
    }


def cube_names(model):
    """Every free-floating body called cube_*, in model order."""
    names = []
    for i in range(model.nbody):
        name = model.body(i).name
        if name.startswith("cube_") and model.body(i).jntnum == 1:
            if model.jnt_type[model.body(i).jntadr[0]] == mujoco.mjtJoint.mjJNT_FREE:
                names.append(name)
    return tuple(names)


class BinPickEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": CONTROL_HZ}

    PRIMITIVES = ("pick", "place", "push")

    def __init__(self, scene_path=SCENE_PATH, max_primitives=12, contact_pairs=DEFAULT_CONTACT_PAIRS,
                 on_frame=None, **env_kwargs):
        # A primitive runs to completion inside one step, so a recorder needs a hook to see the 30 Hz
        # frames underneath it.
        self.on_frame = on_frame
        self.env = SO101Env(scene_path=scene_path, **env_kwargs)
        self.model = self.env.model
        self.data = self.env.data
        self.max_primitives = int(max_primitives)
        self.contact_pairs = int(contact_pairs)

        self.bin = bin_geometry(self.model)
        self.cubes = cube_names(self.model)
        self._cube_qadr = {n: self.model.jnt_qposadr[self.model.body(n).jntadr[0]] for n in self.cubes}
        self._cube_geom = {n: self.model.geom(n).id for n in self.cubes}
        self._jaw_geoms = grasp.jaw_geom_ids(self.model)

        self.observation_space = self.env.observation_space
        self.action_space = spaces.Dict({
            "kind": spaces.Discrete(len(self.PRIMITIVES)),
            "xy": spaces.Box(-0.5, 0.5, (2,), dtype=np.float32),
            "angle": spaces.Box(-np.pi, np.pi, (), dtype=np.float32),
        })

        self._n_steps = 0
        self._in_bin = set()

    # ---- gym api -------------------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.env.reset(seed=seed)
        options = options or {}
        pairs = int(options.get("contact_pairs", self.contact_pairs))
        self._scatter(pairs)
        mujoco.mj_forward(self.model, self.data)
        self._n_steps = 0
        self._in_bin = self._cubes_in_bin()
        return self.env._obs(), self._info()

    def step(self, action):
        """Run one primitive to completion. `action` is a dict, or `(kind, xy, angle)`."""
        kind, xy, angle = self._parse(action)
        self._n_steps += 1
        before = set(self._in_bin)
        failure = None
        frames = 0

        try:
            traj = self._plan(kind, xy, angle)
        except primitives.Unreachable as exc:
            failure = f"unreachable: {exc}"
            traj = None

        if traj is not None:
            traj = primitives.bridge(self.model, self.env.joint_state, traj)
            for row in traj:
                self._drive(row)
            frames = len(traj)
            frames += self._settle(traj[-1])

        self._in_bin = self._cubes_in_bin()
        gained = len(self._in_bin - before)
        terminated = len(self._in_bin) == len(self.cubes)
        truncated = self._n_steps >= self.max_primitives

        info = self._info()
        info["primitive"] = kind
        info["frames"] = frames
        info["failure"] = failure
        return self.env._obs(), float(gained), terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    # ---- task state ----------------------------------------------------------------------------

    def cube_pose(self, name):
        """(x, y, yaw) on the table plane, plus z, for one cube."""
        adr = self._cube_qadr[name]
        pos = self.data.qpos[adr:adr + 3].copy()
        quat = self.data.qpos[adr + 3:adr + 7].copy()
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, quat)
        x_axis = mat.reshape(3, 3)[:, 0]
        yaw = float(np.arctan2(x_axis[1], x_axis[0]))
        return {"pos": pos, "yaw": yaw, "in_bin": name in self._in_bin}

    def is_grasped(self, name):
        """True when both jaws are touching this cube."""
        return grasp.is_grasped(self.model, self.data, self._cube_geom[name], self._jaw_geoms)

    def is_lifted(self, name):
        """RoboMME's pick predicate: high enough, and actually held."""
        return bool(self.cube_pose(name)["pos"][2] > LIFT_Z and self.is_grasped(name))

    def settled(self):
        return bool(np.abs(self.data.qvel).max() <= SETTLE_QVEL)

    def graspable(self, name, clearance=0.012):
        """False when a neighbour sits flush enough that the jaw cannot get between them."""
        me = self.cube_pose(name)["pos"][:2]
        for other in self.cubes:
            if other == name or other in self._in_bin:
                continue
            if np.linalg.norm(self.cube_pose(other)["pos"][:2] - me) < 2 * primitives.CUBE_HALF + clearance:
                return False
        return True

    # ---- internals -----------------------------------------------------------------------------

    def _parse(self, action):
        if isinstance(action, dict):
            kind, xy, angle = action["kind"], action["xy"], action.get("angle", 0.0)
        else:
            kind, xy, angle = action
        if isinstance(kind, (int, np.integer)):
            kind = self.PRIMITIVES[int(kind)]
        if kind not in self.PRIMITIVES:
            raise ValueError(f"unknown primitive {kind!r}, expected one of {self.PRIMITIVES}")
        return kind, np.asarray(xy, dtype=np.float64)[:2], float(angle)

    def _plan(self, kind, xy, angle):
        if kind == "pick":
            return primitives.pick(self.model, xy, angle)
        if kind == "place":
            return primitives.place(self.model, xy)
        return primitives.push(self.model, xy, angle)

    def _drive(self, row):
        self.env.step(row)
        if self.on_frame is not None:
            self.on_frame(self)

    def _settle(self, hold):
        for i in range(SETTLE_MAX_STEPS):
            self._drive(hold)
            if self.settled():
                return i + 1
        return SETTLE_MAX_STEPS

    def _cubes_in_bin(self):
        """Inside the footprint, let go of, and not being carried over the top.

        Height alone cannot decide this. The walls are 25 mm and the cubes are 30 mm, so even one cube
        resting on the bin floor sticks above the rim, and a second stacked on it sits at z=0.049. What
        separates "in the bin" from "dangling above it" is whether the jaws are still holding it.
        """
        c, half = self.bin["center"], self.bin["half"]
        ceiling = self.bin["floor_z"] + STACK_CEILING
        inside = set()
        for name in self.cubes:
            p = self.data.qpos[self._cube_qadr[name]:self._cube_qadr[name] + 3]
            if (abs(p[0] - c[0]) < half and abs(p[1] - c[1]) < half
                    and p[2] < ceiling and not self.is_grasped(name)):
                inside.add(name)
        return inside

    def _info(self):
        return {
            "cubes": {n: self.cube_pose(n) for n in self.cubes},
            "n_in_bin": len(self._in_bin),
            "n_cubes": len(self.cubes),
            "bin": dict(self.bin),
            "steps": self._n_steps,
        }

    def _scatter(self, contact_pairs):
        """Place every cube in the spawn sector, with `contact_pairs` pairs deliberately flush."""
        placed = []
        order = list(self.cubes)
        pairs_left = min(contact_pairs, len(order) // 2)

        while len(placed) < len(order):
            if pairs_left > 0 and len(order) - len(placed) >= 2:
                anchor = self._sample_free(placed, margin=2.6 * primitives.CUBE_HALF)
                yaw = self.np_random.uniform(-np.pi, np.pi)
                # Flush faces: the neighbour sits one edge away along the anchor's own x axis.
                step = 2 * primitives.CUBE_HALF + 0.0015
                offset = step * np.array([np.cos(yaw), np.sin(yaw)])
                placed.append((anchor, yaw))
                placed.append((anchor + offset, yaw))
                pairs_left -= 1
            else:
                placed.append((self._sample_free(placed), self.np_random.uniform(-np.pi, np.pi)))

        for name, (xy, yaw) in zip(order, placed):
            adr = self._cube_qadr[name]
            self.data.qpos[adr:adr + 3] = [xy[0], xy[1], primitives.CUBE_HALF]
            self.data.qpos[adr + 3:adr + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
            self.data.qvel[self.model.jnt_dofadr[self.model.body(name).jntadr[0]]:][:6] = 0.0

    def _sample_free(self, placed, margin=0.05, tries=200):
        for _ in range(tries):
            r = self.np_random.uniform(*SPAWN_RADIUS)
            a = self.np_random.uniform(*SPAWN_ANGLE)
            xy = np.array([r * np.cos(a), r * np.sin(a)])
            if any(np.linalg.norm(xy - p) < margin for p, _ in placed):
                continue
            return xy
        raise RuntimeError("could not find a free spawn position; too many cubes for the sector")
