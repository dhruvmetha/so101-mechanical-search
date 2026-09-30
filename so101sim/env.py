"""Gymnasium env for the SO-101 on a table, with lerobot-style actions and observations.

Actions and observation.state are the six joints in lerobot order and units: degrees for the arm, 0 to 100
for the gripper. Image keys match lerobot's dataset naming, so episodes can be written to a
LeRobotDataset as-is. Reward is always 0 and episodes never end until a task exists.
"""

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from so101sim import SCENE_PATH, units

# Gripper pointing straight down over the table, half open. lerobot units.
HOME = np.array([0.0, -30.0, 30.0, 90.0, 0.0, 50.0], dtype=np.float32)
CONTROL_HZ = 30

# Observation name -> MJCF camera name.
CAMERAS = {"front": "front", "wrist": "wrist_cam"}

# The vendored arm's finger geoms. Their class sets solref 0.01 and priority 1, so they would decide every finger
# contact with a softer setting than the rest of the scene.
GRIPPER_CLASSES = ("collision_gripper", "collision_gripper_mesh")


def load_model(scene_path=SCENE_PATH):
    """Compile a scene with the gripper's contacts as stiff as the scene's main default (models/stage.xml). Anything
    that steps physics loads through here; a bare MjModel.from_xml_path keeps the vendored gripper's softer contacts.
    """
    spec = mujoco.MjSpec.from_file(str(scene_path))
    solref, solimp = list(spec.default.geom.solref), list(spec.default.geom.solimp)
    for geom in spec.geoms:
        if geom.classname.name in GRIPPER_CLASSES:
            geom.solref, geom.solimp = solref, solimp
    return spec.compile()


class SO101Env(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": CONTROL_HZ}

    def __init__(self, scene_path=SCENE_PATH, height=480, width=640, render_mode="rgb_array"):
        self.model = load_model(scene_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode

        joint_ids = [self.model.joint(name).id for name in units.JOINT_NAMES]
        self._qpos_adr = self.model.jnt_qposadr[joint_ids]
        ranges = self.model.jnt_range[joint_ids]
        self._gripper_range = tuple(ranges[5])

        # Clip to joint ranges, not actuator ctrlranges. Menagerie's wrist_roll ctrlrange runs 5.6 degrees
        # past its joint limit on the high side.
        low, high = units.lerobot_limits(ranges)
        self.action_space = spaces.Box(low.astype(np.float32), high.astype(np.float32), dtype=np.float32)
        image = spaces.Box(0, 255, (height, width, 3), dtype=np.uint8)
        self.observation_space = spaces.Dict(
            {
                "observation.state": spaces.Box(-np.inf, np.inf, (6,), dtype=np.float32),
                **{f"observation.images.{name}": image for name in CAMERAS},
            }
        )

        substeps = (1 / CONTROL_HZ) / self.model.opt.timestep
        self._n_substeps = int(round(substeps))
        if abs(substeps - self._n_substeps) > 1e-6:
            raise ValueError(f"timestep {self.model.opt.timestep} doesn't divide the {CONTROL_HZ} Hz control period")

        self._renderer = mujoco.Renderer(self.model, height, width)

    @property
    def joint_state(self):
        """The six joints in lerobot units, the same quantity an action commands."""
        return units.to_lerobot(self.data.qpos[self._qpos_adr], self._gripper_range).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        q = units.to_rad(HOME, self._gripper_range)
        self.data.qpos[self._qpos_adr] = q
        self.data.ctrl[:] = q
        mujoco.mj_forward(self.model, self.data)
        return self._obs(), {}

    def step(self, action):
        self._apply(action)
        return self._obs(), 0.0, False, False, {}

    def _apply(self, action):
        """Clip `action` to the joint ranges and run one 30 Hz control step of physics."""
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:] = units.to_rad(action, self._gripper_range)
        mujoco.mj_step(self.model, self.data, nstep=self._n_substeps)

    def render(self):
        return self._render("front")

    def render_depth(self, name="front"):
        """Metres along the view axis from the named observation camera, at the observation size. The real front
        camera gives depth and the wrist camera doesn't (dm1487, 2026-09-30), so datasets store only the front's."""
        self._renderer.enable_depth_rendering()
        self._renderer.update_scene(self.data, camera=CAMERAS[name])
        depth = self._renderer.render().copy()
        self._renderer.disable_depth_rendering()
        return depth

    def close(self):
        self._renderer.close()

    def _render(self, camera):
        self._renderer.update_scene(self.data, camera=CAMERAS[camera])
        return self._renderer.render()

    def _obs(self):
        state = units.to_lerobot(self.data.qpos[self._qpos_adr], self._gripper_range)
        obs = {"observation.state": state.astype(np.float32)}
        for name in CAMERAS:
            obs[f"observation.images.{name}"] = self._render(name)
        return obs
