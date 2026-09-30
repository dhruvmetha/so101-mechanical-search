import numpy as np
import pytest

from so101sim import units
from so101sim.env import HOME, SO101Env


@pytest.fixture(scope="module")
def env():
    e = SO101Env()
    yield e
    e.close()


def jaw_tip_distance(env):
    a = env.data.geom_xpos[env.model.geom("fixed_jaw_sph_tip1").id]
    b = env.data.geom_xpos[env.model.geom("moving_jaw_sph_tip1").id]
    return np.linalg.norm(a - b)


def test_joints_and_actuators_are_in_lerobot_order(env):
    assert tuple(env.model.joint(i).name for i in range(6)) == units.JOINT_NAMES
    assert tuple(env.model.actuator(i).name for i in range(6)) == units.JOINT_NAMES


def test_reset_puts_arm_at_home(env):
    obs, _ = env.reset(seed=0)
    np.testing.assert_allclose(obs["observation.state"], HOME, atol=1e-4)


def test_observation_and_action_fit_their_spaces(env):
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert env.action_space.shape == (6,)
    assert env.action_space.contains(HOME)


def test_step_advances_one_thirtieth_of_a_second(env):
    env.reset(seed=0)
    t0 = env.data.time
    env.step(HOME)
    assert env.data.time - t0 == pytest.approx(1 / 30)


def test_held_targets_settle_within_one_degree(env):
    env.reset(seed=0)
    target = np.array([20.0, -20.0, 30.0, 40.0, -45.0, 50.0], dtype=np.float32)
    for _ in range(60):
        obs, *_ = env.step(target)
    np.testing.assert_allclose(obs["observation.state"], target, atol=1.0)


def test_actions_past_limits_are_clipped(env):
    env.reset(seed=0)
    for _ in range(90):
        obs, *_ = env.step(np.full(6, 1000.0, dtype=np.float32))
    state = obs["observation.state"]
    # The servos can overshoot a hair under load, so allow 1 unit past the limit.
    assert np.all(state <= env.action_space.high + 1.0)


def test_gripper_0_closes_and_100_opens(env):
    env.reset(seed=0)
    action = HOME.copy()
    action[5] = 0.0
    for _ in range(30):
        env.step(action)
    closed = jaw_tip_distance(env)
    action[5] = 100.0
    for _ in range(30):
        env.step(action)
    assert jaw_tip_distance(env) > closed + 0.05


def test_cameras_render_distinct_nonblank_images(env):
    obs, _ = env.reset(seed=0)
    front = obs["observation.images.front"]
    wrist = obs["observation.images.wrist"]
    for img in (front, wrist):
        assert img.shape == (480, 640, 3) and img.dtype == np.uint8
        assert img.std() > 10
    assert not np.array_equal(front, wrist)


def test_reward_is_zero_and_episode_never_ends(env):
    env.reset(seed=0)
    _, reward, terminated, truncated, _ = env.step(HOME)
    assert reward == 0.0 and not terminated and not truncated
