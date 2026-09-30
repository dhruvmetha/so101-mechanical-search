import mujoco
import numpy as np
import pytest

from so101sim import SCENE_PATH, ik


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(str(SCENE_PATH))


def fk(model, q_arm):
    data = mujoco.MjData(model)
    for name, q in zip(ik.ARM_JOINTS, q_arm):
        data.qpos[model.joint(name).qposadr[0]] = q
    mujoco.mj_kinematics(model, data)
    return data.site_xpos[model.site("gripperframe").id].copy()


def arm_ranges(model):
    return np.array([model.jnt_range[model.joint(name).id] for name in ik.ARM_JOINTS])


def test_ik_reaches_fk_generated_points_within_1mm(model):
    rng = np.random.default_rng(0)
    ranges = arm_ranges(model)
    for _ in range(20):
        target = fk(model, rng.uniform(0.8 * ranges[:, 0], 0.8 * ranges[:, 1]))
        q, err = ik.solve_ik(model, target)
        assert err < 1e-3
        # Check the answer with an independent FK pass, not the solver's own error.
        assert np.linalg.norm(fk(model, q) - target) < 1e-3


def test_ik_solution_stays_within_joint_limits(model):
    rng = np.random.default_rng(1)
    ranges = arm_ranges(model)
    for _ in range(20):
        target = fk(model, rng.uniform(ranges[:, 0], ranges[:, 1]))
        q, _ = ik.solve_ik(model, target)
        assert np.all(q >= ranges[:, 0]) and np.all(q <= ranges[:, 1])


def test_ik_reports_error_for_unreachable_target(model):
    _, err = ik.solve_ik(model, np.array([2.0, 0.0, 0.2]))
    assert err > 1.0
