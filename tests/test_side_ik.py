import numpy as np
import pytest

from so101sim import SCENE_PATH, ik, primitives

import mujoco  # after so101sim, which sets MUJOCO_GL


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(str(SCENE_PATH))


def plan_grip(model):
    return primitives._grip_rad(model, primitives.GRIP_PLAN)


def test_the_pan_axis_sits_in_front_of_the_origin(model):
    np.testing.assert_allclose(ik.pan_axis_xy(model), [0.0388, 0.0], atol=0.001)


@pytest.mark.parametrize("target", [(0.36, 0.0, 0.10), (0.42, 0.10, 0.10), (0.30, -0.12, 0.11)])
def test_solve_side_reaches_level_with_the_mount_up(model, target):
    q, info = ik.solve_side(model, target, gripper=plan_grip(model))
    assert info["pos_err"] < 1e-3 and info["tilt"] < 1.0, info
    data = mujoco.MjData(model)
    data.qpos[model.jnt_qposadr[[model.joint(n).id for n in ik.ARM_JOINTS]]] = q
    data.qpos[model.jnt_qposadr[model.joint("gripper").id]] = plan_grip(model)
    mujoco.mj_kinematics(model, data)
    R = data.site_xmat[model.site("gripperframe").id].reshape(3, 3).copy()
    assert R[2, ik.SIDE_UP_AXIS] > 0.999  # the camera mount points up
    assert abs(R[2, ik.TOOL_AXIS]) < 0.02 and abs(R[2, ik.JAW_AXIS]) < 0.02  # tool and jaws level
    np.testing.assert_allclose(ik.grasp_point(model, data), target, atol=1e-3)


def test_heading_follows_the_ray_from_the_pan_axis(model):
    pan = ik.pan_axis_xy(model)
    for y in (-0.10, 0.0, 0.10):
        _, info = ik.solve_side(model, (0.38, y, 0.10), gripper=plan_grip(model))
        ray = np.degrees(np.arctan2(y - pan[1], 0.38 - pan[0]))
        assert abs(info["heading"] - ray) < 8.0, (y, info)


def test_solve_side_reports_what_it_cannot_reach(model):
    # 0.11 m from the pan axis, well inside level reach's 0.22 m inner limit
    _, info = ik.solve_side(model, (0.15, 0.0, 0.10), gripper=plan_grip(model), restarts=2)
    assert info["pos_err"] > 5e-3 or info["tilt"] > 2.0, info


def test_a_seed_keeps_the_solution_on_its_branch(model):
    q0, _ = ik.solve_side(model, (0.36, 0.0, 0.10), gripper=plan_grip(model))
    q1, info = ik.solve_side(model, (0.364, 0.0, 0.10), gripper=plan_grip(model), q_init=q0, restarts=0)
    assert info["pos_err"] < 1e-3
    assert np.abs(q1 - q0).max() < np.radians(5)
