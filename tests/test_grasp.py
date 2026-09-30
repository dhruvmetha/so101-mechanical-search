import pytest

from so101sim import SCENE_PATH, grasp, ik

import mujoco  # after so101sim, which sets MUJOCO_GL


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(str(SCENE_PATH))


def test_jaw_geoms_cover_both_jaws_and_nothing_else(model):
    names = {model.geom(i).name for i in grasp.jaw_geom_ids(model)}
    assert set(ik.JAW_TIP_GEOMS) <= names
    assert all(n.startswith(grasp.JAW_PREFIXES) for n in names)
    assert any(n.startswith("fixed_jaw") for n in names)
    assert any(n.startswith("moving_jaw") for n in names)


def test_arm_geoms_are_the_moving_links_only(model):
    bodies = {model.body(model.geom_bodyid[g]).name for g in grasp.arm_geom_ids(model)}
    assert {"shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "moving_jaw_so101_v1"} <= bodies
    assert not bodies & {"world", "base", "base_collision", "bin", "cube_red"}


def test_a_resting_scene_has_no_grasp_and_no_arm_contact(model):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    red = model.geom("cube_red").id
    assert not grasp.is_grasped(model, data, red, grasp.jaw_geom_ids(model))
    assert not grasp.touching(data, grasp.arm_geom_ids(model), {red})


def test_touching_sees_a_cube_resting_on_the_table(model):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for _ in range(30):
        mujoco.mj_step(model, data)
    assert grasp.touching(data, {model.geom("cube_red").id}, {model.geom("table").id})
