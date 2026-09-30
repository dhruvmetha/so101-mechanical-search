import numpy as np
import pytest

from so101sim import rest

import mujoco  # after so101sim, which sets MUJOCO_GL

# A free box dropped onto a floor, and a hinged paddle that stands in for the arm.
SCENE = """
<mujoco>
  <option timestep="0.0033333333333333335"/>
  <worldbody>
    <geom type="plane" size="1 1 0.1"/>
    <body name="box" pos="0 0 0.1">
      <freejoint/>
      <geom type="box" size="0.015 0.015 0.015" density="1000"/>
    </body>
    <body name="paddle" pos="0.5 0 0.2">
      <joint name="spin" type="hinge" axis="0 0 1"/>
      <geom type="box" size="0.05 0.01 0.01" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def scene():
    model = mujoco.MjModel.from_xml_string(SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def frame(model, data):
    return lambda: mujoco.mj_step(model, data, nstep=10)


def test_a_falling_box_is_not_still(scene):
    model, data = scene
    dofs = rest.free_dofs(model, ["box"])
    for _ in range(5):
        frame(model, data)()
    assert not rest.still(data, dofs)


def test_a_dropped_box_comes_to_rest(scene):
    model, data = scene
    dofs = rest.free_dofs(model, ["box"])
    assert rest.come_to_rest(frame(model, data), data, dofs)
    assert rest.still(data, dofs)
    assert np.linalg.norm(data.qvel[dofs[0]:dofs[0] + 3]) <= rest.REST_SPEED


def test_only_the_listed_objects_count(scene):
    model, data = scene
    dofs = rest.free_dofs(model, ["box"])
    assert rest.come_to_rest(frame(model, data), data, dofs)
    data.qvel[model.jnt_dofadr[model.joint("spin").id]] = 5.0  # the paddle spins, the box is untouched
    assert rest.still(data, dofs)


def test_something_that_never_settles_is_reported(scene):
    model, data = scene
    dofs = rest.free_dofs(model, ["box"])

    def shove():
        mujoco.mj_step(model, data, nstep=10)
        data.qvel[dofs[0]] = 0.05  # 5 cm/s along x at the end of every frame; friction would stop it within one

    assert not rest.come_to_rest(shove, data, dofs)


def test_free_dofs_rejects_a_body_without_a_free_joint(scene):
    model, _ = scene
    with pytest.raises(ValueError):
        rest.free_dofs(model, ["paddle"])
