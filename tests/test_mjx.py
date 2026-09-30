import mujoco
import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("warp")
from mujoco import mjx  # noqa: E402

from so101sim import mjx as so101_mjx  # noqa: E402
from so101sim import units  # noqa: E402

pytestmark = pytest.mark.mjx

# Arm target in radians, in free space above the table.
TARGET = np.array([0.4, -0.5, 0.6, 0.8, -0.5, 0.3])
STEPS = 300  # 1 s at the scene's 1/300 s timestep
WORLDS = 4


@pytest.fixture(scope="module")
def model():
    return so101_mjx.load_model()


def test_put_drops_noslip_with_a_warning_and_leaves_the_cpu_model_alone(model):
    # The scene turns noslip on (stage.xml) and Warp raises NotImplementedError for it.
    assert model.opt.noslip_iterations > 0
    with pytest.warns(UserWarning, match="noslip"):
        so101_mjx.put(model, 1)
    assert model.opt.noslip_iterations > 0


def test_batched_warp_worlds_match_cpu(model):
    cpu = mujoco.MjData(model)
    cpu.ctrl[:] = TARGET
    for _ in range(STEPS):
        mujoco.mj_step(model, cpu)

    mjx_model, batch = so101_mjx.put(model, WORLDS)
    batch = batch.replace(ctrl=jax.numpy.tile(TARGET, (WORLDS, 1)))
    step = jax.vmap(mjx.step, in_axes=(None, 0))

    def body(d, _):
        return step(mjx_model, d), None

    batch, _ = jax.jit(lambda d: jax.lax.scan(body, d, None, length=STEPS))(batch)

    qpos = np.asarray(batch.qpos)
    assert qpos.shape == (WORLDS, model.nq)
    arm = [model.joint(name).qposadr[0] for name in units.JOINT_NAMES]
    # The cubes rest on the table. Too small a Warp buffer drops their contacts and they sink: 2 mm at 32
    # constraint rows per world, straight through the table at 8. So cube height gets a 1 mm tolerance.
    cube_z = [model.joint(name).qposadr[0] + 2 for name in ("cube_red", "cube_green", "cube_blue")]
    for world in qpos:
        np.testing.assert_allclose(world[arm], cpu.qpos[arm], atol=0.01)
        np.testing.assert_allclose(world[cube_z], cpu.qpos[cube_z], atol=1e-3)
