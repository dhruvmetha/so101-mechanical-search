"""Run the scene in MJX on the GPU, batched, with the MuJoCo Warp backend.

Measured on an RTX 6000 Ada with 1024 worlds, on 2026-09-18:

    backend  contacts           step     env-steps/s
    Warp     full scene          33.6 ms   30,490
    Warp     finger pads only    32.8 ms   31,236
    JAX      finger pads only   240.2 ms    4,263   (and 41 GB of GPU memory)
    JAX      full scene          5.3 s per step for ONE world

So this uses Warp with the scene as-is. Trimming contacts barely helps Warp, and the JAX backend is too slow
for this model even when trimmed. An env-step here is one 1/300 s physics step, so a 30 Hz control step
costs 10 of them.
"""

import copy
import warnings

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from so101sim import SCENE_PATH

IMPL = "warp"


def load_model(scene_path=SCENE_PATH):
    return mujoco.MjModel.from_xml_path(str(scene_path))


def put(model, worlds, contacts_per_world=64, constraints_per_world=256):
    """Returns (mjx_model, data) on the GPU, with data batched over `worlds` identical copies of qpos0.

    Warp preallocates fixed contact and constraint buffers. naconmax counts contacts across all worlds,
    njmax counts constraint rows per world. When a buffer fills, Warp prints "broadphase overflow",
    "narrowphase overflow" or "nefc overflow" and drops the excess, and objects sink into whatever they
    rest on. At 32 constraint rows per world the resting cubes sink 2 mm. If you see those warnings,
    raise the matching argument.

    Warp has no noslip pass and raises if the model asks for one, which stage.xml does. This drops it on a copy,
    with a warning, and leaves `model` alone. Without noslip, resting piles creep on the GPU: measured 2026-09-29,
    bin piles drifted 22 to 47 mm median over 30 s under every GPU-compatible setting tried. Warp computes in
    32-bit floats. Don't trust GPU results that depend on objects staying where they were left.
    """
    if model.opt.noslip_iterations:
        warnings.warn(f"the Warp backend has no noslip; running without the scene's "
                      f"noslip_iterations={model.opt.noslip_iterations}, so resting piles will creep", stacklevel=2)
        model = copy.deepcopy(model)
        model.opt.noslip_iterations = 0
    mjx_model = mjx.put_model(model, impl=IMPL)
    # Warp's make_data takes the MjModel, not the mjx model.
    data = mjx.make_data(model, impl=IMPL, naconmax=worlds * contacts_per_world, njmax=constraints_per_world)
    return mjx_model, jax.vmap(lambda _: data)(jnp.arange(worlds))
