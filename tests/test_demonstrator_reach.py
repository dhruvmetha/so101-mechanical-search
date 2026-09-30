"""The demonstrator's arm clearance check (demonstrator/reach.py)."""
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demonstrator"))

import so101sim  # noqa: E402,F401  before mujoco
import mujoco  # noqa: E402
from so101sim import SEARCH_SCENE_PATH, ik  # noqa: E402

from reach import Clearance  # noqa: E402


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(str(SEARCH_SCENE_PATH))


def dense_box_distance(model, data, a, b, n=25):
    """Reference: a fine grid over box `a`'s surface against box `b`'s exact distance field."""
    lin = np.linspace(-1.0, 1.0, n)
    g = np.array(list(itertools.product(lin, lin, lin)))
    g = g[np.abs(g).max(1) > 0.999] * model.geom_size[a]
    pts = data.geom_xpos[a] + g @ data.geom_xmat[a].reshape(3, 3).T
    q = np.abs((pts - data.geom_xpos[b]) @ data.geom_xmat[b].reshape(3, 3)) - model.geom_size[b]
    return float((np.linalg.norm(np.clip(q, 0, None), axis=1) + np.minimum(q.max(1), 0)).min())


def test_clearance_matches_a_dense_reference_where_mj_geomdistance_reads_zero(model):
    """At this pose the camera mount is about 30 mm from the upper arm, and mj_geomDistance returns 0 for the pair.
    The pose is an exact IK solution; moving any joint by 1e-6 rad gives the right answer, so some alignment of the
    two boxes trips MuJoCo's box-box routine. IK solutions land on such poses."""
    data = mujoco.MjData(model)
    adr = model.jnt_qposadr[[model.joint(n).id for n in ik.ARM_JOINTS]]
    data.qpos[adr] = [0.8394547018291589, -0.6981408625656446, 1.0705653423247807, 1.1983690201188273,
                      0.8881338738059443]
    mujoco.mj_kinematics(model, data)
    cam, upper = model.geom("camera_box2").id, 13
    assert model.body(model.geom_bodyid[upper]).name == "upper_arm"
    assert mujoco.mj_geomDistance(model, data, cam, upper, 0.05, None) == 0.0  # the reason Clearance exists
    ref = dense_box_distance(model, data, cam, upper)
    assert ref > 0.02
    c = Clearance(model)
    pts, owner = c.points(data)
    got = c._sdf(data, pts[owner == cam], upper).min()
    assert abs(got - ref) < 0.003


def test_clearance_goes_negative_when_the_arm_folds_into_itself(model):
    data = mujoco.MjData(model)
    adr = model.jnt_qposadr[[model.joint(n).id for n in ik.ARM_JOINTS]]
    c = Clearance(model)
    rng = np.random.default_rng(0)
    low, high = model.jnt_range[[model.joint(n).id for n in ik.ARM_JOINTS]].T
    signs = set()
    for _ in range(200):
        data.qpos[adr] = rng.uniform(low, high)
        mujoco.mj_kinematics(model, data)
        mine = c.measure(data)[0]
        ref = min(dense_box_distance(model, data, a, b) for a in c.distal for b in c.prox
                  if model.geom_type[a] == mujoco.mjtGeom.mjGEOM_BOX)
        if abs(ref) < 0.05:
            assert mine <= ref + 0.003  # meshes count too, so mine can only be lower
        signs.add(mine < 0)
    assert signs == {True, False}
