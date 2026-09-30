import numpy as np
import pytest

from so101sim import SHELF_SCENE_PATH, ik, primitives, side_primitives as sp, units
from so101sim.shelf_layout import ShelfLayout

import mujoco  # after so101sim, which sets MUJOCO_GL


@pytest.fixture(scope="module")
def setup():
    model = mujoco.MjModel.from_xml_path(str(SHELF_SCENE_PATH))
    layout = ShelfLayout.from_model(model)
    return model, layout, sp.Geometry.from_layout(model, layout), sp.side_open_value(model)


def mids(model, traj, ik_grip):
    """Jaw midpoint of every frame, with the gripper set to the value IK planned at."""
    data = mujoco.MjData(model)
    adr = model.jnt_qposadr[[model.joint(n).id for n in units.JOINT_NAMES]]
    grip_range = tuple(model.jnt_range[model.joint("gripper").id])
    out = []
    for row in traj:
        frame = np.array(row, dtype=np.float64)
        frame[5] = ik_grip
        data.qpos[adr] = units.to_rad(frame, grip_range)
        mujoco.mj_kinematics(model, data)
        out.append(ik.grasp_point(model, data).copy())
    return np.array(out)


def lateral(geo, through_xy, points):
    """Distance of each point from the ray through `through_xy`, in the horizontal plane."""
    u = np.asarray(through_xy) - geo.pan_xy
    u /= np.linalg.norm(u)
    d = points[:, :2] - geo.pan_xy
    return np.abs(d[:, 0] * u[1] - d[:, 1] * u[0])


def centre(layout):
    x_lo, x_hi, y_lo, y_hi = layout.region()
    return np.array([(x_lo + x_hi) / 2, (y_lo + y_hi) / 2 + 0.03])


def max_step(traj):
    return np.abs(np.diff(traj, axis=0)).max()


def test_side_open_value_is_the_smallest_with_the_gap(setup):
    model, _, _, open_value = setup
    assert sp.jaw_gap(model, open_value) >= sp.SIDE_OPEN_GAP > sp.jaw_gap(model, open_value - 1)
    assert 25 <= open_value <= 33


def test_on_ray_stays_on_the_ray(setup):
    _, layout, geo, _ = setup
    p = geo.on_ray([0.40, 0.10], layout.standoff_x)
    assert p[0] == pytest.approx(layout.standoff_x)
    assert lateral(geo, [0.40, 0.10], np.array([[*p, 0.0]]))[0] < 1e-9


def test_pick_goes_in_along_the_ray_and_stops_in_the_aisle(setup):
    model, layout, geo, open_value = setup
    xy = centre(layout)
    traj = sp.pick(model, geo, xy, open_value)
    assert max_step(traj) <= primitives.MAX_JOINT_STEP_DEG + 1e-4
    m = mids(model, traj, primitives.GRIP_PLAN)
    np.testing.assert_allclose(m[0], [*geo.on_ray(xy, layout.standoff_x), layout.grasp_z], atol=2e-3)
    np.testing.assert_allclose(m[-1], [*geo.on_ray(xy, layout.aisle_x), layout.carry_z], atol=2e-3)
    assert np.linalg.norm(m - [*xy, layout.grasp_z], axis=1).min() < 2e-3
    assert lateral(geo, xy, m).max() < 3e-3


def test_place_keeps_the_held_cylinder_over_the_board(setup):
    model, layout, geo, open_value = setup
    here = geo.on_ray(centre(layout), layout.aisle_x)
    xy = centre(layout) + [0.03, -0.08]
    traj = sp.place(model, geo, here, xy, open_value)
    assert max_step(traj) <= primitives.MAX_JOINT_STEP_DEG + 1e-4
    m = mids(model, traj, primitives.GRIP_HOLDING)
    closed = traj[:, 5] <= primitives.GRIP_CLOSE + 0.5
    assert (m[closed, 0] >= layout.aisle_x - 3e-3).all()
    assert np.linalg.norm(m - [*xy, layout.grasp_z], axis=1).min() < 2e-3
    np.testing.assert_allclose(m[-1], [*geo.on_ray(xy, layout.standoff_x), layout.grasp_z], atol=2e-3)


def test_pull_takes_the_cylinder_off_the_board(setup):
    model, layout, geo, _ = setup
    here = geo.on_ray(centre(layout), layout.aisle_x)
    m = mids(model, sp.pull(model, geo, here), primitives.GRIP_HOLDING)
    np.testing.assert_allclose(m[-1], [*geo.on_ray(here, layout.standoff_x), layout.carry_z], atol=2e-3)
    assert layout.front_x - m[-1][0] >= 0.02


def test_push_drops_only_past_the_front_edge(setup):
    model, layout, geo, _ = setup
    xy = centre(layout)
    traj = sp.push(model, geo, xy, -np.pi / 2)
    assert max_step(traj) <= primitives.MAX_JOINT_STEP_DEG + 1e-4
    m = mids(model, traj, primitives.GRIP_CLOSE)
    low = m[:, 2] < layout.grasp_z - 0.005
    assert (m[low, 0] >= layout.front_x - 3e-3).all()
    end = xy + sp.PUSH_STROKE * np.array([0.0, -1.0])
    assert np.linalg.norm(m - [*end, layout.push_z], axis=1).min() < 2e-3
    for p in (m[0], m[-1]):
        assert p[0] == pytest.approx(layout.standoff_x, abs=2e-3)


def test_look_stops_on_the_standoff_line(setup):
    model, layout, geo, _ = setup
    xy = centre(layout)
    m = mids(model, sp.look(model, geo, xy), primitives.GRIP_PLAN)
    np.testing.assert_allclose(m[-1], [*geo.on_ray(xy, layout.standoff_x), layout.grasp_z], atol=2e-3)


def test_bridge_between_standoff_poses_stays_outside_the_shelf(setup):
    model, layout, geo, _ = setup
    x_lo, x_hi, y_lo, y_hi = layout.region()
    here = sp.look(model, geo, [x_hi, y_hi])[-1]
    traj = sp.look(model, geo, [x_hi, y_lo])
    bridged = sp.bridge(model, geo, here, traj)
    assert max_step(bridged) <= primitives.MAX_JOINT_STEP_DEG + 1e-4
    m = mids(model, bridged, primitives.GRIP_PLAN)
    assert (m[:, 0] <= layout.standoff_x + 5e-3).all()


def test_bridge_backs_out_of_the_aisle_along_the_ray(setup):
    model, layout, geo, open_value = setup
    xy = centre(layout)
    here = sp.pick(model, geo, xy, open_value)[-1]
    x_lo, _, y_lo, _ = layout.region()
    bridged = sp.bridge(model, geo, here, sp.look(model, geo, [x_lo, y_lo]))
    m = mids(model, bridged, primitives.GRIP_PLAN)
    inside = m[:, 0] > layout.standoff_x + 5e-3
    assert inside.any()
    assert lateral(geo, xy, m[inside]).max() < 3e-3


def test_a_point_out_of_reach_raises(setup):
    model, _, geo, open_value = setup
    with pytest.raises(primitives.Unreachable):
        sp.pick(model, geo, [0.62, 0.0], open_value)
