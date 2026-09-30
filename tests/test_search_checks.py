import numpy as np
import pytest

from so101sim import SCENE_PATH, SEARCH_SCENE_PATH, grasp, primitives, search_checks as sc, search_layout as sl

import mujoco  # after so101sim, which sets MUJOCO_GL


@pytest.fixture(scope="module")
def arm_model():
    return mujoco.MjModel.from_xml_path(str(SCENE_PATH))


@pytest.fixture(scope="module")
def committed():
    return sl.Layout.from_model(mujoco.MjModel.from_xml_path(str(SEARCH_SCENE_PATH)))


def test_grid_covers_every_cube_centre_that_fits():
    pts = np.array(sc.grid_points(sl.Layout(width=0.12, center_x=0.20)))
    assert len(pts) == 19 * 19
    np.testing.assert_allclose([pts[:, 0].min(), pts[:, 0].max()], [0.155, 0.245], atol=1e-9)
    np.testing.assert_allclose([pts[:, 1].min(), pts[:, 1].max()], [-0.045, 0.045], atol=1e-9)


def test_wall_strip_covers_cubes_near_any_wall_not_just_corners():
    layout = sl.Layout(width=0.12, center_x=0.20)  # cube centres run 0.155 to 0.245 in x, -0.045 to 0.045 in y
    assert sc.near_wall(layout, (0.245, 0.045))  # corner, faces flush
    assert sc.near_wall(layout, (0.245, 0.0))  # flush against the far wall, mid-edge
    assert sc.near_wall(layout, (0.20, -0.025))  # face exactly 2 cm from a side wall counts
    assert not sc.near_wall(layout, (0.20, -0.02))  # face 2.5 cm away
    assert not sc.near_wall(layout, (0.20, 0.0))


def test_reach_check_agrees_with_the_measured_band(arm_model):
    assert sc.reachable(arm_model, (0.20, 0.0, -0.05), primitives.GRIP_PLAN)
    assert not sc.reachable(arm_model, (0.20, 0.0, 0.12), primitives.GRIP_PLAN)


def test_clearance_catches_jaws_driven_into_the_floor():
    layout = sl.Layout()
    model = sl.load(layout)
    data = mujoco.MjData(model)
    arm, scene = grasp.arm_geom_ids(model), sc.scene_geom_ids(model)
    below = sc.grasp_poses(model, (layout.center_x, 0.0, layout.floor_z - 0.01))
    assert below, "IK reaches 1 cm under the floor; the measured band runs down to z = -0.20"
    assert not sc.clearance_ok(model, data, below, arm, scene)
    above = sc.grasp_poses(model, (layout.center_x, 0.0, layout.floor_z + 0.05))
    assert sc.clearance_ok(model, data, above, arm, scene)


def test_committed_layout_came_from_the_sizing_script():
    assert any(n.startswith("Chosen") for n in sl.read_notes())


def test_committed_layout_reaches_its_grid_corners(arm_model, committed):
    reach = committed.width / 2 - primitives.CUBE_HALF
    heights = ((committed.floor_z + primitives.CUBE_HALF, primitives.GRIP_PLAN),
               (committed.floor_z + 5 * primitives.CUBE_HALF, primitives.GRIP_PLAN),
               (committed.travel_z, primitives.GRIP_HOLDING))
    for dx in (-reach, reach):
        for dy in (-reach, reach):
            for z, grip in heights:
                assert sc.reachable(arm_model, (committed.center_x + dx, dy, z), grip), (dx, dy, z)


def test_committed_start_pose_stays_out_of_the_drop_column(committed):
    model = sl.load(committed, probes=True)
    assert sc.drop_column_clear(model, mujoco.MjData(model), committed.home)


def test_committed_cameras_see_enough_of_the_floor(committed):
    model = sl.load(committed, probes=True)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, *sc.IMAGE_SIZE)
    try:
        assert sc.marker_coverage(model, data, renderer, "front", committed.home) >= sc.FRONT_MIN
        assert sc.marker_coverage(model, data, renderer, "wrist_cam", committed.home) >= sc.WRIST_MIN
    finally:
        renderer.close()
