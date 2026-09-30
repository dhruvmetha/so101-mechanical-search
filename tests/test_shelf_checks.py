import numpy as np
import pytest

from so101sim import SCENE_PATH, SHELF_SCENE_PATH, grasp, primitives, shelf_checks as sc, shelf_layout as sl
from so101sim import side_primitives as sp

import mujoco  # after so101sim, which sets MUJOCO_GL


@pytest.fixture(scope="module")
def arm_model():
    return mujoco.MjModel.from_xml_path(str(SCENE_PATH))


@pytest.fixture(scope="module")
def committed():
    return sl.ShelfLayout.from_model(mujoco.MjModel.from_xml_path(str(SHELF_SCENE_PATH)))


def test_centre_points_cover_the_region_on_a_5_mm_grid():
    layout = sl.ShelfLayout()
    pts = np.array(sc.centre_points(layout))
    x_lo, x_hi, y_lo, y_hi = layout.region()
    np.testing.assert_allclose([pts[:, 0].min(), pts[:, 0].max()], [x_lo, x_hi], atol=0.0026)
    np.testing.assert_allclose([pts[:, 1].min(), pts[:, 1].max()], [y_lo, y_hi], atol=0.0026)
    assert sc.region_area(layout) == pytest.approx((x_hi - x_lo) * (y_hi - y_lo))


def test_the_right_wall_strip_is_on_the_right():
    layout = sl.ShelfLayout()
    assert sc.near_right_wall(layout, (0.40, -layout.half_width + 0.05))
    assert not sc.near_right_wall(layout, (0.40, layout.half_width - 0.05))


def test_level_pose_agrees_with_the_measured_band(arm_model):
    assert sc.level_pose(arm_model, (0.36, 0.0, 0.10), primitives.GRIP_PLAN) is not None
    assert sc.level_pose(arm_model, (0.15, 0.0, 0.10), primitives.GRIP_PLAN) is None


def test_clear_catches_a_top_board_at_grasp_height(arm_model):
    low_top = sl.ShelfLayout(opening=0.035)  # the top board's underside 5 mm under the jaw midpoint
    model = sl.load(low_top)
    data = mujoco.MjData(model)
    arm, scene = grasp.arm_geom_ids(model), sc.shelf_geom_ids(model)
    q = sc.level_pose(arm_model, (0.40, 0.0, low_top.grasp_z), primitives.GRIP_PLAN)
    assert not sc.clear(model, data, q, primitives.GRIP_PLAN, arm, scene)
    fine = sl.ShelfLayout()
    model = sl.load(fine)
    q = sc.level_pose(arm_model, (0.40, 0.0, fine.grasp_z), primitives.GRIP_PLAN)
    assert sc.clear(model, mujoco.MjData(model), q, primitives.GRIP_PLAN, grasp.arm_geom_ids(model),
                    sc.shelf_geom_ids(model))


def test_committed_layout_came_from_the_sizing_script():
    assert any(n.startswith("Chosen") for n in sl.read_notes())


def test_committed_layout_reaches_its_corners_level(arm_model, committed):
    x_lo, x_hi, y_lo, y_hi = committed.region()
    for x in (x_lo, x_hi):
        for y in (y_lo, y_hi):
            assert sc.level_pose(arm_model, (x, y, committed.grasp_z), primitives.GRIP_PLAN) is not None, (x, y)
    for y in (y_lo, y_hi):
        assert sc.level_pose(arm_model, (committed.aisle_x, y, committed.carry_z), primitives.GRIP_HOLDING) is not None


def test_committed_cameras_see_enough_of_the_board(committed):
    model = sl.load(committed, probes=True)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, *sc.IMAGE_SIZE)
    try:
        assert sc.marker_coverage(model, data, renderer, "front", committed.home) >= sc.FRONT_MIN
        assert sc.marker_coverage(model, data, renderer, "wrist_cam", committed.home) >= sc.WRIST_MIN
    finally:
        renderer.close()


def test_committed_home_is_level_on_the_standoff_line(committed):
    model = mujoco.MjModel.from_xml_path(str(SHELF_SCENE_PATH))
    point, level = sp.tool_point(model, committed.home)
    assert level and point[0] == pytest.approx(committed.standoff_x, abs=2e-3)


def test_the_wall_strip_covers_both_side_walls():
    layout = sl.ShelfLayout()
    hw = layout.half_width
    assert sc.in_wall_strip(layout, (0.40, hw - 0.03))
    assert sc.in_wall_strip(layout, (0.40, -hw + 0.02))
    assert not sc.in_wall_strip(layout, (0.40, hw - 0.035))
    assert not sc.in_wall_strip(layout, (0.40, 0.0))


def test_clear_keeps_a_margin_under_the_top_board(arm_model):
    """Touching isn't enough: a carried arm bounces about 5 mm, so a pose that clears the top board by less than
    CLEARANCE fails (Task 8 finding). camera_box2 sits about 8 cm over the jaw midpoint when level."""
    layout = sl.ShelfLayout()
    model = sl.load(layout)
    sc.add_clearance(model)
    data = mujoco.MjData(model)
    arm, scene = grasp.arm_geom_ids(model), sc.shelf_geom_ids(model)
    top = layout.board_z + layout.opening
    tight = sc.level_pose(arm_model, (0.40, 0.03, top - 0.084), primitives.GRIP_HOLDING)
    roomy = sc.level_pose(arm_model, (0.40, 0.03, top - 0.10), primitives.GRIP_HOLDING)
    assert not sc.clear(model, data, tight, primitives.GRIP_HOLDING, arm, scene)
    assert sc.clear(model, data, roomy, primitives.GRIP_HOLDING, arm, scene)


def test_the_front_camera_is_the_lowest_spot_that_sees_enough():
    """Lower is better: a low camera sees cylinders from the side, so the red can hide (spec, third amendment)."""
    options = [((0.05, 0.0, 0.40), 1.00), ((0.05, 0.0, 0.20), 0.84), ((0.05, -0.3, 0.20), 0.89),
               ((-0.10, 0.0, 0.20), 0.03), ((0.05, 0.0, 0.25), 0.91)]
    assert sc.lowest_camera(options) == ((0.05, -0.3, 0.20), 0.89)
    assert sc.lowest_camera([((0.05, 0.0, 0.20), 0.3)]) is None


def test_the_wrist_camera_starts_near_the_cylinder_tops(committed):
    """Lower is better at the start pose too: a wrist camera well above the tops sees a hidden red over any gray
    (hidden starts kept 0 of 60 with it 4.5 cm up, 90% with it 1.5 cm up; spec, third amendment)."""
    from so101sim.shelf_layout import CYL_HALF_HEIGHT
    model = sl.load(committed)
    data = mujoco.MjData(model)
    sc.set_pose(model, data, committed.home)
    tops = committed.board_z + 2 * CYL_HALF_HEIGHT
    assert data.cam_xpos[model.camera("wrist_cam").id][2] <= tops + 0.02


def test_add_clearance_matches_a_recompiled_inflated_shelf(arm_model, committed):
    """The check-only model has to flag exactly what a shelf recompiled CLEARANCE bigger would. Growing geom sizes
    in place isn't enough on its own: the midphase's bounding-box tree was built at compile time with the old sizes
    and skipped pairs inside the band (final review, 2026-09-26)."""
    with sl.candidate_scene(committed) as path:
        spec = mujoco.MjSpec.from_file(str(path))
    for g in spec.geoms:
        if g.name.startswith("shelf_") or g.name == "table":
            g.size = np.asarray(g.size) + sc.CLEARANCE
    truth = spec.compile()
    probe = sc.add_clearance(sl.load(committed))
    arm_t, scene_t = grasp.arm_geom_ids(truth), sc.shelf_geom_ids(truth)
    arm_p, scene_p = grasp.arm_geom_ids(probe), sc.shelf_geom_ids(probe)
    d_t, d_p = mujoco.MjData(truth), mujoco.MjData(probe)
    x_lo, x_hi, y_lo, y_hi = committed.region()
    top = committed.board_z + committed.opening
    disagree, checked = [], 0
    for x in np.linspace(x_lo, x_hi, 5):
        for y in np.linspace(y_lo, y_hi, 5):
            for z in (committed.push_z, committed.grasp_z, top - 0.085, top - 0.09):
                q = sc.level_pose(arm_model, (x, y, z), primitives.GRIP_CLOSE)
                if q is None:
                    continue
                checked += 1
                hit_t = not sc.clear(truth, d_t, q, primitives.GRIP_CLOSE, arm_t, scene_t)
                hit_p = not sc.clear(probe, d_p, q, primitives.GRIP_CLOSE, arm_p, scene_p)
                if hit_t != hit_p:
                    disagree.append((round(x, 3), round(y, 3), round(z, 3), hit_t))
    assert checked > 50
    assert not disagree, disagree[:5]


def test_sizing_checks_carry_height_at_every_centre():
    """Place carries a cylinder in to its spot at carry height, so sizing has to check carry height at the centres,
    not only on the aisle and pull lines. Without it a 13 cm opening passed while the camera mount scraped the top
    board during a place (2026-09-26)."""
    layout = sl.ShelfLayout()
    geo = sp.Geometry(np.array([0.0388, 0.0]), layout.front_x, layout.aisle_x, layout.standoff_x, layout.grasp_z,
                      layout.carry_z, layout.push_z)
    checks = {c[0]: c for c in sc.sizing_checks(layout, geo, open_value=28.0)}
    name, points, z, ik_grip, grip, exempt = checks["carry"]
    assert z == pytest.approx(layout.carry_z) and ik_grip == grip == primitives.GRIP_HOLDING
    assert set(sc.centre_points(layout)) <= set(points)
