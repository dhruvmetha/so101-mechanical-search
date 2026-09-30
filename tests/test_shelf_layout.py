import dataclasses

import numpy as np
import pytest

from so101sim import SHELF_SCENE_PATH, shelf_layout as sl
from so101sim.env import SO101Env

import mujoco  # after so101sim, which sets MUJOCO_GL

LAYOUT = sl.ShelfLayout(board_z=0.06, opening=0.13, front_x=0.30, depth=0.18, width=0.24,
                        cam_pos=(0.05, -0.3, 0.4), n_cyls=6)


@pytest.fixture(scope="module")
def model():
    return sl.load(LAYOUT)


def geom_box(model, name):
    g = model.geom(name)
    return g.pos - g.size, g.pos + g.size


def test_layout_round_trips_through_the_xml(model):
    back = sl.ShelfLayout.from_model(model)
    for field in dataclasses.fields(sl.ShelfLayout):
        np.testing.assert_allclose(getattr(back, field.name), getattr(LAYOUT, field.name), atol=1e-6)


def test_the_shelf_sits_where_the_layout_says(model):
    lo, hi = geom_box(model, "shelf_board")
    assert hi[2] == pytest.approx(0.06) and lo[0] == pytest.approx(0.30)
    lo, _ = geom_box(model, "shelf_top")
    assert lo[2] == pytest.approx(0.19)
    lo, _ = geom_box(model, "shelf_back")
    assert lo[0] == pytest.approx(0.48)
    assert geom_box(model, "shelf_left")[0][1] == pytest.approx(0.12)
    assert geom_box(model, "shelf_right")[1][1] == pytest.approx(-0.12)


def test_heights_lines_and_region():
    assert LAYOUT.grasp_z == pytest.approx(0.10) and LAYOUT.carry_z == pytest.approx(0.11)
    assert LAYOUT.push_z == pytest.approx(0.085)
    assert LAYOUT.aisle_x == pytest.approx(0.32) and LAYOUT.standoff_x == pytest.approx(0.27)
    np.testing.assert_allclose(LAYOUT.region(), [0.355, 0.46, -0.10, 0.10], atol=1e-9)
    assert LAYOUT.on_board((0.31, 0.11)) and not LAYOUT.on_board((0.29, 0.0))


def test_cylinders_wait_off_stage_and_only_grays_are_in_the_gray_group(model):
    names = [model.body(b).name for b in range(model.nbody) if model.body(b).name.startswith("cyl_")]
    assert names[0] == sl.TARGET_NAME and len(names) == 6
    for name in names:
        g = model.geom(name)
        assert g.type == mujoco.mjtGeom.mjGEOM_CYLINDER
        np.testing.assert_allclose(g.size[:2], [0.015, 0.04])
        assert g.group == (0 if name == sl.TARGET_NAME else sl.GRAY_GROUP)
        assert model.body(name).pos[0] == pytest.approx(sl.PARK_X)


def test_probes_add_markers_on_the_board_that_collide_with_nothing():
    probed = sl.load(LAYOUT, probes=True)
    markers = [g for g in range(probed.ngeom) if probed.geom(g).name.startswith(sl.MARKER_PREFIX)]
    assert len(markers) > 100
    for g in markers:
        assert probed.geom_contype[g] == 0 and probed.geom_conaffinity[g] == 0
        assert LAYOUT.on_board(probed.geom_pos[g][:2])


def test_notes_with_a_double_dash_are_refused():
    with pytest.raises(ValueError):
        sl.objects_xml(LAYOUT, notes=("bad -- note",))


def test_the_committed_scene_loads_in_the_base_env():
    env = SO101Env(scene_path=SHELF_SCENE_PATH, height=64, width=64)
    try:
        env.reset(seed=0)
    finally:
        env.close()
