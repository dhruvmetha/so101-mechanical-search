import numpy as np
import pytest

from so101sim import SEARCH_SCENE_PATH, search_layout as sl
from so101sim.binpick import bin_geometry, cube_names
from so101sim.env import SO101Env

import mujoco  # after so101sim, which sets MUJOCO_GL


def test_default_layout_compiles_and_reads_back():
    layout = sl.Layout()
    assert sl.Layout.from_model(sl.load(layout)) == layout


def test_bin_geometry_matches_the_layout():
    layout = sl.Layout(depth=0.08, width=0.12, center_x=0.21, wall=0.06)
    b = bin_geometry(sl.load(layout))
    np.testing.assert_allclose(b["center"], [0.21, 0.0], atol=1e-9)
    assert b["half"] == pytest.approx(0.06)
    assert b["floor_z"] == pytest.approx(layout.floor_z)
    assert b["rim_z"] == pytest.approx(layout.rim_z)


def test_table_and_riser_put_the_base_on_a_step():
    model = sl.load(sl.Layout(depth=0.10))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    table, riser = model.geom("table").id, model.geom("riser").id
    assert data.geom_xpos[table][2] + model.geom_size[table][2] == pytest.approx(-0.10)
    assert data.geom_xpos[riser][2] + model.geom_size[riser][2] == pytest.approx(0.0)
    assert data.geom_xpos[riser][2] - model.geom_size[riser][2] == pytest.approx(-0.10)


def test_cubes_are_one_red_then_grays_parked_off_the_table():
    model = sl.load(sl.Layout(n_cubes=7))
    names = cube_names(model)
    assert names[0] == "cube_red" and len(names) == 7
    assert all(n.startswith("cube_gray_") for n in names[1:])
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert all(data.body(n).xpos[0] > 1.0 for n in names)


def test_probe_scenes_add_markers_and_a_hidden_drop_column():
    model = sl.load(sl.Layout(), probes=True)
    markers = [g for g in range(model.ngeom) if model.geom(g).name.startswith(sl.MARKER_PREFIX)]
    assert len(markers) > 100
    assert all(model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0 for g in markers)
    assert model.geom_group[model.geom("drop_column").id] > 2  # the renderer draws groups 0 to 2
    plain = sl.load(sl.Layout())
    assert not any(plain.geom(g).name.startswith(sl.MARKER_PREFIX) for g in range(plain.ngeom))


def test_candidate_files_are_cleaned_up():
    before = set(sl.MODELS_DIR.glob(".candidate_*"))
    sl.load(sl.Layout())
    assert set(sl.MODELS_DIR.glob(".candidate_*")) == before


def test_notes_round_trip_and_reject_double_hyphens():
    xml = sl.objects_xml(sl.Layout(), notes=["first line", "second line"])
    assert sl.parse_notes(xml) == ["first line", "second line"]
    with pytest.raises(ValueError):
        sl.objects_xml(sl.Layout(), notes=["a -- b"])


def test_travel_height_clears_a_full_pile_and_the_rim():
    for wall in (0.06, 0.08, 0.10):
        layout = sl.Layout(depth=0.10, wall=wall)
        cube_bottom = layout.travel_z - 0.015
        assert cube_bottom >= layout.floor_z + 0.09 + 0.005 - 1e-9
        assert cube_bottom >= layout.rim_z + 0.015 - 1e-9


def test_committed_search_scene_loads_in_the_base_env():
    env = SO101Env(scene_path=SEARCH_SCENE_PATH, height=64, width=64)
    try:
        assert sl.Layout.from_model(env.model).n_cubes == len(cube_names(env.model))
        assert SEARCH_SCENE_PATH.read_text() == sl.scene_xml(sl.OBJECTS_PATH.name)
    finally:
        env.close()


def test_gray_cubes_have_their_own_geom_group():
    model = sl.load(sl.Layout(n_cubes=5))
    assert model.geom_group[model.geom("cube_red").id] != sl.GRAY_GROUP
    assert all(model.geom_group[model.geom(f"cube_gray_{i}").id] == sl.GRAY_GROUP for i in range(4))
