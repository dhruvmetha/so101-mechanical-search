import sys
from pathlib import Path

import numpy as np
import pytest

import so101sim  # noqa: F401
from so101sim.env import CAMERAS
from so101sim.shelf import TARGET, ShelfPrimitiveEnv

import mujoco  # after so101sim, which sets MUJOCO_GL

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import shelf_play  # noqa: E402

W, H = 640, 480


@pytest.fixture(scope="module")
def prim():
    e = ShelfPrimitiveEnv()  # 640 x 480, the size the planner sees
    yield e
    e.close()


@pytest.mark.parametrize("camera", list(CAMERAS.values()))
def test_project_and_pixel_ray_are_inverses(prim, camera):
    prim.reset(seed=0)
    point = np.array([0.40, 0.03, prim.env.layout.grasp_z])
    u, v = shelf_play.project(prim.model, prim.data, camera, point, W, H)
    origin, d = shelf_play.pixel_ray(prim.model, prim.data, camera, u, v, W, H)
    t = (point - origin) @ d
    assert np.linalg.norm(origin + t * d - point) < 1e-3


def test_the_front_lens_matches_the_renderer(prim):
    """A standing cylinder's axis projects to a vertical line, so its silhouette's horizontal centre is the axis's
    u. The front camera sits 11 cm over the board and sees it edge-on, so board markers can't check it: 1 px of
    height there is several mm on the board."""
    offsets = []
    for seed in range(3):
        prim.reset(seed=seed, options={"hidden": False})
        seg = prim.env._seg
        seg.update_scene(prim.data, camera="front")
        ids = seg.render()
        for name in prim.env.cyls:
            v, u = np.nonzero((ids[..., 0] == prim.env._geom[name]) & (ids[..., 1] == 5))
            if len(u) >= 400:
                pu, _ = shelf_play.project(prim.model, prim.data, "front", prim.env.cylinder_pose(name)["pos"], W, H)
                offsets.append(pu - u.mean())
    assert len(offsets) >= 15
    assert abs(np.median(offsets)) < 1.5, np.median(offsets)


def test_the_wrist_lens_matches_the_renderer():
    """Deproject the pixel centroid of each board marker the wrist camera sees onto the board plane and compare with
    where the marker is. Markers are 6 mm tiles, so their centroid is their centre. The wrist camera renders its
    16:9 sensor stretched over 640 x 480; square pixels would put it off by centimetres."""
    camera = "wrist_cam"
    from so101sim import SHELF_SCENE_PATH, shelf_layout as sl
    from so101sim.search_checks import set_pose
    layout = sl.ShelfLayout.from_model(mujoco.MjModel.from_xml_path(str(SHELF_SCENE_PATH)))
    model = sl.load(layout, probes=True)
    data = mujoco.MjData(model)
    set_pose(model, data, layout.home)
    renderer = mujoco.Renderer(model, H, W)
    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera=camera)
    seg = renderer.render()
    renderer.close()
    errors = []
    for g in range(model.ngeom):
        if not model.geom(g).name.startswith(sl.MARKER_PREFIX):
            continue
        v, u = np.nonzero((seg[..., 0] == g) & (seg[..., 1] == 5))
        if len(u) < 6:
            continue
        origin, d = shelf_play.pixel_ray(model, data, camera, u.mean(), v.mean(), W, H)
        z = data.geom_xpos[g][2]
        hit = origin + ((z - origin[2]) / d[2]) * d
        errors.append(np.linalg.norm(hit[:2] - data.geom_xpos[g][:2]))
    assert len(errors) > 50
    assert np.median(errors) < 0.003, np.median(errors)


def test_a_pick_pixel_lands_on_the_cylinder_axis(prim):
    prim.reset(seed=0, options={"hidden": False})
    red = prim.env.cylinder_pose(TARGET)["pos"]
    eye = prim.data.cam_xpos[prim.model.camera("front").id]
    toward = (eye - red)[:2] / np.linalg.norm((eye - red)[:2])
    face = np.r_[red[:2] + 0.015 * toward, prim.env.layout.grasp_z]  # the face point nearest the camera
    u, v = shelf_play.project(prim.model, prim.data, "front", face, W, H)
    xy = shelf_play.deproject(prim, {"kind": "pick", "cam": "front", "px": [u, v]}, W, H)
    assert np.linalg.norm(xy - red[:2]) < 0.012


def test_wrist_pixels_deproject_from_the_current_pose(prim):
    prim.reset(seed=0, options={"hidden": False})
    red = prim.env.cylinder_pose(TARGET)["pos"]
    prim.step(("look", red[:2] + [0.0, 0.06], 0.0))  # the wrist has moved since reset
    spot = np.array([red[0], red[1], prim.env.layout.board_z])
    u, v = shelf_play.project(prim.model, prim.data, "wrist_cam", spot, W, H)
    xy = shelf_play.deproject(prim, {"kind": "place", "cam": "wrist", "px": [u, v]}, W, H)
    assert np.linalg.norm(xy - red[:2]) < 0.005


def test_a_pixel_off_the_board_is_refused(prim):
    prim.reset(seed=0)
    action = shelf_play.to_env_action(prim, {"kind": "pick", "cam": "front", "px": [320, 5]})
    _, _, terminated, _, info = prim.step(action)
    assert info["refused"] and not terminated


def test_grid_image_keeps_the_size():
    img = np.zeros((H, W, 3), np.uint8)
    assert shelf_play.grid_image(img).shape == (H, W, 3)
