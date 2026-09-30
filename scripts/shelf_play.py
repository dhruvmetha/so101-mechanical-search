"""Play ShelfPrimitiveEnv from camera images, as a planner that sees only what a policy sees.

    python scripts/shelf_play.py --seed 300 --start hidden --actions outputs/shelf_play_300/actions.json \
        --out outputs/shelf_play_300 [--video]

Replays every action in the JSON list from reset, since the sim is deterministic, and writes front.png and
wrist.png of the latest observation with a labelled pixel grid. Prints what a policy gets back per step and nothing
else: refused and why, success, the failure message. Never a pose.

An action names a pixel, not a position:

    {"kind": "pick" | "place" | "push" | "pull" | "look", "cam": "front" | "wrist", "px": [u, v], "heading": k}

The ray through that pixel comes from the camera's pose and lens. The wrist camera's pose follows from the joint
angles, which the real arm knows too. Pick intersects the ray with the grasp-height plane, which lands on the
cylinder's visible face, then steps one radius further along the ray's horizontal direction to its axis. Place,
push and look intersect the board's plane.

Act on wrist pixels. The front camera sits about 11 cm over the board and sees it nearly edge-on, so a front pixel
fixes left and right well and depth badly: 1 px of height is several mm along the ray. Use the front view to decide
where to `look`, then pick, push and place from the wrist view. `heading` is one of 16 push directions, k * 22.5 degrees anticlockwise
from +x, which points away from the robot. `pull` needs no pixel.
"""

import argparse
import json
import sys
from pathlib import Path

import imageio
import numpy as np
from PIL import Image, ImageDraw

import so101sim  # noqa: F401  before mujoco, which reads MUJOCO_GL at import
from so101sim import side_primitives as sp
from so101sim.env import CAMERAS, CONTROL_HZ
from so101sim.shelf import ShelfPrimitiveEnv
from so101sim.shelf_layout import CYL_RADIUS

GRID_STEP = 40
LABEL_STEP = 80


def _lens(model, cam, width, height):
    """Pixel focal lengths and principal point.

    A camera with a sensor size, like the wrist camera, renders its sensor stretched over the image, so its pixels
    aren't square at 640 x 480. Otherwise fovy is the vertical field of view and pixels are square.
    """
    if model.cam_sensorsize[cam].any():
        fx, fy, px, py = model.cam_intrinsic[cam]
        sw, sh = model.cam_sensorsize[cam]
        return fx / sw * width, fy / sh * height, width / 2 + px / sw * width, height / 2 - py / sh * height
    f = 0.5 * height / np.tan(np.radians(model.cam_fovy[cam]) / 2)
    return f, f, width / 2, height / 2


def pixel_ray(model, data, camera, u, v, width, height):
    """World origin and unit direction of the ray through pixel (u, v), u across and v down from the top left."""
    cam = model.camera(camera).id
    fx, fy, cx, cy = _lens(model, cam, width, height)
    local = np.array([(u + 0.5 - cx) / fx, -(v + 0.5 - cy) / fy, -1.0])
    d = data.cam_xmat[cam].reshape(3, 3) @ local
    return data.cam_xpos[cam].copy(), d / np.linalg.norm(d)


def project(model, data, camera, point, width, height):
    """Pixel (u, v) where a world point lands."""
    cam = model.camera(camera).id
    fx, fy, cx, cy = _lens(model, cam, width, height)
    p = data.cam_xmat[cam].reshape(3, 3).T @ (np.asarray(point, dtype=np.float64) - data.cam_xpos[cam])
    return cx + fx * p[0] / -p[2] - 0.5, cy - fy * p[1] / -p[2] - 0.5


def deproject(prim, action, width=640, height=480):
    """Board xy for a pixel action, or None for pull or a ray that never comes down to the plane."""
    if action["kind"] == "pull":
        return None
    origin, d = pixel_ray(prim.model, prim.data, CAMERAS[action.get("cam", "front")], *action["px"], width, height)
    layout = prim.env.layout
    z = layout.grasp_z if action["kind"] == "pick" else layout.board_z
    if d[2] > -1e-6 or origin[2] <= z:
        return None
    hit = origin + ((z - origin[2]) / d[2]) * d
    if action["kind"] == "pick":
        return hit[:2] + CYL_RADIUS * d[:2] / np.linalg.norm(d[:2])
    return hit[:2]


def to_env_action(prim, action):
    """The env action for a pixel action. A pixel that misses the board becomes a point the env will refuse."""
    xy = deproject(prim, action)
    if xy is None and action["kind"] != "pull":
        xy = prim.action_space["xy"].high + 1.0
    heading = sp.push_directions()[int(action.get("heading", 0)) % sp.PUSH_DIRECTIONS]
    return {"kind": action["kind"], "xy": np.zeros(2) if xy is None else xy, "angle": heading}


def grid_image(img):
    """A copy with a line every GRID_STEP pixels and u,v labels every LABEL_STEP."""
    out = Image.fromarray(np.ascontiguousarray(img))
    draw = ImageDraw.Draw(out)
    w, h = out.size
    for u in range(0, w, GRID_STEP):
        draw.line([(u, 0), (u, h)], fill=(255, 255, 0) if u % LABEL_STEP == 0 else (120, 120, 0))
    for v in range(0, h, GRID_STEP):
        draw.line([(0, v), (w, v)], fill=(255, 255, 0) if v % LABEL_STEP == 0 else (120, 120, 0))
    for u in range(0, w, LABEL_STEP):
        for v in range(0, h, LABEL_STEP):
            draw.text((u + 2, v + 1), f"{u},{v}", fill=(255, 255, 255))
    return np.asarray(out)


def run(seed, start, actions, out, video=False):
    out.mkdir(parents=True, exist_ok=True)
    frames = []

    def record(p):
        frames.append(np.concatenate([p.env._render("front"), p.env._render("wrist")], axis=1))

    prim = ShelfPrimitiveEnv(on_frame=record if video else None)
    try:
        obs, _ = prim.reset(seed=seed, options={"hidden": start == "hidden"})
        print(f"reset: seed {seed}, {start} start, {len(actions)} actions to replay")
        for i, action in enumerate(actions):
            obs, _, terminated, truncated, info = prim.step(to_env_action(prim, action))
            where = "" if action["kind"] == "pull" else f" {action.get('cam', 'front')} px {action['px']}"
            result = f"refused ({info['refused']})" if info["refused"] else f"ran {info['frames']} frames"
            if info["is_success"]:
                result += ", SUCCESS"
            if info["failure"]:
                result += f", FAILURE: {info['failure']}"
            print(f"{i}: {action['kind']}{where}"
                  f"{' heading ' + str(action['heading']) if 'heading' in action else ''} -> {result}")
            if terminated or truncated:
                print("episode over" + (" (primitive limit)" if truncated else ""))
                break
        Image.fromarray(grid_image(obs["observation.images.front"])).save(out / "front.png")
        Image.fromarray(grid_image(obs["observation.images.wrist"])).save(out / "wrist.png")
        print(f"wrote {out / 'front.png'} and {out / 'wrist.png'}")
    finally:
        prim.close()
    if video and frames:
        path = out.parent / f"{out.name}.mp4"
        imageio.mimwrite(path, frames, fps=CONTROL_HZ, macro_block_size=1)
        print(f"wrote {len(frames)} frames to {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--start", choices=("hidden", "partial"), required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    actions = json.loads(args.actions.read_text()) if args.actions.exists() else []
    if not isinstance(actions, list):
        sys.exit(f"{args.actions} must hold a JSON list of actions")
    run(args.seed, args.start, actions, args.out, args.video)


if __name__ == "__main__":
    main()
