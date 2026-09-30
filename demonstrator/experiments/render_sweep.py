"""Before/after pictures and a short video for chosen sweeps. Throwaway."""
import json
import sys

import numpy as np

SCR = "/common/home/dm1487/robotics_research/rearrange/demonstrator/experiments"
sys.path.insert(0, SCR)
import sweep_test as T  # noqa: E402

sys.path.insert(0, T.REPO + "/demonstrator")
import so101sim  # noqa: E402,F401
import mujoco  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from so101sim import primitives  # noqa: E402
from so101sim.mechsearch import TARGET  # noqa: E402

W = H = 480


class TopView:
    def __init__(self, model):
        self.m = model
        self.r = mujoco.Renderer(model, H, W)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat[:] = [0.20, 0.0, -0.07]
        self.cam.distance = 0.40
        self.cam.elevation = -90
        self.cam.azimuth = 0  # image up = +x (far wall), base at the bottom

    def render(self, data, arm_aside=False):
        """Top view. With arm_aside, render a copy of the scene with the arm swung to the side, for stills only."""
        if arm_aside:
            d = mujoco.MjData(self.m)
            d.qpos[:] = data.qpos
            d.qpos[self.m.jnt_qposadr[self.m.joint("shoulder_pan").id]] = np.radians(110)
            mujoco.mj_forward(self.m, d)
            data = d
        self.r.update_scene(data, camera=self.cam)
        return self.r.render().copy()

    def project(self, p):
        """Looking straight down with azimuth 0: +x is up in the image and +y is left."""
        f = (H / 2) / np.tan(np.radians(self.m.vis.global_.fovy) / 2)
        cam_z = self.cam.lookat[2] + self.cam.distance
        depth = cam_z - p[2]
        dx, dy = p[0] - self.cam.lookat[0], p[1] - self.cam.lookat[1]
        return W / 2 - f * dy / depth, H / 2 - f * dx / depth


def front(prim):
    r = mujoco.Renderer(prim.model, H, W)
    r.update_scene(prim.env.data, camera="front")
    img = r.render().copy()
    r.close()
    # crop to the bin, as observe does
    return np.asarray(Image.fromarray(img).crop((170, 110, 470, 335)).resize((W, int(W * 225 / 300))))


def label(img, text, font):
    im = Image.fromarray(img).convert("RGB")
    ImageDraw.Draw(im).rectangle((0, 0, im.width, 26), fill=(0, 0, 0))
    ImageDraw.Draw(im).text((8, 4), text, fill=(255, 255, 255), font=font)
    return im


def main():
    rows = json.load(open(f"{T.OUT}/sweep_rows.json"))
    picks = [(1006, "mid", 0), (1012, "rim", 90), (1012, "mid", 180)]
    font = ImageFont.load_default(size=16)
    for seed, kind, deg in picks:
        row = next(r for r in rows if r["seed"] == seed and r["kind"] == kind and r["deg"] == deg)
        prim = T.setup(seed)
        env, m = prim.env, prim.model
        top = TopView(m)
        red = env.cube_pose(TARGET)["pos"]
        before_top = top.render(env.data, arm_aside=True)
        # check the projection: the highest gray's top centre should land on that cube's pixels
        hi = max(env.grays, key=lambda n: env.cube_pose(n)["pos"][2])
        seg_r = mujoco.Renderer(m, H, W)
        seg_r.enable_segmentation_rendering()
        dd = mujoco.MjData(m); dd.qpos[:] = env.data.qpos
        dd.qpos[m.jnt_qposadr[m.joint("shoulder_pan").id]] = np.radians(110)
        mujoco.mj_forward(m, dd)
        seg_r.update_scene(dd, camera=top.cam)
        seg = seg_r.render()
        seg_r.close()
        hp = top.project(env.cube_pose(hi)["pos"] + np.array([0, 0, 0.015]))
        got = seg[int(round(hp[1])), int(round(hp[0])), 0]
        assert got == m.geom(hi).id, f"projection off: {hi} at {hp} shows geom {got}"
        before_front = front(prim)
        start = np.array(row["start"])
        u = np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg))])
        end = start + row["stroke"] * u
        a, b = top.project([*start, row["z"]]), top.project([*end, row["z"]])
        rp = top.project(red)
        T.unpark(prim)
        traj = primitives.push(m, row["start"], np.radians(deg), stroke=row["stroke"], hover_z=prim.travel_z,
                               push_z=row["z"], flip=row["flip"])
        traj = primitives.bridge(m, env.joint_state, traj, hover_z=prim.travel_z)
        frames, failure = [], None
        for k, fr in enumerate(traj):
            _, term, _, info = env.advance(fr)
            if k % 2 == 0:
                frames.append(top.render(env.data))
            if term:
                failure = info.get("failure")
                break
        if failure is None:
            T.park(prim)
        after_top = top.render(env.data, arm_aside=True)
        after_front = front(prim)
        state = T.red_state(env) if failure is None else None

        # compose: 2 rows (before, after) x 2 cols (top, front)
        panels = []
        for img, txt, draw in ((before_top, "before, from above (base at the bottom)", True),
                               (before_front, "before, front camera", False),
                               (after_top, "after, from above", False),
                               (after_front, "after, front camera", False)):
            im = label(img, txt, font)
            d = ImageDraw.Draw(im)
            if draw:
                d.line([a, b], fill=(255, 60, 60), width=5)
                ang = np.arctan2(b[1] - a[1], b[0] - a[0])
                for s in (0.5, -0.5):
                    d.line([b, (b[0] - 18 * np.cos(ang + s), b[1] - 18 * np.sin(ang + s))], fill=(255, 60, 60),
                           width=5)
                d.ellipse((rp[0] - 16, rp[1] - 16, rp[0] + 16, rp[1] + 16), outline=(255, 230, 0), width=3)
            panels.append(im.resize((W, H)))
        sheet = Image.new("RGB", (2 * W, 2 * H + 60), (255, 255, 255))
        for i, im in enumerate(panels):
            sheet.paste(im, ((i % 2) * W, 60 + (i // 2) * H))
        dsh = ImageDraw.Draw(sheet)
        where = "rim, 8 mm above it, 12 cm" if kind == "rim" else "layer-2 height, 9 cm"
        outcome = (f"cubes on the red {row['over_side_before'][0]} -> {state[0]}, flush neighbours "
                   f"{row['over_side_before'][1]} -> {state[1]}, {row['cubes_moved']} cubes moved"
                   if state else f"SPILL: {failure}")
        dsh.text((8, 6), f"seed {seed}: one sweep from the {where}, heading {deg} deg (red arrow). "
                         f"Yellow ring: where the hidden red is.", fill=(0, 0, 0), font=font)
        dsh.text((8, 32), outcome, fill=(160, 0, 0) if not state else (0, 110, 0), font=font)
        out = f"{T.OUT}/sweep_{seed}_{kind}_{deg}.png"
        sheet.save(out)
        imageio.mimsave(f"{T.OUT}/sweep_{seed}_{kind}_{deg}.mp4", frames, fps=15)
        print(out, outcome)
        top.r.close()
        prim.close()


if __name__ == "__main__":
    main()
