"""Render an episode to video: the front and wrist cameras side by side at 30 Hz, captioned with the move under way.

    python demonstrator/video.py d2000 --out outputs/videos/d2000.mp4 [--size 480]

The episode is replayed from its record, as tools.py rebuilds it, so the video shows exactly the demonstration.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import tools  # noqa: E402  sets up so101sim before mujoco
import imageio  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402


def caption(rec, index):
    ops = rec["ops"]
    if index is None or index >= len(ops):
        return "reset"
    op = ops[index]
    if op["op"] != "act":
        return op["op"]
    move = sum(1 for o in ops[:index + 1] if o["op"] == "act")
    res = op.get("result", {})
    tail = "  SOLVED" if res.get("success") else ""
    return f"move {move}: {op['kind']} {tools.fmt(op['xyz'])} {op['deg']:.0f} deg{tail}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=480)
    a = ap.parse_args()
    frames = []
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", max(12, a.size // 30))
    except OSError:
        font = ImageFont.load_default()

    def on_advance(ep, row):
        obs = ep.env._obs()
        both = np.concatenate([obs["observation.images.front"], obs["observation.images.wrist"]], axis=1)
        img = Image.fromarray(both)
        draw = ImageDraw.Draw(img)
        text = f"{ep.name}  seed {ep.rec['seed']}  |  {caption(ep.rec, ep.op_index)}  |  robot {ep.robot_s:.1f} s"
        draw.rectangle((0, 0, img.width, a.size // 14), fill=(0, 0, 0))
        draw.text((8, 4), text, fill=(255, 255, 255), font=font)
        frames.append(np.asarray(img))

    ep = tools.Episode(a.name, size=a.size, on_advance=on_advance)
    outcome = ep.outcome()
    ep.close()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(a.out, frames, fps=30, macro_block_size=1)
    print(f"{a.out}: {len(frames)} frames, {len(frames) / 30:.1f} s, outcome {outcome}")


if __name__ == "__main__":
    main()
