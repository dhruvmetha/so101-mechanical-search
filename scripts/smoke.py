"""Move the arm over each cube and the bin, and write both camera views side by side to an mp4.

This is a look-at-it check, not a grasp. IK is position only, so the wrist ends up wherever the solver
leaves it.

    python scripts/smoke.py --out outputs/smoke.mp4
"""

import argparse
from pathlib import Path

import imageio
import numpy as np

from so101sim import ik
from so101sim.env import CONTROL_HZ, HOME, SO101Env

HOVER = 0.08  # metres above an object's centre
MOVE_S = 1.5  # seconds per waypoint move
HOLD_S = 0.5  # seconds to sit at each waypoint


def waypoints(env):
    """(label, target xyz or None, gripper 0..100). None keeps the arm where it is."""
    out = []
    for cube in ("cube_red", "cube_green", "cube_blue"):
        above = env.data.body(cube).xpos + [0, 0, HOVER]
        out += [(f"over {cube}", above, 100.0), (f"close over {cube}", None, 0.0)]
    out.append(("over bin", env.data.body("bin").xpos + [0, 0, 0.12], 0.0))
    out.append(("open over bin", None, 100.0))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("outputs/smoke.mp4"))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    env = SO101Env()
    env.reset(seed=0)
    frames = []
    action = HOME.copy()

    def run(start, end, seconds):
        for alpha in np.linspace(0, 1, int(seconds * CONTROL_HZ)):
            step_obs, *_ = env.step((1 - alpha) * start + alpha * end)
            frames.append(np.concatenate([step_obs["observation.images.front"], step_obs["observation.images.wrist"]], axis=1))

    site = env.model.site("gripperframe").id
    for label, target, gripper in waypoints(env):
        start = action.copy()
        if target is not None:
            q, err = ik.solve_ik(env.model, target)
            action[:5] = np.degrees(q)
        action[5] = gripper
        run(start, action, MOVE_S)
        run(action, action, HOLD_S)
        if target is not None:
            miss = np.linalg.norm(env.data.site_xpos[site] - target) * 1000
            print(f"{label:24s} IK error {err * 1000:5.2f} mm, tool settled {miss:5.2f} mm from target")
        else:
            print(label)

    env.close()
    imageio.mimwrite(args.out, frames, fps=CONTROL_HZ, macro_block_size=1)
    print(f"wrote {len(frames)} frames to {args.out}")


if __name__ == "__main__":
    main()
