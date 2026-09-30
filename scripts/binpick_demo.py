"""Greedy scripted policy for the bin-picking task, with an mp4 of both cameras.

Place whatever can be grasped. When nothing can, push something until it can. That is the whole policy,
and it is deliberately not clever: it exists to show the primitives work and to give a baseline number
for how often they do.

    python scripts/binpick_demo.py --out outputs/binpick.mp4 --episodes 3

The policy reads exact object poses out of the simulator. That is fine for a planner and useless for a
policy you intend to deploy, which is why the env keeps those poses in `info` and out of the observation.
"""

import argparse
from pathlib import Path

import imageio
import numpy as np

from so101sim import binpick, primitives
from so101sim.env import CONTROL_HZ

# A push moves a cube 26 to 56 mm for an 80 mm stroke, because contact starts partway along. Used only to
# guess where a cube will end up when choosing a heading.
PUSH_GAIN = 0.045
PUSH_BACKOFF = 0.045
MAX_PICK_ATTEMPTS = 2
# Carrying a cube, top-down reach is an annulus that gives out around here. Measured, see objects.xml.
CARRY_REACH = 0.27
SEPARATION = 2 * primitives.CUBE_HALF + 0.018


def held(env):
    for name in env.cubes:
        if env.is_grasped(name):
            return name
    return None


def in_sector(xy, pad=0.0):
    r = float(np.hypot(*xy))
    a = float(np.arctan2(xy[1], xy[0]))
    return (binpick.SPAWN_RADIUS[0] + pad <= r <= binpick.SPAWN_RADIUS[1] - pad
            and binpick.SPAWN_ANGLE[0] <= a <= binpick.SPAWN_ANGLE[1])


def loose_cubes(env):
    return [n for n in env.cubes if not env.cube_pose(n)["in_bin"]]


def choose_push(env, skip=(), stuck=None):
    """Heading that drives some stuck cube furthest from its neighbours.

    Tries to keep the cube inside the spawn sector first. If no heading manages that, which happens when
    a jammed pair has drifted to the edge, it settles for anywhere the arm can still reach.
    """
    for strict in (True, False):
        best = None
        others = loose_cubes(env)
        for name in others:
            if env.graspable(name) and (stuck is None or name not in stuck):
                continue
            here = env.cube_pose(name)["pos"][:2]
            neighbours = [env.cube_pose(o)["pos"][:2] for o in others if o != name]
            for heading in primitives.push_directions():
                if (name, round(float(heading), 3)) in skip:
                    continue
                step = np.array([np.cos(heading), np.sin(heading)])
                landing = here + PUSH_GAIN * step
                start = here - PUSH_BACKOFF * step
                if strict and not (in_sector(landing) and in_sector(start)):
                    continue
                if not strict and not (reachable(landing) and reachable(start)):
                    continue
                if np.abs(landing - env.bin["center"]).max() - env.bin["half"] < 0.03:
                    continue
                score = min((np.linalg.norm(landing - n) for n in neighbours), default=1.0)
                if best is None or score > best[0]:
                    best = (score, name, float(heading), start)
        if best is not None:
            return best
    return None


def reachable(xy, lo=0.14, hi=0.28):
    """Crude stand-in for the measured annulus, used only to reject hopeless push targets."""
    return lo <= float(np.hypot(*xy)) <= hi


def drop_spot(env):
    """Somewhere in the bin clear of what is already in there.

    Dropping every cube on the bin centre stacks them, and a cube falling 16 mm onto another one bounces,
    sometimes over a rim that is shorter than the cube is tall. Spreading them out is most of the fix.
    """
    occupied = [env.cube_pose(n)["pos"][:2] for n in env.cubes if env.cube_pose(n)["in_bin"]]
    if not occupied:
        return env.bin["center"]
    inset = env.bin["half"] - primitives.CUBE_HALF - 0.004
    best, best_gap = env.bin["center"], -1.0
    for dx in np.linspace(-inset, inset, 5):
        for dy in np.linspace(-inset, inset, 5):
            spot = env.bin["center"] + np.array([dx, dy])
            # The far corner of the bin is often the emptiest spot and also outside the arm's carrying
            # reach, so candidates get filtered before they get scored.
            if not reachable(spot, hi=CARRY_REACH):
                continue
            gap = min(np.linalg.norm(spot - o) for o in occupied)
            if gap > best_gap:
                best, best_gap = spot, gap
    return best


def next_action(env, skip=(), pick_fails=None):
    """(action, note) for the current state, or (None, reason) when there is nothing sensible to do."""
    pick_fails = pick_fails or {}
    carrying = held(env)
    if carrying is not None:
        return ("place", drop_spot(env), 0.0), f"place {carrying}"

    loose = loose_cubes(env)
    if not loose:
        return None, "table clear"

    # A cube the jaws keep sliding off is not going to become pickable by trying again. Push it instead
    # and let the changed pose reset its count.
    free = [n for n in loose if env.graspable(n) and pick_fails.get(n, 0) < MAX_PICK_ATTEMPTS]
    if free:
        name = min(free, key=lambda n: np.hypot(*env.cube_pose(n)["pos"][:2]))
        pose = env.cube_pose(name)
        return ("pick", pose["pos"][:2], pose["yaw"]), f"pick {name}"

    push = choose_push(env, skip, stuck={n for n, c in pick_fails.items() if c >= MAX_PICK_ATTEMPTS})
    if push is None:
        return None, "everything jammed and no push available"
    _, name, heading, start = push
    return ("push", start, heading), f"push {name} at {np.degrees(heading):.0f} deg"


def run_episode(env, seed, frames=None, verbose=True):
    env.reset(seed=seed)
    skip = set()
    pick_fails = {}
    where = {n: env.cube_pose(n)["pos"][:2].copy() for n in env.cubes}
    log = []
    while True:
        action, note = next_action(env, skip, pick_fails)
        if action is None:
            break
        _, reward, terminated, truncated, info = env.step(action)
        target = note.split()[1]
        if info["failure"]:
            note += "  UNREACHABLE"
            if action[0] == "push":
                skip.add((target, round(float(action[2]), 3)))
        if action[0] == "pick":
            if held(env) is None:
                pick_fails[target] = pick_fails.get(target, 0) + 1
                note += f"  MISSED ({pick_fails[target]})"
            else:
                pick_fails.pop(target, None)
        # Anything that actually moved is a new situation, so stop holding its past against it.
        for n in env.cubes:
            now = env.cube_pose(n)["pos"][:2]
            if np.linalg.norm(now - where[n]) > 0.005:
                pick_fails.pop(n, None)
            where[n] = now.copy()
        log.append((note, reward, info["n_in_bin"]))
        if verbose:
            print(f"    {note:34s} reward {reward:.0f}  in bin {info['n_in_bin']}/{info['n_cubes']}  "
                  f"{info['frames']} frames")
        if terminated or truncated:
            break
    return info["n_in_bin"], info["n_cubes"], log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("outputs/binpick.mp4"))
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    frames = None if args.no_video else []
    env = binpick.BinPickEnv(height=240, width=320)
    if frames is not None:
        env.on_frame = lambda e: frames.append(
            np.concatenate([e.env._render("front"), e.env._render("wrist")], axis=1)
        )

    cleared = 0
    total = 0
    for i in range(args.episodes):
        print(f"episode {i} (seed {args.seed + i})")
        n, of, _ = run_episode(env, args.seed + i)
        print(f"  -> {n}/{of} in the bin")
        cleared += n
        total += of
    env.close()

    print(f"\n{cleared}/{total} cubes binned over {args.episodes} episodes")
    if frames:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(args.out, frames, fps=CONTROL_HZ, macro_block_size=1)
        print(f"wrote {len(frames)} frames to {args.out}")


if __name__ == "__main__":
    main()
