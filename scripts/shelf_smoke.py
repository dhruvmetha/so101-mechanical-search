"""Reset the shelf env many times: starting shelves on video, reset cost, and the cylinder count.

    python scripts/shelf_smoke.py --episodes 8 --out outputs/shelf_smoke.mp4
    python scripts/shelf_smoke.py --episodes 100 --no-video
    python scripts/shelf_smoke.py --tune-count --write

Acceptance is the share of attempted shelves that reset keeps, 1 / mean attempts. The spec wants at least half kept
for hidden and for partial starts, and a reset under 3 s on average. --tune-count starts at 18 cylinders and takes
grays away until both pass, and --write records the count in models/shelf_objects.xml.
"""

import argparse
import dataclasses
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import imageio
import numpy as np

import so101sim  # noqa: F401  before mujoco, which reads MUJOCO_GL at import
from so101sim import SHELF_SCENE_PATH, shelf_layout as sl
from so101sim.env import CONTROL_HZ
from so101sim.shelf import MAX_RESET_ATTEMPTS, ShelfEnv

import mujoco

START_COUNT = 18
MIN_COUNT = 8
MIN_ACCEPTANCE = 0.5
RESET_BUDGET_S = 3.0
HOLD_S = 1.0

_ENVS = {}


def _measure(job):
    """One reset in a worker process, reusing that worker's env for the same scene."""
    scene, seed, hidden = job
    if scene not in _ENVS:
        for old in _ENVS.values():
            old.close()
        _ENVS.clear()
        _ENVS[scene] = ShelfEnv(scene_path=scene, height=64, width=64)
    started = time.perf_counter()
    try:
        _, info = _ENVS[scene].reset(seed=seed, options={"hidden": hidden})
        attempts = info["reset_attempts"]
    except RuntimeError:
        attempts = MAX_RESET_ATTEMPTS
    return hidden, time.perf_counter() - started, attempts


def measure(scene, seeds, workers):
    jobs = [(str(scene), seed, hidden) for hidden in (True, False) for seed in seeds]
    with mp.get_context("spawn").Pool(workers) as pool:
        rows = pool.map(_measure, jobs, chunksize=2)
    stats = {}
    for hidden, label in ((True, "hidden"), (False, "partial")):
        mine = [r for r in rows if r[0] == hidden]
        stats[label] = {"acceptance": len(mine) / sum(r[2] for r in mine),
                        "reset_s": float(np.mean([r[1] for r in mine]))}
    return stats


def report(stats):
    return "; ".join(f"{label} kept {s['acceptance']:.0%}, {s['reset_s']:.1f} s per reset"
                     for label, s in stats.items())


def video(args):
    env = ShelfEnv()
    frames = []
    try:
        for episode in range(args.episodes):
            obs, info = env.reset(seed=episode, options={"hidden": episode % 2 == 0})
            print(f"episode {episode}: {'hidden' if info['hidden'] else 'partial'}, red pixels "
                  f"{info['target_pixels']}, {info['reset_attempts']} attempts")
            frame = np.concatenate([obs["observation.images.front"], obs["observation.images.wrist"]], axis=1)
            frames += [frame] * int(HOLD_S * CONTROL_HZ)
    finally:
        env.close()
    imageio.mimwrite(args.out, frames, fps=CONTROL_HZ, macro_block_size=1)
    print(f"wrote {len(frames)} frames to {args.out}")


def tune(args):
    committed = sl.ShelfLayout.from_model(mujoco.MjModel.from_xml_path(str(SHELF_SCENE_PATH)))
    for count in range(START_COUNT, MIN_COUNT - 1, -1):
        layout = dataclasses.replace(committed, n_cyls=count)
        with sl.candidate_scene(layout) as scene:
            stats = measure(scene, range(args.seeds), args.workers)
        print(f"{count} cylinders: {report(stats)}", flush=True)
        if all(s["acceptance"] >= MIN_ACCEPTANCE for s in stats.values()):
            break
    else:
        sys.exit(f"STOP: no count down to {MIN_COUNT} keeps half the hidden and half the partial shelves. "
                 "Bring this back to dm1487.")
    slow = [label for label, s in stats.items() if s["reset_s"] > RESET_BUDGET_S]
    if slow:
        print(f"REPORT TO dm1487: reset averages over {RESET_BUDGET_S:.0f} s for {', '.join(slow)} starts. "
              "The spec says report it, not cut cylinders to meet it.")
    if args.write:
        notes = [n for n in sl.read_notes() if not n.startswith("Cylinder count")]
        notes.append(f"Cylinder count {count}, the most with at least half of shelves kept over {args.seeds} "
                     f"seeds each: {report(stats)}.")
        sl.write(layout, notes)
        print(f"wrote {sl.OBJECTS_PATH}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("outputs/shelf_smoke.mp4"))
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--tune-count", action="store_true")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.tune_count:
        return tune(args)
    if not args.no_video:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        video(args)
    print(f"over {args.episodes} seeds: {report(measure(SHELF_SCENE_PATH, range(args.episodes), args.workers))}")


if __name__ == "__main__":
    main()
