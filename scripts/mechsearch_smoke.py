"""Reset the mechanical-search env many times: starting piles on video, reset cost, and the cube count.

    python scripts/mechsearch_smoke.py --episodes 8 --out outputs/mechsearch.mp4
    python scripts/mechsearch_smoke.py --episodes 100 --no-video
    python scripts/mechsearch_smoke.py --tune-count --write

Acceptance is the share of dropped piles that reset keeps, 1 / mean attempts. The spec wants at least half
kept for buried starts, and a reset under 3 s on average; partial starts keep about 12% by nature and are
judged by the reset time alone. --tune-count starts at 30 cubes
and takes grays away until acceptance passes, and --write records the count in models/search_objects.xml.
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
from so101sim import SEARCH_SCENE_PATH, search_layout as sl
from so101sim.env import CONTROL_HZ
from so101sim.mechsearch import MAX_RESET_ATTEMPTS, MechSearchEnv

import mujoco

START_COUNT = 30
MIN_COUNT = 10
MIN_ACCEPTANCE = 0.5
RESET_BUDGET_S = 3.0
HOLD_S = 1.0

_ENVS = {}


def _measure(job):
    """One reset in a worker process, reusing that worker's env for the same scene."""
    scene, seed, buried = job
    if scene not in _ENVS:
        for old in _ENVS.values():
            old.close()
        _ENVS.clear()
        _ENVS[scene] = MechSearchEnv(scene_path=scene)
    started = time.perf_counter()
    try:
        _, info = _ENVS[scene].reset(seed=seed, options={"buried": buried})
        attempts = info["reset_attempts"]
    except RuntimeError:
        attempts = MAX_RESET_ATTEMPTS
    return buried, time.perf_counter() - started, attempts


def measure(scene, seeds, workers):
    jobs = [(str(scene), seed, buried) for buried in (True, False) for seed in seeds]
    with mp.get_context("spawn").Pool(workers) as pool:
        rows = pool.map(_measure, jobs, chunksize=2)
    stats = {}
    for buried, label in ((True, "buried"), (False, "partial")):
        mine = [r for r in rows if r[0] == buried]
        stats[label] = {"acceptance": len(mine) / sum(r[2] for r in mine),
                        "reset_s": float(np.mean([r[1] for r in mine]))}
    return stats


def report(stats):
    return "; ".join(f"{label} kept {s['acceptance']:.0%}, {s['reset_s']:.1f} s per reset"
                     for label, s in stats.items())


def video(args):
    env = MechSearchEnv()
    frames = []
    try:
        for episode in range(args.episodes):
            obs, info = env.reset(seed=episode, options={"buried": episode % 2 == 0})
            print(f"episode {episode}: {'buried' if info['buried'] else 'partial'}, red pixels "
                  f"{info['target_pixels']}, {info['reset_attempts']} attempts")
            frame = np.concatenate([obs["observation.images.front"], obs["observation.images.wrist"]], axis=1)
            frames += [frame] * int(HOLD_S * CONTROL_HZ)
    finally:
        env.close()
    imageio.mimwrite(args.out, frames, fps=CONTROL_HZ, macro_block_size=1)
    print(f"wrote {len(frames)} frames to {args.out}")


def tune(args):
    committed = sl.Layout.from_model(mujoco.MjModel.from_xml_path(str(SEARCH_SCENE_PATH)))
    for count in range(START_COUNT, MIN_COUNT - 1, -1):
        layout = dataclasses.replace(committed, n_cubes=count)
        with sl.candidate_scene(layout) as scene:
            stats = measure(scene, range(args.seeds), args.workers)
        print(f"{count} cubes: {report(stats)}", flush=True)
        # Buried starts must keep half their piles. Partial starts keep far fewer by nature, about 12%, so they
        # are judged by the reset budget instead (spec, third amendment).
        if stats["buried"]["acceptance"] >= MIN_ACCEPTANCE and stats["partial"]["reset_s"] <= RESET_BUDGET_S:
            break
    else:
        sys.exit(f"STOP: no count down to {MIN_COUNT} keeps half the buried piles with partial resets under "
                 f"{RESET_BUDGET_S:.0f} s. Bring this back to dm1487.")
    slow = [label for label, s in stats.items() if s["reset_s"] > RESET_BUDGET_S]
    if slow:
        print(f"REPORT TO dm1487: reset averages over {RESET_BUDGET_S:.0f} s for {', '.join(slow)} starts. "
              "The spec says report it, not cut cubes to meet it.")
    if args.write:
        notes = [n for n in sl.read_notes() if not n.startswith("Cube count")]
        notes.append(f"Cube count {count}, the most with at least half of piles kept over {args.seeds} "
                     f"seeds each: {report(stats)}.")
        sl.write(layout, notes)
        print(f"wrote {sl.OBJECTS_PATH}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("outputs/mechsearch.mp4"))
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
    print(f"over {args.episodes} seeds: {report(measure(SEARCH_SCENE_PATH, range(args.episodes), args.workers))}")


if __name__ == "__main__":
    main()
