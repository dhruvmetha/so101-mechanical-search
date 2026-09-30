"""Check that a scanned-object pile replays exactly after a snapshot and restore, with and without sleeping islands.

Written 2026-09-30 for docs/reports/2026-09-30-realistic-objects.md. The demonstrator is about to try candidate moves
by saving the sim, trying one, and rolling back, so the rollback has to reproduce the pile bit for bit.

For each seed it builds the pile from scripts/objects_pile_test.py, saves it three ways, then runs the same probe
(a 0.5 s shove on one object, then 1.5 s free) from each saved copy and compares every qpos with the first run:
  getstate   mj_getState(mjSTATE_INTEGRATION) into the same MjData with mj_setState
  fresh      the same saved vector into a new MjData
  copydata   mj_copyData of the whole MjData
MuJoCo's docs say the first two capture everything unless sleeping is on, when only mj_copyData does.

    python scripts/objects_snapshot_check.py --seeds 0 1 2
    python scripts/objects_snapshot_check.py --seeds 0 1 2 --sleep 1e-3
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import objects_pile_test as pile  # noqa: E402  (imports so101sim first, so MUJOCO_GL is set)
import mujoco  # noqa: E402

SPEC = int(mujoco.mjtState.mjSTATE_INTEGRATION)
PUSH_FRAMES, FREE_FRAMES = 15, 45


def probe(model, data, obj, target):
    """Shove one object sideways for PUSH_FRAMES, then let everything run FREE_FRAMES. Returns the final qpos."""
    sub = int(round(1 / pile.FRAME_HZ / model.opt.timestep))
    body = obj.bodies[target]
    for k in range(PUSH_FRAMES + FREE_FRAMES):
        data.xfrc_applied[body, :3] = [0.0, 0.3, 0.0] if k < PUSH_FRAMES else 0.0
        mujoco.mj_step(model, data, nstep=sub)
    data.xfrc_applied[body] = 0.0
    return data.qpos.copy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--sleep", type=float, nargs="?", const=1e-3, default=0.0)
    ap.add_argument("--hulls", choices=["vhacd", "coacd"], default="vhacd")
    args = ap.parse_args()
    scene = pile.write_scene(pile.OBJECTS, args.hulls)
    for seed in args.seeds:
        model, data, obj, _ = pile.build_pile(scene, seed, args.sleep)
        asleep = int(np.sum(data.tree_asleep >= 0)) if args.sleep else 0
        size = mujoco.mj_stateSize(model, SPEC)
        saved = np.empty(size)
        mujoco.mj_getState(model, data, saved, SPEC)
        copy = mujoco.MjData(model)
        mujoco.mj_copyData(copy, model, data)
        target = int(np.argmax([data.xpos[b][2] for b in obj.bodies]))  # the object on top

        reference = probe(model, data, obj, target)
        moved = float(np.linalg.norm(reference[obj.qadr[target]:obj.qadr[target] + 3]
                                     - saved[1 + obj.qadr[target]:1 + obj.qadr[target] + 3]))

        results = {}
        mujoco.mj_setState(model, data, saved, SPEC)
        results["getstate"] = probe(model, data, obj, target)
        fresh = mujoco.MjData(model)
        mujoco.mj_setState(model, fresh, saved, SPEC)
        results["fresh"] = probe(model, fresh, obj, target)
        restored = mujoco.MjData(model)
        mujoco.mj_copyData(restored, model, copy)
        results["copydata"] = probe(model, restored, obj, target)
        diffs = {k: float(np.abs(v - reference).max()) for k, v in results.items()}
        print(f"seed {seed} sleep {args.sleep:g}: {asleep} trees asleep at the snapshot, shoved object moved "
              f"{moved * 1000:.1f} mm; max |qpos - first run| after 2 s: "
              + ", ".join(f"{k} {v:.3g}" for k, v in diffs.items()), flush=True)


if __name__ == "__main__":
    main()
