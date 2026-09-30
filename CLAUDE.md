# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. It loads in
every session, in every subagent except Explore and Plan, and again after each compaction, so it holds only what
every session needs. Everything else lives in `docs/knowledge/`, and its README says where each kind of fact goes.

## What this is

The SO-101 arm (TheRobotStudio / Hugging Face lerobot arm) simulated in MuJoCo on arrakis, headless. There is
no physical arm on this machine. The end use is still undecided: VLA fine-tuning or RL. The task was
decided on 2026-09-25: mechanical search. A 22 cm bin sunk 10 cm below the base holds a pile of 29 gray cubes
and one red one, up to three layers deep, and the robot has to find the red cube and lift it out without any
cube leaving the bin. Spec: `docs/superpowers/specs/2026-09-25-mechanical-search-design.md`, including two
amendments made after measurements. `BinPickEnv`, which moves loose cubes into a small bin, came first and
still works. A second search task was added on 2026-09-26: a shelf bay of 14 upright 3 by 8 cm cylinders, one red,
reached sideways with the gripper level. Nothing may topple, no gray may leave the board, and the arm may not touch
the shelf. Spec: `docs/superpowers/specs/2026-09-26-shelf-search-design.md`, with three amendments.

Design spec: `docs/superpowers/specs/2026-09-18-so101-sim-design.md`.

The cube bin is a proof of concept. The expert demonstrator should later handle a family of containers: deeper bins up
to sink-like basins, bins set into the table with the rim flush, and contents from cubes to YCB objects and toys.

## Memory: boot and write-back

- At startup, resume, `/clear` and after every compaction, `.claude/hooks/knowledge.sh` puts
  `docs/knowledge/state.md`, the knowledge index and the latest journal headings into context. Read them before acting.
  If they aren't in context, read those files yourself.
- Write back as you go, not at the end. When anything starts, finishes, gets decided or gets blocked, edit `state.md`
  and append an entry to `docs/knowledge/journal.md`. A lesson goes to its one home, in general form, and anywhere
  else gets a pointer. Correct a wrong entry where it lives and date the correction.
- Log every decision in the journal, dm1487's and Claude's, with what, why, who and the evidence. A run's entry names
  its commit, seeds, rulebook version and commands, so another person with an LLM can replicate the demos.
- Right after a compaction, compare the summary with `state.md` and write back whatever it lacks before new work.
- Subagents write only their own output files and return a summary. The main session merges results into
  `docs/knowledge/`.
- When the boot checks report a budget over its limit, consolidate: `docs/knowledge/README.md`, "Consolidation".

## How to work here

In short, from `docs/knowledge/operating-principles.md`:

- Rebuild the picture from the files, not the last message. Check the available skills and use the ones that fit.
- Judge each step by the current goal (in `state.md`). As of 2026-09-30 it is an expert that makes good demos fast.
- Measure before asserting, and check rules against geometry before writing them down.
- Budget long or parallel runs (usage limits, recovery), freeze tools for a batch, one unit of work per agent.
- Put recurring rules into the tools as checks, not more prose.

## Commands

The env is a conda prefix, not a named env:

```bash
conda activate /common/users/dm1487/envs/so101sim
pip install -e ".[mjx]"          # already done; the mjx extra adds jax 0.6.2, mujoco-mjx 3.8.1, warp-lang

python -m pytest -q -m "not mjx"  # CPU suite, about 4 minutes, most of it search resets
python -m pytest -q               # adds the MJX test: ~10 s with Warp's kernel cache warm, ~2 min cold
python -m pytest tests/test_env.py::test_held_targets_settle_within_one_degree -q

python scripts/smoke.py --out outputs/smoke.mp4   # arm hovers over each cube and the bin, both cameras side by side
python scripts/binpick_demo.py --episodes 10 --no-video   # scripted push/pick/place baseline, ~29/30 cubes binned
python scripts/binpick_demo.py --out outputs/binpick.mp4  # same, with video
python scripts/size_search_bin.py --write          # re-pick the search bin, riser, walls, start pose, camera
python scripts/mechsearch_smoke.py --episodes 8    # starting piles from both cameras, reset cost, acceptance
python scripts/mechsearch_smoke.py --tune-count --write   # re-pick the cube count
python scripts/size_shelf.py --write               # re-pick the shelf, start pose and front camera
python scripts/shelf_smoke.py --episodes 8         # starting shelves on video, reset cost, acceptance
python scripts/shelf_smoke.py --tune-count --write # re-pick the cylinder count
python scripts/shelf_play.py --seed 300 --start hidden --actions A.json --out DIR [--video]  # pixel actions, replayed
```

arrakis has 5 GPUs shared with other people's jobs. Check `nvidia-smi` and pin work to a free card.
`MUJOCO_EGL_DEVICE_ID` picks the GPU MuJoCo renders on. `CUDA_VISIBLE_DEVICES` picks it for jax and torch but
does not affect EGL. jax grabs 75% of a card's memory on first use, so set
`XLA_PYTHON_CLIENT_PREALLOCATE=false` unless you mean to own the card.

## Map

One line per part. `docs/knowledge/codebase.md` has what each one does, its conventions and its measured numbers.

- `models/`: `stage.xml` holds the shared scene pieces and each task has its own objects and scene XML. Never edit
  `models/robotstudio_so101/` (menagerie's model, vendored) or the generated `search_*.xml` and `shelf_*.xml`.
- `so101sim/units.py`: lerobot units outside the env, radians inside, joints in `units.JOINT_NAMES` order.
- `so101sim/env.py`: `SO101Env`, the 30 Hz Gymnasium env the task envs build on.
- `so101sim/ik.py`: `solve_ik`, `solve_topdown` (the bin) and `solve_side` (the shelf).
- `so101sim/primitives.py` and `side_primitives.py`: primitives as 30 Hz joint trajectories, top-down and level.
- `so101sim/binpick.py`: `BinPickEnv`, with `scripts/binpick_demo.py` as its scripted baseline.
- `so101sim/search_layout.py`, `search_checks.py`, `mechsearch.py`: the bin task's scene, sizing and envs.
- `so101sim/shelf_layout.py`, `shelf_checks.py`, `shelf.py`: the same for the shelf task.
- `so101sim/grasp.py` and `rest.py`: contact checks and the at-rest rule both tasks share.
- `so101sim/mjx.py`: the scene on the GPU with MuJoCo Warp. Always go through `so101sim.mjx.put`.
- `demonstrator/`: Claude playing the bin task from camera data. `tools.py` is the CLI, `instructions.md` the rulebook.

## Traps for every script

- `import so101sim` before `mujoco` in scripts. The package sets `MUJOCO_GL=egl`, and MuJoCo reads it at
  import time.
- Call `env.close()`. Otherwise the EGL context gets freed during interpreter shutdown and prints
  `EGLError` tracebacks.

Traps for one subsystem live in `.claude/rules/`, and Claude Code loads each when you Read a file it covers:
`scenes-and-physics.md` (scene XML, timestep, noslip, chaotic piles), `arm-and-motion.md` (IK, gripper, reach,
bridging), `gpu.md` (MJX and Warp), `bin-task.md` and `shelf-task.md`. If you read code with `cat` or `sed`, open the
matching rule file yourself before changing code in that area.

## The real arm

The physical SO-101 is on host RobotLearning, driven by a separate Claude session named `so101`. The sim deliberately
ignores things that only matter on hardware. Before anything that moves toward hardware, read
`docs/knowledge/real-arm.md`: what that session has measured, and what still blocks transfer.
