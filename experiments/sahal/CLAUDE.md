# experiments/sahal: rules for this folder

Sahal's studies on the shelf task. Never modify anything outside `experiments/sahal/`; the rest of the repo is
dm1487's. Subclass or wrap repo code here instead.

## Layout

- `common/`: anything two studies use. `shelf_env.py` (SingleRedShelf, placing the red, the shared push run),
  `measures.py` (grasp conditions, tilt outcome, contacts, clearance, glitch detection), `plotting.py` (colours,
  shelf outline, maps), `runner.py` (parallel runs, CSV, config.json), `retrieval_check.py` (can the red be picked and pulled
  now: hybrid, exact, kinematic), `scenes.py` (scenes of the red and grays as JSON in `scenes/`, study 01's graspable zone), `cylinder_size.py`
  (other cylinder sizes without repo changes), `random_sweeps.py` (random scenes and random single sweeps, studies 10
  and 11), `sweep_videos.py` (demo videos of saved trials; they go in `videos/`, the only mp4s kept in git).
- `skills/`: new skills built on the repo's primitives, reusing its IK, retiming and path check (`sweep.py`).
- `studies/NN_name/`: `run.py` (the only entry point, settings at the top, `--quick`), `README.md` (question,
  method, how to run, results, caveats), `outputs/` (CSVs, PNGs, summary.json, config.json).

## Coding rules

- Named constants with units in the name (`SWEEP_HEIGHT_M`, `GLITCH_SPEED_M_S`); no magic numbers. A study's settings
  sit at the top of its `run.py`.
- Short functions with docstrings and type hints; clear names, no abbreviations.
- Anything used by two studies lives in `common/`. No copy-paste between studies.
- Fixed seeds. Every job carries everything that decides its result, so results don't depend on the worker or run
  order. Save `config.json` next to every output (`runner.save_config`).
- Delete dead code, old duplicates and scratch files; no unused imports.
- Ask before installing packages. The venv is `experiments/sahal/.venv` (native arm64; an x86 Python under Rosetta
  can't import mujoco).

## Measuring

- Flag simulator glitches with `measures.is_glitch` and exclude them from topple counts and means: arm-cylinder
  contact force over 100 N (1000 N while the jaws squeeze the cylinder, which normally reaches 147 N) or the
  cylinder faster than 1 m/s. Glitches seen so far reached 428 N to 44 kN, or 6.9 to 46 m/s.
- Near walls, check `measures.is_wall_jam` first: the arm on a side wall, the back wall or the top board, or over
  100 N while the red touches one, is a jam against a wall. It is a failure, never a glitch (study 05: 101 to 1404 N).
- Read contacts every physics substep (`SingleRedShelf` does). A 30 Hz frame-end sample misses most push contacts.
- Pushes are chaotic: a float32 instead of float64 interpolation of the same trajectory flipped outcomes. Report
  rates over jittered repeats, not single runs.
- Numbers in READMEs and HANDOFF.md come from the output files, never estimates.
