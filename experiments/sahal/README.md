# Sahal's shelf studies

The shelf task's board (`so101sim/shelf.py`): the red cylinder alone (studies 01 to 06), then with two grays (07, 08).
Full simulator state, no repo files changed.
`HANDOFF.md` is the summary for Dhruv. Each study's README has its question, method, results and caveats.

| study | question |
|---|---|
| `studies/01_grasp_pull` | where can the red be grasped and pulled out, and how exact must the grasp be |
| `studies/02_sweep_height` | how low can the level gripper push, and does a pushed cylinder slide or topple |
| `studies/03_sweep_factors` | which of direction, height, arm friction, speed and board friction changes tilt most |
| `studies/04_sweep_speed` | what push speed keeps the cylinder upright, and what happens at quarter speed |
| `studies/05_sweep_skill` | does the sweep skill move a lone cylinder as told without toppling it, and where is it refused |
| `studies/06_entry_above` | is there room for the level gripper to enter above the cylinders and drop behind one |
| `studies/07_retrieval_check` | does the kinematic "can the red be retrieved now?" check agree with pick + pull in physics |
| `studies/08_scenes` | ten hand-made scenes from easy to hard: does each hand-written solution (sweeps, then retrieve) work |

Latest, in brief (details in HANDOFF.md, sections 10 and 11):

- **07, retrieval check.** `can_retrieve(env, mode)`. `"hybrid"` (default, for search): the env's own refusals,
  then the plan's smallest gap to a gray: −5 mm or more is yes, an overlap over 33 mm is no, in between it simulates
  pick + pull and restores the state. `"exact"` always simulates (for verifying scenes); `"kinematic"` never does
  and is too strict (refused 86 of 148 real successes). Hybrid on three sets of 300 scenes: tuning 299 of 300 right
  (fast-no cutoff then −25 mm), held-out 292 of 300 (all 8 errors fast no's, so the cutoff moved to −33 mm), test
  300 of 300, as was exact. Mean time per check on the test set: hybrid 884 ms, exact 1104 ms (10 workers).
- **08, ten hand-made scenes** (L0 to L3, red + 2 grays, in `scenes/`) with hand-written solutions: 24 of 30 runs work
  at μ 0.55, 0.4, 0.7. The layouts were tuned at μ 0.55, so this shows the scenes are solvable, not how reliably.
  Order matters only through physics (the sweep doesn't check other cylinders); L1b (the red too near the right
  wall) can't be solved with the current skills; one long sweep for two grays is fragile (a gray toppled at 0.4 and
  0.7).

`common/`: the env (`shelf_env.py`), what is measured (`measures.py`), plots (`plotting.py`), and running and saving
(`runner.py`), the retrieval check (`retrieval_check.py`) and hand-made scenes (`scenes.py`, stored in `scenes/`). `skills/`: new skills built on the repo's primitives (`sweep.py`).

## Run

From the repo root, with the venv in this folder (native arm64 Python 3.11, mujoco 3.8.1, gymnasium, numpy,
imageio[ffmpeg], matplotlib). On macOS `common` sets `MUJOCO_GL=cgl`; no PYTHONPATH is needed.

```bash
experiments/sahal/.venv/bin/python experiments/sahal/studies/01_grasp_pull/run.py --quick   # seconds
experiments/sahal/.venv/bin/python experiments/sahal/studies/01_grasp_pull/run.py            # full, ~4 min
```

Every `run.py` takes `--quick` (a subset into `outputs/quick/`) and `--workers` (default 10), and writes CSVs,
PNGs, `summary.json` and `config.json` to its `outputs/`.
