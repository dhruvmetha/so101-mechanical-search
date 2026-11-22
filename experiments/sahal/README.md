# Sahal's shelf studies

The red cylinder alone on the shelf task's board (`so101sim/shelf.py`), full simulator state, no repo files changed.
`HANDOFF.md` is the summary for Dhruv. Each study's README has its question, method, results and caveats.

| study | question |
|---|---|
| `studies/01_grasp_pull` | where can the red be grasped and pulled out, and how exact must the grasp be |
| `studies/02_sweep_height` | how low can the level gripper push, and does a pushed cylinder slide or topple |
| `studies/03_sweep_factors` | which of direction, height, arm friction, speed and board friction changes tilt most |
| `studies/04_sweep_speed` | what push speed keeps the cylinder upright, and what happens at quarter speed |

`common/`: the env (`shelf_env.py`), what is measured (`measures.py`), plots (`plotting.py`), and running and saving
(`runner.py`).

## Run

From the repo root, with the venv in this folder (native arm64 Python 3.11, mujoco 3.8.1, gymnasium, numpy,
imageio[ffmpeg], matplotlib). On macOS `common` sets `MUJOCO_GL=cgl`; no PYTHONPATH is needed.

```bash
experiments/sahal/.venv/bin/python experiments/sahal/studies/01_grasp_pull/run.py --quick   # seconds
experiments/sahal/.venv/bin/python experiments/sahal/studies/01_grasp_pull/run.py            # full, ~4 min
```

Every `run.py` takes `--quick` (a subset into `outputs/quick/`) and `--workers` (default 10), and writes CSVs,
PNGs, `summary.json` and `config.json` to its `outputs/`.
