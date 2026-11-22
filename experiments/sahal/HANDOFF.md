# Handoff: grasp, pull and push on the shelf, red cylinder alone

For Dhruv, from Sahal, 2026-11-21. Results come from the output files named in each section; parameter values come
from the repo's code (`so101sim/side_primitives.py`, `so101sim/shelf.py`, `so101sim/shelf_layout.py`) and the
generated scene. Nothing outside `experiments/sahal/` was changed.

## 1. Purpose

You asked: on an empty shelf with one red cylinder, grasp it and pull it out; should that be one skill or two? This
is groundwork for a SPAR-style planner that needs a test of "can the red be grasped and pulled out from here?". It
grew into a second question, from replacing the shelf push with a lower sweep: how low can the gripper push, and
what makes a pushed cylinder slide instead of topple?

Everything uses full simulator state (exact poses and geometry) and the repo's `ShelfPrimitiveEnv` with its
primitives unchanged. The gray cylinders stay parked on the floor where `mj_resetData` puts them and are dropped from
the task's checks; `common/shelf_env.py` explains why that rather than moving them or turning off their collisions.

## 2. Setup

The venv in `experiments/sahal/.venv` is a native arm64 CPython 3.11 made with uv (an x86 Python under Rosetta can't
import mujoco on Apple Silicon). To make it again, from the repo root:

```bash
uv python install cpython-3.11-macos-aarch64-none
uv venv --python ~/.local/share/uv/python/cpython-3.11.17-macos-aarch64-none/bin/python3.11 experiments/sahal/.venv
uv pip install --python experiments/sahal/.venv/bin/python "mujoco==3.8.1" gymnasium numpy "imageio[ffmpeg]" matplotlib
```

`so101sim` is not installed into it; `common/__init__.py` puts the repo root on `sys.path`. Run each study from the
repo root (on Linux, drop `MUJOCO_GL=cgl` and the repo's default, egl, applies; `common` also sets cgl itself on
macOS):

```bash
MUJOCO_GL=cgl experiments/sahal/.venv/bin/python experiments/sahal/studies/01_grasp_pull/run.py     # ~4 min, 10 workers
MUJOCO_GL=cgl experiments/sahal/.venv/bin/python experiments/sahal/studies/02_sweep_height/run.py   # ~1.5 min
MUJOCO_GL=cgl experiments/sahal/.venv/bin/python experiments/sahal/studies/03_sweep_factors/run.py  # ~2 min
MUJOCO_GL=cgl experiments/sahal/.venv/bin/python experiments/sahal/studies/04_sweep_speed/run.py    # ~1 min
```

Every `run.py` takes `--quick` (a subset into `outputs/quick/`, seconds), `--workers N` (default 10) and, where it
has parts, `--parts`. Seeds are fixed and each job carries its own seed, so reruns give the same numbers.

## 3. Folder map

| path | what it is |
|---|---|
| `HANDOFF.md` | this document |
| `README.md` | one line per study and how to run it |
| `CLAUDE.md` | coding rules for this folder, read by Claude Code sessions |
| `.gitignore` | `.venv/`, `__pycache__/`, `*.mp4` |
| `common/shelf_env.py` | `SingleRedShelf` (the env with only the red, its options) and the shared push run |
| `common/measures.py` | grasp conditions, tilt outcome, contacts, clearance, glitch detection |
| `common/plotting.py` | colours, top-down shelf outline, rate and category maps |
| `common/runner.py` | parallel runs, CSV writing, `config.json` |
| `studies/01_grasp_pull/` | grasp and pull-out over the floor, grasp offsets |
| `studies/02_sweep_height/` | lowest push height, slide or topple vs height and friction; `tilted_gripper_proposal.md` |
| `studies/03_sweep_factors/` | tilt vs direction, height, arm friction, speed, board friction |
| `studies/04_sweep_speed/` | tilt vs speed in fine steps, quarter-speed traces |
| `studies/*/outputs/` | CSVs, PNGs, `summary.json` (the numbers quoted here), `config.json` (settings, versions) |

## 4. Definitions

**Grasp success**, checked at the end of `pick`, every condition required (`measures.grasp_conditions`):

1. the pick was not refused;
2. no failure during the pick (no arm-shelf contact, no topple, nothing off the board);
3. both jaws touch the red (`grasp.is_grasped`, the env's `held`);
4. the red's centre is within 1 cm of the jaw midpoint;
5. the red is lifted more than 5 mm above its resting height;
6. the red tilts less than 5°.

**Pull-out success** is the env's own: `info["is_success"]` after `pull`, which needs both jaws on the red, its centre
at least 2 cm in front of the board's front edge, tilt under 20°, all held for 5 consecutive frames, with no failure.
A run counts as a pull-out success only if it gets there, so it includes the grasp.

**Push outcome** (studies 02 to 04): slid (max tilt ≤ 5°), leaned (5° to 20°, stayed up), toppled (> 20°), or glitch
(section 9).

## 5. Skills and parameters

Scene (generated `models/shelf_objects.xml`): board top 0.09 m above the base plane, front edge x = 0.30 m, back wall
0.48 m, side walls at y = ±0.16 m, opening 0.15 m. Cylinders r = 1.5 cm, 8 cm tall. The action box (the reset
region): x 0.355 to 0.46 m, y −0.14 to 0.14 m, with 5 mm tolerance.

A level gripper can only point along rays from the pan axis at (0.0388, 0), so every move runs in or out along the
ray through its target. The lines it moves on (jaw midpoint): standoff x = 0.27 (3 cm outside), aisle x = 0.32 (2 cm
inside), grasp height 0.13 (board + 4 cm), carry height 0.14 (board + 5 cm), push height 0.115 (board + 2.5 cm).

| skill | input | what it does |
|---|---|---|
| `pick` | xy | standoff → in along the ray to xy at grasp height, jaws open to 28 (43 mm gap), hold 3 frames → close to 10, hold 8 → rise 1 cm → back out to the aisle line, holding. Pick already draws the cylinder most of the way out. |
| `place` | xy | from the aisle point, sideways along the aisle line at carry height → in to xy → lower to grasp height, hold 3 → open to 28, hold 8 → out to the standoff |
| `push` | xy, heading | jaws closed (10), in along the ray at grasp height, drop to push height at the front edge → to xy → 5 cm along heading (one of 16) → back the same way to the standoff |
| `pull` | (none) | from the aisle point straight out along its ray at carry height to the standoff; the cylinder's centre ends 3 cm past the front edge |
| `look` | xy | go to the standoff on the ray through xy and stop (points the wrist camera) |

How a path is built (`side_primitives._trace`): waypoints joined by straight lines in tool space, level IK solved
every 4 mm along them, each solve seeded by the last; a solve must land within 2 mm and 2° of level, and a fallback
solve that jumps more than 20° on any joint is refused (another IK branch). IK plans against a fixed gripper value
(22 for pick, 19.4 while holding, 10 for push), because the jaw midpoint moves 65 mm over the gripper's range. The
joint path is then retimed so no joint moves more than 2° per 30 Hz frame. There is no path search: if a straight
segment fails IK or the collision check, the primitive is refused. Between primitives `bridge` backs out along the
current ray to the standoff line and slides along it; anything else is a joint-space ramp.

What the env checks:

- Before running (kinematic, every frame of the planned path): arm geoms against the shelf panels, legs and the
  table, with the shelf grown by 6 mm. **Cylinders are not checked, and neither is a held cylinder against the shelf.**
- While running (every physics substep): any arm geom touching the shelf or table ends the episode as a failure.
  Every frame: a cylinder tilted over 20° (toppled) or a cylinder off the board footprint (unless it is the red and
  held by both jaws) ends it too.
- "Holding" (required for `pull` and `place`) is the gripper joint: last command 11 or less and the joint reading
  11 or more. An empty close reads 10.
- After each primitive the last pose is held up to 60 frames to settle, with the checks running.

## 6. Grasp and pull-out (study 01)

Outputs: `studies/01_grasp_pull/outputs/` (summary.json, heatmap_runs.csv, offsets_runs.csv, single_record.json).

- **Pick does most of the extraction.** At the centre, pick moves the red from x = 0.4075 to 0.3205 m (87 mm) and
  pull moves it the last 47 mm to 0.2739 m. Pick 163 frames, pull 32, max tilt 1.67°.
- **Where it works.** Over every 1 cm cell where a cylinder fits on the board (480 cells, 5 physics draws each,
  2400 runs), grasp and pull-out succeed in 396 cells (82%) at 100% and fail in the other 84 at 0%. No cell is
  partial: friction and density never changed an outcome. Pull-out works for y from −0.115 to +0.125 m at every
  depth, and only to +0.085 m in the back row (x = 0.465).
- **How it fails.** Every failure is a refusal before the arm moves: path would hit the right wall (225 runs), left
  wall (150), IK (30), back wall (15). The right side needs more room than the left because the moving jaw opens to
  the robot's right.
- **The env's action box contains unreachable cells.** 296 of its 360 cells work; the other 64 are refused (of its
  1800 runs: right wall 165, left wall 110, IK 30, back wall 15). Reset keeps the red only 3 cm from a side wall, so
  a red there needs a push before a pick.
- **Successful grasps are well inside the thresholds**: red centre 0.3 to 3.7 mm from the jaw midpoint (limit 10),
  pick tilt at most 1.87° (limit 5). Lift is the closest: commanded 10 mm, measured 9.9 mm at the front row and
  6.5 mm at the back (lowest single run 6.0 mm; limit 5).
- **Grasp accuracy.** Grasp point offset from the true axis (56 runs): 47 pulled out. Back (away from the robot) never
  failed up to 20 mm; front and right fail from 10 mm; left at 20 mm.
- **The only pull failures follow a bad grasp the env calls "holding".** In 2400 runs no grasp succeeded and then the
  pull failed. At 10 mm front or right the gripper joint reads 14.25 and 15.34 (11 or more counts as holding) without
  both jaws on the red; the env allows the pull and the red is dragged off the board.

**Recommendation: two skills in execution, one check in planning.** Keep `pick` and `pull` as separate skills,
because the grasp is reused on gray cylinders (pick then place) where pulling would be a failure. For the planner,
use one combined test, "graspable and pullable from here": reach, wall clearance and axis accuracy decide both, and
the pull added no failures of its own. Gate `pull` on both jaws touching the red (`grasp.is_grasped`), not on the
gripper reading, which passed the two bad grasps above.

## 7. Push height (study 02)

Outputs: `studies/02_sweep_height/outputs/` (reach.csv, slide_topple_runs.csv, summary.json).

- **How low the level gripper can push.** The env's path check allows 22 to 24 mm above the board at every start and
  heading tested (today's push height is 25 mm). Level IK solves down to 5 mm; the limit is clearance: the gripper
  body (wrist-roll housing) at 6.3 to 6.8 mm near the front and the forearm at 6.3 to 6.5 mm on deep pushes. In the
  dynamic runs the arm touched the board in all 15 runs at 1.75 cm and in none at 2.0 cm.
- **h < r/μ doesn't separate slide from topple here.** 120 pushes (8 heights, 5 frictions, 3 directions; 1 glitch
  excluded): with first contact below r/μ, 20 slid, 43 leaned and 6 toppled; above it, 5 slid, 38 leaned and 7
  toppled.
- **Direction matters.** Away from the robot: 1 slid, 31 leaned, 8 toppled. Left: 9 / 27 / 3. Right: 15 / 23 / 2.
  Away and right pushes hit with the jaw; left pushes hit with the gripper body.
- `studies/02_sweep_height/tilted_gripper_proposal.md`: what tilting the gripper down for a lower sweep would take
  (not built), with the files and functions it touches.

## 8. What reduces tilt (studies 03 and 04)

Outputs: `studies/03_sweep_factors/outputs/summary.json`, `studies/04_sweep_speed/outputs/summary.json` and
sweep_cells.csv. Repeated runs with the push start jittered up to 1 mm; glitches excluded.

| factor (study 03, 750 runs) | mean max tilt, topple rate |
|---|---|
| direction | away 14.2°, 26%; left 7.3°, 8%; right 7.8°, 9% |
| height 2.25 → 3.0 cm | 11.2°, 21% → 9.3°, 16% (2.5 cm 9.9°, 2.75 cm 8.9°) |
| cylinder-board friction 0.4 → 0.7 | 6.8°, 3% → 13.7°, 31% |
| arm friction set to the cylinder's μ (normally 1.0) | away 14.2° → 8.0° and 26% → 3%; left 7.3° → 7.0°; right 7.8° → 7.2° |
| speed at 2.5 cm, full / half / quarter | away 13.9° / 7.4° / 1.0°; left 7.1° / 3.4° / 0.9°; right 8.8° / 3.6° / 0.8° |

Speed in finer steps (study 04, 288 runs at 2.5 cm, μ 0.55 and 0.7):

| speed | toppled runs | worst cell mean tilt | worst single run | robot time per push |
|---|---|---|---|---|
| 0.25 | 0 | 1.67° | 3.11° | 24.5 s |
| 0.33 | 0 | 5.15° | 6.22° | 18.9 s |
| 0.5 | 0 | 9.15° | 13.62° | 12.8 s |
| 0.67 | 6 | 16.71° | 27.53° | 9.3 s |
| 0.75 | 2 | 12.89° | 22.08° | 8.8 s |
| 1.0 | 6 | 16.59° | 29.32° | 6.6 s |

- **Speed matters most.** The push is an impact: the trajectory runs at the 2°/frame joint cap throughout, the
  median non-glitch push touches the cylinder for 30 physics substeps (50 ms) and sends it off at a median 0.23 m/s.
  No topples at half speed or slower; at 1/3 speed every cell's mean tilt is at most 5.15°.
- **Arm friction comes second**, mostly for pushes away; height matters least in the range the gripper allows.
- The robot times slow the whole primitive, so they overstate the cost of slowing only the stroke.

## 9. Decisions and deviations

- **Widened action box** (studies 01 heatmap, 02, 03, 04): picks and pushes are accepted anywhere a cylinder fits on
  the board, not only in the env's box, to measure what the arm can do. `in_action_box` records whether the
  unmodified env would have accepted each heatmap cell.
- **Bypassed checks, marked** (studies 02 to 04): the path check runs but does not refuse; its verdict is saved as
  `would_refuse`. The arm touching the shelf is recorded (`arm_touched_shelf`) and does not end the run. Study 01 uses
  the env's checks unchanged.
- **Fixed physics where used**: the single run and the offsets use friction 0.55 and density 900 (the ranges'
  middle); the heatmap draws friction and density as ShelfEnv does for seeds 0 to 4; pushes set friction per cell
  through reset's physics option, density 900.
- **Glitch exclusion**: a run is a glitch when the arm-cylinder force passes 100 N (1000 N while the jaws squeeze the
  cylinder, which normally reaches 147 N) or the cylinder passes 1 m/s. Glitches are excluded from every rate and
  mean. Counts: study 01 0 of 2456, study 02 1 of 120, study 03 8 of 750, study 04 5 of 288. The traced one: tilt
  under 1.5° for most of a quarter-speed push, then 94.86° with a 15,492 N contact force.
- **Jaw-friction test is a diagnostic only**: study 03 sets the arm geoms' friction for single runs and restores it;
  the env is not changed.
- **Slower pushes** resample the push trajectory (float64 linear interpolation), the whole primitive, not only the
  stroke.

Other simulator caveats:

- **Chaos.** Slowing the same trajectory with float32 instead of float64 arithmetic changed 78 of 150 runs. Read push
  results as rates over jittered repeats, not single runs.
- **30 Hz tapping.** At quarter speed the jaw touches the cylinder about once per 30 Hz frame (median gap 0.032 to
  0.038 s), sliding on it at the friction limit; the position servos surge toward each new target and stop. A real
  servo may behave differently.
- **Gripper friction.** The gripper geoms have friction 1.0 and MuJoCo uses the larger of a contact's two frictions,
  so the jaw-cylinder friction is 1.0 whatever the episode's friction is.

## 10. Open questions for Dhruv

1. **Sweep parameters.** Start and end points, or start, direction and duration?
2. **What "duration" means.** Stroke length, time, or both (which would set the speed)?
3. **Tilting the gripper.** Worth building a pitched sweep (`tilted_gripper_proposal.md`), or is ~2.3 cm low enough
   given that speed mattered more than height?
4. **Jaw friction 1.0.** Is it realistic for the real gripper's surface? It decides much of the lean on pushes away.
5. **Sweep and push.** Should the sweep replace the push or join it as another action?
6. **Entry height.** Should the sweep go in low (at sweep height from the front edge) or above the cylinders and drop
   down at the start point?

## 11. Next steps

1. A sweep skill at about 2.3 to 2.5 cm and about 1/3 of today's speed, in both parameterisations (start and end;
   start, direction and duration).
2. Tests with several cylinders: neighbours, sweeping one cylinder into another, grasping beside a gray.
3. A depth-1 search baseline: try every candidate action one step ahead with the true state and pick the best.
4. Then MCTS (UCT) over the skills.
5. Later, a SPAR-style ranker as the prior for the search.
