# 04 Push speed

**Question.** What push speed keeps a lone cylinder upright, and what happens at quarter speed?

**Method.**

- `sweep`: the study 02 push at 2.5 cm, speed 0.25, 0.33, 0.5, 0.67, 0.75 and 1.0 of normal, μ 0.55 and 0.7, all
  three directions, 8 jittered runs per cell (288 runs). A speed fraction s resamples the whole push trajectory at
  frame spacing s, so the same path takes 1/s as many frames.
- `traces`: four quarter-speed runs (left μ 0.7 reps 1 and 2, right μ 0.6 rep 0, left μ 0.55 rep 0) recorded every
  physics substep: tilt, arm-cylinder normal and tangential force, slip, the base's contact with the board, the jaw
  midpoint's and the cylinder's speed.

Glitch runs (arm-cylinder force over 100 N or the red faster than 1 m/s) are excluded from rates and means.

**Run.** `experiments/sahal/.venv/bin/python experiments/sahal/studies/04_sweep_speed/run.py [--quick]
[--parts sweep traces] [--workers 10]`. Full: about 1 minute.

**Results** (outputs/summary.json, sweep_cells.csv, traces.csv).

| speed | toppled runs | worst cell mean tilt | worst single run | glitches | robot time per push |
|---|---|---|---|---|---|
| 0.25 | 0 | 1.67° | 3.11° | 2 | 24.5 s |
| 0.33 | 0 | 5.15° | 6.22° | 2 | 18.9 s |
| 0.5 | 0 | 9.15° | 13.62° | 0 | 12.8 s |
| 0.67 | 6 | 16.71° | 27.53° | 0 | 9.3 s |
| 0.75 | 2 | 12.89° | 22.08° | 0 | 8.8 s |
| 1.0 | 6 | 16.59° | 29.32° | 1 | 6.6 s |

- No topples at half speed or slower. At 1/3 speed every cell's mean tilt is at most 5.15°.
- Traces: the quarter-speed runs that stayed up show about one touch per 30 Hz frame (median gap 0.032 to 0.038 s,
  40 to 52 touches), the arm sliding on the cylinder at its friction limit (tangential/normal 0.97 to 0.99), and the
  base on the board in only 30 to 38% of the substeps with a touch (100% before the push). The 30 Hz position
  targets make the jaw surge and stop every frame, so it taps the cylinder, which slides and stops between taps.
- The trace of left μ 0.7 rep 1 is a glitch: tilt under 1.5° for most of the push, then 94.86° with a 15,492 N
  contact force.

**Caveats.** The whole primitive is slowed, so the robot times overstate the cost; only the stroke needs to be slow.
The 30 Hz surge is how the simulated position servos follow targets; a real servo may differ.
