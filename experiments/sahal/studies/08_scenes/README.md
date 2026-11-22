# 08 Hand-made scenes with hand-written solutions

**Question.** For ten scenes of the red and two grays, from easy to hard, does a hand-written solution (sweeps with
`skills/sweep.py`, then the repo's pick + pull) work in physics, and at which frictions?

**Scenes** (`common/scenes.py`, files in `experiments/sahal/scenes/`). Each is a JSON file: name, level, one-line
description, friction, positions of `cyl_red`, `cyl_gray_0`, `cyl_gray_1`. `place_scene` refuses overlaps (under
5 mm between surfaces), cylinders not fully on the board, and scenes where a cylinder moves over 2 mm or tilts over
2° while settling.

**Solutions** (`SCENE_SPECS` in run.py): each sweep is (cylinder, direction, distance), planned with
`sweep_cylinder` from that cylinder's pose at the time, so it starts 5 mm short of it; then pick + pull. Only
directions the skill supports (no 180°), none ending at the back wall.

**Method.** For each scene at friction 0.55, 0.4 and 0.7 (density 900): place it; `can_retrieve(mode="exact")` must
say no at the start, except L0; run the solution; record success, sweeps run and the first failure (sweep refused,
toppled, jammed against a wall, off the board, retrieve failed). Glitches are flagged (arm force on any cylinder over
100 N, 1000 N during the retrieve, or a cylinder over 1 m/s). The two L2b scenes also run their sweeps in reverse
order. Front-camera images before and after each run at μ 0.55 go into the gallery.

**Run.** `experiments/sahal/.venv/bin/python experiments/sahal/studies/08_scenes/run.py [--quick] [--workers 10]`.
About 35 s.

**Results** (outputs/table.csv, solutions.csv, summary.json, gallery.png, images/).

| scene | level | needs | μ 0.55 | μ 0.4 | μ 0.7 | reversed order |
|---|---|---|---|---|---|---|
| L0_open | L0 | retrieve | works | works | works | |
| L1_front | L1 | 1 sweep + retrieve | works | works | works | |
| L1_front_offset | L1 | 1 sweep + retrieve | works | works | works | |
| L1b_right_wall | L1b | sweep the red off the right wall | refused | refused | refused | |
| L2_two_in_line | L2 | 2 sweeps + retrieve | works | works | works | |
| L2_side_by_side | L2 | 1 long sweep + retrieve | works | gray toppled | gray toppled | |
| L2b_order | L2b | 2 sweeps in order + retrieve | works | works | works | works / gray toppled / gray toppled |
| L2b_wall_order | L2b | 2 sweeps in order + retrieve | works | works | gray toppled (glitch) | works at all three |
| L3_right_wall | L3 | 1 sweep + retrieve | works | works | works | |
| L3_left_wall_two | L3 | 2 sweeps + retrieve | works | works | works | |

- 24 of 30 runs work. The exact check said no at the start of every scene except L0, as intended.
- **L1b can't be solved with the current sweep.** Where the red is too near the right wall to be retrieved (from
  y = −0.125 m here; at −0.122 m and above it is still retrievable), no sweep moving it off the wall plans: at 90°
  the start is off the board (y −0.1742), at 45° the path check refuses (probed at x 0.36 to 0.44). The same holds for
  any cylinder: starting from home, a sweep away from the right wall plans only from about y ≥ −0.09 m, and away
  from the left wall only from about y ≤ +0.09 m, because the sweep must start between the cylinder and the wall.
- **L2_side_by_side**: one 10 cm sweep pushing both grays works at μ 0.55 only; at 0.4 and 0.7 a gray topples
  during the pick that follows.
- **Order.** In L2b_order (gray B at the front edge, on the gripper's way to gray A's start) the reversed order
  fails at μ 0.4 and 0.7 (the gripper knocks B over on the way in) and works at 0.55. In L2b_wall_order the reversed
  order works at all three frictions: the arm shoves B aside harmlessly, so order does not matter there. The skill's
  checks ignore cylinders, so order can only matter through physics.
- **L2b_wall_order at μ 0.7** failed on its first sweep (gray B toppled), and that sweep is flagged as a glitch.

**Caveats.** One run per scene and friction; pushes are chaotic, so a "works" is one sample, not a rate. Positions
were tuned at μ 0.55 until the solutions worked there, except L2b_order: its layout was picked from three tried at
all three frictions, as the one whose order worked forward and failed reversed. So these results are not a fair test
of the solutions, only a check that they work.
