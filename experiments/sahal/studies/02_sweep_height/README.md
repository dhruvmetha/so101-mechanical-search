# 02 Push height: how low, and slide or topple

**Question.** How low can the level gripper push, and does a lone cylinder pushed at height h slide or topple?
Theory: it slides if h < r/μ (r = 1.5 cm).

**Method.**

- `reach`: kinematic. `side_primitives.push` with push height 5 to 35 mm above the board, at five start points and
  four headings; level IK, then the env's path check (shelf grown by 6 mm). Clearance at the lowest passing height
  by growing the shelf boxes until they touch the arm (the path check's own definition).
- `slide_topple`: the red alone at (0.4075, 0); a 5 cm push away from the robot, left or right, starting 5 mm clear
  of it; heights 1.75 to 3.5 cm, friction 0.4 to 0.7 (density 900). The path check is bypassed and its verdict
  recorded (`would_refuse`); an arm-board touch is recorded and does not end the run. Contacts are read every physics
  substep. Glitch: arm-cylinder force over 100 N or the red faster than 1 m/s.

**Run.** `experiments/sahal/.venv/bin/python experiments/sahal/studies/02_sweep_height/run.py [--quick]
[--parts reach slide_topple] [--workers 10]`. Full: about 1.5 minutes.

**Results** (outputs/reach.csv, slide_topple_runs.csv, summary.json).

- Lowest push height the path check allows: 22 to 24 mm everywhere. Level IK solves down to 5 mm; the limit is
  clearance: the gripper body (wrist-roll housing, `gripper#32`) at 6.3 to 6.8 mm near the front, the forearm
  (`lower_arm#18`) at 6.3 to 6.5 mm on deep pushes. In the dynamic runs the arm touched the board in all 15 runs at
  1.75 cm and none at 2.0 cm.
- 120 runs, 1 glitch (left, 3.25 cm, μ 0.55: 428 N, 29.8 m/s).
- Outcome by direction: away 1 slid / 31 leaned / 8 toppled; left 9 / 27 / 3; right 15 / 23 / 2.
- h = r/μ does not separate the outcomes: with first contact below r/μ, 20 slid, 43 leaned, 6 toppled; above it, 5
  slid, 38 leaned, 7 toppled.
- First contact sits 2.1 mm below the commanded height pushing away, 2.6 mm above pushing left and 1.1 mm above
  pushing right.

`tilted_gripper_proposal.md`: what it would take to tilt the gripper down a few degrees for a lower sweep (not built).

**Caveats.** One run per cell (study 03 repeats with jitter); the push is an impact at full trajectory speed (study 04).
