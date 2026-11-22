# 05 Sweep skill on a lone cylinder

**Question.** Does `skills/sweep.py` move a lone cylinder where it is told without toppling it, and where is it
refused?

**The skill.** The repo's shelf push route with the jaws closed and the gripper level: in at grasp height to the front
edge, drop, approach the start point, stroke, retrace, rise, out. Normal speed except the stroke and the last 1 cm of
the approach, which run at 1/3 of it (0.020 m/s; normal is 0.060 m/s, measured on the repo's push). Height: the lowest
of 23 to 26 mm (1 mm steps) that passes the env's path check. `sweep(x, y, direction, duration)` and
`sweep_points(x0, y0, x1, y1)`; length = 0.020 m/s × duration. Refused if the direction is toward the robot (within
22.5° of 180°; not supported), the start or end is off the board, level IK fails, or the path check fails at every
height. Reuses the repo's level IK, retiming, bridge and path check.

**Method.** The red alone at five spots: centre (0.4075, 0), near the left wall (0.4075, 0.12), near the right wall
(0.4075, −0.12), front (0.33, 0), back (0.45, 0). The sweep starts 5 mm short of touching it (found kinematically),
jittered up to 1 mm, and runs in the skill's 7 default directions (0°, 45°, 90°, 135°, 225°, 270°, 315°; 0° = away
from the robot, 90° = robot's left) for 3, 6 and 10 cm, at friction 0.55 and 0.7, 3 repeats: 630 runs. All of the
env's checks are on. Intended movement = length − 5 mm along the direction. Distance ratio = movement along the
direction / intended; sideways drift = movement perpendicular to it (positive to the left of the direction).

Outcomes, in order: **glitch** (arm-cylinder force over 100 N or the red faster than 1 m/s, and not a jam: a solver
blow-up, excluded from rates and means); **jammed** against a wall (the arm touches a side wall, the back wall or the
top board, or pushes with more than 100 N while the red touches one; a failure); toppled (> 20°); leaned (5° to 20°);
slid. Movement is measured over clean runs: no glitch, no jam, no env failure.

The `toward_robot` part runs the same sweeps at 180°, planned with `allow_toward_robot=True`, to keep the evidence
for refusing them.

**Run.** `experiments/sahal/.venv/bin/python experiments/sahal/studies/05_sweep_skill/run.py [--quick]
[--parts sanity sweeps toward_robot] [--workers 10]`.

**Results** (outputs/summary.json, sweep_runs.csv, sweep_cells.csv, toward_robot_runs.csv, sanity.json).

- Sanity: `sweep()` and `sweep_points()` give identical joint trajectories for all 3 test motions (difference 0.0°).
- 630 runs: 374 executed, 256 refused. Executed: 312 slid, 21 leaned, 1 toppled, 29 jammed, 11 glitches. 333 clean.
- **Jams** (29, all with the red against the back wall): back spot at 315° (8), 0° (6), 45° (6), 270° (3); centre at
  45° (5) and 315° (1). Force while the red touched the wall 101 to 1404 N. 4 jams ended with the arm on the top board,
  and 2 jams also toppled. Jam rate by direction: 0° 10%, 45° 23%, 315° 18%, 270° 5%, others 0%.
- **Glitches** (11, solver blow-ups, the red at 6.9 to 45.7 m/s): centre 90° 6 cm (3), centre 225° (4), centre 270°
  (3), front 45° 10 cm (1).
- **Toppled outside jams: 1** (centre, 315°, 10 cm, μ 0.55).
- **Refusals** (256): IK 72, end off the board 66, start off the board 36, path check left wall 36, right wall 36,
  back wall 10. By spot (of 126 each): centre 6, near left wall 90, near right wall 90, front 0, back 70.
- **Distance and drift** (clean runs, median by direction; ratio and drift ranges over all clean runs):

| direction | clean runs | distance ratio, median (range) | sideways drift mm, median (range) |
|---|---|---|---|
| 0° away | 54 | 0.938 (0.718 to 1.01) | 1.4 (−9.0 to 6.5) |
| 45° | 36 | 0.75 (0.574 to 0.821) | −9.3 (−26.3 to −5.0) |
| 90° left | 57 | 0.849 (0.659 to 0.928) | −8.9 (−17.9 to −3.0) |
| 135° | 48 | 0.315 (0.181 to 0.494) | −16.0 (−29.3 to −4.9) |
| 225° | 44 | 0.538 (0.365 to 0.734) | 9.4 (4.5 to 23.7) |
| 270° right | 54 | 0.821 (0.597 to 0.949) | 9.9 (3.8 to 19.8) |
| 315° | 40 | 0.706 (0.552 to 0.797) | 13.7 (6.0 to 28.6) |

  In every direction the drift is away from the robot (+x): the red loses any component toward the robot and gains
  one away from it. Sideways and diagonal drift grows with length (e.g. 315°: 7.35, 14.3 and 26.65 mm medians at 3,
  6 and 10 cm). Sweeps at 135° and 225° hardly move the red toward the robot (135°: median dx 0.45 mm against an
  intended −38.9 mm). Overall: distance ratio median 0.744 (p10 to p90: 0.345 to 0.919), absolute drift median 8.9 mm
  (p90 21.3), absolute direction error median 16.7°.
- **Toward the robot (180°), now refused**: of 90 runs, 18 executed, all from the front spot: 15 toppled, 3 leaned.
  The rest: IK 54, start off the board 18. The route enters low at the front edge and must pass the cylinder to get
  behind it. Study 06: there is no room to enter above the cylinders instead.
- Heights used: 23 mm in 246 executed runs, 24 mm in 128. Robot time per sweep: 8.2 s (3 cm), 9.7 s (6 cm), 11.9 s
  (10 cm), of which 2.2, 3.8 and 5.9 s run slow. Planning median 0.15 s.

**Caveats.** The jam rule uses the same 100 N as the glitch rule; all 29 jams had the red on the back wall. The start
point is found from the red's true pose. The 11 glitches are all at 6 m/s or more and are not wall jams, but 6 of them
have arm forces under 100 N (the red flew without a hard hit).
