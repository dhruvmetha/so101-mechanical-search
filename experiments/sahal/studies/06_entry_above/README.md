# 06 Entering above the cylinders

**Question.** Could the sweep enter the shelf above the 8 cm cylinders and drop down behind one, instead of entering
low at the front edge (which makes sweeps toward the robot pass through the cylinder, study 05)? Is there room
between the cylinder tops and the top board, and does level IK reach those heights inside the shelf?

**Method.** Kinematic only; the sweep skill is unchanged. Jaw-midpoint heights 80 to 150 mm above the board (5 mm
steps) at every 1 cm target where a cylinder centre could stand (480 targets per height, 7200 poses). Level IK with
the jaws closed (`ik.solve_side`, accepted with `side_primitives`' own 2 mm and 2° tolerances). For each solved pose,
every point of every arm geom (mesh vertices, box corners, sphere and capsule extremes) over the board
(0.30 ≤ x ≤ 0.48 m) gives the arm's lowest and highest point there. A height fits if IK solves and the arm clears the
cylinder tops and the top board's underside by the path check's 6 mm.

**Run.** `experiments/sahal/.venv/bin/python experiments/sahal/studies/06_entry_above/run.py [--quick]
[--workers 10]`. Seconds.

**Results** (outputs/summary.json, poses.csv, entry_above.png).

- **Room**: the cylinder tops are 80 mm above the board and the top board's underside is 150 mm above it, leaving
  70 mm.
- **The level arm needs 95.4 to 108.5 mm of height** over the board at targets in the env's action box (x ≥ 0.355 m,
  4860 solved poses): 14.8 to 28.7 mm below the jaw midpoint (gripper body and jaws) and 79.7 to 80.6 mm above it.
  The top is always the wrist camera mount (`camera_mount`), which stands up from the level gripper.
- **So there is no room**: no pose in the action box fits. Where the arm clears the cylinder tops by 6 mm, the camera
  mount is at best 35.5 mm into the top board.
- **Level IK is not the limit**: it solves at 400 to 470 of the 480 targets at each height; all failures are at
  x ≥ 0.435 m (the back), more at greater heights.
- The only poses that fit (heights 95 to 135 mm) have the jaw midpoint at x 0.315 to 0.345 m, 4.5 cm or less past the
  front edge, where the camera mount is still outside the shelf. That is not deep enough to get behind a cylinder in
  the action box.

**Caveats.** Arm points are exact for boxes, spheres and capsules and use mesh vertices for meshes (the collision
shape of a mesh is its convex hull, whose vertices are among these). Only the level gripper with the camera mount up
(the repo's `solve_side`) was checked; the wrist rolled to put the mount sideways was not.
