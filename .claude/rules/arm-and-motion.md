---
paths:
  - "so101sim/ik.py"
  - "so101sim/primitives.py"
  - "so101sim/side_primitives.py"
  - "so101sim/grasp.py"
  - "so101sim/units.py"
  - "demonstrator/*.py"
  - "demonstrator/experiments/*.py"
  - "scripts/smoke.py"
  - "scripts/sag_experiment.py"
  - "tests/test_ik.py"
  - "tests/test_side_ik.py"
  - "tests/test_side_primitives.py"
  - "tests/test_grasp.py"
  - "tests/test_demonstrator_reach.py"
  - "tests/test_units.py"
---

# Traps: IK, gripper and motion

Claude Code loads this file when Claude reads a file matching the paths above. It is the only home of these traps,
which moved here from CLAUDE.md on 2026-09-30. Add a trap here when you find one in this area, dated, with the
evidence. The load follows the Read tool; if you read code with `cat` or `sed`, open this file yourself.

- `gripperframe` is not where a grasped object ends up. In the site frame the jaw tip midpoint is
  [+1.4, 0, -18.6] mm at the closed stop and [-36.8, 0, +33.5] mm fully open. Target `ik.grasp_point`.
- The gripper is a rotating single jaw, not a parallel one. The jaw midpoint travels 64.7 mm between
  lerobot gripper 0 and 100, so IK has to be solved at the gripper value the arm will actually be at.
  Plan a grasp at the approach width and closing drags the cube 12.6 mm into the fixed jaw.
- Holding a 30 mm cube, the gripper joint sits near 19.4 no matter what you command, because the cube
  blocks it. IK for a loaded arm must use `primitives.GRIP_HOLDING`, not the commanded 10. (Corrected
  2026-09-30: it read 17.7 and this said 18 under the old soft contacts, where the jaws sank into the cube.)
- Fit a circle over a full sweep, not an arc. Measuring the wrist_roll circle over ±86 degrees and taking
  the mean distance from the centroid gives 8.7 mm; over the full ±180 it gives 12.00 mm. The partial
  version's standard deviation was 3.1 mm against 0.05 mm, which is the tell. The `so101` session caught
  this one, along with two offsets I had mislabelled.
- The wrist_roll axis passes through the jaw midpoint at MJCF gripper 0, to 0.21 mm. `ik.ROLL_AXIS_TCP`
  is that point. Define a tool there and roll is pure yaw. At the gripper value that closes on a 30 mm
  cube the midpoint is 6.92 mm off axis again, so grasping still needs position and yaw solved together.
- Top-down reach is an annulus and it shrinks with height: roughly radius 0.13 to 0.29 at table level,
  out to 0.27 carrying a cube at 5 cm, and nothing at all above z = 0.10. Rectangular workspace bounds
  will include corners the arm cannot reach. The bin sits at radius 0.261 for this reason; the obvious
  spot straight ahead at (0.24, 0.15) is 0.283 and unreachable.
- Every top-down grasp has two wrist rolls half a turn apart that close the jaws on the same faces. IK takes
  whichever it finds first, and near the base that is often the one that folds the wrist camera mount into the
  shoulder: bin picks at x <= 0.14 failed that way. `primitives.pick`, `place` and `push` take `flip` (0 or 1) to
  choose, and the demonstrator's check picks the roll that stays clear. With the right roll the near bin wall is no
  worse than the others.
- IK from the zero seed gets stuck on targets behind the base, where pan hits its 110 degree limit. It
  retries from seeded random poses, so results repeat run to run.
- Menagerie's `wrist_roll` actuator ctrlrange runs 5.6 degrees past the joint limit. The env clips actions
  to joint ranges, not ctrlranges.
- In the search scene the table is below the base, so grasp heights are negative. `primitives.bridge`
  decides whether the arm is hovering relative to `hover_z`. Before 2026-09-25 it used a fixed z = 0.03,
  which would have sent every crossing in the sunken bin through a joint ramp and the pile.
- Never bridge between primitives by interpolating in joint space. A straight line in joint space is an
  arbitrary curve in tool space and will sweep the gripper through the bin. `primitives.bridge` crosses
  at hover height in tool space instead, falling back to a joint ramp only from the non-vertical HOME.
- lerobot runs the follower with `max_relative_target=4.0` and silently clips anything larger, so the arm
  lags a plan that violates it. Trajectories here cap at 2 deg per frame, which is 60 deg/s at 30 Hz.
- `data.site_xmat[...].reshape(3, 3)` is a view into `data`. Copy it before comparing two poses, or both
  variables alias the same buffer and every difference reads as zero.
- Don't trust `mj_geomDistance` for a box against a box. At an exact IK pose it returned 0 for two boxes 30 mm apart,
  and moving one joint by 1e-6 rad gave the right answer. `demonstrator/reach.Clearance` samples points instead.
