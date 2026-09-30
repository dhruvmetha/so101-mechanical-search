---
paths:
  - "so101sim/shelf*.py"
  - "so101sim/side_primitives.py"
  - "models/shelf_*.xml"
  - "scripts/*shelf*.py"
  - "tests/test_shelf*.py"
  - "tests/test_side_*.py"
---

# Traps: the shelf task

Claude Code loads this file when Claude reads a file matching the paths above. It is the only home of these traps,
which moved here from CLAUDE.md on 2026-09-30. Add a trap here when you find one in this area, dated, with the
evidence. The load follows the Read tool; if you read code with `cat` or `sed`, open this file yourself.

- A level gripper points only along rays from the pan axis at (0.0388, 0), not from the origin. Level reach runs
  0.22 to 0.44 m from that axis; the inner limit moves out to 0.225 m holding a cylinder and 0.24 m at push height.
- The wrist camera mount decides the side-grasp roll. Mount down, it hits a board 6 cm above the base. Mount up,
  the moving jaw always opens to the robot's right, so a side grasp needs room on the right.
- Cylinders need condim 6. Under condim 4 the rolling coefficient is ignored, and a pushed cylinder wobbled on its
  rim forever with its tilt growing.
- Push shelf cylinders 2.5 cm above the board. Re-measured with noslip on 2026-09-29: 2.5 cm slides a lone cylinder
  45 mm and leaves it upright, 3 cm topples it, and 2 cm is now refused because the forearm comes within 6 mm of the
  board. Before noslip, opening the jaws to lerobot 33 on release dragged a cylinder 14 mm and 28 dragged it 1.4 mm;
  with noslip the two are 2.5 and 2.2 mm, so 28 stays for the narrower swing. A held cylinder reads 12.7 to 17.8 on
  the gripper joint and an empty close reads 10.0.
- A grasp that lands short of a cylinder's axis holds a chord near its front, and the pull swings it past 20 degrees.
  That's what failed Claude's seed-300 run.
- The shelf's front camera sits 11 cm above the board and sees it edge-on, so a front pixel fixes left and right
  well and depth badly. Act on wrist pixels. Any camera well above the cylinder tops sees over the grays, which is
  why the front camera and start pose are the lowest that pass.
