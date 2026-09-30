---
paths:
  - "so101sim/mechsearch.py"
  - "so101sim/search_*.py"
  - "so101sim/binpick.py"
  - "models/search_*.xml"
  - "scripts/mechsearch_smoke.py"
  - "scripts/size_search_bin.py"
  - "scripts/binpick_demo.py"
  - "demonstrator/*.py"
  - "tests/test_mechsearch.py"
  - "tests/test_search_*.py"
  - "tests/test_binpick.py"
---

# Traps: the bin task

Claude Code loads this file when Claude reads a file matching the paths above. It is the only home of these traps,
which moved here from CLAUDE.md on 2026-09-30. Add a trap here when you find one in this area, dated, with the
evidence. The load follows the Read tool; if you read code with `cat` or `sed`, open this file yourself.

- A straight-down grasp can't reach a cube whose face is within 2 cm of a bin wall: the gripper is wider
  than the cube. Walls over 6 cm are worse, because the wrist's `camera_mount` and the gripper housing
  catch the rim while the jaws reach the floor. The sizing rule exempts that 2 cm strip; cubes there need
  a push first.
- A buried red cube is boxed in by four grays flush against its sides. To pick it, clear the grays on its
  jaw axis first, gripping each across its own width.
- Search-env cubes wait on the ground plane at x = 1.2 until reset drops them, so a cube there in a debug
  view hasn't fallen off anything.
