# 01 Grasp and pull-out on an empty shelf

**Question.** With only the red cylinder on the shelf and its true pose known, where on the board can
`("pick", xy)` then `("pull", ...)` take it out, and how exact must the grasp point be?

**Method.** `SingleRedShelf` (common/shelf_env.py) is the repo's `ShelfPrimitiveEnv` with the gray cylinders left
parked on the floor and dropped from the task's checks. The repo's own primitives run unchanged.

- `empty`: places the red at the centre and three other spots; images from both cameras; counts grays on the board.
- `single`: one pick and pull at the region's centre (0.4075, 0), mid-range physics (friction 0.55, density 900),
  with the 30 Hz trace (`--video` also saves an mp4, kept out of git).
- `heatmap`: every 1 cm cell where an upright cylinder fits on the board (480 cells), 5 runs each with friction and
  density drawn as ShelfEnv draws them for seeds 0-4. The env's action box is widened to the whole floor;
  `in_action_box` records whether the unmodified env would accept the cell.
- `offsets`: grasp point moved 2 and 5 mm back, front, left and right of the true axis at four spots, and 8 to 20 mm
  at the centre, in the approach ray's frame (back = away from the robot along the ray).

Grasp success, checked at the end of pick, each condition its own column: not refused, no failure, both jaws on the
red, red centre within 1 cm of the jaw midpoint, lifted more than 5 mm, tilt under 5 degrees. Pull-out success is the
env's `info["is_success"]`. Glitch: contact force over 1000 N (closed jaws normally squeeze with up to 147 N here) or
the red faster than 1 m/s.

**Run.** `experiments/sahal/.venv/bin/python experiments/sahal/studies/01_grasp_pull/run.py [--quick] [--parts
empty single heatmap offsets] [--workers 10] [--video]`. Full: about 4 minutes on 10 workers.

**Results** (outputs/summary.json, heatmap_runs.csv, offsets_runs.csv).

- Empty shelf: 0 grays on the board at every placement.
- Single run: success; pick 163 frames, pull 32. Pick already moves the red from x = 0.4075 to 0.3205 m; pull takes it
  to 0.2739 m. Max tilt 1.67 degrees.
- Heatmap: 2400 runs, 0 glitches. Grasp and pull-out both 100% in 396 of 480 cells and 0% in the other 84; no cell
  is partial, so physics draws never changed an outcome. Inside the env's action box: 296 of 360 cells.
  Every failure is a refusal before the arm moves: right wall 225 runs, left wall 150, IK 30, back wall 15.
  No run grasped and then failed the pull.
- Offsets: 47 of 56 pulled out. Back offsets never failed (up to 20 mm). Front and right fail from 10 mm, left at
  20 mm. At 10 mm front or right the gripper joint still reads "holding" (above 11) without both jaws on the red,
  so the env allows the pull and the red is dragged off the board: "holding" alone doesn't mean a good grasp.

**Caveats.** One cylinder, no neighbours; true poses; one seed per offset run.
