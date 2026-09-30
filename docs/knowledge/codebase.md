# Codebase: layout and conventions

What each part of the repo does and the conventions it follows. Moved here from CLAUDE.md on 2026-09-30; CLAUDE.md
keeps a one-line map. Traps specific to a subsystem are in `.claude/rules/`. Update this file when a module's
behaviour or a measured number in it changes.

- `models/robotstudio_so101/` is menagerie's model copied verbatim at a pinned commit. Never edit it. Shared
  scene pieces go in `models/stage.xml`. Each task's table, camera and objects go in its own objects file.
  `models/README.md` has the commit and re-vendoring steps.
- `so101sim/units.py` owns the unit boundary. Everything outside the env is in lerobot units: five arm joints
  in degrees, gripper 0 closed to 100 open. Everything inside MuJoCo is radians. Joint order is always
  `units.JOINT_NAMES`, which matches lerobot's servo IDs 1 to 6.
- `so101sim/env.py` holds `SO101Env`, a Gymnasium env. Observation keys follow lerobot's dataset naming
  (`observation.state`, `observation.images.front`, `observation.images.wrist`). Control is 30 Hz, 10
  physics steps per control step. `SO101Env` itself has reward 0 and never ends; the task envs subclass or
  wrap it.
- `so101sim/ik.py` has two solvers. `solve_ik` is position only on the MJCF site `gripperframe`, leaving
  the wrist wherever the solver drops it. `solve_topdown` is what the task uses: it points the gripper
  straight down and targets the midpoint between the jaw tips, with jaw yaw as a soft objective. Pointing
  down costs 2 of the 5 DOF, so yaw is only partly controllable, and the solver reports what it achieved
  rather than pretending. Measured across the workspace, every distinct cube yaw is reachable inside
  x <= 0.26; authority drops off past that at negative y.
- `so101sim/primitives.py` turns `pick(xy, yaw)`, `place(xy)` and `push(xy, direction)` into 30 Hz joint
  trajectories in lerobot units. The push parameterization is VPG's (start point, one of 16 headings,
  fixed stroke) rather than two endpoints, because pan and push direction are coupled on this arm.
  `bridge` gets the arm from wherever it is to the start of the next primitive.
- `so101sim/binpick.py` holds `BinPickEnv`, where one `step` runs a whole primitive. Object poses go in
  `info`, never the observation, so a policy sees what hardware would. Success thresholds follow RoboMME
  (arXiv:2603.04639) where they apply.
- `scripts/binpick_demo.py` is a greedy scripted policy: place what you can grasp, push when you can't.
  It clears about 29 of 30 cubes. Treat that as the baseline a learned policy has to beat.
- `so101sim/search_layout.py` generates `models/search_objects.xml` and `models/search_scene.xml` from a
  `Layout`: riser height, bin width and centre, wall height, front camera height, cube count, start pose.
  The numbers live in the XML's `<custom>` fields and `Layout.from_model` reads them back, so never edit the
  generated files. `scripts/size_search_bin.py` picks the layout with the checks in
  `so101sim/search_checks.py`, and the notes at the top of the XML say what it measured.
- `so101sim/mechsearch.py` holds `MechSearchEnv`, a 30 Hz env with `SO101Env`'s actions and observations,
  and `MechSearchPrimitiveEnv`, where one step is a pick, place or push inside the bin. Any cube leaving the
  bin ends the episode as a failure. Success is the red cube held for 5 frames with its lowest corner 1 cm over
  the rim and 2 cm above where a top-layer cube rests; with 5 cm walls the rim alone let a 2 mm lift count. Starts are 50/50 buried and partial; a partial start shows at most half of the red cube
  to each camera (compared with a render that hides the gray cubes, which sit in geom group 1), because a fully
  visible red cube is a plain pick. Only about 12% of dropped partial piles pass, so a partial reset takes about
  2.5 s against 0.5 s for a buried one, and reset tries up to 200 piles. A buried start boxes the red cube in, and "buried" means no face of it
  shows in either camera: the count comes from a 640x480 segmentation render at reset, with a 5x5 erosion so the seams
  between touching cubes don't count.
- `so101sim/grasp.py` holds the contact checks both task envs share. `so101sim/rest.py` holds the rule for when
  loose objects count as at rest: every object under 2 mm/s and 0.1 rad/s for 15 frames in a row, checked on free
  joints only. The bin reset hands over a pile and the shelf reset a shelf only once it holds.
- `tests/test_mechsearch.py::dig_out` is a scripted dig-out, not a policy. It clears a buried red cube on 15
  of seeds 0 to 19 with noslip on (12 before 2026-09-29); seed 0 spills, so `SOLVABLE_SEED` is 1.
- The shelf task. `ik.solve_side` holds the gripper level with the wrist camera mount on top.
  `so101sim/side_primitives.py` builds pick, place, pull, push and look as straight lines along rays from the
  pan axis. Held cylinders only move along the aisle line, so only `pull` takes anything off the board.
  `so101sim/shelf_layout.py` and `shelf_checks.py` generate and size the scene the way the search files do for the
  bin. `so101sim/shelf.py` holds `ShelfEnv`, the 30 Hz env, and `ShelfPrimitiveEnv`. The primitive env refuses any
  path that comes within 6 mm of the shelf, checked on a copy of the model with the shelf boxes grown by 6 mm and
  the midphase off.
  `tests/test_shelf.py::dig_out` clears a hidden red on 1 of seeds 0 to 19: 14 cylinders crowd the board. Claude
  played seeds 300 and 301 from images on 2026-09-26 and failed both; `outputs/shelf_play_30*.md` says why.
- Physics: `so101sim.env.load_model` compiles a scene with the gripper as stiff as the scene (models/stage.xml); use
  it for anything that steps physics. `SO101Env.render_depth` gives a camera's depth at observation size.
  `scripts/contact_study.py` measures penetration and pile stability under contact settings.
- `demonstrator/` is Claude playing the bin task from camera data, to make demonstrations. `tools.py` is the CLI it
  plays through (new, observe, check, act, status), with episodes replayed exactly from `demonstrator/episodes/`.
  `perception.py` turns depth and segmentation from the front camera and the wrist camera (at the park pose) into
  cube poses (corrected 2026-09-30: this line said front camera only). `reach.py` has the arm clearance check.
  `instructions.md` is the rulebook and memory: read it before playing, and add each episode's lessons after.
  `live.py` keeps an episode in memory with `snapshot()`/`restore()` (MuJoCo integration state including the warm
  start, the env's counters, the episode's bookkeeping); a restored line replays bit for bit
  (`tests/test_demonstrator_live.py`). Restore order matters: `mj_setState`, `mj_forward`, `mj_setState` again.
  A snapshot costs about 2 µs with `mj_getState`, against 35 ms for a deepcopy of `MjData`. `planner.py` is the expert as a program (see `planner.md`):
  `python demonstrator/planner.py --seeds S... --procs N --out DIR`, one process per seed, a record per seed with every
  candidate it tried. `video.py` renders an episode (front and wrist side by side, captioned). `export.py` turns
  solved episodes into a LeRobot dataset, with front depth in millimetres as `observation.depth.front`; `report.py` and `postmortem.py` tally
  and replay finished ones; `experiments/` holds the sweep pilot.
- `MechSearchPrimitiveEnv` (since f5790cd) also takes a `sweep` kind (a push with its own `stroke`, starting up to 5 mm
  from a wall and up to 3 cm above the rim, refused if it would end against a wall) and an optional `lift_z` on pick
  (a higher final lift). `primitives.pick/place/push` take `flip` (the wrist roll) and `pick` takes `lift_z`. Records
  without these keys replay unchanged.
  Cubes are the proof-of-concept rig; the expert should later handle YCB or other realistic objects. Record general
  lessons in `docs/knowledge/mechanical-search-learnings.md`, written for any objects, with cube-only details kept apart.
- `so101sim/mjx.py` puts the unmodified scene on the GPU with the MuJoCo Warp backend, batched over N
  worlds. The module docstring has the benchmark behind that choice: Warp runs the full scene at about 30k
  physics steps/s for 1024 worlds, and the JAX backend is 7x slower even with contacts trimmed. MJX defaults
  to JAX on GPU, so always go through `so101sim.mjx.put` or pass `impl="warp"` yourself. Image
  observations come from the CPU env for now, but not because MJX can't render: the Warp backend has a
  batched ray-traced renderer (`mjx.create_render_context`, `mjx.refit_bvh`, `mjx.render_with_segmentation`).
  On 2026-09-25 it rendered this scene's front and wrist cameras, meshes and segmentation included, at
  about 7,000 images/s at 128x128 for 1024 worlds. Shadows are off by default and the lighting is
  brighter than the CPU renderer's, so never train on one renderer and evaluate on the other.
