---
paths:
  - "models/*.xml"
  - "so101sim/env.py"
  - "so101sim/*_layout.py"
  - "so101sim/*_checks.py"
  - "so101sim/rest.py"
  - "so101sim/mechsearch.py"
  - "so101sim/shelf.py"
  - "so101sim/binpick.py"
  - "scripts/size_*.py"
  - "tests/test_env.py"
  - "tests/test_rest.py"
  - "tests/test_*_layout.py"
  - "tests/test_*_checks.py"
---

# Traps: scene files and physics

Claude Code loads this file when Claude reads a file matching the paths above. It is the only home of these traps,
which moved here from CLAUDE.md on 2026-09-30. Add a trap here when you find one in this area, dated, with the
evidence. The load follows the Read tool; if you read code with `cat` or `sed`, open this file yourself.

- The scene timestep has to divide 1/30 s exactly. The env raises if it doesn't.
- `stage.xml` overrides `meshdir` after including the vendored arm. MuJoCo resolves `meshdir` against the
  top-level file, so without the override it looks for meshes in `models/assets/`.
- The MJCF base has no collision geometry. `base_collision` in `stage.xml` is a stand-in box.
- Every top-level scene has to sit in `models/`, because `meshdir` resolves against the top-level file.
  `search_layout.candidate_scene` writes its temporary scenes there for that reason, named `.candidate_*`
  and ignored by git.
- Don't use menagerie's `trs_so_arm100`. That's the older 5-DOF SO-100.
- Editing `geom_pos` on a compiled model doesn't move a world-body geom; recompile. Growing `geom_size` in place
  isn't enough either: the midphase's bounding-box tree is built at compile time and skips pairs outside the old
  boxes. Turn the midphase off on a check-only model (`shelf_checks.add_clearance`). An earlier note here blamed
  the box-box collider for ignoring margins; that was wrong, it was the midphase.
- Without noslip, resting objects creep. On 2026-09-29, bin piles held still for 30 s with nothing touching them
  drifted 24 mm median, and some were still moving 2 to 3 mm a second at the end. `stage.xml` turns noslip on
  (`noslip_iterations="10"`), which cuts 10 s of idle drift to 0.85 mm median over seeds 0 to 19 and costs about
  2.5x the CPU time. Solver iterations, impratio, stiffer solref or solimp, pyramidal cones and a 1/600 s timestep
  did not fix it; impratio 100 with a 1/600 s timestep holds piles still but breaks reset, which then accepts 3 buried
  piles in 60.
- Piles are chaotic. Moving one gray cube 1 micrometre at reset changed where 21 cubes ended up after seed 304's 10
  moves, and turned a success into a miss. A saved action list replays exactly only under the same code and physics
  settings. Every plan has to look again after each move.
- Contacts are stiff since 2026-09-30 (dm1487: "we should not be penetrating"): a 1/600 s step, the main default's
  solref at two steps and solimp 0.95 to 0.99, in `stage.xml`. Before, a finger driven onto a 27 g cube's top pressed
  it 11.9 mm into the floor and sweeps pressed cubes 13 mm into each other, because MuJoCo's default contact scales
  its stiffness to the mass it touches while the servo pushes about 12 N. Now the deepest anywhere is 0.74 mm
  (`scripts/contact_study.py`, 16 seeds). Pushing further breaks the solver: solimp 0.999 flung cubes.
- Load anything that steps physics with `so101sim.env.load_model`. It copies the scene's contact stiffness onto the
  vendored gripper, whose class sets solref 0.01 with priority 1 and so decides every finger contact. A bare
  `MjModel.from_xml_path` keeps the soft fingers; that is fine for geometry-only checks, not for dynamics.
- Episodes recorded before the stiff contacts (2026-09-30, about 16:00) do not replay under them; piles are chaotic.
- Reset holds a finished pile still for 5 s and rejects it if any cube moves over 0.5 mm (`CREEP_SECONDS`,
  `CREEP_MM` in `mechsearch.py`). The rest rule alone passed a cube creeping at 0.18 mm/s that later fell 32 mm
  (d1031).
