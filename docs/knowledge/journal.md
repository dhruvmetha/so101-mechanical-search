# Journal

The project's log of what happened, what was decided and why, oldest first, newest at the bottom. Append only. Never
edit or delete an entry; to correct one, append a new entry that names the old one and says what was wrong.
`state.md` says where things stand now; this file says how they got there.

It is also the record that lets another person, with an LLM, reproduce the demonstrations. dm1487, 2026-09-30:
"maintain a log of every decision we make and also you make to maintain memory, because that is what helps another
person replicate the demos (information to instruct an LLM)". So every decision goes in, dm1487's and Claude's, large
or small: a rule adopted or dropped, a number picked, a tool feature added, a run's settings, an approach rejected.
Per-move choices inside an episode are the one exception: every act in `demonstrator/episodes/*.json` stores its own
"why".

## Entry format

Each entry starts with a heading in one fixed form, so `grep '^## \[' docs/knowledge/journal.md | tail -5` lists the
last five; the form follows Karpathy's LLM Wiki log. Take the time from `date`.

    ## [YYYY-MM-DD HH:MM] kind | title

Kinds: `decision`, `direction` (dm1487's words, quoted where possible), `run`, `result`, `mistake`, `compaction` (a
write-back after one), `consolidation` (what moved where), `note`.

A decision or direction entry has these lines:

    - Decided: what, exactly, with the values.
    - Why: the reason, with numbers, and the alternatives turned down.
    - By: dm1487 | Claude (main session) | Claude, approved by dm1487 | agent dNNNN.
    - Evidence: code, commit, notes, outputs, as paths.
    - Replaces: the earlier entry it reverses or amends, if any.

A run entry, for a batch of episodes or any job whose output someone may want to reproduce, has:

    - Commit: `git rev-parse --short HEAD`, and whether `git status --short` showed changes.
    - Simulator: the MuJoCo version (`pyproject.toml` pins 3.8.1) and any change to `models/stage.xml`.
    - Seeds and start kinds; the player (the agent's model, or Claude by hand); how many ran in parallel.
    - Rulebook and brief: the `demonstrator/instructions.md` version, and `AGENT_BRIEF.md` as of the commit.
    - Commands: exactly as run, with the conda env.
    - Directions from dm1487 that shaped the run.

A result entry closes the run: outcomes with numbers, and where the files are.

Entries with "(reconstructed)" in the title were written after the fact, on 2026-09-30, from the files named in their
Evidence line; where the files give no time, the heading has the date only, and entries of the same date without a
time are in no known order. When this file passes 800 lines, entries from earlier months move unchanged to
`journal/YYYY-MM.md`.

## [2026-09-18] decision | Sim design spec (reconstructed)

- Decided: the SO-101 in MuJoCo on arrakis, headless, with lerobot units at the env boundary.
- Why and alternatives: in the spec.
- By: not recorded in this journal; see the spec.
- Evidence: `docs/superpowers/specs/2026-09-18-so101-sim-design.md`.

## [2026-09-22] result | Hardware measurements from the `so101` session (reconstructed)

- Sag, backlash and per-joint stiffness fits on the physical arm, as of this date.
- Evidence: `docs/knowledge/real-arm.md`.

## [2026-09-25] decision | Bin task definition (reconstructed)

- Decided: a bin holds a pile of 30 mm cubes up to three layers deep, one red and the rest gray, and the robot lifts
  the red one out. Cubes only, no language instruction. Starts are 50/50 buried and partly visible. A cube centre
  leaving the bin fails the episode, with no buffer area. Two interfaces: a 30 Hz joint env and a pick, place and push
  primitive env. CPU MuJoCo only; `BinPickEnv` stays as it is.
- Why: it settles the task CLAUDE.md had left open, following mechanical search (Danielczuk et al., ICRA 2019). The
  individual choices have no recorded reasons.
- By: dm1487 approved the spec on 2026-09-25.
- Evidence: `docs/superpowers/specs/2026-09-25-mechanical-search-design.md`:3, 7-19, 24-30.

## [2026-09-25] decision | Sink the bin below the base, and the height rules (reconstructed)

- Decided: the arm stands on a riser, so the bin floor sits below the base. The floor is 4 mm thick with its top at
  f = -D + 0.004. Layer centres at f + 1.5, 4.5 and 7.5 cm. TRAVEL_Z = max(f + 11 cm, rim + 3 cm).
- Why: top-down reach narrows with height and ends about 9 cm above the base. On the base's own table a two-layer pile
  needs an 8.4 cm crossing where the band is 5 cm wide, which caps the bin at 10 cm with 5% still unreachable. With the
  floor 10 cm down, 12, 15 and 18 cm bins reached 100% (without collision checks). TRAVEL_Z keeps a carried cube 5 mm
  over the pile and a lifted cube 1.5 cm over the rim.
- By: dm1487 approved the spec.
- Evidence: `docs/superpowers/specs/2026-09-25-mechanical-search-design.md`:18, 32-64.

## [2026-09-25] decision | Bin sizing, camera and start-pose rules (reconstructed)

- Decided: search depth D in {6, 8, 10, 12} cm, width in {18, 15, 12} cm, centre x in {0.19, 0.20, 0.21}, walls in
  {6, 8, 10} cm. Checks on a 5 mm grid: layer-1 and layer-3 grasp, travel at GRIP_PLAN and GRIP_HOLDING, open-jaw
  clearance for at least one of 4 yaws, corners exempt. Rank by widest, then shallowest D, tallest wall, centre
  nearest 0.20. Front camera at x = 0.70, at the lowest z in {0.45, 0.55, 0.65, 0.75} that sees at least 70% of the
  floor. Start pose HOME by default, clear of the drop column below f + 13 cm, with the wrist camera seeing at least
  50%. If nothing passes at 12 cm, stop and bring it to dm1487.
- Why: a lower riser is easier to build, and a low camera sees the pile from the side.
- By: dm1487 approved the spec.
- Evidence: `docs/superpowers/specs/2026-09-25-mechanical-search-design.md`:82-101, 117-127;
  `docs/superpowers/plans/2026-09-25-mechanical-search.md`:1181-1197.

## [2026-09-25] decision | Bin reset, acceptance and cube-count rules (reconstructed)

- Decided: cubes drop in waves of at most 4, at least 4.5 cm apart, 1 cm over the highest cube, each wave settling to
  max|qvel| <= 0.2 (capped at 1 s). A pile is kept if every centre is in the footprint, none sits above f + 8.5 cm,
  a free floor spot exists 1.8 cm from the walls and 4.5 cm from bottom-layer centres, the pile has settled, no cube
  touched the arm, and the red rule holds (buried: 0 red pixels in both cameras; visible: at least 100 in one, at
  640x480). Retries reuse the same `np_random`, up to 50 attempts. The count starts at 30 and drops until at least 50%
  of piles are kept for each start type over 100 seeds, with a floor of 10. Reset should average under 3 s; if not,
  report the number rather than cut cubes.
- Why: a seed always gives the same pile. f + 8.5 cm caps the pile at three layers. The 4.5 cm spacing is two
  half-diagonals plus 2.6 mm.
- By: dm1487 approved the spec.
- Evidence: `docs/superpowers/specs/2026-09-25-mechanical-search-design.md`:129-152, 166;
  `docs/superpowers/plans/2026-09-25-mechanical-search.md`:1531-1558.

## [2026-09-25] decision | Bin env contracts and the solvability gate (reconstructed)

- Decided: `MechSearchEnv` terminates when a cube leaves the bin. Success is both jaws on the red with its lowest
  corner at least 1 cm over the rim for 5 frames. max_steps 1800. The primitive env's action: kind Discrete(3), an xyz
  box pulled in by a cube half-width with z from f + 1.5 to f + 7.5 cm, and an angle. Push stroke 0.05 m, a 60-frame
  settle, refused moves leave the arm still, max_primitives 15. Gate: a scripted dig-out from `info` poses must
  succeed on some buried seed in 0 to 19.
- Why: one xy can hold a cube on any of three layers, so z is in the action. A red under a two-cube stack needs 5
  primitives, and jammed neighbours add pushes.
- By: dm1487 approved the spec.
- Evidence: `docs/superpowers/specs/2026-09-25-mechanical-search-design.md`:168-218, 245;
  `docs/superpowers/plans/2026-09-25-mechanical-search.md`:34-38, 2372-2397.

## [2026-09-25] decision | Where the bin plan departed from the spec (reconstructed)

- Decided: the layout and SEARCH_HOME live in the XML's `<custom>` fields, and TRAVEL_Z is computed from the bin.
  Camera coverage uses 6 mm marker tiles 1 cm apart in a segmentation render instead of `mj_ray`. The drop column is
  an invisible collision box. `bridge` ramps joints only below hover_z - 0.025. `info["cubes"]` is added.
- Why: the chosen layout can't drift from the code. Hand-written camera maths would get the wrist camera's
  sensorsize intrinsics wrong. A fixed z = 0.03 threshold would ramp every crossing in the sunken bin through the pile.
- By: not recorded.
- Evidence: `docs/superpowers/plans/2026-09-25-mechanical-search.md`:46-56, 425-433, 494-498.
- Replaces: the spec's constants and its `mj_ray` wording (spec lines 119, 225).

## [2026-09-25] decision | Bin amendment 1: walls at most 6 cm, and a 2 cm wall strip (reconstructed)

- Decided: walls in {6, 5, 4} cm. The clearance check skips any cube whose face is within 2 cm of a wall; those
  cubes need a push first. The pile keeps three layers.
- Why: the first sizing run found no layout, because clearance failed everywhere. With a 10 cm riser, 6 cm walls on a
  15 cm bin blocked 41 of 169 positions, all at the wall; 8 cm walls blocked 82 of 169, up to the middle, on the
  camera mount; 10 cm walls on an 18 cm bin blocked 65 of 225.
- By: not recorded; the amendment sits in the spec dm1487 approved.
- Evidence: `docs/superpowers/specs/2026-09-25-mechanical-search-design.md`:20, 90, 95, 103-115;
  `docs/superpowers/plans/2026-09-25-mechanical-search.md`:916.
- Replaces: walls of 6, 8 or 10 cm and the corner-only exemption.

## [2026-09-25] decision | Bin amendment 2: face pixels, a boxed burial, bins up to 24 cm (reconstructed)

- Decided: a red pixel counts only if its whole 5x5 patch is red. A buried red sits on the floor with four grays flush
  against its sides, 1 mm apart, and a fifth on top, all settled as one wave. Widths of 24, 22 and 20 cm join the
  search.
- Why: with one gray on top, the red still showed through 1 mm seams. Piles kept out of 30, one-on-top against boxed:
  0 vs 26 at 14 cubes, 3 vs 20 at 18, 1 vs 9 at 24. Most losses at 24 had no free floor spot, which a wider bin eases.
- By: not recorded; the amendment sits in the spec dm1487 approved.
- Evidence: `docs/superpowers/specs/2026-09-25-mechanical-search-design.md`:21, 88, 141, 152, 154-164;
  `docs/superpowers/plans/2026-09-25-mechanical-search.md`:1388.
- Replaces: the one-gray-on-top recipe and the raw zero-pixel rule.

## [2026-09-25] decision | The bin layout the sizing script picked (reconstructed)

- Decided: a 22 cm bin at x = 0.20, 10 cm riser, 5 cm walls, travel z +0.014. Front camera at (0.70, 0, 0.45), seeing
  86% of the floor. Start pose [0, -30, 30, 90, 0, 50], which is HOME (`so101sim/env.py`:16), with a wrist view of 79%.
- Why: the first candidate by rank to pass, found by `scripts/size_search_bin.py` in 35 s after 57 better-ranked
  candidates failed.
- By: the script, under the rules above.
- Evidence: `models/search_objects.xml`:5-8, 12-13, 19.

## [2026-09-25] decision | A stricter success height for the bin (reconstructed)

- Decided: the red's lowest corner must also sit 2 cm above where a top-layer cube rests, floor + 8 cm.
- Why: with 5 cm walls, rim + 1 cm equals that resting height, so a grasp and a 2 mm lift counted. A primitive pick
  holds the corner at about floor + 8.8 cm.
- By: not recorded ("changed after the final review"). Date from the spec's filename; the spec gives none.
- Evidence: `docs/superpowers/specs/2026-09-25-mechanical-search-design.md`:175.
- Replaces: rim + 1 cm alone (`docs/superpowers/plans/2026-09-25-mechanical-search.md`:25).

## [2026-09-25] result | Warp's batch renderer draws this scene (reconstructed)

- The MuJoCo Warp backend rendered the front and wrist cameras, meshes and segmentation included, at about 7,000
  images/s at 128x128 for 1024 worlds. Shadows are off by default and the lighting is brighter than the CPU
  renderer's, so training and evaluation must use the same renderer.
- Evidence: `docs/knowledge/codebase.md`, the `so101sim/mjx.py` entry.

## [2026-09-26] decision | Bin amendment 3: partial starts replace visible ones (reconstructed)

- Decided: a non-buried start needs at least 100 face pixels in one camera, and no camera may show more than half of
  what it would see of the red with the grays hidden (grays sit in geom group 1). The red drops in a random middle
  wave. The mix stays 50/50.
- Why: "a fully visible red cube is a plain pick and tests nothing about search."
- By: dm1487 ("dm1487's call").
- Evidence: `docs/superpowers/specs/2026-09-25-mechanical-search-design.md`:22, 142, 148;
  `models/search_objects.xml`:33.
- Replaces: visible starts with the red in the last wave.

## [2026-09-26] decision | 30 cubes, and partial starts judged by reset time (reconstructed)

- Decided: the half-kept rule covers buried starts only; partial starts are judged by the 3 s reset budget. Reset
  attempts go from 50 to 200. The count stays at 30.
- Why: partial starts kept about 12% at every count from 30 down to 10, averaging 2.7 s at 30. The recorded run kept
  59% of buried piles at 0.5 s per reset and 14% of partial piles at 2.3 s.
- By: dm1487 ("dm1487 chose").
- Evidence: `docs/superpowers/specs/2026-09-25-mechanical-search-design.md`:131; `models/search_objects.xml`:9. No
  date of its own; it goes with amendment 3.
- Replaces: the 50% rule for both start types, and 50 reset attempts.

## [2026-09-26] decision | Shelf task definition (reconstructed)

- Decided: a shelf bay with a top board holds upright cylinders, all 3 by 8 cm, one red. The gripper enters level
  through the front. Success is the red held in both jaws and pulled past the front edge. A tilt past 20 degrees
  fails, as does any gray leaving the board. Two interfaces, CPU only. Claude plays the primitive env from images
  alone: seeds 300 (hidden) and 301 (partial), fixed before looking, no undo, at most 20 primitives, pixel actions
  through `scripts/shelf_play.py`.
- Why: follows LAX-RAY (Huang et al., 2020). atan(3/8) = 20.6 degrees is the tilt past which a cylinder can't right
  itself.
- By: dm1487 approved the spec on 2026-09-26.
- Evidence: `docs/superpowers/specs/2026-09-26-shelf-search-design.md`:3, 15-29, 363-382;
  `docs/superpowers/plans/2026-09-26-shelf-search.md`:3059-3070.

## [2026-09-26] decision | Shelf above the base, mount-up roll, fixed heights (reconstructed)

- Decided: the board sits above the base plane. Every grasp uses the roll with the camera mount on top, so the moving
  jaw opens to the right. GRASP_Z = b + 4 cm, CARRY_Z = b + 5 cm. The aisle line at f + 2 cm and the standoff line at
  f - 3 cm; reset keeps centres at x >= f + 5.5 cm. Primitives run along rays: pick, place, pull (the only move off
  the board), push (jaws closed, 5 cm, 16 headings), look.
- Why: level reach is wide above the base and absent below -0.08 m. With the mount down it hangs 8 to 9 cm under the
  gripper, hits the board, and the gripper gets only 4.5 cm into the bay.
- By: dm1487 approved the spec.
- Evidence: `docs/superpowers/specs/2026-09-26-shelf-search-design.md`:50-122, 247-271.

## [2026-09-26] decision | Shelf sizing, reset and count rules (reconstructed)

- Decided: search b in {3, 6, 9} cm, opening in {13, 15, 18} cm, f in {0.27, 0.285, 0.30} m, depth 16 to 22 cm, width
  20 to 32 cm. Six checks on a 5 mm grid, with a 6 cm right-wall exemption for the open jaw. Rank by graspable area,
  then smallest opening, then nearest edge; stop below 12x18 cm. Front camera sees at least 60% of the board, wrist
  camera at least 40%. A hidden red goes in the back two thirds with a gray 3.5 to 5 cm in front on each camera's
  sight line; a partial red goes last, at random. Cylinders at least 3.5 cm apart, upright within 2 degrees, 50
  attempts. The count starts at 18 with a floor of 8, and at least 50% must be kept per start type.
- Why: reach bottoms out near 0.255 m, so f can't come closer. GRIP_OPEN swings the moving jaw 55 mm and would knock
  over the right-hand neighbour.
- By: dm1487 approved the spec.
- Evidence: `docs/superpowers/specs/2026-09-26-shelf-search-design.md`:141-167, 224-237, 273-303.

## [2026-09-26] decision | Shelf env contracts (reconstructed)

- Decided: checks in order: topple (over 20 degrees, held or not), off-board (any gray, or the red when not held),
  success (both jaws, 2 cm past the edge, tilt under 20 degrees, 5 frames). max_steps 2700. The primitive env's kind is
  Discrete(5) and it takes an xy with no z. max_primitives 20. Holding is read from the gripper joint.
- Why: moving in and out along rays is slower than the bin's hover moves, hence 90 s. Every cylinder stands on one
  board, so no z. The joint stops near 18 on a cylinder, and the real arm reads the same joint.
- By: dm1487 approved the spec.
- Evidence: `docs/superpowers/specs/2026-09-26-shelf-search-design.md`:319-361;
  `docs/superpowers/plans/2026-09-26-shelf-search.md`:2550-2554.

## [2026-09-26] decision | Where the shelf plan departed after the reach spike (reconstructed)

- Decided: rays start at the pan axis (0.0388, 0). A push enters at grasp height and drops past the edge. Push height
  is fixed at b + 2.5 cm. The "12x18 cm" stop rule means 216 cm² of area. Holding means the last command was at most
  GRIP_CLOSE + 1 and the joint reads at least 14. The side-open value is the smallest lerobot value with a gap of at
  least 5 cm, which is 33.
- Why: the spike measured reach inner limits of 0.220, 0.225 and 0.240 m and an outer limit of 0.440 m, so only
  f = 0.30 works. A 5 cm push at 2 cm left a cylinder leaning 9 degrees, at 2.5 cm slid it 4 cm upright, at 3 cm
  toppled it.
- By: not recorded.
- Evidence: `docs/superpowers/plans/2026-09-26-shelf-search.md`:60-88, 481-483, 898-900.
- Replaces: the spec's "lowest of b + 2, 2.5 or 3 cm" push height (spec line 115).

## [2026-09-26] decision | Shelf amendment 1: arm contact fails, a 3 cm wall strip (reconstructed)

- Decided: the arm touching any shelf panel, leg or the table ends the episode, and the primitive env refuses such
  paths. Sizing skips centres within 3 cm of a side wall, and the red always starts outside that strip. Digs try a
  push before a pick.
- Why: the first sizing run found no layout. At 2 to 2.5 cm from a wall the closed gripper hits the walls. A red the
  gripper can't fit around would make the start impossible rather than hard.
- By: dm1487 ("dm1487 decided").
- Evidence: `docs/superpowers/specs/2026-09-26-shelf-search-design.md`:31-33, 169-189.
- Replaces: sizing with no wall exemption.

## [2026-09-26] decision | Shelf amendment 2: partial starts occluded on purpose (reconstructed)

- Decided: a partial red goes in the back two thirds with a gray on each camera's line 3.5 to 5 cm in front, shifted
  0.8 to 1.6 cm to one side.
- Why: with a random red, partial starts kept at most 22% at 14 cylinders; 79 of 150 attempts showed more than half
  the red. Over 200 seeds a 0.8 to 1.6 cm shift kept 57%, against 49% for 1.2 to 2.2 cm.
- By: dm1487, over a reset-time rule.
- Evidence: `docs/superpowers/specs/2026-09-26-shelf-search-design.md`:35-36, 305-317.
- Replaces: the random partial red (spec line 284).

## [2026-09-26] decision | Shelf amendment 3: 6 mm clearance, 15 cm opening, lowest camera and pose (reconstructed)

- Decided: every collision check needs 6 mm of clearance, run on a model with the shelf boxes grown and the midphase
  off, and sizing adds a carry check at every centre. Result: b = 9 cm, a 15 cm opening, f = 0.30 m, 18x32 cm. The
  front camera takes the lowest height in {0.20, 0.25, 0.30, 0.40} m that sees at least 60%, the start pose the lowest
  whose wrist camera sees at least 40%. Hidden starts add up to three grays wherever a camera still sees the red.
- Why: every place hit the top board, and the arm bounces about 5 mm at carry height; at 8 mm no shelf passes. The
  midphase skipped pairs in the grown band. dm1487 wants starts that are really hidden: a camera at 0.40 m kept 0% of
  hidden starts and one at 0.20 m kept 40 to 50%; with the wrist camera 4.5 cm over the cylinder tops no hidden start
  survived, at 1.5 cm 90% did.
- By: dm1487 chose 6 mm and set the hidden-start goal.
- Evidence: `docs/superpowers/specs/2026-09-26-shelf-search-design.md`:38-40, 191-222.
- Replaces: contact-only checks, the 13 cm opening, spec lines 230-236 and the one-gray-per-camera recipe (281-283).

## [2026-09-26] decision | The shelf layout and count the scripts picked (reconstructed)

- Decided: board +9 cm, opening 15 cm, f 0.300 m, 18x32 cm, centres over 10.5x28 cm. Front camera at (-0.10, -0.30,
  0.20), seeing 95%. Start pose [-21.43, -37.73, 83.26, -45.53, -87.21, 22.0], wrist view 75%. 14 cylinders: 83% of
  hidden starts kept at 0.1 s, 54% of partial at 0.0 s. The open jaw at lerobot 28.
- Why: the first candidate by rank to pass, found in 70 s after 124 better-ranked candidates failed. On 28 against 33:
  `so101sim/side_primitives.py`:19-23 says 33 dragged a released cylinder 14 mm and 28 dragged it 1.4 mm (undated).
- By: `scripts/size_shelf.py` and `scripts/shelf_smoke.py --tune-count`, under the rules above.
- Evidence: `models/shelf_objects.xml`:5-14, 30.
- Replaces: a side-open value of 33.

## [2026-09-26] decision | Cylinders use condim 6 (reconstructed)

- Decided: condim 6, friction "1 0.005 0.0001", density 1000.
- Why: under condim 4 a pushed cylinder wobbled on its rim without end; condim 4 ignores the rolling coefficient.
- By: not recorded. `so101sim/shelf_layout.py`:38-40 dates the measurement 2026-09-26.
- Evidence: `docs/superpowers/specs/2026-09-26-shelf-search-design.md`:135-137; `models/shelf_objects.xml`:33.
- Replaces: condim 4.

## [2026-09-26] decision | Shelf seeds 300 and 301 played from images only (reconstructed)

- Decided: Claude plays the shelf from the front and wrist images and the printed step results, with no poses and no
  undo: seed 300 hidden, 301 partial. Both failed; a grasp short of the cylinder's axis swung the red past 20 degrees
  on 300.
- Why: not recorded.
- By: Claude.
- Evidence: `outputs/shelf_play_300.md`:1-3, 91-100; `outputs/shelf_play_301.md`:1-3, 26-33.

## [2026-09-26] decision | Shelf pull's holding threshold lowered from 14 to 11 (reconstructed)

- Decided: `pull` counts the red as held at gripper 11 or more instead of 14, with a test. Seed 300's action list was
  replayed unchanged.
- Why: an off-centre grasp with both jaws on the red read 12.7 and pull refused it; an empty close reads 10.0.
- By: Claude ("I set the threshold to 11 with a test").
- Evidence: `outputs/shelf_play_300.md`:83-89.

## [2026-09-26] decision | Shelf trials on seeds 302 to 305 (reconstructed)

- Decided: seeds fixed before looking (302 and 304 hidden, 303 and 305 partial), at most 20 primitives, no poses, no
  undo. Aids limited to pixel deprojection (`where.py`) and a path check with a stand-in cylinder (`plancheck.py`).
  Result: 3 of 4 solved.
- Why: the path check "would have flagged the pick that failed seed 301".
- By: Claude.
- Evidence: `outputs/shelf_trials.md`:1-7, 61-63.
- Replaces: the 300 and 301 setup. The shelf has since shrunk (18 to 15 cm opening, 8 to 6 mm path check), so 300 and
  301 no longer replay (`outputs/shelf_play_300.md`:5-7).

## [2026-09-28] decision | Hand-played bin seeds 301 to 310 become the teacher's spec (reconstructed)

- Decided: Claude hand-played bin seeds 301 to 310 as planner from 2026-09-26 to 09-28, and its lessons became "the
  spec for the teacher program that is to come next": the 42 mm grasp budget across two faces, a 15 mm slot behind a
  push, a layer-2 push toward the base to slide a cover off, reading the gripper (about 18 holding, 10 empty) before
  queuing a place.
- Why: 10 of 10 solved, 5.8 moves on average (buried 7.6, partial 4.0), 24 of 37 grasps missed.
- By: Claude.
- Evidence: Claude's memory file `mechsearch-planner-playbook.md`; videos `outputs/llm_*_seed30*.mp4` are likely from
  these runs, but no file says so.

## [2026-09-28] decision | One artifact for the weekly advisor meetings (reconstructed)

- Decided: dm1487 kept one artifact, "Mechanical Search Log", for weekly advisor meetings, first published 2026-09-28,
  covering both tasks.
- Why: one stable link to present every week, with results, videos and findings.
- By: dm1487.
- Evidence: Claude's memory file `weekly-advisor-log.md`.

## [2026-09-29] decision | noslip on; piles handed over at rest; old action lists retired (reconstructed)

- Decided: `models/stage.xml` sets `noslip_iterations="10"`. Resets hand over a pile only once it holds still
  (`so101sim/rest.py`: every object under 2 mm/s and 0.1 rad/s for 15 frames). The action lists of bin seeds 301 to 310
  and shelf seeds 302 to 305 no longer replay, and their videos are the record.
- Why: without noslip, piles left alone for 30 s drifted 24 mm median; with it, 10 s of idle drift is 0.85 mm median,
  at about 2.5x the CPU time. Solver iterations, impratio, stiffer solref or solimp, pyramidal cones and a 1/600 s
  timestep did not fix it. The scripted dig-out went from 12 to 15 of seeds 0 to 19.
- By: not recorded.
- Evidence: `.claude/rules/scenes-and-physics.md`; `mechsearch-planner-playbook.md` (memory);
  `outputs/shelf_trials.md`:65-66.

## [2026-09-29] direction | Advisor log paused (reconstructed)

- Decided: no advisor log for now; focus on the demonstrator.
- By: dm1487, as recorded in Claude's memory: "no need for the advisor log for now and to focus on the demonstrator".
- Evidence: Claude's memory file `weekly-advisor-log.md`:11.

## [2026-09-29] direction | The rulebook is the memory (reconstructed)

- Decided: `demonstrator/instructions.md` replaces the playbook as the working rulebook. Claude follows it when
  playing, and adds each episode's lessons and a change-log line after every episode.
- By: dm1487 ("dm1487 asked for it to act as memory").
- Evidence: `mechsearch-planner-playbook.md` (memory):11-13; `demonstrator/instructions.md`:3-5.
- Replaces: the playbook memory file as the rulebook.

## [2026-09-29] decision | Episodes are replayed op lists; the expert decides from camera data only (reconstructed)

- Decided: an episode is a JSON op list (`demonstrator/episodes/NAME.json`: seed, buried, ops, undone), rebuilt by
  replay from reset on every tool call. Only `act` counts as a move; parks cost robot time and are real motions;
  `check` never blocks. The player never reads true poses.
- Why: replay was "checked 2026-09-29, bit-identical across processes". True poses "would make the demonstration
  something a camera-based policy can't copy."
- By: not stated in the files. `state.md` lists it as a 2026-09-29 decision.
- Evidence: `demonstrator/tools.py`:1-19; `demonstrator/instructions.md`:68; `demonstrator/AGENT_BRIEF.md`:31-33.

## [2026-09-29] decision | Rulebook v1 (reconstructed)

- Decided: the first rulebook, with a 15-move cap and a loss on any cube leaving the bin.
- Why: built from hand-played seeds 301 to 310 and "the program test on seeds 400 to 403" (no file describes that
  test, and there is no e403).
- By: not stated; Claude wrote it.
- Evidence: `demonstrator/instructions.md`:12, 258.

## [2026-09-29] decision | Rulebook v2, after e400 (reconstructed)

- Decided: the success line and the set-down-and-regrasp rule, nudging a boxed red, grasping a tilted cube high, where
  to put covers, reading the red line in the image. Tools: observe doesn't park while holding, and check leaves the
  target cube out of its hits.
- Why: e400 (Claude, buried): solved in 11 moves and 68 s. Move 9 held the red tilted and missed success by 0.7 mm;
  then the tool's park opened the gripper and dropped it (a bug, fixed), and the next park swing shook it out.
- By: Claude.
- Evidence: `demonstrator/instructions.md`:240-246, 259-261.

## [2026-09-30] decision | Rulebook v3: the wrist flip, after e401 and the reach study (reconstructed)

- Decided: primitives take `flip` (0 or 1) to pick the wrist roll. check tries both rolls over the full path and
  reports the arm against itself, fingers within 3 mm of a wall, and the camera mount against the scene. An unpark
  step. `reach.Clearance` samples points instead of trusting `mj_geomDistance`.
- Why: e401 (partial) failed at move 3 by knocking a cube out. Move 2 missed by 4.5 cm because the wrist had to roll 93
  degrees from home, grounded its camera mount on the shoulder and lagged; move 3's IK chose a roll that folded the arm
  into the shoulder, and the lagging arm swept a layer-3 cube over the rim. On an empty bin, 182 picks: 54 of 55 landed
  within 6 mm with the arm 10 mm or more from itself and the fingers 3 mm from every wall; fingers within 3 mm of a wall
  tracked in 8 of 80. `mj_geomDistance` returned 0 for boxes 30 mm apart.
- By: Claude.
- Evidence: `demonstrator/instructions.md`:183-191, 248-254, 262-264; `demonstrator/REPORT.md`:40-45.
- Replaces: the near-base "dead strip" (x at or below 0.14), which was the wrist roll, not reach.

## [2026-09-30] decision | Rulebook v4: undo, and the red-pixel line (reconstructed)

- Decided: `undo`, which keeps the attempt under "undone" and out of the demo. Always undo a spill, undo a miss that
  made things worse, keep a useful miss, never repeat an undone move exactly, at most 8 undos. observe places red
  pixels in 3D.
- Why: "A demonstration should show good moves." The sim is exact, so an undone line replays as if it never happened,
  and e400's misses had opened gaps.
- By: not stated; Claude wrote it. (`state.md` files undo under 2026-09-29; the change log says v4, 2026-09-30.)
- Evidence: `demonstrator/instructions.md`:28-30, 59-68, 265; `demonstrator/REPORT.md`:93-94.

## [2026-09-30] run | Overnight agent run, seeds 1000 to 1034 (reconstructed)

- Commit: none. The code was not committed until b0455b2 at 11:26, after the run, and tools changed during the run.
- Simulator: MuJoCo 3.8.1 on CPU, noslip on (since 2026-09-29).
- Seeds and start kinds: 1000 to 1034, episode dSEED, even seeds buried and odd partial (true of every d episode
  file). d1023 and d1026 were never started.
- Player: eight Claude agents in parallel. Which model they ran on, and which agent got which seeds, is not recorded.
- Rulebook and brief: `demonstrator/instructions.md` from v4 at the start to v8 during the run (see the mistake entry
  below); `demonstrator/AGENT_BRIEF.md` (read the rulebook first; decide only from observe, check and images; write
  no Python; edit only `episodes/` and `notes/`; under 60 tool calls; a note of 20 lines or fewer per episode).
- Commands: `python demonstrator/tools.py new|observe|check|act|undo|places|status|list ...`, each run as in
  `AGENT_BRIEF.md`:19 with the `so101sim` conda env and a 600000 ms Bash timeout.
- Directions from dm1487: none recorded for the run itself.
- Evidence: `demonstrator/AGENT_BRIEF.md`; `demonstrator/REPORT.md`; Claude's memory `common-sense-every-turn.md`:15.

## [2026-09-30] mistake | Tools and rules changed under running agents (reconstructed)

- What: rulebook v5 to v8 and tool changes went live mid-run, with no freeze per batch, so early and late episodes
  were played under different rules. d1000 saw v5 to v6.2 arrive and re-read the rulebook at move 14; observe's
  output changed partway through d1005, d1013 and d1016. The brief also changed during the run, and neither its old
  versions nor the spawn prompt were kept.
- Cost: which rulebook version each overnight episode followed is not recorded. From the change log: d1003, d1004,
  d1006, d1009, d1012 and d1015 ran under v4; d1000, d1005, d1013 and d1016 saw changes mid-episode; the rest fall
  between v5 and v8. An agent read the rulebook once, before its first episode, so later episodes may have followed an
  older copy.
- By: Claude.
- Evidence: `demonstrator/notes/d1000.md`:15, `d1005.md`:19-20, `d1013.md`:21-22, `d1016.md`:20;
  `demonstrator/REPORT.md`:91-92.

## [2026-09-30] decision | Rulebook v5 and v5.1 (reconstructed)

- Decided: clear both ends of one jaw axis instead of aiming around neighbours; nudging only as a last resort; a 2 mm
  limit on shifting away from the fixed finger (worded backwards until v9); 40 mm or more of unseen space on a finger's
  path counts as a hit; a diagonal red is taken across its corners parallel to the wall; find place spots by hand; the
  red is under the cube that shows the red line. v5.1: observe lists the cubes at each end of each jaw axis.
- Why: from d1003, d1004, d1006, d1009, d1012 and d1015, all solved. Shifts of 6 to 13 mm or turns of 9 to 15 degrees
  never cleared a flush neighbour. 47 mm of unseen space hid a cube.
- By: Claude, from the agents' notes.
- Evidence: `demonstrator/instructions.md`:100-102, 118-125, 129-130, 142-145, 206-208, 266-270.

## [2026-09-30] decision | Rulebook v6, v6.1 and v6.2: the wrist camera joins observe (reconstructed)

- Decided: observe fuses the wrist camera with the front camera. check leaves the held cube out of a place's hits and
  names the fixed finger's side. `places` lists checked spots 6 cm from the red at two yaws.
- Why: unseen floor fell from about 60% to 38%, pose error from 1.4 to 0.6 mm median. The wrist camera's pixels are
  not square (16:9 sensor in a 4:3 render), which had put x off by a third. e402 lost two picks to a cube the front
  camera couldn't see.
- By: Claude.
- Evidence: `demonstrator/instructions.md`:34-37, 154-156, 271-273; `demonstrator/REPORT.md`:46-49;
  `demonstrator/notes/e402.md`:21, `d1008.md`:17. The brief lacked `places` for part of the night (`d1018.md`:19,
  `d1024.md`:14-15).

## [2026-09-30] decision | Rulebook v7 and v7.1 (reconstructed)

- Decided: slide the grasp along the faces, up to 12 mm, before digging out an offset neighbour (d1010); a nudge after
  one cleared end costs two more moves (d1008); gaps of 3 mm or less count as flush (d1005); raise a layer-2 grasp by a
  wall (d1011); near the base, use the axis with 30 mm of clearance; check both axes on a partial start, since one-move
  starts exist (d1011); places need finger room. v7.1: check samples the finger meshes every 2 mm.
- Why: from d1005, d1007, d1008, d1010 and d1011. The finger meshes have faces up to 60 mm across with vertices only at
  the corners, and a vertex-only check passed a finger through a flush neighbour (d1005). Over 76 scored picks, a worst
  hit under 3 mm lifted 96%, 3 to 10 mm 80%, 10 mm or more 75%.
- By: Claude.
- Evidence: `demonstrator/instructions.md`:88-91, 103-109, 126-128, 192-198, 220-221, 274-277.

## [2026-09-30] decision | Rulebook v8 (reconstructed)

- Decided: raise a layer-2 grasp by a wall in 1 mm steps (5 to 9 mm worked four times); a nudge can pinch a gray; set a
  tilted red down in an empty slot; a grip that reads over 20 needs more than 5 mm of room; count moves against the cap.
- Why: d1000 failed at move 15 with the red held tilted 15 degrees; it needed about 16 moves. Also from d1016, d1017,
  d1021 and e402.
- By: Claude.
- Evidence: `demonstrator/instructions.md`:110-113, 167-168, 192-196, 216-219, 278-279; `demonstrator/notes/d1000.md`:1,
  8.

## [2026-09-30 02:33] mistake | No usage budget or restart plan for the overnight run (reconstructed)

- What: eight agents ran with no pacing and no recovery. The account's session limit stopped every agent mid-episode
  at 02:33, and the run sat dead until morning. d1002, d1014, d1020, d1022, d1025, d1029, d1032 and d1034 are half
  played; their moves are saved. `MUJOCO_LOG.TXT` logged "Insufficient arena memory" at 00:02; which episode it hit is
  unknown.
- Result: 24 agent episodes solved (10 of 11 buried and 14 of 14 partial among those finished), d1000 failed. About 38
  requests, 4.8M cached tokens and 15 to 20 minutes per episode.
- By: Claude.
- Evidence: `demonstrator/REPORT.md`:9-12; `demonstrator/consult/fable-2026-09-30.md`:11-12; Claude's memory
  `common-sense-every-turn.md`:15-16.

## [2026-09-30] decision | Rulebook v9 (reconstructed)

- Decided: the 2 mm shift rule now says away from the fixed finger, in the rulebook and in check's printout. Also:
  clear the moving finger's end first (d1018); check a neighbour's own grasp first (d1033); nudge a red at the near wall
  first (d1013); grip leaning cubes on the upright pair or expect a drop; "far" means away from the red's neighbours;
  `places` doesn't screen for leaning neighbours; a bridging layer-2 cube or a seam can mark a buried red (d1024,
  d1028); don't plan axes until the red's top is fully seen.
- Why: d1033's agent noticed the shift wording was backwards; it had been copied from d1009's note instead of derived,
  and the tools printed it for hours. Grips on tilted pairs dropped cubes five times.
- By: Claude; d1033's agent caught the error.
- Evidence: `demonstrator/instructions.md`:97-99, 114-125, 134-137, 159-164, 211-215, 280-283;
  `demonstrator/REPORT.md`:52-54; `demonstrator/notes/d1033.md`:22-23.
- Replaces: v5's shift wording; v5's "both ends" rule, now relaxed.

## [2026-09-30] decision | Ground truth only after the fact (reconstructed)

- Decided: `demonstrator/report.py` tallies episodes (a kept pick that ends at gripper 14 or below counts as missed).
  `demonstrator/postmortem.py` replays finished episodes against true poses, "never for playing one".
- Why: to measure where moves and time go without leaking true poses into play. Results: 205 moves in the solved agent
  episodes, 80% shuttling grays, 12 of 128 picks lifted nothing, parking took 14% of robot time.
- By: Claude.
- Commands: `python demonstrator/report.py [--time] [--json OUT] [--prefix d]`;
  `python demonstrator/postmortem.py --out outputs/postmortem_2026-09-30.json [names]`.
- Evidence: `demonstrator/report.py`:1-9; `demonstrator/postmortem.py`:1-8; `demonstrator/REPORT.md`:16-36.

## [2026-09-30 08:22] result | 26 solved demos exported (reconstructed)

- Decided: the 24 solved agent episodes plus Claude's e400 and e402 go to `outputs/mechsearch_demos_v1`: LeRobot
  format, front and wrist images at 224, 43,392 frames at 30 Hz, 282 MB. Parks and settling are frames; undone moves
  are not. An episode whose replay disagrees with its recorded results is skipped; all 26 matched. Episodes played
  before v6 stayed in.
- By: not stated; Claude ran it. The exact command is not recorded; the form is
  `python demonstrator/export.py --out DIR NAMES|--all-solved [--size 224]`.
- Evidence: `demonstrator/REPORT.md`:5-8, 91-97; `demonstrator/export.py`:1-12;
  `outputs/mechsearch_demos_v1/demos.json`.

## [2026-09-30] direction | The goal is the expert, and cubes are a proof of concept (reconstructed)

- Decided: judge demonstrator changes by demos per hour and cost per demo; the student is out of scope until dm1487
  says otherwise. YCB objects or toys, deeper bins up to sink-like basins and bins set into the table come later, so
  keep object-agnostic lessons apart from cube-specific ones.
- Why: Claude had framed choices around the student while briefing the consult.
- By: dm1487: "It's not about the student. I want the expert to give me good demos and fast."
- Evidence: Claude's memory `demonstrator-goal.md`; `state.md`, "Goal right now".

## [2026-09-30] direction | Common sense every turn (reconstructed)

- Decided: start each turn from the skills, the state and a decision log, not the last message. Budget long parallel
  runs (limits, a wakeup after resets, tools frozen per batch). Encode recurring rules as checks in the tools.
- Why: the overnight lapses above.
- By: dm1487: "I really like that you have common sense, but you seem to not activate it every turn."
- Evidence: Claude's memory `common-sense-every-turn.md`; `docs/knowledge/operating-principles.md`.

## [2026-09-30 10:57] decision | Sweep pilot and a planner proposal (reconstructed)

- Decided: a pilot of 16 long strokes on the piles of seeds 1006 and 1012, aimed with true poses. Proposed order: add
  stroke length and height to the push, run a stroke study on 20 + 20 seeds, fit a corridor rule, build snapshot and
  restore last.
- Why: 8 of 16 strokes took every object off the target; 2 of 16 pushed a cube out, both toward a nearby wall. Buried
  starts need a median of 5 relocations offline.
- By: Claude.
- Evidence: `demonstrator/RELATED_WORK.md`:38-50, 71-74, 149-155; `demonstrator/experiments/sweep_test.py`,
  `render_sweep.py`; `outputs/sweep_pilot_2026-09-30/`.

## [2026-09-30 11:12] decision | The Fable consult's order becomes Claude's proposal (reconstructed)

- Decided: `demonstrator/consult/fable-2026-09-30.md`, from Fable 5.1, proposes: snapshots in a long-lived env; a
  depth-1 planner where perception proposes moves and the sim judges each on a copy; relocate as one decision and a
  higher red lift; perturbation scoring, cost logging and an export gate; a capped Claude rescue. No search tree.
  Claude agreed and filed it as proposed, not yet approved.
- Why: replay costs 5 to 30 s per tool call; about 38 requests and 4.8M tokens go into each episode; a tree would only
  serve the fallback. Measured the same day: a pick is 1.9 s of CPU for 3.6 s of robot time, a place 0.7 s, saving the
  state 2 microseconds, copying the full MjData 35 ms; about 25 to 45 episodes per core-hour at 8 candidates per move.
- By: Claude, on Fable's advice.
- Evidence: `demonstrator/consult/fable-2026-09-30.md`:9-72, 242-258.
- Replaces: the morning's plan (a snapshot search tree plus a scripted planner) and the sweep-first order above.

## [2026-09-30 11:13] decision | Knowledge moves into docs/knowledge/ (reconstructed)

- Decided: README, state, operating principles and the mechanical-search learnings, kept current so the common sense
  survives compaction.
- By: Claude, on dm1487's request.
- Evidence: the files' write times; `docs/knowledge/`.

## [2026-09-30] direction | A memory system along the lines of Karpathy's LLM OS

- Decided: set up the project's documentation and memory so what it learns, and the working common sense, survives
  compaction, restarts and new sessions: tiers with size budgets, a boot hook in the project's settings, an
  append-only journal kept apart from `state.md`, a consolidation routine, rules for subagents, and docs that stay
  general beyond cubes. No changes to code, the rulebook's rules or user-level settings; no commits.
- By: dm1487, before 11:22; carried out by a subagent of the main session.
- Evidence: `docs/knowledge/README.md`.

## [2026-09-30] direction | Follow the Fable build order, as a pilot on a few seeds

- Decided: build in the consult's order (snapshots in a long-lived env; the depth-1 planner with relocate and a
  higher final red lift; perturbation scoring, cost logging and an export gate; a capped Claude rescue), first as a
  pilot on a few seeds. The order's first step is to commit and stamp versions.
- Why: the consult's case, `demonstrator/consult/fable-2026-09-30.md`; Claude agreed with the order (state.md).
- By: dm1487, before 11:26. Recorded from the main session's report, not from a file.
- Evidence: `demonstrator/consult/fable-2026-09-30.md`; `state.md`, "Next".
- Replaces: "Proposed next, not yet approved" in `state.md`.

## [2026-09-30 11:26] decision | First commit on mechanical-search: b0455b2

- Decided: commit everything up to then as b0455b2, "Bin mechanical search, shelf search, and the Claude
  demonstrator": the sim, both search tasks, their sizing scripts and tests, the demonstrator (tools, perception, arm
  clearance, export, report and post-mortem scripts), the rulebook, 36 episode records and their notes, the knowledge
  docs and the sweep pilot scripts. Observe PNGs in `demonstrator/episodes/` and `MUJOCO_LOG.TXT` stay out of git;
  replaying an episode regenerates the PNGs.
- Why: dm1487 approved a build order whose first step was committing, so later runs can name the code they ran on.
- By: Claude (main session), approved by dm1487.
- Evidence: `git show --stat b0455b2`; `.gitignore`.
- Replaces: "Nothing is committed" in `state.md`.

## [2026-09-30 11:28] decision | A sweep primitive, and picks that can lift higher

- Decided: `MechSearchPrimitiveEnv` takes a fourth kind, `sweep`: a push with its own stroke of 0.03 to 0.18 m
  (`SWEEP_MIN_STROKE`, `SWEEP_MAX_STROKE`), starting anywhere in the bin up to 5 mm from a wall (`SWEEP_INSET`), at any
  height from a floor cube's centre to 3 cm above the rim (`SWEEP_MAX_ABOVE_RIM`), and refused if it would end against
  a wall. `pick` takes an optional `lift_z` that ends the lift above travel height. Both are optional keys, so old
  episode records replay unchanged.
- Why: the sweep pilot cleared the target in 8 of 16 strokes, and both of its spills headed toward a nearby wall. The
  consult's items 4 and 10 ask for a higher final lift for the red and for sweeps as a candidate move.
- By: Claude (main session), inside the build order dm1487 approved.
- Evidence: `so101sim/mechsearch.py` (the `SWEEP_*` constants, `_plan_sweep`), `so101sim/primitives.py` (`pick`),
  and in `tests/test_mechsearch.py`: `test_a_sweep_drags_the_closed_jaws_the_whole_stroke_at_the_chosen_height`,
  `test_a_sweep_that_would_run_into_a_wall_is_refused`, `test_a_pick_can_end_higher_than_travel_height`. Committed
  in f5790cd at 11:34.

## [2026-09-30 11:29] decision | Episodes kept alive in memory, with snapshot and restore

- Decided: `demonstrator/live.py`, class `Live(tools.Episode)`. `snapshot()` saves MuJoCo's integration state
  (`mjSTATE_INTEGRATION`, which includes the solver's warm start), the env's step, success-streak and move counters,
  and the episode's robot time, arm state and op list; `restore()` puts them all back.
- Why: `tools.py` replays every op from reset on each call, 5 to 30 s, while saving the state takes 2 microseconds.
  The consult's item 1.
- By: Claude (main session), inside the build order dm1487 approved.
- Evidence: `demonstrator/live.py`; `tests/test_demonstrator_live.py` checks that restoring a snapshot replays the
  rest of an episode bit for bit. Committed in f5790cd at 11:34.

## [2026-09-30 11:34] decision | Planner pilot p0 committed as f5790cd

- Decided: `demonstrator/planner.py`, planner version p0.1. Each decision parks and looks, proposes candidates from
  camera data only (a pick on the red on both jaw axes with a higher final lift, sweeps across the red's spot, and
  relocations of covers and axis blockers), runs each on a snapshot (`live.py`), scores it with the simulator's true
  poses (a spill is out, success wins, else cubes taken off the red, flush neighbours freed, robot time, disturbance),
  keeps the best, and stops after two decisions in a row find nothing that helps. Caps: 20 decisions, 200 s of robot
  time. Episode records carry the git commit, the MuJoCo version and the planner version.
- Why: step 2 of the approved build order. Candidates come from perception, so every kept move has a reason a camera
  could see; true poses only judge outcomes.
- By: Claude (main session), inside the build order dm1487 approved. The main session will log the pilot's runs.
- Commands: `python demonstrator/planner.py --seeds 1000 1001 --out outputs/planner_pilot` (one process per seed).
- Evidence: `git show f5790cd`; the docstring of `demonstrator/planner.py`.

## [2026-09-30 11:35] direction | Log every decision, so another person can replicate the demos

- Decided: the journal records every decision, dm1487's and Claude's, with what was decided, why, who decided and
  where the evidence or code is, in enough detail that another person with an LLM could reproduce the demonstrations.
  Run entries carry the commit, simulator version, seeds, rulebook version, commands, and which directions came from
  dm1487. Decisions from 2026-09-25 on get backfilled from the files, marked reconstructed.
- By: dm1487: "maintain a log of every decision we make and also you make to maintain memory, because that is what
  helps another person replicate the demos (information to instruct an LLM)". Relayed by the main session; the time is
  when it arrived.
- Evidence: this file's header; `docs/knowledge/README.md`, "The journal as a replication record".

## [2026-09-30 11:40] result | What a replication of the overnight demos lacks

- No episode file records the rulebook, tool or code version it was played under; the records hold only seed,
  buried, ops and undone. The overnight code was never committed as it stood: b0455b2 came after the run, and
  `tools.py`, `mechsearch.py` and `primitives.py` have changed since.
- No file names the model the agents ran on, or which agent played which seeds.
- The brief changed during the run and its old versions were not kept. The plan Fable reviewed is not in the repo.
- Old acts rely on `tools.py`'s fallbacks for a missing `flip` or `unpark` (14 and 13 acts).
- What does replay: every episode record replays exactly under the current code (export checked all 26 before it kept
  them), and every act stores its "why".
- Going forward: run entries in this journal carry the commit and versions, and planner episodes stamp them into
  their records (see 11:34).
- Evidence: the reconstruction on 2026-09-30 from the files named in the entries above.

## [2026-09-30 11:40] decision | How the memory system works

- Decided: (1) `CLAUDE.md` keeps only what every session needs, 111 lines from 311: the layout section moved to
  `docs/knowledge/codebase.md`, the hardware notes to `real-arm.md`, and the traps to five files in `.claude/rules/`
  that Claude Code loads when Claude Reads a file they cover. (2) A SessionStart hook with no matcher runs
  `.claude/hooks/knowledge.sh boot`: `state.md`, the README's index section, the headings of the last 8 journal
  entries and budget checks, under 9,800 characters. (3) A UserPromptSubmit hook, `knowledge.sh nudge`, prints one line
  when `state.md` is over 20 minutes old and project files changed after it. (4) Budgets: `CLAUDE.md` 150 lines,
  `state.md` 6,000 characters, the index section 2,000; consolidate after 25 journal entries; move earlier months out
  of the journal past 800 lines. (5) Only the main session writes `docs/knowledge/` and `.claude/rules/`; subagents
  write their own files and return summaries. (6) Project facts live in the repo, not in Claude's memory directory.
- Why: Claude Code swaps hook output over 10,000 characters for a file path and a 2,000-character preview; it re-reads
  the project `CLAUDE.md` after compaction; every subagent except Explore and Plan loads `CLAUDE.md`, so its size is
  paid per agent. Rule files keep traps out of sessions that never touch the code, such as demonstrator play.
- Turned down: a PreCompact hook (it can block compaction but can't reach Claude, and blocking a compaction forced by a
  context-limit error fails the request); a PostCompact hook (it can't add context, and the transcript keeps the
  summary); a Stop hook forcing a write-back (fires every turn, needs a loop guard, invites filler); injecting
  `state.md` into subagents; `@` imports in `CLAUDE.md` (they load in full at launch); search or embeddings.
- By: Claude (a subagent of the main session), on dm1487's request.
- Evidence: `.claude/settings.json`, `.claude/hooks/knowledge.sh`, `.claude/rules/`, `docs/knowledge/README.md`
  ("Boot" has the tests); the hooks and memory pages of the Claude Code docs.

## [2026-09-30 11:40] consolidation | Knowledge restructured for the memory system

- `CLAUDE.md`: layout to `codebase.md`, hardware to `real-arm.md`, traps to `.claude/rules/`
  (`scenes-and-physics.md`, `arm-and-motion.md`, `gpu.md`, `bin-task.md`, `shelf-task.md`). Every sentence of the
  old file was checked against the new files.
- `state.md`: the consult's 11-point summary left for the consult file, the timing numbers for the learnings file and
  this journal; it records the two commits and the approved pilot.
- Learnings: the simulator section points to the rule files and no longer copies their numbers; its noslip line had
  mixed a 30 s and a 10 s window and now reads correctly; the mj_geomDistance line points to its trap.
- `operating-principles.md`: the knowledge habits now name the journal and the one-home rule.
- Claude's memory: `demonstrator-goal.md` dropped its copy of the state; `common-sense-every-turn.md` names the journal
  and the boot hook; new `project-knowledge-home.md` says project facts go in the repo.
- This journal: backfilled with reconstructed decisions from 2026-09-18 to 2026-09-30, from the specs, plans, layout
  XML, rulebook change log, REPORT, consult, episode notes and memory files.

## [2026-09-30 11:46] run | Planner pilot p0, p0.1 and p0.2 on 13 seeds

- Commit: f5790cd for p0 (clean); p0.1 and p0.2 were edits to `demonstrator/planner.py` on top of it, uncommitted
  while they ran, so their episode records stamp `f5790cd+dirty`.
- Simulator: MuJoCo 3.8.1, `models/stage.xml` unchanged (noslip on).
- Seeds: 1000, 1003, 1004, 1005, 1006, 1009, 1012 (played by agents overnight, for comparison) and 1036 to 1041 (new);
  even seeds buried, odd partial. Player: the planner, no LLM. 7 to 13 processes in parallel on arrakis.
- Commands: `conda activate /common/users/dm1487/envs/so101sim; python demonstrator/planner.py --seeds ... --procs N
  --out outputs/planner_pilot_p0` (and `_p0.1`, `_p0.2`).
- Directions from dm1487: "Lets follow the build order on a few seeds to check performance" (the build order in
  `state.md`, from the Fable consult).

## [2026-09-30 11:46] result | Planner pilot: 5 of 7, then 9 of 13 twice, with one- and two-move solutions

- p0 (7 seeds): 5 solved in 2 to 4 moves against the agents' 7 to 11, 38 to 119 s of wall clock each and no tokens.
  Two bugs: it committed moves that scored at or below zero (seed 1000's line took a spill that way) and let a pick
  that lifted nothing into a kept line (seed 1005).
- p0.1 (13 seeds): 9 solved, every kept line free of empty picks, spills and harmful moves; solved episodes took 1 or 2
  moves and 5 to 18 s of robot time. Stuck: 1000, 1003, 1036, 1038.
- p0.2 (13 seeds): 9 solved; 1003 now solved, 1004 now stuck. The greedy choice took a relocation scoring 8.97 over a
  sweep scoring 8.75 that p0.1 had followed to success: one-step scoring is myopic.
- A single sweep across the red's spot was the first move in 4 of the 5 p0 solutions.
- Files: `outputs/planner_pilot_p0*/` (one record per seed with every candidate tried and its score, and summary.json).

## [2026-09-30 11:46] decision | Planner rules from the pilot

- Decided: commit a move only if it scores above zero, else stop the seed as stuck (a held cube is the exception: it
  must be set down, so the best non-spilling set-down is committed); reject any pick that lifts nothing, and any red
  pick that lifts a gray; score progress by cubes over the red (10), blocked ends of its better jaw axis (6), flush
  neighbours (2), minus 0.05 per robot second and 0.3 per disturbed cube; offer red picks slid 8 mm either way along
  the faces, blockers from both axes, and sweeps over the highest spot when no red is seen.
- Why: the p0 bugs above, and the stuck seeds' logs (the four-sided flush count hid progress on one axis; a "red pick"
  on seed 1003 lifted a gray cover and scored as progress).
- By: Claude (main session), within the pilot dm1487 approved.
- Evidence: `demonstrator/planner.py` (PLANNER = "p0.2"), `outputs/planner_pilot_p0.1/`, `outputs/planner_pilot_p0.2/`.

## [2026-09-30 11:46] decision | Next: a two-step lookahead for the planner

- Decided: score the top three first moves by the best second move each allows, instead of the best single move.
- Why: seed 1004 (above). Fable's consult ranked a depth-2 loop below a full search tree in cost, and recommended the
  tree only if the planner stalls on the neighbour step in more than about a fifth of buried seeds; 4 of 7 buried
  seeds stalled or regressed in p0.2.
- By: Claude (main session).
- Evidence: this journal's p0.2 result; `demonstrator/consult/fable-2026-09-30.md`.

## [2026-09-30 11:46] decision | Keep the memory system's hooks

- Decided: commit `.claude/settings.json` (SessionStart runs `knowledge.sh boot`; UserPromptSubmit runs `knowledge.sh
  nudge`) and the restructured docs.
- Why: dm1487 asked for the system; the hook script only reads files (cat, find, stat, wc, awk) and prints context.
- By: Claude (main session), after reading the script.
- Evidence: `.claude/hooks/knowledge.sh`, `.claude/settings.json`.

## [2026-09-30 11:54] result | Realistic objects: Google Scanned Objects in MuJoCo, stay on MuJoCo CPU

- Report: docs/reports/2026-09-30-realistic-objects.md (agent). kevinzakka's MuJoCo port of Google Scanned Objects
  (CC BY 4.0) compiles in 3.8.1; about 80 objects fit the gripper, about 20 of them toys. Its masses come out about 4x
  too high and 32 hulls per object is costly. YCB (elpis-lab port) is the second choice, since the real objects exist.
- Loading test, 20 objects, 12 cm walls, 8 seeds: rest in 0.5 to 2.1 s, nothing out, 10 s idle drift median 0.3 mm
  (worst 18.7 mm), 4.1 ms per step against 0.76 for cubes. Sleeping islands froze 7 of 8 piles at 0.05 ms per step but
  then snapshots need mj_copyData; restoring the state vector drifted. Piles reach 13 to 18 cm, into the jaws' start
  pose: a sink-like bin needs the arm raised.
- Simulator: stay on MuJoCo CPU. Isaac Sim needs driver 595 (arrakis has 570), ManiSkill has only the SO-100,
  Genesis's piles are untested.

## [2026-09-30 11:54] note | Usage limit reached mid-work

- Planner p0.3 (commit 9859fed, two-step lookahead) was running on 13 seeds into outputs/planner_pilot_p0.3/ as a
  plain script; check its summary.json. The bin-depth agent was stopped before finishing; its partial scripts are
  scripts/bin_depth*.py and outputs/bin_depth_2026-09-30/. Not yet committed: the objects agent's scripts and models,
  and these journal entries.

## [2026-09-30 12:01] result | Planner p0.3 (two-step lookahead) on the same 13 seeds: 11 solved

- Commit 9859fed. Per-seed lines, as printed:
  seed 1004 buried  solved         moves  2 empty picks 0 robot  13.5 s  wall  274.7 s  sims 80
  seed 1005 partial solved         moves  1 empty picks 0 robot   5.9 s  wall  104.4 s  sims 28
  seed 1006 buried  solved         moves  2 empty picks 0 robot  11.2 s  wall  299.0 s  sims 66
  seed 1009 partial solved         moves  2 empty picks 0 robot  18.1 s  wall  228.8 s  sims 46
  seed 1012 buried  solved         moves  2 empty picks 0 robot  16.1 s  wall  185.7 s  sims 42
  seed 1036 buried  solved         moves  3 empty picks 0 robot  21.2 s  wall  322.7 s  sims 74
  seed 1037 partial solved         moves  1 empty picks 0 robot   6.7 s  wall   68.0 s  sims 17
  seed 1038 buried  solved         moves  6 empty picks 0 robot  43.0 s  wall  844.4 s  sims 220
  seed 1039 partial solved         moves  1 empty picks 0 robot   5.5 s  wall   42.0 s  sims 14
  seed 1040 buried  solved         moves  2 empty picks 0 robot  18.0 s  wall  355.7 s  sims 106
  seed 1041 partial solved         moves  2 empty picks 0 robot  13.2 s  wall  230.2 s  sims 50
  
  [exited with code 0]
- Files: outputs/planner_pilot_p0.3/. Compare p0.2: 9 of 13.

## [2026-09-30 14:14] result | Planner p0.3 solved all 13 seeds (correction: the earlier entry listed only 11)

- 1000 solved in 6 moves (38.8 s robot, 682 s wall), 1003 in 2 moves (14.5 s, 297 s). The earlier result entry was
  written from the last 11 lines of the run's output. All 13: buried 2 to 6 moves, partial 1 or 2; no empty picks.
- Files: `outputs/planner_pilot_p0.3/summary.json` and one record per seed.

## [2026-09-30 14:14] direction | Taller walls, and not throwing things out by mistake (reconstructed from the chat, about 12:50)

- Decided: no assumption; the planner's rule is that a move is judged in the simulator first and any spilling move is
  rejected, so the expert doesn't throw things out by mistake. Taller walls are a layout parameter, to be measured,
  since they raise the success line and the travel height and may hit the gripper housing near walls.
- By: dm1487 asked; Claude (main session) answered.
- Evidence: this journal's planner rules; `docs/knowledge/mechanical-search-learnings.md`, "Containers".

## [2026-09-30 14:14] direction | Spawn agents for bin depth and for realistic objects (reconstructed, about 12:55)

- Decided: one agent measures how deep the end effector can work and which bins are doable; one surveys toy and
  household assets, tests loading them, and weighs switching simulators ("If you think we should switch to IsaacLab or
  something is also okay").
- By: dm1487.
- Evidence: `docs/reports/2026-09-30-realistic-objects.md`; `scripts/bin_depth*.py`, `outputs/bin_depth_2026-09-30/`.

## [2026-09-30 14:14] result | Bin depth, partial: the gripper tip reaches from 22.5 cm below the base to 8.5 cm above it

- The bin-depth agent was stopped at the usage limit before writing its report. Its envelope log: straight-down grasps
  reach z -0.225 to +0.085 m relative to the base, clear of the arm with the right wrist roll, also holding a cube;
  leaning grasps reach up to +0.15. The lattice of 7,000 bin configurations finished computing (`lattice.npz`) and was
  never analysed. `reach_map.png` is in the same folder.
- Evidence: `outputs/bin_depth_2026-09-30/envelope.log`, `scripts/bin_depth.py`.

## [2026-09-30 14:14] direction | Prepare for compaction first (dm1487)

- Decided: write back what the session holds before new work: this entry, `state.md`, the learnings and codebase
  files, and a commit of everything pending (the objects agent's scripts, test models and report, and the bin-depth
  scripts).
- By: dm1487.

## [2026-09-30 14:15] direction | Planning must not read hidden state

- dm1487: "But the planning moves (backtracking) should not be hacking into active sensing?"
- Decided: yes. p0.3 judged every candidate with the true pile (true poses in the score, and candidates tried on the
  real hidden world), which leaks what the cameras can't see; its 13 of 13 is not honest play and its episodes are not
  demos. The planner will judge candidates in several worlds rebuilt from perception instead, and the real world will
  answer only the move actually made; looks become candidate moves.
- By: dm1487 raised it; Claude (main session) agreed and set it as the next step.
- Evidence: `demonstrator/planner.py` (`truth()` feeds `score()`; `decide()` runs candidates on `live`, the real
  pile); `outputs/planner_pilot_p0.3/p1038.json` (no red pixels seen, scored by the red's true position).
- Replaces: the planner caveat that treated this as luck to be fixed by aim-error scoring.

## [2026-09-30 14:25] note | Compaction done; write-back check
- What: dm1487 ran `/compact` once (a second run was interrupted). Compared the summary with the knowledge files.
  Everything was there except the snapshot restore order and its cost, now in `codebase.md`, and a stale line in
  `codebase.md` that said perception uses the front camera only, corrected there.
- By: Claude (main session).
- Next: state.md Next #1, the honest planner.

## [2026-09-30 14:41] finding | Undo leaks hidden state too; a perception-built twin may not pay
- What: dm1487 asked why the planner needs a digital twin at all, and whether a memory document read every turn would
  do. Measured before answering:
  - p0.3 simulated 348 candidates over 27 decisions: 127 helped, 100 gained nothing, 89 picks lifted nothing,
    21 red picks lifted a gray, 11 spilled. It ran them on the true pile, so these catches don't show what a twin
    rebuilt from perception would catch.
  - The 26 solved player episodes (228 kept moves) hold 13 undone attempts in 10 episodes, 12 of them misses.
    Undo on the real pile is the same leak as simulating on it, in a smaller dose.
  - No solved player episode spilled; the players used camera checks and the rulebook, no simulator.
- Proposal (Claude, not yet decided): a per-episode memory document as the expert's belief (seen-empty space, where
  moved objects went, where the target was last seen, a map of where it could be), camera checks and rules as code,
  no twin and no undo. Add a twin only if measurement shows a failure the rules can't catch.
- Evidence: `outputs/planner_pilot_p0.3/p*.json` (`log[].tried`), `demonstrator/episodes/*.json` (`undone`).
- Learnings: "The planner", two bullets added.

## [2026-09-30 14:46] decision | Undo only resets a spill: privileged, allowed for now, discouraged
- What: dm1487: "We can also allow spill resets (but we discourage the behavior, and we want you to understand this
  and remember this) -- this can be priveleged for now to get decent demos." So undo may take back a move that knocked
  an object out of the container and nothing else. Misses and moves that made things worse stay in the demo, and
  the expert recovers from them.
- Why: undo picks the next move after seeing how the hidden pile answered (entry 14:41). A spill ends the episode,
  so without a reset one spill wastes a whole demo; dm1487 accepts that privilege for now and no other.
- How it's enforced: `demonstrator/tools.py` `cmd_undo` refuses unless the last act's `result.failure` says an object
  left the bin (`spilled()`); `tests/test_demonstrator_undo.py`. Rulebook v10 and `AGENT_BRIEF.md` say the same. Every
  record keeps its undone attempts, and `export.py` already writes the undo count per episode. The honest planner
  will follow the same rule.
- Also: Next #1 in state.md now names Claude's design from 14:41 (memory of what the cameras saw, rules as code, no
  twin unless a measured failure needs one). dm1487 hasn't objected to it and added this allowance on top of it.
- Open: the 10 overnight demos with pre-v10 miss undos. Keep, mark or redo.
- By: dm1487 decided; Claude enforced it in the tool.

## [2026-09-30 14:52] correction | What the 13 overnight undos were
- The entries at 14:41 and 14:46 said 12 of the 13 undos were misses. Read one by one, they were 10 missed picks
  (d1001, d1005, d1009, d1012, d1016, d1031, e402 four times) and 3 cubes that fell out of the gripper on the swing to
  a place (d1013, d1027, d1030). None was a spill. The rule they used, rulebook v4 to v9, allowed undoing "a missed
  pick or a push when it made things worse": the red buried deeper, cubes tipped onto it, or the red wedged against a
  wall. Two undos (d1012, d1016) took back misses that moved nothing, as wasted moves.
- Corrected in the learnings file and rulebook v10. Evidence: `demonstrator/episodes/*.json`, `undone[].why`.

## [2026-09-30 14:55] decision | A missed pick resets with probability eps
- What: dm1487: "lets allow reset on missed picks too but with a probabilty \epsilon ... Because this may happen
  sometime sno?" Read as: real picks miss sometimes, so most misses stay in the demo with their recovery, and a
  fraction eps get reset.
- Claude's choices, open to change: eps = 0.1. A miss is a pick whose gripper closes on nothing (grip <= 14), which
  the robot reads from its own joint. The tool draws, from sha256(seed, move, resets so far), so a rerun draws the
  same and a retry draws again. The tool resets by itself below eps; the player never chooses, because choosing would
  reset only the harmful misses. Cubes dropped on the way to a place are not picks and stay.
- Code: `demonstrator/tools.py` (`MISS_RESET_EPS`, `draw`, `take_back`, `missed`, `reset_miss`, the hook in
  `cmd_act`); new records store `miss_reset_eps`; each undone entry records `granted` (spill or miss draw);
  `export.py` writes the split. Tests: `tests/test_demonstrator_undo.py`, 6 pass. Rulebook v10.1, agent brief.
- By: dm1487 decided; Claude implemented.

## [2026-09-30 14:56] start | Why picks miss: replay study
- What: dm1487 asked "why did our pick miss?". The overnight demos hold 22 picks that closed on nothing: 12 kept,
  10 undone. The postmortem's xy aim error was under 2 mm for 7 of the 12 kept ones, so xy aim isn't the main cause.
  A subagent replays each miss and 20 or so good picks, and measures z, yaw, tilt, how far the jaws got down, and
  what the fingers touched. Script `demonstrator/experiments/miss_study.py`, output `outputs/miss_study_2026-09-30/`.
- By: Claude (main session), after dm1487's question.

## [2026-09-30 14:59] decision | eps = 0.5 for missed-pick resets
- What: dm1487 set eps to 0.5 ("Maybe 5 in 10 gets reset"), replacing Claude's default of 0.1 from the entry above.
- Expected effect, from the overnight rate (22 misses in 138 picks, 16%): about 8% of picks reset, so demos split
  roughly evenly between a clean retry and a shown recovery after a miss.
- Code: `MISS_RESET_EPS = 0.5` in `demonstrator/tools.py`; rulebook v10.1 text and state.md updated.
- By: dm1487.

## [2026-09-30 14:59] direction | Why eps: it sets the demos' miss rate
- What: dm1487 confirmed the reason for eps: resetting every miss gives demos where the arm never misses, resetting
  none keeps every miss and makes demos longer, and eps sets the point between. This replaces Claude's reading at
  14:55 ("most misses stay, so most demos show the recovery"), which no longer holds at eps = 0.5.
- Updated: the comment on `MISS_RESET_EPS` in `demonstrator/tools.py`, rulebook v10.1, the learnings file, Claude's
  memory note.
- By: dm1487.

## [2026-09-30 15:02] decision | eps back to 0.1; cut the miss rate at its causes
- What: dm1487 asked what value Claude thought right. Claude argued for lower than 0.5: eps can only push the demo
  miss rate below the expert's own 16% (16% x (1 - eps)), and a real arm will likely miss more than the sim expert,
  since it droops about 9 mm per 100 mm of reach and repeats to about 5 mm peak to peak at full reach (real-arm.md).
  A reset buys shorter demos and fewer runs into the 15-move cap, and leaks a little each time. dm1487: "0.1 is fine
  then."
- Also dm1487: reduce how often the expert misses, and learn and record why it misses. The replay study (entry at
  14:56) finds the causes; fixes go in at the source, and the miss rate gets measured before and after.
- Code: `MISS_RESET_EPS = 0.1`; rulebook v10.1 and state.md updated.
- By: dm1487 decided; Claude argued the value.

## [2026-09-30 15:09] decision | The wrist camera has no depth
- What: dm1487: "No depth on the wrist camera (we dont have that in reality)". The wrist camera gives colour only.
- Why it matters: `Episode.see()` in `demonstrator/tools.py` back-projects wrist depth into 3D points and uses them
  for cube poses, the height map and the red pixels' positions. That fusion cut unseen floor from about 60% to 38%
  and trusted pose error from 1.4 to 0.6 mm median. It was a privileged sensor, so the 26 overnight demos and p0.3
  decided with information the real arm lacks.
- Plan (Claude, not built yet): the wrist image keeps what colour alone can give. Red pixels become rays, placed where
  they meet the front camera's surface. Floor pixels meet the known floor plane. A cube top of known size gives its
  pose from one image. Measure each against front-only perception before keeping it. Wait until the miss study stops
  replaying through `tools.py`, so its numbers come from one version of the tools.
- Open, asked of dm1487: does the real rig's front camera give depth, and what stands in for the sim's exact
  segmentation (`perception.py` calls it "the optimistic part")?
- By: dm1487 decided.

## [2026-09-30 15:18] direction | Park the wrist depth camera question
- What: Claude offered to test in sim whether a depth camera on the real wrist would pay. dm1487: "Dont worry about
  this fornow." The sim's wrist camera stays colour only; no hardware study.
- By: dm1487.

## [2026-09-30 15:19] decision | The real front camera gives depth
- What: dm1487: "Yes, front camera can give depth." So the front camera stays RGB-D in the sim and the height map
  keeps its metric source. Still open: what segments objects in reality (dm1487 asked whether SAM would be used).
- By: dm1487.
