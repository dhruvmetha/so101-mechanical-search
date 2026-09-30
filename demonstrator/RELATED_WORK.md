# Related work: sweeping, pushing and scripted planners for the bin task

Written 2026-09-30. V means I read the claim in a paper text, a repo, or the earlier notes `sota_baselines.md` and
`demo_baselines.md`, which were themselves written from full texts. U means I saw it only in an abstract, a search
snippet or a citation, and did not confirm it. "No license" means the repo has no LICENSE file.

## What the literature says, in short

Nobody has published a test of multi-cube sweeps against pick-and-place in a 3D pile where nothing may leave the
container. Pushing work is planar or shallow and lets objects go off the table. Removal-order work assumes a buffer or
a shelf. So the evidence below gives parts and warnings, not a number we can borrow. The number has to come from our
own sim, and the main session's sweep test is the first one.

## Table

| Work | Motion | How it chooses | Reported results | Code, license | Checked |
|---|---|---|---|---|---|
| Mechanical Search, Danielczuk et al., ICRA 2019 | Grasp, suction, or a linear push planned in free space | Rule: target if reachable, else the occluder with the largest visible area, push when no grasp | Sim, 1000 heaps of 15: largest-first plus push 93.3%, 6.51 actions. Grasped distractors leave the bin | none found | V |
| X-Ray, Danielczuk et al., IROS 2020 | Grasp only | CNN gives a per-pixel occupancy map for the hidden target, grasp the object overlapping it most | Sim 82%, median 5 actions; real 20 of 20 | BerkeleyAutomation/xray, MIT | V |
| VPG, Zeng et al., IROS 2018 | 10 cm push at 16 headings, or top-down grasp | Two FCN Q maps over a 224x224 heightmap | Sim 30 objects: 100% completion, 67.7% grasp success | andyzeng/visual-pushing-grasping, BSD-2 | V |
| Grasping the Invisible, Yang et al., RA-L 2020 | VPG push and grasp | Bayesian explorer pushes while the target is unseen, learned critic once seen | Sim buried cases 93.5% in 2.7 motions; real 85% of 40 | choicelab/grasping-invisible, BSD-2 | V |
| Novkovic et al., ICRA 2020 (1911.07482) | End-effector displacements in a walled workspace | PPO on a TSDF, against a hand-written grid sweep | Red cube hidden in piles of cubes. Learned 88% in 26.9 steps. The grid sweep that pushes every stacked cube 99.9% in 54.8. Success means seeing the target | none found | V (earlier note) |
| DIPN, Xu et al., ICRA 2021 (2011.04692) | Push of several objects, image of the outcome predicted, joined to a grasp net | Search over pushes scored by the imagined image | Reports gains over prior push-grasp methods; numbers not read | rutgers-arc-lab/dipn, license unread | abstract V, rest U |
| Learning to singulate, Eitel et al., ISRR 2017 (1707.08101) | Pushes proposed on over-segmented RGB-D | Push proposal network trained on real robot data | Singulates up to 8 unknown objects on a table; rates not read | none found | abstract V, rest U |
| Linear push policies, Danielczuk et al., CASE 2018 | Straight pushes in a bin, "when grasps are not available, pushing or sweeping" | Heuristics on grasp access | Pushing lifts grasp success by over 15% (earlier note) | none found | abstract V, number U |
| Dogar and Srinivasa, AuRo 2012 | Push-grasp and sweep, planned with pose uncertainty | Capture-region planner, plans monotone, each object moves once | Real, 5 scenes: planned 4, executed all | none found | V |
| Transporter Networks and Ravens, Zeng et al., CoRL 2020 | Brush or spatula pushes of piles toward a goal zone | Learned spatial displacement, closed loop | Real Go pieces 98.3 (metric not confirmed) | google-research/ravens, Apache-2.0 | V, metric U |
| PMBS and MORE, Huang et al., 2022 | 5 cm pushes on object contours, top-down grasp | Parallel MCTS over cloned simulators | PMBS 100% in 3.91 actions, planar, visible target, about 10 objects | arc-l, no license | V |
| Nakhimovich, Miao, Bekris, ICRA 2023 | Pick and place inside the workspace | Dependency graph ("below", "grasp blocked by", "hidden by") plus voxel occlusion | Motoman sim, about 19/20 at 6 objects, 13/20 at 14 | none found | V |
| Cheong et al., ICRA 2020 | Pick and place | Choose the free slot that kills the fewest other slots (the beta rule) | 90% at 25 objects | none | V |
| Srivastava et al., ICRA 2014 | Pick and place, no free space | Motion layer reports obstructions as facts, task layer replans | 100% at 15 objects, 84% at 30, 600 s budget | dead link | V |
| STAX-RAY, Huang et al., ISRR 2022 | Blade push, suction lift onto another object | MCTS over an occupancy net | Sim 82 to 100%; real 67 to 100% | none found | V via summary |
| Block relocation problem, Caserta et al., EJOR 2012 | Restack blockers inside one bay | Exact branch and bound, lower bounds | n/a | many papers | citation V, solvers U |
| Excavation learning for rigid objects in clutter (2107.04171); GEOTACT (2402.04536) | Digging in granular media | Learned excavation and tactile occupancy | Granular media, not cubes | not checked | title and snippet only, U |

## Sweeping, single pushes and pick-and-place in our bin

Our numbers first. Last night's 24 agent episodes used 205 moves. 80% were pick-and-place of grays, 14% picks of the
red, 1 was a push. Buried starts took a median of 10.5 moves and 65 s of robot time. Places are 41% of the moves and
disturb the pile more than picks do (21% moved three or more other cubes, against 14%). The offline count of what a
buried start needs, from the contact graph, is a median of 5 relocations, so about 11 primitives if each cover costs a
pick and a place. That barely fits the 15-move cap.

The main session's sweep test, on seeds 1006 and 1012, changes the picture for covers. Sixteen single strokes of the
closed jaws across the red's spot (four headings, two recipes: from the rim at 8 mm above the spot with a 12 cm
stroke, or from 5 cm before the red at layer-2 centre height with a 9 cm stroke) cleared every cube off the red in 8 of
16. Two of 16 pushed a gray out of the bin, and both were layer-2-height strokes heading toward a wall. One rim stroke
cleared both covers on seed 1012 and moved only 5 cubes. Pick-and-place spends 2 moves per cover. Two piles and 16
strokes is a pilot, not a rate. Treat 50% as an upper-middle guess and the 12.5% spill rate as a floor on
layer-2-height strokes toward walls.

What the literature adds to that:

- **Sweeps have a precedent as the cheap move.** Novkovic et al. is the closest scene in print, a red cube hidden in
  piles of cubes in a walled workspace. A hand-written exhaustive sweep beat their learned agent on success, 99.9% to
  88%. It was slower, 54.8 steps against 26.9, and it only had to reveal the target, not lift it. The same order of
  result is plausible here: sweeps are reliable at uncovering and pick-and-place is reliable at extracting.
- **Sweeps against single pushes.** VPG and its descendants push one cluster a fixed 10 cm to make grasp room. They
  never plan a stroke to clear a stack, and they let objects fall off the table. A long stroke at a chosen height
  is closer to Dogar and Srinivasa's sweep, which they treat as a capture action, or to the Ravens sweep, which pushes a
  pile toward a goal. Neither has a wall to fall over.
- **The spill is the real cost.** Every method above either has a buffer or ignores the edge. Our bin has 5 cm walls
  and layer-3 cubes stand 4 cm over the rim, so a stroke that ends in a wall lifts the cubes it carries over it.
  The two spills in the pilot fit that.
- **A sweep cannot open the floor.** The red's four flush neighbours are on layer 1, level with the red. A stroke
  above the red's top passes over them. Sweeps remove covers only. Neighbours still need picks or low pushes.
- **Push versus pick, measured elsewhere.** The trade is rarely measured. Mechanical Search reports 93.3% with
  pushes allowed. Danielczuk 2018 reports a grasp success gain from pushing above 15%. Xu et al. report 4.82 motions
  per object. All three count moves in single-layer heaps with a buffer. None counts a move that undoes another.

## Proposal: a program for the bin task

The goal is a scripted planner, no LLM, that reads the same inputs the agents had. It combines the Nakhimovich
dependency graph, Srivastava's obstruction-then-replan loop, Cheong's placement rule, and the pilot's sweep.

### Inputs

- Deployable mode. `Episode.see()` in `demonstrator/tools.py`: cube poses with layer, yaw, tilt and top-face
  coverage, a 2 mm heightmap with an owner per cell, the unseen mask, red pixels in 3D, and `axis_ends` for the red.
  `check` for finger, wall, self and camera-mount clearance.
- Teacher mode. The same, plus true poses from `info`, contact-graph "rests on" edges, and snapshot and restore of the
  env. Restore does not exist yet.

### Decision loop, one primitive at a time, re-observing after each

1. **Extract.** If the red's top is seen and one jaw axis has 4 to 14 mm free on the fixed side and 20 to 28 mm on the
   moving side (about 42 mm), pick it, and lift to a height above the success line to avoid the tilted-lift cost (5 of
   29 lifts).
2. **Locate.** If the red is not seen, put a red-region estimate on the cube whose edge shows the red line, within
   1.5 cm of the pixel centroid (held in every buried episode so far). If there are no red pixels, use a level layer-2
   cube bridging two floor cubes one cube width apart.
3. **Cover set.** Take every cube above the red's footprint through the graph. Cubes on layer 2 or 3 over the red are
   the covers.
4. **Cover removal, sweep first.** Enumerate candidate strokes: 8 or 16 headings, start beyond the cover set, jaw
   height just over the top of the layer below the covers, stroke long enough to pass the red's footprint plus one cube.
   Reject a stroke by rule when any cube in the swept corridor is within the stroke length plus 3 cm of a wall in the
   heading direction, when any corridor cube is layer 3 and the heading goes toward a wall, or when the corridor's end
   is inside the rim margin. In teacher mode, run the survivors on a cloned sim and keep the first that clears the
   covers with no spill and no cube left leaning on the red. In deployable mode, pick the survivor with the fewest
   corridor cubes and the roomiest exit, then verify with `observe`. If nothing survives, fall back to pick-and-place
   of the top cover, placed on a flat layer-1 top or open floor by the `places` rule, at least 6 cm from the red and
   away from its neighbours.
5. **Open one jaw axis.** With the red uncovered, pick the axis whose two end cubes are cheapest, using the axis
   report. Each end cube is gripped across its own free pair of faces, which is the axis perpendicular to the red's
   axis, and needs its side cells free, diagonals included. Lift them away with the same placement rule.
6. **Nudge fallback.** If an end cube has no free pair of faces, slide the grasp along the red's face or push a
   neighbour sideways with the closed jaws at layer-1 height, which needs a 15 mm slot beside it. This step is
   weakest and cost most in the agent notes.
7. **Guard every move.** Refuse moves the `check` rules refuse: fingers within 3 mm of a wall, arm self-overlap past
   10 mm, camera mount on the scene, 40 mm or more of unseen space on a finger path. After a pick, read the gripper
   (14 to 30 holding, about 10 empty) before the place.

### The four flush neighbours

A boxed red has four grays within 3 mm on layer 1. The program opens one axis, not four sides. Sweeps do nothing here.
The order is covers, then the end cube with the freer sides, then its partner. The 42 mm budget means both ends of
one axis must go, unless the moving-finger end alone is clear (d1018 succeeded that way). Expect 4 moves for the two
neighbours, 5 or 6 if a layer-2 cube rests on one.

### What it needs that we do not have

- A sweep primitive. `MechSearchPrimitiveEnv` fixes the stroke at 5 cm (`PUSH_STROKE` in `mechsearch.py`), and the
  pilot used 9 to 12 cm. It needs stroke length and start height in the action, a rim-start reach check (jaws over the
  wall, camera mount clear), and the same 6 mm style wall refusal the shelf has.
- Snapshot and restore in the env, checked for determinism, for the teacher mode.
- A corridor and spill predictor. That is a rule table now, and 2 spills in 16 strokes is too few to fit it. It needs
  a few hundred strokes on random piles, with layer of the swept cubes, distance to the wall and heading logged.
- A pile-disturbance measure per move, so sweeps and picks can be compared in cubes moved, not only moves.
- A rate for "sweep clears all covers" over 20 or more buried seeds and 20 partial ones, per heading, with the
  pile recipe held fixed.

### What will be hard

- **Sweeps randomise the pile.** Piles are chaotic. A 1 micrometre shift changed 21 cube positions in one seed, so no
  stroke outcome can be trusted without a look. The program must observe after every stroke, and the model of stroke
  results is a prior, not a prediction.
- **Sweeps can rebury the red.** Cubes pushed along a corridor pile up at its end, and if the red's spot is near the end
  they land back on it. The corridor rule should end the stroke beyond the red by a fixed margin toward open floor.
- **Pose error.** Poses are 0.6 mm median for trusted cubes, but about 1 in 10 box-fit poses is off by up to 16 mm.
  A 46 mm jaw pair with 15 mm clearance needs pose error under a few mm at the red's edge.
- **The neighbour step has no known-good recipe.** The agent notes call a nudge a last resort, and it cost two extra
  moves each time. Most of the remaining move count sits here, so a sweep saves less than it seems.
- **The move cap.** 15 moves against a median 5 relocations. Sweeps help only if they remove two covers per move. The
  pilot's 8 of 16 clears mean an expected 0.5 covers cleared per stroke attempt on those piles, before counting
  strokes that fail and need a follow-up.
- **The imitation gap if the program becomes a teacher.** A teacher that reads true poses digs where the red is. A
  camera-only student sees 30 gray cubes. The deployable mode above avoids this, at some cost in success.

### Order of work

1. Add stroke length and height to the push action, and log cubes moved, cubes over the rim margin, and red uncovered.
2. Run the 16-stroke test on 20 buried and 20 partial seeds, with pick-and-place as the control for the same seeds.
3. Fit the corridor rule to those results, then write the loop above in deployable mode and compare its moves and robot
   time with the agent medians (10.5 moves, 65 s buried).
4. Add snapshot and restore only if the deployable version stalls on the neighbour step.

## Sources

Papers in the scratchpad: 1903.01588, 1904.10348, 1912.07024, 2110.12325, 2207.02347, 2306.14251, Dogar 2010, 2012,
2013, Danielczuk 2018, Haustein 2015, King 2016, Papallas 2020, PMBS, Ravens. Abstracts read this session:
2011.04692 (DIPN), 1707.08101 (singulate). Search only: 2107.04171, 2402.04536. Others from `sota_baselines.md` and
`demo_baselines.md` in the session scratchpad.
