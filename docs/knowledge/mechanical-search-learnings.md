# Mechanical search: what we have learned

Started 2026-09-30. The bin of 30 cubes is the proof-of-concept rig; the same expert should later work with YCB or
other realistic objects, in deeper bins and bins set into the table. So each lesson here is stated for any objects and
containers, with the cube evidence behind it and a note on what changes for other shapes. Cube-only details live in
`demonstrator/instructions.md`, which the players follow, and the episode notes in `demonstrator/notes/`. Simulator
and code traps, with their full numbers, live in `.claude/rules/`. Add to this file whenever something is learned, and
say where the evidence is. When a section passes about 40 lines, split it into its own file and add it to the index.

## The arm and gripper

- A top-down grasp has two wrist rolls half a turn apart that close the jaws on the same object. Near the arm's base,
  one of them folds the wrist camera mount into the shoulder. IK took whichever it found first, and that made a strip
  near the base look unreachable. Choosing the roll that keeps the arm 10 mm or more from itself fixed it: 54 of 55
  empty-bin picks landed within 6 mm (2026-09-30, `demonstrator/reach.py`). Holds for any object.
- The gripper needs free space beside the object along the jaw axis, and the two fingers need different amounts. On
  the SO-101 the fixed finger lands 4 to 14 mm outside one side and the moving finger 20 to 28 mm outside the other.
  Shifting the aim more than about 2 mm away from the fixed finger puts it on the object's own edge. For any object
  the question is the same: are both finger landing zones free, down to the grasp height? Only the numbers are the
  gripper's.
- Container walls are the real reach limit. Fingers within 3 mm of a wall landed where planned in 8 of 80 empty-bin
  picks. A grasp near a wall needs its jaw axis parallel to the wall, or a higher grasp point when the object stands
  above the rim. For tall or irregular objects the higher grasp is often the natural one.
- Grasps that close across tilted or off-centre surfaces can read as holding and then drop the object on the swing
  to the place (five drops overnight). Prefer grasp surfaces that stay upright, and keep the carry short when you can't.
  Irregular objects will make this more common, not less.
- A top-down pick misses when a finger comes down on a top surface instead of beside the object: the object's own top,
  a flush neighbour's top, or a tilted object's upper face. The jaws then close above it. Replaying 32 misses and 97
  hits (2026-09-30, `outputs/miss_study_2026-09-30/`): the jaws started closing a median 12 mm higher than planned in
  misses and 0 mm in hits. Perception error did not differ (0.4 mm against 0.3). So misses come from where the
  fingers land, which a height map can check before the move, not from seeing the object wrong.

## Containers

- The rig will vary: deeper bins up to sink-like basins, bins set into the table with the rim flush, and contents
  from cubes to YCB objects and toys (dm1487, 2026-09-30). A bin sitting 10 cm below the arm's base, as in the sim, is
  physically a bin set 10 cm into the table.
- Depth is limited by the gripper, not only by reach. With walls over about 6 cm, the wrist camera mount and the
  gripper housing catch the rim while the jaws reach the floor (found while sizing the bin, 2026-09-25). Top-down reach
  also shrinks with depth. How deep a sink the SO-101 can work has to be measured with `so101sim/search_layout.py` and
  its checks before building one; leaning grasps (`demonstrator/reach.py`) may extend it.
- Measured, partly (2026-09-30): straight down, the jaw tip reaches from 22.5 cm below the arm's base to 8.5 cm above
  it, clear of the arm with the right wrist roll and also holding a cube. That bounds depth only; which floor areas a
  deep bin leaves reachable is in `outputs/bin_depth_2026-09-30/lattice.npz`, not analysed yet.

## Seeing the pile

- One fixed camera leaves most of a pile hidden. Adding the wrist camera, from the parked arm, cut unseen floor from
  about 60% to 38% and pose error from 1.4 mm to 0.6 mm median. A second viewpoint from the other side is worth more
  than any pose-fitting cleverness. (Corrected 2026-09-30 15:21: those numbers used wrist depth, which the real wrist
  camera lacks; what colour alone recovers is not measured yet.)
- Give the expert only what the student's observations hold (dm1487, 2026-09-30). The expert runs in sim and a
  policy distilled from its demos runs on the robot, so perfect perception of what the student's cameras see is fair,
  exact segmentation included. A channel the robot lacks is not: the sim renders depth for every camera, and using one
  the robot lacks leaks as surely as reading true poses. Store every channel the expert used in the dataset. The SO-101's
  wrist camera gives colour only, so wrist pixels reach 3D only through something known: the floor plane, another
  camera's surface, or an object's known size.
- Check every camera's intrinsics. The wrist camera has a 16:9 sensor, so its pixels are not square in a 4:3 render,
  and assuming square pixels put x off by a third.
- A few pixels of the target are enough to know where to dig. In every buried episode the target was under the object
  whose edge showed a sliver of it, within 1.5 cm of the sliver's centre. With no sliver, look for structure: an object
  bridging a gap a target-width wide (`demonstrator/notes/d1024.md`). This carries over to any target with a
  distinctive appearance; for YCB it means detecting the target's colour or instance, not just "red".
- Deep unseen space on a finger's path usually hides an object. 40 mm or more of it caused misses on hidden cubes that
  a second view later showed. Unknown space is an obstacle until seen.
- Segmentation here is perfect, which makes perception optimistic. Real segmentation merges and splits objects,
  especially cluttered ones; plan for that before trusting a pose.

## Checking a move before making it

- Test the fingers' swept volume against a height map of what the cameras see, over the whole path including the
  crossing from where the arm is. This is object-agnostic: it needs only points and a height map. Scored over 76 real
  picks, a clean check lifted the target 96% of the time.
- Sample the gripper densely. The finger meshes have faces up to 60 mm across with vertices only at the corners, and a
  vertex-only check passed a finger through a flush neighbour (`demonstrator/notes/d1005.md`).
- Hits predict failure less than you would think: with 10 mm or more of overlap, 75% of picks still worked, because a
  finger landing on a loose neighbour shoves it aside. A hit on something that can't move (boxed in, against a wall) is
  the one that matters.
- Distance queries can lie. MuJoCo's box-box distance returned 0 for boxes 30 mm apart at an exact IK pose
  (`.claude/rules/arm-and-motion.md`), so clearance checks sample points on both bodies instead.

## Strategy

- Two phases: expose the target, then open a grasp on it. Covers come off first. Then the target's grasp needs both
  finger zones free, and the moving finger's side matters most. Aiming around a flush neighbour never worked; lifting it
  did. Sliding the grasp along the object's face, to dodge a neighbour that sits offset, did work.
- Where you put a removed object matters as much as which one you remove. Objects parked beside the target's
  neighbours blocked later grasps in three episodes, and spots next to leaning objects tipped them. A good spot is far
  from the target and everything touching it, flat, and away from anything leaning.
- 80% of the expert's moves were pick-and-place of distractors, two moves per object. A sweep can move several at once:
  in a pilot on two piles, one long stroke across the target's spot took every object off it in 8 of 16 strokes, and 2 of
  16 pushed an object out of the container, both heading toward a nearby wall. Never sweep toward a close wall. The
  pilot aimed with true poses; it needs repeating with camera-based aiming on more piles.
- Nudging (a grasp on the target through its neighbours) sometimes opens a gap and often tilts the target, which then
  costs a set-down and a second grasp. It is a last resort.
- Count moves. A target under three objects and boxed on four sides needed about 16 moves, over the 15-move cap.

## The simulator

The measurements and every setting tried are in `.claude/rules/scenes-and-physics.md` and `.claude/rules/gpu.md`.
What they mean for search:

- Resting piles creep without MuJoCo's noslip: 24 mm median over 30 s with nothing touching them. With noslip, 10 s
  of idle drift is 0.85 mm median. A planner that leaves objects and comes back to them needs noslip or its equal.
  (Corrected 2026-09-30: this line had read "10 s of idle drift fell from 24 mm to 0.85 mm", mixing the two windows.)
- Piles are chaotic: moving one object by a micrometre changes the outcome of later moves. Replaying the same moves in
  the same build is exact, which is what makes undo, replay and export possible. Anything that searches over moves will
  find lucky outcomes; decide whether that matters for the use.
- Warp (GPU) has no noslip and its piles creep under every setting tried. Don't trust GPU results that need objects
  to stay put.
- Check penetration before trusting any contact-rich result. MuJoCo's default contact scales stiffness to the mass it
  touches, so a light object under a strong servo sinks: a 27 g cube went 11.9 mm into the floor under a finger. Soft
  contacts flattered the expert: fingers pressed neighbours down and still grasped, and pushes slid tall objects that
  real physics tips (a 15 mm radius cylinder with base friction 1 tips when pushed above about 15 mm). Under stiff
  contacts (2026-09-30), a finger 10 mm or more into a neighbour lifted the target in 33 of 209 picks.
- Varying hidden physics (friction, mass) per episode makes the task need memory (dm1487, 2026-09-30): acting well
  means remembering what objects did earlier in the episode. And whatever the expert remembers, the student must be
  able to remember too, or it can't copy the choice from the current frame. An expert that looks after every move
  corrects without that memory, at the cost of more moves; a narrow, realistic range keeps a memoryless policy viable.
- Judge a push by what it moves and where things end, not by what it touches. A pick-style check that flags every
  contact made every push look dangerous, and the players pushed once in 260 moves (2026-09-30). Measured on 768
  strokes: carrying a cube that rides above the lower layer to within 15 mm of the wall ahead spilled 18%; other
  strokes 4.5%, and 2% with little unseen space in the path; longer strokes spill more. Most cubes a stroke moves are
  ones the cameras didn't predict, so push toward open space and away from walls.
- Hand over a scene only once nothing in it creeps. A speed threshold passes slow creep; hold the scene still for a
  few seconds and measure displacement instead (0.5 mm in 5 s caught both creepers in 24 piles).

## Running the expert

- The overnight run cost about 40 agent requests per demo, and eight agents in parallel used up the account's session
  limit in a few hours with no recovery. Budget usage, pace agents, and schedule a restart after a limit resets.
- Freeze tools and rules during a batch. Changing them under running agents made early and late demos differ.
- Each agent should play one episode in a fresh context; later episodes in a shared context paid for earlier ones.
- Rules written as prose drift and can be wrong for hours (the fixed-finger shift was written backwards). Where a rule
  can be checked, put the check in the tools instead.
- Parking the arm to look cost 14% of robot time.
- Physics is cheap enough to judge candidate moves in the simulator. A pick costs 1.9 s of CPU for 3.6 s of robot time
  and a place 0.7 s, at 1 ms per step with noslip on; saving the state takes 2 microseconds and copying the full
  MjData 35 ms, so save and restore the state rather than copying data (2026-09-30). At 8 candidates per move that is
  about 25 to 45 episodes per core-hour. Every tool call that replays an episode from the start wastes this; keep the
  env alive and restore snapshots instead.
- Watch MuJoCo's log. `MUJOCO_LOG.TXT` recorded an arena memory overflow on 2026-09-30, which silently drops contacts.

## What is cube-specific and must change for realistic objects

- Pose fits: `perception.box_pose` fits a 30 mm box and `top_face_pose` assumes a square top. Realistic objects need
  grasps computed from points (antipodal pairs on the segment's point cloud), not from a fitted pose.
- Layers: the tools report "layer 1 to 3" from fixed cube heights. Use heights and support relations instead.
- Grasp axes: "yaw or yaw + 90" works for cubes. Other shapes have many candidate axes, some much better than others.
- The success line is defined by the cube's lowest corner. For other objects, define it by the object's lowest point.
- Placement spots are "tops of level cubes". Realistic objects need stable-placement checks: a flat enough, big enough
  patch, and objects that don't roll.

## The planner

In `planner.md`: what the expert-as-program measured, and what counts as honest play.
