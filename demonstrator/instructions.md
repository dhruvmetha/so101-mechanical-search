# Bin task: instructions for the demonstrator

These are the rules Claude follows when it plays the bin task as the demonstrator. It is also the memory: after every
episode, the lessons go into "What experience has taught" and the change log at the bottom, and later episodes follow
the updated text. Keep rules short and concrete, and put each rule where it applies.

**IMPORTANT (dm1487, 2026-10-01): play as the teacher, by the five rules in "IMPORTANT: how the teacher plays" at
the top of `docs/knowledge/mechanical-search-learnings.md`. Read them before every episode.**

## The task

- The bin is 22 cm square with 5 cm walls, sunk 10 cm below the arm's base, centred at (0.200, 0.000). The floor is
  at z = -0.096 and the rim at z = -0.046. The 30 cubes are 30 mm. Layer 1 centres sit at z = -0.081, layer 2 at
  -0.051, layer 3 at -0.021.
- Win by lifting the red cube clear of the bin. Lose if any cube leaves the bin. The cap is 15 moves.
- Buried starts: the red cube is invisible to the front camera, boxed in on the floor. Partial starts: part of it shows.

## Tools

Run from the repo root with the so101sim conda env active.

    python demonstrator/tools.py new EP SEED buried|partial
    python demonstrator/tools.py observe EP
    python demonstrator/tools.py check EP KIND X Y Z DEG
    python demonstrator/tools.py act EP KIND X Y Z DEG "why"
    python demonstrator/tools.py status EP
    python demonstrator/tools.py undo EP "why"
    python demonstrator/tools.py list
    python demonstrator/tools.py places EP [X Y]

- observe also prints where red pixels show in the colour image, placed in 3D by depth. A few pixels are a thin red
  line at some cube's edge; their region says where to dig. The z range is loose because edge pixels mix depths, but
  the lowest z tells you whether the red is on the floor (about -0.096) or higher.
- When the red is seen, observe lists the seen cubes at both ends of each of its two jaw axes, with the gap to each.
  An axis is ready once both ends are clear. That is the axis to work on: pick the one whose listed cubes are easiest
  to lift, and lift them all.
- observe uses two cameras: the fixed front camera (colour, depth, masks) and the wrist camera on the parked arm
  (colour and masks, no depth, as on the real arm). All 3D comes from the front camera. The wrist camera adds floor
  it sees empty and cubes the front can't see. Measured on 12 piles (v10.2): 52% of the bin seen against 41% for the
  front camera alone; trusted poses within 1.5 mm median. Each listed cube says which cameras saw it: F, FW or W.
  A W-only cube is placed along its pixels' rays ("ray"), about 15 mm off median and up to 33: find it again after
  the cube hiding it moves before planning a grasp on it.
- The image has two panels. Left, the front camera cropped to the bin: top is the base side (low x), bottom the far
  wall, left -y. Right, the wrist camera: left is the base side, right the far wall, top +y, bottom -y.
- observe parks the arm and lists every cube the cameras see, with its pose. Trust a pose only when at least
  60% of its top face is seen. It also lists flat spots to set a cube on, and saves a labelled image. Look at the
  image when the list alone doesn't explain the scene.
- check plans a move without running it, over the whole path the arm will take, crossing included. It reports which
  seen cubes the fingers would pass through, and how deep, and how far they go into space the camera can't see. It
  never refuses a move. Contact is sometimes the point.
- Every grasp can be made with two wrist rolls, half a turn apart, that close the jaws on the same faces. check tries
  both, and act uses the one check keeps. When both keep the arm 10 mm or more from itself, check takes the one with
  fewer finger hits, so it may keep the smaller distance. That is intended. The "arm to itself" line
  gives the closest the moving end of the arm comes to the shoulder, upper arm and base.
- For a pick, check says which way the fixed finger points from the target. Use it with the 2 mm shift rule below.
- check also reports the wrist camera mount against the scene, fingers passing within 3 mm of a wall, and jaws that
  end off the asked yaw because the wrist roll is at its limit.
- act runs the move, lets the scene settle, and reports the gripper reading. Above 14 means holding, and about 10
  means empty.
- For pick, XYZ is the cube's centre and DEG is the jaw yaw. Use the cube's yaw or yaw + 90. For place, XYZ is where
  the held cube's centre goes, using a spot from observe. For push, XYZ is where the closed jaws start, at the height
  to push at. DEG is the heading. The stroke is 5 cm.

## Resets: spills, and a few missed picks by chance

Resets are privileged (dm1487, 2026-09-30). A real robot can't take a move back, and choosing the next move after
seeing how the hidden pile answered uses what the cameras never saw. dm1487 allows two, discourages both, and the
record and the dataset count every one. Plan so you never need them.

- Undo every move that knocks a cube out of the bin. The episode ends on that frame, and undo is the only way on.
  `tools.py undo` refuses any other move.
- A missed pick (the gripper closes on nothing) gets reset by chance, not by choice. After the miss, act draws a
  number from the seed and the move; below eps (0.1 unless the record says otherwise) it resets the move itself and
  says so. Otherwise the miss stays. eps sets how often demonstrations miss: resetting every miss would show an arm
  that never misses, and resetting none keeps every miss and makes demonstrations longer (dm1487).
- Keep every other result: a kept miss, a push that made things worse, a cube dropped on the way to a place. Recover
  from it in the next moves. Until v10, a move that made things worse could be undone by choice
  (`docs/knowledge/planner.md` says what those undos were and why they leak).
- After a reset, choose a different move. Repeating the same move repeats the same result, since the sim is exact.
- At most 8 resets per episode. Every reset also costs the robot time of the move taken back, in real life.
- Never read the simulator's true cube poses while playing. Decide from observe, check and the image only.

## The loop

1. observe.
2. Decide on one move, using the rules below.
3. check it. If it hits cubes you didn't mean to touch, adjust it or pick another move.
4. act.
5. If a cube left the bin, undo it and pick another move. If act reset a missed pick, observe again. Keep any
   other result, good or bad, and plan from it.
6. If it was a pick, read the gripper before doing anything else. Place only when holding. While holding, observe
   looks without parking, because a swing home shook a held cube out even with the grip closed (e400).
7. Go back to 1. After the episode, update this file.

## Rules

### Grasping

- The jaws need room on both sides of the cube along the jaw axis. The fixed finger lands 4 to 14 mm outside one
  face, usually the face toward the base. The open moving finger lands 20 to 28 mm outside the other face. That is
  about 42 mm of free space in total. Check both jaw axes, yaw and yaw + 90, and take the one with fewer hits.
- Hits matter more since the stiff contacts of 2026-09-30. A neighbour no longer gives way under a finger, so the
  jaws stop on its top and close above the target. Over 342 calibration picks: a finger 10 mm or more into a
  neighbour lifted the target 33 times in 209, and act now refuses it; picks that pass every refusal lifted it 50 of
  60 times. Clear the side a finger needs before picking; nudging no longer works. (Before, on soft contacts, 76
  scored picks gave 96%, 80% and 75% for worst hits under 3, 3 to 10 and over 10 mm.)
- A pick that comes up holding two cubes, or holding a gray stuck to the red, won't count. Place and re-pick.
- The red only counts once its lowest corner is 2 cm above where a top-layer cube rests. The pick lifts to travel
  height, which leaves about 2 mm to spare for a level cube. A red held tilted 10 deg missed by 0.7 mm (e400 move 9).
  If a pick on the red reads HOLDING but not success, set it down level on an open flat spot and grasp it again,
  square to its faces.
- Which end matters most is the moving finger's: it needs 20 to 28 mm. In d1018 only that end was clear, check put
  the fixed finger 18 mm into a flush g9, and the pick held and counted. So clear the moving finger's end first, then
  try the pick before digging out the fixed finger's neighbour too. check names which end is which.
- A boxed-in red needs both ends of one jaw axis cleared. Clearing one end is not enough (d1012), and shifting the
  aim 6 to 13 mm along the jaw axis or turning the jaws 9 to 15 deg does not get past a flush neighbour: check's hit
  barely changes, and a big shift puts a finger on the red itself (d1006, d1012, d1015).
- Sliding the aim along the faces, across the jaw axis, is different and worth trying first. When the flush
  neighbours sit offset toward one side of the red's face, sliding the grasp up to 12 mm the other way along the
  face can clear them all, and a grasp 12 mm off the face centre still lifted the red level (d1010). The axis report's
  gaps don't show that offset; check does.
- A nudge after clearing one end of an axis does work, but it shoves the other neighbour about 10 mm and lifts the
  red tilted, short of the line. Budget two more moves to set it down level and grasp again (d1008). A nudge on a red
  boxed on all four sides can also come up with a gray pinched on (d1016), which costs a set-down and more.
- To set a tilted red down level, use an empty floor slot or an open layer-2 spot with room on one axis. On a layer-2
  top next to its old hole it slid back into the hole tilted 30 deg (d1016). In e402 an open flat spot at layer 2
  left it level with one clean axis, and the regrasp counted. Budget 4 moves for the two neighbours, 5 or 6 if
  a layer-2 cube sits over one of them. Pick the axis whose two neighbours are easiest to lift.
- Before spending moves on a neighbour, check that it has a clean grasp itself. Some have none: every g21 grasp in
  d1033 put a finger on the red or over the wall. When a neighbour has open floor behind it, a nudge whose moving
  finger lands on it can shove it clear (g21 moved 7 mm). Near the near wall, nothing within about 2 cm can be picked,
  and a nudge moved the red from x 0.109 to 0.129 (d1013). Try the nudge there first.
- Pushes and sweeps are first-class moves (v10.4). One push can clear what pick-and-place needs two moves or more
  for: a pick and a place per cube, about 12 s of robot time each. Use `push` (5 cm) or `sweep X Y Z DEG STROKE`
  (0.03 to 0.18 m, may start 5 mm from a wall and 3 cm above the rim). check lists what a stroke moves and how far.
  Calibrated on 768 strokes (2026-09-30): 70% moved the pushed cube 15 mm or more its way; act refuses a stroke that
  carries a layer-2 or higher cube to within 15 mm of the wall ahead (18% spilled); allowed strokes spilled 4.5%, 2%
  when the path had under 30 mm of unseen space. Those strokes were aimed at random cubes in random directions, so
  they measure risk, not usefulness. Where each point below comes from:
  - Push away from walls (measured: the refused class above). Push along a wall rather than into it (reasoning,
    untested).
  - Use the shortest stroke that does the job (measured on random strokes: 16 cm sweeps spilled 18 to 28%, 5 to 8 cm
    strokes 8 to 9%).
  - Push a flush neighbour out of a jaw's path along the red's face from a slot a removed cube left (one episode,
    b5000).
  - Purposeful strokes, measured on 32 piles where the red was seen but not pickable (2026-09-30,
    \`outputs/push_goal_2026-09-30\`): one stroke made the red pickable in 25 of 29 piles. A blocker at an end of one of
    the red's jaw axes, pushed sideways off the axis, did it in 23 of 29 piles, but only 1 such stroke in 4 works: the
    whole cluster tends to move together, the red included. Prefer a blocker on an axis whose other end is already
    clear: those strokes worked 19 times in 38, the rest 38 in 194. Open floor where the blocker would land did not
    help (3 in 29; this line said to prefer it until the same night, push_choice.py). Pushing the
    red itself toward room worked in 12 of 28 piles; sweeping a cover off in 5 of 14. 5 cm did as well as 8 cm.
  - Pushes reveal: half of all strokes near the red turned a rough red into a trusted pose. Use one to see.
  - Once the red is pickable, pick it. A push beside it kept it pickable only 10 times in 22.
  - Pick-and-place versus push for the same blocker is not measured yet; when every push is refused or blocked,
    pick-and-place (reasoning, untested).
  - Before 2026-09-30 the tools had no sweep and check called every touch a hit, so players pushed once in 260 moves.
- Under stiff contacts both ends of a jaw axis need clearing (b5002): the finger that can't be avoided lands on the
  flush neighbour and act refuses it. Where one roll is unreachable, the fixed finger's side is fixed too; plan for it.
- Look for a window along the face before digging out a neighbour. Two cubes at one end can leave a gap: in b5000 g7
  and g27 left 18 mm, and sliding the aim 4 mm along the face (not the jaw axis) put the finger in it. In b5002 a 2 mm
  gap took 1 mm steps along the face to find.
- A push that slides a flush neighbour along the red's face, started from a slot a removed cube left, clears that end
  in one move (b5000, g3 moved 44 mm). In a dense pile most pushes fail check (b5002: 20 to 50 mm hits on every one, from the old pick-style check; see the push bullet above).
- Treat a rough neighbour at the fixed-finger end as flush: check said no hits and the pick missed (b5000 move 10).
- Nudging, a pick on the red that lands the fingers on its neighbours, is gone since v10.2: act refuses a finger 10 mm
  or more into a neighbour. Under soft contacts it opened a 12 mm gap once (e400); under stiff ones the neighbour
  doesn't give. To move a cube a little, push it (the tools have no sweep yet; corrected 2026-09-30, this said "push or
  sweep"). The nudges described in the bullet above predate this.
- Never shift a grasp centre more than about 2 mm away from the fixed finger, toward the moving finger's side. The
  fixed finger lands only 4 mm outside its face, so a 4 mm shift that way puts it on the cube's own top edge, and check
  does not flag that (d1009, whose aim moved 4 mm toward the moving finger). Replayed on 2026-09-30, all 10 picks
  aimed more than 4 mm toward the moving finger missed, and no hit was shifted more than 3.4 mm. Until 2026-09-30 this rule and check's
  printout said "toward the fixed finger", which was backwards (d1033 noticed). Shifting toward the fixed finger is
  safe for the target and pushes that finger into whatever is on its side. Sliding
  the aim along the face, not across it, is safe: 4 mm along a face cut a tilted neighbour's hit from 10 to 4 mm.
- Two trusted cubes on one layer with a gap of 3 mm or less between them are flush. Before 2026-09-30 check could
  pass a finger through such a gap and report no hits (d1005). The finger check is denser now, but treat those gaps
  as closed.
- Deep unseen space on a finger's path usually hides a cube. 47 mm hid g19 in d1004, and 73 to 79 mm hid a tilted g5
  in e402, where two picks missed on it. Treat 40 mm or more of unseen space as a hit.
- Hits of 40 mm or more mean the finger goes through a cube above the neighbour, not a flush floor cube (d1015). A
  graze of 6 mm on a higher cube did no harm (d1006). Hits on the moving finger don't shrink with offsets, because
  its open arc sweeps wider than where it lands (d1009).
- A leaning cube: grip its upright pair of faces, the box-fit yaw, even when the other axis checks clean. Grasps across
  the tilted pair, or high, or 10 mm off centre can read HOLDING and then drop the cube on the swing to the place:
  d1013 twice, d1027 twice, d1030 once, against two holds on the upright pair in d1031. If a grasp like that is the
  only one, place the cube close by, with a short swing.
- A cube that sticks up tilted can be grasped higher than its centre, above its neighbours' tops. That kept every
  finger clear in e400 move 10. A red tilted 18 deg, grasped 7 mm above its box-fit centre on the box-fit yaw, held
  but stayed tilted and didn't count. Setting it on a well-seen layer-1 top at its own yaw and grasping it again
  worked first time (d1004).
- A cube turned about 45 deg with a corner near a wall can't be grasped on its faces: every face grasp puts a finger
  outside the bin. Close across its corners with the jaws parallel to the wall and expect a miss that squares it up.
  In d1003 that turned the red from 43 to 106 deg and moved it 10 mm, and the next pick, parallel to the wall, lifted
  it.

### Clearing

- Take cubes off the top first. A top-layer cube with clear jaws is the cheapest move there is.
- Put a removed cube on the lowest flat spot that is far from where you are digging. Never put it where the red might
  be. The top of a well-seen layer-1 cube is the best spot, and you often have to find it yourself: observe's list is
  often next to the dig or fails check (d1003, d1006, d1015). Floor spots beside stacks often send the fingers into
  space the camera can't see.
- While holding a cube, run places first. It checks the tops of trusted, level layer-1 cubes and the open floor
  spots, farthest from the red first (or from X Y if you give it), at two yaws each, and lists the best with their
  hits, unseen space and arm clearance. Pick from its list, or check a spot of your own if none suits.
- Check a place at two yaws 90 deg apart. The wrist roll changes with the yaw, and so do the hits and the arm's
  distance to itself (d1006, d1015).
- "Far from the dig" means not beside any of the red's neighbours or the cubes over them, not just 6 cm from the red.
  Covers parked 5 cm from the red, beside its neighbours, later blocked three picks in d1024 and made g3 unliftable in
  d1033. places keeps 6 cm from the red but doesn't know about the neighbours; read its list against the red's axes.
- places doesn't screen for leaning neighbours either. Its clean spots on g24, g7 and g7 again tipped the placed cube
  and its neighbours 22 to 43 deg (d1019, d1028, d1031). Skip a top with a tilted cube beside it, and a layer-1 top
  beside a layer-2 cube (d1018: g28 slid 25 mm off g7).
- Avoid layer-2 tops: a cube set on g0 slid to the far wall (d1006). Avoid layer-1 cubes with a leaning neighbour:
  the placed cube ends up tilted (d1012). Don't park a cover beside a cube you may need to lift later (d1009).
- A cube held askew (gripper over 20) swings wider when released. Leave more than 5 mm to neighbours: a place on g21
  knocked g4 off g1 next door (d1016).
- A place needs room for the open fingers too. Setting the red down beside g11 and g18 hit one or the other until the
  spot was 5 mm from each (d1008).
- Taking a floor cube out beside the red can let a layer-2 cube that rested on both slide onto the red (d1009), and a
  place near a leaning stack by a wall can shift it up over the rim (d1015). Read the list again after every move.
- A cover cube on layer 2 or 3 comes off in two moves, pick and place. It is the cheapest progress there is.
- A tilted layer-2 cube that fails a pick can be pushed: heading 0 at layer-2 height, starting 3 cm behind it, moved g4
  12 mm and dropped it a layer without touching anything else (d1001).
- When nothing near the target can be grasped, push. The closed jaws are 46 mm wide and about 15 mm thick along the
  stroke. They need a 15 mm slot to come down in, and whatever is in the 5 cm stroke moves.
- A push at second-layer height from the front, heading toward the base (180 deg), slides a covering cube off
  whatever is under it.

### Reach and walls

- The near wall (x = 0.09) is no different from the others once the wrist roll is right. On an empty bin, 182 picks at
  x 0.11 to 0.16: with the fingers 3 mm or more from every wall, an "arm to itself" of 10 mm or more tracked to 6 mm
  in 54 of 55, and an overlap past 10 mm tracked in none of 17. Before check chose the roll, x at or below 0.14
  failed (e401).
- Walls are the real limit. Fingers passing within 3 mm of a wall tracked in 8 of 80 picks: the open finger catches
  the wall and the grasp slides. A cube with a face within 2 cm of a wall needs its jaws parallel to that wall: 90 deg
  against the near and far walls, 0 deg against the side walls. If its neighbours along that axis are flush, push it
  out from the wall first. The wall warning turns on within a few mm: aiming 3 mm away from the wall cleared it with
  no new hits, and that pick held (d1006).
- A layer-2 cube by a wall has its centre at rim height, so a grasp at the centre can put a finger over the rim.
  Raise the grasp in 1 mm steps until check's wall line goes away: 5 mm cleared it in d1011, 6 mm in d1016, 8 mm in
  e402, 9 mm in d1021, and every one of those picks held (d1021's lifted the red to success). Try height before any
  big sideways shift: a 9 mm shift at centre height missed in d1016. Raised like this, a layer-2 cube can take jaws
  that point at the wall, which the floor-cube rule below forbids.
- Near the base, one jaw axis may leave the arm only 3 to 5 mm from itself (d1005). Plan the clearing around the axis
  whose check shows 30 mm or more.
- Layer-3 cubes near a wall sit above the rim. Anything that sweeps sideways near them can knock one out, and that
  ends the episode (e401 move 3). Before a move near a wall, look for layer-3 cubes on that side.
- The wrist camera mount must stay off the scene. check names what it would hit.

### Finding the red

- If the red isn't seen, it's under or behind something. The red pixels line in observe, and the image, say where.
  In every buried episode so far the red was under the cube whose edge showed the red line, within 1.5 cm of the
  pixel centroid (e400, d1004, d1006, d1012, d1015). Red pixels as low as z -0.066, the top of a floor cube, put the
  red on the floor right under that cube (d1006). Lift that cube first.
- Red pixels can also mean the red is behind a cube, not under it: in e402 the red's top showed as a line past g27's
  back edge, and lifting g27 showed the red uncovered.
- Four flush floor cubes around one spot, with a cube stacked over the middle, mark a boxed-in red. So does a level
  layer-2 cube whose floor neighbours sit a full cube width apart under it, even with no red pixels (d1024). With no
  red pixels, look at the image around seams before digging blind: a seam a few pixels wide showed red in d1028.
- The red's yaw can change once its top is fully seen: d1019's read 28 at 71% of the top and 82 once the cover came
  off. Don't plan its axes until the top is fully seen.
- Count moves. A red under three covers and boxed on four sides needs about 16 (d1000 ran out at 15 with the red
  held tilted). Covers cost two moves each and each blocker on an axis two more, so pick the axis with the fewest
  blockers, as the axis report lists them (d1017), and don't spend moves on nudges or retries that the budget can't
  carry.
- On a partial start, check both of the red's axes before digging at all. A red on top with one clean axis is a
  one-move episode (d1011).
- Dig where the pile is highest and busiest first, and observe after every move. Poses sharpen once a cover comes
  off: g27 went from rough, tilted 26 deg, to trusted and flat, 12 mm away from the rough guess (d1007).
- If the red is partly seen, clear the cubes on top of it and on its better jaw axis, one at a time. If one axis is
  blocked by a floor cube backed by others or a wall, clear the other axis (d1015).
- A miss can still help: in d1015 a miss knocked two cubes down a layer and bared the red's side, and the next pick
  was clean. Since v10 every miss stays, so plan the recovery from one that buried or tilted the red.

### Rough poses

- A cube whose top face is less than 60% seen gets a box fit to all its visible faces, marked "rough, box fit" with a
  tilt. The yaw given is for the pair of faces that stays most nearly vertical, which is what the jaws should close
  on. In e401 the top-face estimate of a leaning cube was off by 4 to 5 mm and the pick missed; the box fit was right.

## What experience has taught

(The rules above already include these lessons. From d1000 on, each episode's own notes are in demonstrator/notes/,
and the entries here are only the practice seeds that changed the tools.)

- e400, buried, solved in 11 moves, 68 s of robot time. The red line at g9's edge put the red at about
  (0.253, -0.038), and it was at (0.255, -0.042). Two covers came off in four moves, and the red was uncovered at
  move 4. It was boxed on all four sides. A nudge across x did little; a nudge across y opened a 12 mm gap. Clearing
  g10 took two moves. Move 9 held the red tilted and missed success by 0.7 mm. Then the tool's park opened the
  gripper and dropped it, a tool bug since fixed. Move 10 held it again, and the park swing shook it out even with
  the grip closed. It landed level on top of the pile, and move 11 lifted it. Luck helped at the end. Without the two
  drops, the red would have been out by move 10 at the latest.

- e401, partial, failed at move 3 by knocking a cube out of the bin. The red sat near the base wall under g3, a
  leaning layer-3 cube at x = 0.13. Move 1 aimed at g3's top-face pose, which the lean had pulled 4 mm off, and missed.
  Move 2 used the box fit and missed by 4.5 cm: the wrist had to roll 93 deg from home and ground its camera mount
  on the shoulder, so the roll lagged. The tools gained an unpark step for that. Move 3 got there, but the wrist
  roll IK chose folded the arm into the shoulder, the arm lagged and swept sideways, and g25, a layer-3 cube above
  the rim by the near wall, went over. The lessons: take the wrist roll that keeps the arm off itself (check does now),
  keep sweeps away from layer-3 cubes by a wall, and use box-fit poses for leaning cubes.

## Change log

- v1, 2026-09-29: first version, from the hand-played seeds 301 to 310 and the program test on seeds 400 to 403.
- v2, 2026-09-29, after e400: the success line and the set-down-and-regrasp rule, nudging a boxed red, grasping a
  tilted cube high, where to put covers, reading the red line in the image. Tools: observe doesn't park while
  holding, and check leaves the target cube out of its hits.
- v3, 2026-09-30, after e401 and the reach study: check tries both wrist rolls over the full path and reports the arm
  against itself, fingers near walls, and the camera mount against the scene. Near-base reach rewritten: the near
  wall is like the others once the roll is right. Layer-3 cubes by a wall, and box-fit poses for leaning cubes.
- v4, 2026-09-30: undo, and the red-pixel line in observe.
- v5, 2026-09-30, after the first six agent episodes (d1003, d1004, d1006, d1009, d1012, d1015, all solved): clear both
  ends of one jaw axis instead of aiming around neighbours; nudging demoted to a last resort; the fixed-finger shift
  limit; deep unseen space counts as a hit; tilted and diagonal reds; finding places by hand; the red is under the cube
  showing the red line.
- v5.1, 2026-09-30: observe lists the cubes at each end of the red's jaw axes.
- v6, 2026-09-30: observe fuses the wrist camera; check leaves the held cube out of a place's hits.
- v6.1, 2026-09-30: check names the fixed finger's side.
- v6.2, 2026-09-30: places lists checked spots for a held cube.
- v7, 2026-09-30, after d1005, d1007, d1008, d1010, d1011: slide along the faces before digging out an offset
  neighbour; nudge after one cleared end costs two more moves; flush same-layer gaps; raise the grasp by a wall; plan
  around the axis with room near the base; one-move partial starts; places need finger room.
- v7.1, 2026-09-30: check samples the finger meshes densely; what hits mean, from 76 scored picks.
- v8, 2026-09-30, after d1000, d1016, d1017, d1021, e402: raise a layer-2 grasp by a wall in 1 mm steps; nudges can
  pinch a gray; where to set a tilted red down; askew cubes need room; count moves against the cap.
- v9, 2026-09-30 morning, after the rest of the overnight notes (d1001, d1013, d1018, d1019, d1024, d1027, d1028,
  d1030, d1031, d1033): the 2 mm shift rule was backwards and now says away from the fixed finger; the moving finger's
  end first; check a neighbour's own grasp; grip leaning cubes on the upright pair or expect a drop; "far" means away
  from the red's neighbours; places doesn't screen for leaning neighbours; more ways to find a buried red.
- v10, 2026-09-30 afternoon (dm1487): undo only resets a spill, and the tool refuses anything else. Misses stay in the
  demonstration and get recovered from.
- v10.1, 2026-09-30 afternoon (dm1487): act resets a missed pick with probability eps, by a fixed draw; eps = 0.1.
- v10.2, 2026-09-30 evening: stiff contacts. observe takes 3D from the front camera only (the wrist gives colour and
  masks). act refuses aims over 3 mm toward the moving finger, fingers on the target's own top, fingers 10 mm or
  more into a neighbour, and arm self-overlap past 10 mm.
- v10.3, 2026-09-30 evening, after b5000 and b5002: both ends of an axis, windows along the face, the push along
  the red's face from a slot, rough neighbours at the fixed-finger end.
- v10.4, 2026-09-30 night (dm1487: pushes were sidelined): sweeps in the tools, a push report in check, act
  refuses strokes that carry a high cube to the wall, push guidance from 768 calibration strokes.