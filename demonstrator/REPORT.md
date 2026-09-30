# Demonstrator: overnight report, 2026-09-30

## What we have

- 26 solved episodes, exported as a LeRobot dataset in `outputs/mechsearch_demos_v1`: 43,392 frames at 30 Hz, 24 minutes
  of robot motion, 282 MB. Each frame has front and wrist images at 224 by 224, joint state, commanded joints and the
  index of the op it belongs to. `demos.json` in that folder lists the episodes. Every episode replayed exactly
  against its record before it went in.
- 24 were played by agents on seeds 1000 to 1033, and 2 by me (e400, e402). Agents solved 10 of 11 buried starts and
  14 of 14 partial starts that they finished. The one failure, d1000, ran out of the 15 moves with the red held tilted.
- 8 episodes are half played: d1002, d1014, d1020, d1022, d1025, d1029, d1032, d1034. At 02:33 the account hit its
  session limit and every agent stopped mid-episode. Their moves are saved, and a new agent can finish each one.
- The rulebook `demonstrator/instructions.md` went from v4 to v9 overnight. Every player wrote notes in
  `demonstrator/notes/`, one file per episode, and each rule in the rulebook names the episode it came from.

`python demonstrator/report.py` tallies the episodes. `python demonstrator/postmortem.py --out F.json` replays the
finished ones against the true poses: robot time per op, what each pick really lifted, and how much each move moved
the rest of the pile. The numbers below come from those two.

## Where the moves and the time go

| | buried (10 solved) | partial (14 solved) |
|---|---|---|
| moves, median (range) | 10.5 (7 to 13) | 7.5 (1 to 13) |
| robot time, median (range) | 65 s (36 to 82) | 46 s (5 to 97) |

- Every buried red started boxed in on all four sides, under one or two cubes. Partial starts had 0 to 4 cubes over
  the red and at most one flush neighbour, but still took up to 13 moves.
- Of 205 moves in the solved agent episodes, 80 lifted a gray off the red or off one of its axes and 85 put one down.
  So 80% of the moves are shuttling grays. 29 lifted the red, 10 picks missed, and there was 1 push. The players
  preferred picking to pushing almost every time.
- The first pick aimed at the red worked in 17 of 24 episodes. 5 of 29 red lifts came up tilted, short of the success
  line, and each cost a set-down and a second grasp.
- 12 of 128 picks lifted nothing. None lifted the wrong cube.
- Places disturb the pile more than picks. 21% of places moved three or more other cubes, against 14% of picks.
- Parking the arm to look took 199 s of the 1,390 s of robot time, 14%.

## What changed in the tools

- The near-base "dead strip" was the wrist. Every top-down grasp has two wrist rolls half a turn apart, and IK took the
  first it found, which near the base folds the wrist camera mount into the shoulder. The primitives now take `flip`,
  and check tries both rolls. On an empty bin, 54 of 55 picks landed within 6 mm when the arm stayed 10 mm from itself
  and the fingers 3 mm from every wall. What's left is walls.
- `mj_geomDistance` returned 0 mm for two boxes 30 mm apart at an exact IK pose. `demonstrator/reach.Clearance`
  samples points instead. CLAUDE.md has the trap.
- observe fuses the wrist camera on the parked arm with the front camera. Unseen floor went from about 60% to 38%,
  trusted poses from 1.4 mm to 0.6 mm median error. The wrist camera has non-square pixels (a 16:9 sensor), and
  treating them as square had put x off by a third.
- New: undo, places, list, red pixels in 3D, the cubes at each end of the red's two jaw axes, the fixed finger's side,
  and a finger check sampled 2 mm apart across the finger meshes' large faces. Scored against 76 real picks, a clean
  check lifted the target 96% of the time, and 10 mm or more of hits still 75%.
- A rule was wrong for most of the night. "Shift the aim at most 2 mm toward the fixed finger" should have said away
  from it; check printed the same wrong line. d1033 caught it this morning and both are fixed. Players followed check's
  hit numbers more than the wording, and no episode failed on it that I can find.

## What the players learned

The rulebook has all of it. The rules that decided most episodes:

- The red is under the cube whose edge shows the red line, in every buried episode. With no red pixels at all, a
  level layer-2 cube bridging two floor cubes a cube width apart marked it (d1024).
- A boxed red opens along one jaw axis. The moving finger's end matters most, since it needs 20 to 28 mm. Aiming
  along the jaw axis never gets past a flush neighbour; sliding along the face gets past an offset one.
- A layer-2 cube by a wall comes out if the grasp rises in 1 mm steps until check's wall line clears, 5 to 9 mm in
  four episodes.
- Grip a leaning cube on its upright faces. Grips across the tilted faces, or high, or off centre, dropped the cube
  on the swing five times.
- Parking a removed cube 5 cm from the red, but beside one of its neighbours, blocked later picks in three episodes.

## What would make the task cheaper, ranked

1. Make "move this cube out of the way" one action. Places are 41% of all moves, and each is a second decision about a
   cube already dealt with. A skill that picks a cube and drops it in a chosen spot away from the red's neighbours,
   screened for leaning cubes, would cut the move count by about 40%. It would also stop the knock-on disturbance
   from hand-picked spots, which places cause more often than picks.
2. Lift the red higher. 5 of 29 red lifts held it tilted just under the success line, and each cost two moves. The
   pick lifts to the fixed travel height. A higher final lift for the red would turn most of those into successes. The
   reach limit above z 0.10 leaves room, but I haven't tested it.
3. Raise the 15-move cap, or cap robot time instead. The spec chose 15 because a red under a two-cube stack needs 5.
   Buried starts need 7 to 13 in practice, and d1000 needed about 16.
4. Look without parking. Parking is 14% of robot time. The wrist camera sees the pile from any hover pose, so a look
   from the last hover would save most of that, at the cost of the front camera's view past the arm.
5. Use pushes or drop them. One push in 205 moves. Either they need work (the players found picks more predictable)
   or the action set is simpler without them.
6. Spend less per demonstration. One episode per agent, shorter observe output (list only the cubes near the red and
   on its axes), four agents at a time, Sonnet for partial starts, and code for the routine calls (where to put a
   cube, lifting an obvious cover). The overnight run averaged about 40 agent requests per episode.

## Things to know before training on this

- The demos span rulebook v4 to v8 and three versions of perception. Early episodes were played without the wrist
  camera in observe and without the axis report, so their style differs from later ones.
- Undone moves are in each episode's record under "undone" but not in the demonstration. 9 of the 24 solved agent episodes
  used an undo, and the undone attempts tell a policy nothing, since it never sees them.
- Looks are real motions: every observe parks the arm, and those frames are in the data.
- e402 was played by me with 4 undos while the tools changed under it. Its notes are honest about how messy it was.
  Drop it if you want the cleanest set.
