# Mechanical search on a shelf

Approved by dm1487 on 2026-09-26.

## Goal

A shelf bay in front of the arm holds a few rows of upright cylinders. One is red and the rest are gray. The
robot reaches in sideways through the open front, under a board overhead, finds the red cylinder and pulls it
out. Getting to it usually means moving grays aside first, and no cylinder may tip over or leave the shelf on
the way. Huang et al. studied the same setting as "mechanical search on shelves" (LAX-RAY, 2020).

This adds a second search task next to the bin. It replaces nothing. `MechSearchEnv`, `BinPickEnv`, their
scenes and `models/stage.xml` stay as they are.

## Decided in the brainstorm

- The gripper enters the bay horizontally through the front. There is a top board, so nothing comes from
  above.
- All cylinders are the same size, 3 cm across and 8 cm tall. The target is the only red one, with no language
  instruction.
- Success is the red cylinder held in both jaws and pulled out past the shelf's front edge.
- A cylinder tilting past the angle where it can no longer stand back up ends the episode as a failure. For a
  3 by 8 cm cylinder that's atan(3/8) = 20.6 degrees. Wobbles below that are fine.
- Every gray stays on the shelf board for the whole episode, including while it's held. A gray leaving the
  board ends the episode as a failure, the same rule as cubes and the bin.
- Two interfaces over one task, as with the bin. A 30 Hz joint env, and a primitive env on top.
- Claude plays the primitive env as the high-level planner, from camera images only, the way the bin was
  played on 2026-09-25.
- CPU MuJoCo only.

- Amended the same day, after the first sizing run found no shelf: the arm touching the shelf is a failure,
  cylinders may still stand 2 cm from a side wall, and digs prefer pushes. The amendment under "Sizing rule" has
  the numbers.

- Amended a second time, after the first count tuning: partial starts cover part of the red on purpose. The
  second amendment under "Reset" has the numbers.

- Amended a third time, after the first primitive runs: the arm keeps 6 mm of clearance from the shelf, sizing
  checks carry height at every centre, the opening is 15 cm, and the front camera and start pose are the lowest
  that pass. The third amendment under "Sizing rule" has the numbers.

## Not in this step

- Training anything, reward shaping, and recording demonstrations to a LeRobotDataset.
- A batched Warp version. Before that happens, someone has to check that `mujoco_warp` collides cylinders
  with boxes and with each other the way CPU MuJoCo does.
- Mixed sizes, other shapes, and more than one shelf level.
- Building the real shelf. The layout this spec picks becomes a requirement for it.

## Why the shelf sits where it does

All measurements below come from scratch probes run on 2026-09-26. None of them is the final answer, since
`scripts/size_shelf.py` redoes them with every check in this spec.

### A horizontal gripper only reaches far out

Pan is the only joint that turns the arm about vertical. So with the gripper level, it can only point along a
ray from the base, give or take a few degrees for the jaw midpoint sitting off the roll axis. Lift, elbow and
wrist flex then have to sum to zero, in this MJCF's sign convention, to keep the tool level. A forward
kinematics sweep over those joints gives where the jaw midpoint can go:

| jaw midpoint height vs base plane | distance from base |
|---|---|
| −0.06 m | 0.33 to 0.39 m |
| 0.00 m | 0.34 to 0.46 m |
| +0.04 m | 0.32 to 0.47 m |
| +0.08 m | 0.27 to 0.48 m |
| +0.12 m | 0.26 to 0.48 m |
| below −0.08 m | nothing |

This is the opposite of the bin. Top-down reach is wide below the base, so the bin sank 10 cm. Side reach is
wide above it, so the shelf board goes up.

### Collisions set the board height and the opening

A second probe put a board, a top board and a back wall in the scene and counted cylinder centres straight
ahead that a level gripper reaches with no arm link touching the shelf or the table. The jaw midpoint was 4 cm
above the board, halfway up a cylinder.

| board top vs base | opening | front edge 0.24 m | front edge 0.27 m |
|---|---|---|---|
| +3 cm | 13 cm or more | 0.30 to 0.48 m | 0.30 to 0.48 m |
| +6 cm | 11 cm | 0.255 to 0.30 m | 0.285 to 0.33 m |
| +6 cm | 13 cm or more | 0.255 to 0.48 m | 0.285 to 0.48 m |
| +9 or +12 cm | 13 cm or more | 0.255 to 0.48 m | 0.285 to 0.48 m |

The opening has to be 13 cm for 8 cm cylinders because of the next finding.

### The wrist camera mount decides the roll

Rolling the wrist 90 degrees either way puts the jaws horizontal. One way, the wrist `camera_mount` hangs 8 to
9 cm below the gripper and hits the board, and the gripper gets only 4.5 cm into the bay. The other way, the
mount points up into the opening and the gripper reaches everything in the table above. So every side grasp
uses the same roll, with the mount on top. In that roll the open moving jaw always swings to the robot's
right. A cylinder can only be grasped with room on its right, which becomes part of the search.

CLAUDE.md already notes that nobody knows whether this mount exists on the real arm. Here it shapes the
whole task.

### Physics check

Upright cylinders stand still on a box board in MuJoCo 3.8.1 with default flags. Three test cylinders, two of
them touching, made 15 contacts. Tilt stayed at 0.00 degrees over 5 s, and a cylinder started 3 degrees off
settled back to vertical.

## Frames and heights

World z = 0 is the plane the arm's base stands on, which is also the table top. The board's top is at z = b,
its front edge at x = f. The opening H runs from the board's top to the top board's underside. Right means the
robot's right looking out along a ray, which is −y straight ahead.

- Cylinder radius r = 1.5 cm, height 8 cm.
- Grasp height `GRASP_Z` = b + 4 cm, the jaw midpoint at a cylinder's middle.
- Carry height `CARRY_Z` = b + 5 cm. A held cylinder rides 1 cm off the board.
- Push height `PUSH_Z`, the lowest of b + 2, 2.5 or 3 cm where the closed gripper clears the board
  everywhere on the push grid. Pushing below the centre of mass slides a cylinder instead of tipping it.
- The aisle line at x = f + 2 cm. A held cylinder only moves sideways with its centre on this line, which keeps
  it over the board with 5 mm to spare.
- The standoff line at x = f − 3 cm. Every primitive starts and ends with the jaw midpoint on it, outside the
  bay.
- Reset keeps every cylinder centre at x ≥ f + 5.5 cm, so a cylinder on the aisle line clears the front row by
  5 mm.

## Scene

`so101sim/shelf_layout.py` generates `models/shelf_objects.xml` and `models/shelf_scene.xml` from a
`ShelfLayout`, the same way `search_layout.py` does for the bin. The numbers live in `<custom>` fields and
`ShelfLayout.from_model` reads them back. Nobody edits the generated files.

`models/shelf_scene.xml` includes `stage.xml` and `shelf_objects.xml`. The objects file holds:

- The table, with its top at z = 0, as in `objects.xml`.
- The shelf unit on the table, facing the arm. A board, two side walls, a back wall and a top board, all
  1.5 cm thick, on four legs. Geom names start with `shelf_`.
- The cylinders. `cyl_red` and `cyl_gray_0` onward, in a `cylinder` default class defined in this file:
  r = 1.5 cm, half height 4 cm, density 1000, friction "1 0.005 0.0001", the cube's values, with condim 6 so
  the rolling coefficient applies. Under condim 4 a pushed cylinder wobbled on its rim without end.
  Cylinders wait on the ground plane at x = 1.2 until reset places them, as the bin's cubes do.
- The front camera, named `front`, at the pose the sizing script picks.

## Sizing rule

`scripts/size_shelf.py` picks the layout and stays in the repo, because the XML notes quote its numbers.

- Candidates:
  - Board top b of 3, 6 or 9 cm.
  - Opening H of 13, 15 or 18 cm.
  - Front edge f of 0.27, 0.285 or 0.30 m. The standoff line at f − 3 cm has to be reachable with the gripper
    level, and level reach bottoms out near 0.255 m, so f can't go much closer.
  - Board depth of 16, 18, 20 or 22 cm and inner width of 20, 24, 28 or 32 cm, centred on y = 0.
- Checks, with the empty shelf, on a 5 mm grid, with an arm geom touching the shelf or the table counting as
  a fail. Level means the gripperframe's y axis within 2 degrees of straight up.
  1. Grasp. Every cylinder centre at least r + 5 mm from the side and back walls and at x ≥ f + 5.5 cm has a
     level solution at `GRASP_Z` with the gripper at `GRIP_PLAN`, within 1 mm.
  2. Open jaw. The same, with the gripper at the side-open value. Centres within 6 cm of the right wall are
     exempt, because the moving jaw opens toward it. They need a push first, like the bin's 2 cm wall strip.
  3. Aisle. Every point on the aisle line at `CARRY_Z` with the gripper at `GRIP_HOLDING`.
  4. Standoff. Every point on the standoff line at `GRASP_Z`, `CARRY_Z` and `PUSH_Z`.
  5. Rays. Every point on the straight segment from the aisle line out to the standoff line, at `CARRY_Z`.
  6. Push. The closed gripper at `PUSH_Z` over the whole grid of checks 1 and 3.
- Among passing layouts it picks the largest area of centres passing check 1, then the smallest opening,
  then the front edge nearest the base.
- If no layout reaches 12 by 18 cm of graspable centres, stop and bring it back to dm1487.

The side-open gripper value is the smallest lerobot value that leaves 1 cm between each jaw and a cylinder
centred between them, measured by the script. `GRIP_OPEN` swings the moving jaw 55 mm out, which would knock
over the right-hand neighbour of nearly every cylinder.

### Amendment after the first sizing run, 2026-09-26

The first run of `scripts/size_shelf.py` found no layout. Level reach passed with the front edge at 0.30 m.
Collisions failed. At cylinder centres 2 to 2.5 cm from a side wall the closed gripper is wider than the
cylinder: the fixed jaw and the gripper body hit the left wall, and the moving jaw hits the right wall. The bin
hit the same thing on 2026-09-25. A scratch rerun with centres kept 3 cm from each side wall chose a 32 by 18 cm
shelf with the board 6 cm up, a 13 cm opening and the front edge at 0.30 m. Its front camera, behind the base at
(−0.10, 0, 0.40), saw 98% of the board, and the wrist camera saw 99%.

dm1487 decided:

- The arm touching the shelf is very bad behaviour. In `ShelfEnv` any arm geom touching a shelf panel, a leg or
  the table ends the episode as a failure, checked first each frame, with the geom named in `info["failure"]`.
  `ShelfPrimitiveEnv` checks every frame of a planned path kinematically, cylinders ignored, and refuses a
  primitive whose path would touch the shelf or the table.
- Cylinders aren't kept away from the walls. Reset keeps the original region, centres r + 5 mm from the walls.
  The sizing checks skip centres within 3 cm of a side wall, `WALL_STRIP`, in every collision check, as the
  bin's did. A cylinder there needs a push before a pick. The red cylinder always starts outside that strip,
  because a red the gripper can't fit around would make the start impossible rather than hard.
- Push keeps the jaws closed along any of 16 headings. The scripted dig-out and Claude's planner runs try a push
  before a pick-and-place when clearing the way.

### Third amendment, after the first primitive runs, 2026-09-26

Every place hit the top board. With a 13 cm opening the wrist camera mount scraped it at carry height, and a
carried arm bounces about 5 mm around its plan. Sizing only counted real contacts, and it never checked carry
height at the cylinder centres, where place carries a cylinder in. Now sizing has a carry check at every centre,
and every collision check, in sizing and in `ShelfPrimitiveEnv`'s path check, needs clearance from the shelf and
the table. The check runs on a copy of the model that never steps physics, with the shelf boxes grown by the
clearance and MuJoCo's midphase off, because the midphase's bounding-box tree is built at compile time and skipped
pairs inside the grown band (found by the final review). At 8 mm no shelf passes: pushing deep in the bay takes the
forearm 6 to 8 mm over the board's front edge. dm1487 chose 6 mm, which still covers the measured 5 mm bounce.
Sizing then picks the board 9 cm up, a 15 cm opening, the front edge at 0.30 m, 18 by 32 cm.

The camera sweep below was taken with an 18 cm opening, the layout an earlier, faulty check had picked. The
conclusion carries over: any camera well above the cylinder tops sees over the grays.

With the top board 18 cm up, the front camera 40 cm up saw over every cylinder, and hidden starts kept 0 to 2% of
attempts. A sweep at 14 cylinders:

| front camera | board seen | hidden kept | partial kept |
|---|---|---|---|
| (0.05, 0, 0.40) | 100% | 0% | 62% |
| (0.05, 0, 0.25) | 91% | 0% | 65% |
| (0.05, 0, 0.20) | 84% | 40% | 62% |
| (0.05, −0.30, 0.20) | 89% | 50% | 52% |

dm1487 wants the robot to find the red by pushing things out of the way, which needs a start where it's really
hidden. So the front camera follows the bin's rule: the lowest candidate spot that sees at least 60% of the board,
most board first among spots at that height. Candidate heights are 0.20, 0.25, 0.30 and 0.40 m. The start pose
follows the same rule, the lowest candidate whose wrist camera sees 40%, because with the wrist camera 4.5 cm above
the cylinder tops no hidden start survived, and 1.5 cm above them 90% did. A hidden start also places its grays
where a camera still sees the red, up to three, instead of one per camera at a fixed distance: the two cameras'
sight lines can run within 15 degrees of each other, and the wrist camera's gray kept losing its spot.

### Cameras and start pose

A camera sees a point on the empty board if the point projects inside its image and a ray from the camera hits
the board first. The script tests a 1 cm grid of 6 mm markers in a segmentation render, as the bin script
does.

- Front camera. The bin's camera spot at x = 0.70 sits behind the shelf's back wall. The script tries x of −0.10
  and 0.05, y of −0.30, 0 and 0.30, and z of 0.30 and 0.40, aimed at the board's centre at `GRASP_Z`. It keeps
  the spot that sees the most board with the arm at `SHELF_HOME`, and needs at least 60%. The throwaway render
  on 2026-09-26 showed the arm blocking much of the opening from beside it, so this check may fail.
- `SHELF_HOME` holds the gripper level with its jaw midpoint on the standoff line, at y of 0 or ±8 cm and at
  `GRASP_Z` or 3 cm above it. The wrist camera has to see at least 40% of the board. Among poses that pass, the
  script keeps the one that leaves the front camera the most board.
- If no camera spot or start pose passes, stop and bring it back to dm1487. Don't relax a threshold quietly.

## Side IK

`ik.solve_side(model, target, gripper=0.0, q_init=None)` puts the jaw midpoint at `target` with the
gripperframe's y axis straight up, which is the level, mount-up pose. Position is 3 constraints and the axis is
2, which uses all 5 arm joints. Heading is whatever the arm gives, a few degrees off the ray. It returns
`(q, info)` with `pos_err` in metres and `tilt` and `heading` in degrees, the same shape as `solve_topdown`,
and retries from seeded poses so results repeat.

## Primitives

`so101sim/side_primitives.py` builds each primitive as waypoints solved with `solve_side` and retimed with
`primitives._retime`, so every trajectory keeps the 2 degree per frame cap. The ray through a point is the
line from the base's pan axis through it.

- `pick(xy)`. At the standoff on the ray through `xy`, open to the side-open value. Go in along the ray at
  `GRASP_Z` until the jaw midpoint is at `xy`. Close to `GRIP_CLOSE` and hold 8 frames. Rise to `CARRY_Z`. Back
  out along the ray until the jaw midpoint is on the aisle line, and stop there, holding.
- `place(xy)`. From the aisle, carry the held cylinder sideways along the aisle line at `CARRY_Z` to the ray
  through `xy`. Go in along it to `xy`, lower to `GRASP_Z`, open to the side-open value, hold, and back out to
  the standoff.
- `pull()`. From the aisle, take the held cylinder straight out along its ray to the standoff line. That puts
  its centre 3 cm past the front edge. This is the only move that takes anything off the board. On the red
  cylinder it's the success move. On a gray it's the failure a policy has to learn to avoid.
- `push(xy, heading)`. Jaws closed, from the standoff in along the ray to `xy` at `PUSH_Z`. Move 5 cm along
  `heading`, one of 16. Retrace to `xy`, then out to the standoff.
- `look(xy)`. Move to the standoff on the ray through `xy` and stop. It changes nothing physical, and it points
  the wrist camera down that ray. With the target hidden, where to look is half the search.
- `bridge(model, current, traj)`. From a level pose on the standoff line, slide along it to the next start.
  From the aisle, the next primitive already starts there. From anything else, including a fresh reset, fall
  back to a joint-space ramp, the same fallback `primitives.bridge` uses.

Going in along a ray moves the gripper through whatever sits on that ray between the front edge and `xy`.
That's deliberate. Clearing the ray and the target's right side is the task.

## Reset

1. Reset `SO101Env` and put the arm at `SHELF_HOME`.
2. Choose hidden or partial. `options["hidden"]` if it's given, otherwise a draw from `np_random` with
   probability 0.5.
3. Place every cylinder upright at a sampled xy, then settle for up to 0.5 s of sim time or until
   max|qvel| ≤ 0.2. Positions stay inside the region of sizing check 1, at least 3.5 cm apart centre to centre,
   which leaves a 5 mm gap.
   - Hidden. The red goes first, in the back two thirds of that region. Then a gray goes on the front camera's
     sight line to the red, and another on the wrist camera's, each 3.5 to 5 cm in front of the red. The rest go
     at random.
   - Partial. The grays go at random, and the red goes last at random.
4. Accept the start only if all of these hold:
   - Every cylinder is upright within 2 degrees and still.
   - Every centre is inside the region of sizing check 1.
   - There's a free spot for a place. That's a point in that region at least 4.5 cm from every centre.
   - Hidden. Zero face pixels of the red cylinder in both cameras, at 640 by 480 with the bin's 5 by 5
     erosion, `mechsearch.face_pixels`.
   - Partial. At least 100 face pixels in one camera, and at most half of what that camera shows of the red
     with the grays hidden, in each camera. That's the bin's current rule, for the same reason: a fully visible
     red is a plain pick.
   - No cylinder touched the arm while settling.
5. On a failed check, go back to step 3 with fresh draws from the same `np_random`, so a seed always gives the
   same shelf. Give up after 50 attempts.

### Cylinder count

Start at 18, one red and 17 gray. Take away one gray at a time until at least half of all resets are accepted
over 100 seeds, for hidden and partial starts separately. If the count drops below 8, stop and discuss.
`scripts/shelf_smoke.py` does this, as `mechsearch_smoke.py` did for the bin. Reset should average under 3 s of
wall time on one core, rejected attempts included. If it's slower, report the number.

### Second amendment, after the first count tuning, 2026-09-26

With the red placed at random, partial starts kept at most 22% of attempts, at 14 cylinders, against the 50% the
count rule needs. Resets took 0.1 s. Of 150 partial attempts, 79 showed more than half of the red and 34 showed
fewer than 100 pixels. The front camera sees the whole board from above, so a red placed at random is usually in
plain view. Hidden starts kept 51%. dm1487 chose to build partial occlusion rather than judge partial starts by
reset time as the bin does:

- A partial red goes in the back two thirds of the region, like a hidden one, outside the wall strip.
- A gray goes on each camera's sight line to it, 3.5 to 5 cm in front, shifted sideways by 0.8 to 1.6 cm to a
  random side, so it covers part of the red instead of all of it. A sweep over 200 seeds picked the shift: 0.8
  to 1.6 cm kept 57% of partial attempts at 14 cylinders, against 49% for 1.2 to 2.2 cm.
- The rest go at random. The acceptance rules and the 50% count rule stay as they are for both start types.

## `ShelfEnv`, the 30 Hz env

A subclass of `SO101Env` that loads `shelf_scene.xml` and starts at `SHELF_HOME`. Actions and observations are
`SO101Env`'s: six joint targets in lerobot units in, and `observation.state` plus the front and wrist images
out. Cylinder poses never enter the observation.

Each `step` runs the parent step, then checks these in order:

1. Topple. Any cylinder, held or not, tilted more than 20 degrees from vertical ends the episode. Reward 0,
   and `info["failure"]` names the cylinder.
2. Off the board. A gray whose centre leaves the board's footprint ends the episode, and so does the red if
   it leaves while not held by both jaws.
3. Success. The red is held by both jaws, its centre at least 2 cm in front of the front edge, and it's tilted
   less than 20 degrees. After 5 such frames in a row the episode ends with reward 1.0. A frame that fails the
   test resets the count.
4. Time limit. The episode truncates after `max_steps` frames. The default is 2700, 90 s, because going in
   and out along rays takes longer than the bin's hover moves. `None` turns it off.

`info` carries `target` (the red's position and quaternion), `cylinders`, `hidden`, `target_pixels`,
`n_on_board`, `is_success`, `failure`, `steps`, `reset_attempts`, and `task`, which is always "retrieve the red
cylinder from the shelf".

Constructor arguments: `scene_path`, `max_steps=2700`, `hidden_prob=0.5`, plus `SO101Env`'s `height` and
`width`.

## `ShelfPrimitiveEnv`, the primitive env

It wraps a `ShelfEnv` with `max_steps=None`, as `MechSearchPrimitiveEnv` wraps `MechSearchEnv`. One `step`
runs one whole primitive.

- The action is a dict. `kind` is `Discrete(5)`: pick, place, push, pull, look. `xy` is a `Box` over the region
  of sizing check 1. `angle` is a `Box` from −π to π, the push heading, ignored by the other kinds. `pull`
  ignores `xy`. There's no z, since every cylinder stands on the same board.
- The env feeds every trajectory row to the inner env, stops the moment it terminates, then holds the last
  row for up to 60 frames to settle with the checks still running.
- A primitive that can't run is refused. The arm doesn't move, the step counts, and `info["refused"]` says why.
  Refusals cover an `xy` outside the box, a push ending outside it, IK failing, a pick, push or look while
  holding a cylinder, and a place or pull while holding nothing. Whether the arm holds
  something comes from the gripper joint, which stops near 18 on a 3 cm cylinder instead of reaching the
  commanded 10. The real arm reads the same joint.
- Reward is 1.0 if success happened during the primitive. The episode truncates after `max_primitives`, 20 by
  default.
- An `on_frame` hook sees every 30 Hz frame.

## Claude as the planner

`scripts/shelf_play.py` lets Claude play `ShelfPrimitiveEnv` from images, as the bin was played. It stays in
the repo, where the bin's version lived in scratch, so the runs can be replayed.

- `python scripts/shelf_play.py --seed S --start hidden|partial --actions actions.json --out DIR [--video]`
  replays every action from reset, since the sim is deterministic. It writes the latest front and wrist
  images at 640 by 480 with a labelled pixel grid. It prints only what a policy gets back per step: refused and
  why, success, and the failure message. It never prints a pose.
- An action names a pixel instead of a position:
  `{"kind": "pick", "cam": "front" | "wrist", "px": [u, v], "heading": k}`. The script casts the ray through
  that pixel from the camera's pose. For the wrist camera that pose comes from the joint angles, which the real
  arm knows too. For pick and push it intersects the ray with the plane at `GRASP_Z`, which lands on the
  cylinder's visible face, and steps r further along the ray's horizontal direction to the axis. For place and
  look it intersects the board's plane.
- The seeds are fixed before looking: 300 hidden and 301 partial.
- There's no undo. The action list only grows, a refused primitive still counts, and a failed episode is
  reported as a failure.
- `--video` writes both cameras side by side for the whole run to `outputs/shelf_play_<seed>.mp4`. A log of
  what Claude saw at each step and why it chose each action goes in `outputs/shelf_play_<seed>.md`.

## Other code changes

- `so101sim/__init__.py` gains `SHELF_SCENE_PATH`.
- `so101sim/shelf.py` holds both envs, `SHELF_HOME` and the height constants. It imports `face_pixels` from
  `mechsearch` and the contact checks from `grasp`, and doesn't change either file.
- `so101sim/shelf_checks.py` holds the sizing checks, as `search_checks.py` does for the bin.
- `scripts/shelf_smoke.py` resets N episodes, writes both camera views of each start side by side to an mp4,
  prints reset time, acceptance and red pixel counts, and with `--tune-count --write` picks the cylinder count.
- CLAUDE.md gets the new files and commands, and the shelf's b and f as a hardware requirement. It also gets
  the traps found here: the camera mount fixing the roll, the moving jaw always opening right, and that
  editing `geom_pos` on a compiled model's world geoms doesn't move them.

## Tests

`tests/test_side_ik.py` and `tests/test_shelf.py`, CPU only.

- `solve_side` reaches a grid of points inside the bay level and mount up, and reports failure honestly
  outside level reach.
- Every side primitive keeps each frame within 2 degrees of the previous one.
- `shelf_scene.xml` loads, and its timestep divides 1/30 s.
- The chosen layout passes every sizing check. `SHELF_HOME` sees at least 40% of the board from the wrist and
  the front camera at least 60%.
- The same seed gives the same shelf.
- A hidden reset has zero red face pixels in both cameras. A partial reset meets the partial rule.
- After reset every cylinder is upright, still, inside the region and 3.5 cm from its neighbours, and a free
  spot exists.
- Tilting a gray 25 degrees by hand ends the next step, with its name in `info["failure"]`.
- Setting a gray's pose off the board ends the next step.
- Pulling a held gray ends the episode as a failure.
- Success needs 5 frames. With the held-and-out predicate stubbed, 4 true frames then a false one give no
  success, and 5 give success.
- `max_steps` truncates.
- No pose appears in the observation.
- The action bounds match the region of sizing check 1. Refused primitives leave the arm where it was.
- Solvability, the gate. On one fixed hidden seed, primitives chosen from `info` poses clear the red's ray and
  right side, placing each gray at a free spot, then pick and pull the red. The env has to report success with
  nothing toppled and every gray on the board. If this can't pass, work stops until dm1487 decides what
  changes.

## Risks

- The front camera. Beside the arm, the arm hides much of the opening, and the top board hides the back rows
  from anything higher than the board. The 60% rule may pass nowhere. Then it comes back to dm1487, with the
  wrist camera and `look` carrying most of the search.
- A held cylinder can swing. Two jaw contacts on a round surface make a horizontal pivot, and torsional
  friction of 0.005 resists very little. Accelerating in or out along a ray swings the cylinder about that
  pivot, and past 20 degrees it fails as a topple. If the smoke runs show it, the options are slower moves near
  the ends of each segment, gripping lower, or raising torsional friction on the cylinder class. The last one
  changes the physics and needs dm1487's say.
- The right-wall strip. Anything within 6 cm of the right wall needs a push before a pick. If the red lands
  there in a hidden start, the dig gets longer.
- The camera mount on hardware. If the real arm has no mount, the other roll opens up and the opening can
  shrink. If it has a different mount, the 13 cm opening may be wrong. Measure it before building anything.
- Reach at the standoff. The standoff line sits close to level reach's inner limit of about 0.255 m, so the
  front edge can't come much closer than 0.285 m.
- Pushes into neighbours. A pushed cylinder that runs into another one can still tip, and random pushes will
  fail a lot. That's part of the task, and it will show in the solvability test.
- The success check uses privileged state, the same as the bin. That's fine for labels in sim and matters
  again only when evaluating on hardware.
