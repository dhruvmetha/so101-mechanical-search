# Mechanical search in a bin

Approved by dm1487 on 2026-09-25.

## Goal

A bin holds a pile of 30 mm cubes up to three layers deep. One cube is red and the rest are gray. The robot has to find the red cube and lift it out, which often means moving gray cubes off it first. All of that moving happens inside the bin. The literature calls this mechanical search (Danielczuk et al., ICRA 2019).

This settles the task that CLAUDE.md left open. It adds to the repo and replaces nothing. `BinPickEnv`, its scene and its demo keep working as they are.

## Decided in the brainstorm

- 30 mm cubes only.
- The target is the only red cube. There is no language instruction to pick among colors.
- Half the episodes start with the red cube buried, meaning neither camera sees a pixel of it. The other half start with it partly visible.
- Every cube stays in the bin for the whole episode. There is no buffer area. If any cube's center leaves the bin, the episode ends as a failure. That includes carrying a gray cube out and bringing it back.
- Two interfaces over one task. A 30 Hz joint env, and a primitive env on top of it with pick, place and push.
- The bin floor sits below the arm's base. In sim the arm stands on a riser. On hardware that means a riser under the arm or a bin set into the table.
- CPU MuJoCo. The same scene can later run batched on MuJoCo Warp, which has a batch renderer (see the correction in the 2026-09-18 spec), but that is not this step.
- Amended the same day, after the first sizing run found no bin: walls at most 6 cm, and the clearance check skips any cube within 2 cm of a wall, not only corners. Three layers stay. The amendment under "Sizing rule" has the numbers.
- Amended a second time, after the first resets: buried means no face of the red cube shows, so the thin seams between touching cubes don't count. Buried starts box the red cube in. Bins up to 24 cm wide are candidates. The second amendment under "Reset" has the numbers.
- Amended a third time, on 2026-09-26: fully visible starts are gone. A non-buried start must be partially visible, showing at most half of what each camera would see of the red cube with every gray cube removed. dm1487's call: a fully visible red cube is a plain pick and tests nothing about search. The mix stays 50/50 buried and partial.

## Not in this step

- A scripted planner or baseline, and recording demonstrations to a LeRobotDataset.
- Training anything, and reward shaping.
- A batched Warp version of the task.
- Objects other than 30 mm cubes, and piles deeper than three layers.
- Building the real setup. The riser height this spec picks becomes a requirement for it, and the real gripper mapping is still unmeasured.

## Why the bin sits below the base

Pointing straight down, the arm's reach narrows as the gripper rises and runs out about 9 cm above the base. It stays wide below the base. A scratch probe on 2026-09-25 ran `ik.solve_topdown` straight out from the base at a range of jaw-midpoint heights, with the gripper at the pick value. It counted a point as reachable if IK hit it within 1 mm and 1 degree of vertical for at least one of four jaw yaws.

| height vs base | reachable distance from base |
|---|---|
| +8 cm | 0.16 to 0.25 m |
| +5 cm | 0.12 to 0.29 m |
| +2 cm | 0.10 to 0.31 m |
| −4 cm | 0.09 to 0.33 m |
| −10 cm | 0.05 to 0.32 m |
| −16 cm | 0.05 to 0.30 m |
| −20 cm | 0.09 to 0.26 m |

The 0.05 m readings are probably inside the riser, because the probe ignored collisions.

On the table the base stands on, a two-layer pile needs the gripper to cross at about 8.4 cm, where the band is 5 cm wide. That capped the bin at 10 cm, and even there 5% of positions were out of reach. With the bin floor 10 cm below the base, the heights a three-layer pile needs all fall in the wide part of the band. A second probe tested every cube-center position on a 1 cm grid inside square bins centered at (0.20, 0), with the floor 10 cm down:

| inner width | layer-1 grasp, −8.5 cm | layer-3 grasp, −2.5 cm | carrying over 3 layers, +1 cm |
|---|---|---|---|
| 12 cm | 100% | 100% | 100% |
| 15 cm | 100% | 100% | 100% |
| 18 cm | 100% | 100% | 100% |

The 18 cm bin also passed at 100% centered at x = 0.21. Neither probe checked collisions, so a forearm hitting the near wall would not have shown up. `scripts/size_search_bin.py` redoes the probes with collision checks, and the XML cites its output rather than these tables.

## Heights

All heights are z in the world frame, where the base plane is z = 0. The table top is at z = −D, where D is the riser height. The bin floor is 4 mm thick, so its top is at f = −D + 0.004. The bin's wall height w is measured from f, so the rim is at f + w.

- Layer k's cube center is at f + 1.5 cm + 3 cm × (k − 1). That's f + 1.5, f + 4.5 and f + 7.5 cm.
- The pile top is at f + 9 cm.
- `TRAVEL_Z` is the jaw-midpoint height for all motion across the bin. It's the larger of f + 11 cm and rim + 3 cm. The first keeps a carried cube's bottom 5 mm above the pile top. The second lifts a picked cube's bottom 1.5 cm over the rim.

## Scene

`models/stage.xml` is new and holds what every task shares. That's the arm include and its `meshdir` override, the timestep, visuals, assets, lights, the ground plane, the base collision box and its contact exclude. The table and the front camera move out of it, because this task puts the table lower and the camera somewhere else.

- `models/objects.xml` gains the table at z = 0, the `front_cam_target` body and the `front` camera, unchanged from today's `scene.xml`.
- `models/scene.xml` becomes an include of `stage.xml` plus an include of `objects.xml`, and compiles to the same model as before.
- `models/search_scene.xml` includes `stage.xml` and `models/search_objects.xml`.

`models/search_objects.xml` holds:

- **The table**, with its top at z = −D.
- **A riser** named `riser`, a box under the base from −D up to 0, 12 cm square, centered under the base collision box.
- **The bin.** The body is called `bin` and uses the same geom names as `objects.xml`, so `binpick.bin_geometry` reads it unchanged. It sits on the table, with walls 4 mm thick like the current bin.
- **The cubes.** `cube_red`, plus `cube_gray_0` onward in a gray material, all in the existing `cube` default class, so `binpick.cube_names` finds them.
- **The front camera**, named `front`, since `SO101Env` looks cameras up by name. Its pose comes from the sizing script.

### Sizing rule

`scripts/size_search_bin.py` picks D, the bin's inner width and center, and w. It stays in the repo, because the XML comments quote its numbers.

- Candidates:
  - D of 6, 8, 10 or 12 cm.
  - Inner width of 24, 22, 20, 18, 15 or 12 cm. The first three were added in the second amendment.
  - Center x of 0.19, 0.20 or 0.21 m, with y = 0.
  - w of 6, 5 or 4 cm. This was 6, 8 or 10 cm until the amendment below.
- Checks, run over a 5 mm grid of cube-center positions inside the bin:
  1. Layer-1 grasp at f + 1.5 cm, gripper at `GRIP_PLAN`.
  2. Layer-3 grasp at f + 7.5 cm, gripper at `GRIP_PLAN`.
  3. Travel at `TRAVEL_Z`, once with the gripper at `GRIP_PLAN` and once at `GRIP_HOLDING`.
  4. Clearance at the layer-1 grasp pose with the gripper at `GRIP_OPEN`. No arm geom touches the bin, the riser or the table, for at least one of four yaws. This catches the jaw hitting a wall and the forearm hitting the near wall. The check skips positions where the cube's face is within 2 cm of any wall. A cube that close to a wall has to be pushed out first, and push exists for exactly that. Before the amendment the check skipped only corners.
- Checks 1 to 3 must pass at every grid point, and check 4 at every point it covers. Among the passing configurations the script picks:
  1. The widest bin.
  2. The shallowest D, since a lower riser is easier to build.
  3. The tallest wall.
  4. The center nearest 0.20 m.
- If nothing passes at 12 cm wide, stop and bring it back to dm1487 before building anything else.

### Amendment after the first sizing run, 2026-09-25

The first run of `scripts/size_search_bin.py`, with walls of 6, 8 or 10 cm and only corners exempt from check 4, found no layout. Reach passed for many candidates. Clearance failed for all of them. A follow-up probe with the riser at 10 cm showed why:

| bin | walls | blocked cube positions | where | what hits |
|---|---|---|---|---|
| 15 cm | 6 cm | 41 of 169 | face within 2 cm of a wall | gripper, only at the wall |
| 18 cm | 6 cm | 50 of 225 | face within 1.5 cm of a wall | same |
| 15 cm | 8 cm | 82 of 169 | up to the middle | `camera_mount`, near wall |
| 18 cm | 8 cm | 58 of 225 | face up to 4.5 cm in | gripper housing and `camera_mount` |
| 18 cm | 10 cm | 65 of 225 | face up to 2.5 cm in | gripper housing, near wall |

The gripper is wider than a cube, so no straight-down grasp fits a cube against any wall, not only in a corner. Walls taller than 6 cm add a second failure: reaching the floor, the wrist's `camera_mount` and the gripper housing catch the top of the wall. CLAUDE.md already notes that nobody knows whether that camera mount exists on the real arm. dm1487 chose walls of at most 6 cm with a 2 cm wall strip exempt from check 4, keeping three layers.

### Cameras and start pose

A camera "sees" a point on the empty bin floor if the point projects inside its image and `mj_ray` from the camera hits `bin_floor` first. The script tests this on a 1 cm grid over the floor, with no cubes in the bin.

- **Front camera.** It stays on the x axis at x = 0.70, aimed at the middle of the bin floor. The script tries heights of 0.45, 0.55, 0.65 and 0.75 m and keeps the lowest that sees at least 70% of the floor. Lower is better because it sees the pile from the side, which the wrist camera can't.
- **Start pose.** `SEARCH_HOME` defaults to `SO101Env`'s `HOME`. There the jaw midpoint is at z = 0.078, which should clear anything dropped into a sunken bin. It has to meet two conditions:
  - No arm geom reaches below f + 13 cm inside the column above the bin's inner footprint. That's the height where reset drops cubes.
  - The wrist camera sees at least 50% of the floor.

  If `HOME` fails either one, the script searches a small set of candidate poses.
- If no front camera height or start pose passes, stop and bring it back to dm1487. Don't relax a condition quietly.

### Cube count

Start at 30 cubes, one red and 29 gray. Take away one gray at a time until at least half of all reset attempts are accepted over 100 seeds, for buried and visible starts separately. After the third amendment the half-kept rule applies to buried starts only. Partial starts keep about 12% of dropped piles at every count from 30 down to 10, so dm1487 chose to judge them by the 3 s reset budget instead; they average 2.7 s at 30 cubes. The reset attempt limit went from 50 to 200 so that low keep rate almost never exhausts it. If the count falls below 10, stop and discuss.

## Reset

1. Reset `SO101Env` and put the arm at `SEARCH_HOME`.
2. Choose buried or visible. Use `options["buried"]` if it's given. Otherwise draw from `np_random` with probability 0.5.
3. Drop the cubes in waves of up to 4.
   - Cubes in a wave land at xy positions at least 4.5 cm apart, and each gets a random yaw.
   - Each cube starts with its bottom 1 cm above the highest cube already in the bin, or 1 cm above the floor if the bin is empty.
   - Each wave settles until max|qvel| is at most 0.2, capped at 1 s of sim time.
   - Buried: the red cube goes on the floor first, at a random yaw and a random position far enough from the walls for neighbours to fit around it. Four grays go flush against its four sides, 1 mm apart, and a fifth sits on top. That box settles as one wave. The rest land at random positions inside the bin. This replaced the first recipe, one gray dropped on top, in the second amendment.
   - Partial: the red cube drops in a random middle wave, never the first or the last, so later cubes land on and around it. Until the third amendment this was "visible", with the red cube in the last wave.
4. Accept the pile only if all of these hold:
   - Every cube's center is inside the bin's inner footprint.
   - No cube's center is higher than f + 8.5 cm, so there are at most three layers.
   - The floor has a free spot. That's a point at least 1.8 cm from every wall and at least 4.5 cm from the center of every bottom-layer cube. A bottom-layer cube is one with its center below f + 3 cm.
   - The scene has settled.
   - The red condition holds. Buried means zero face pixels of the red cube in both cameras. Partial means at least 100 face pixels in at least one camera, at 640×480, and no camera showing more than half the face pixels it shows with the gray cubes hidden. The gray cubes sit in their own geom group, so one extra render gives that unoccluded count.
   - No cube touched an arm geom while settling.
5. If any check fails, go back to step 3 with fresh draws from the same `np_random`, so a given seed always gives the same pile.

The pixel counts come from a segmentation render at 640×480, whatever the observation size. A red pixel counts as a face pixel only if the 5×5 patch around it is all red, which drops the 1 to 4 pixel seams between touching cubes and keeps any visible face. The counts label the episode and never enter the observation.

### Second amendment, after the first resets, 2026-09-25

With the 18 cm bin, buried starts almost never passed. Under the first recipe, one gray dropped on top, the red cube showed from the side. Boxed in on every side, it still showed as thin lines through the 1 mm seams, which zero red pixels counted as visible. Buried piles kept, out of 30 per row, with seams not counted:

| cubes | one gray on top | boxed in |
|---|---|---|
| 14 | 0 | 26 |
| 18 | 3 | 20 |
| 24 | 1 | 9 |

Most of the losses at 24 cubes were piles with no free floor spot, which a wider bin eases. dm1487 approved the face-pixel rule, the boxed recipe and bins up to 24 cm. A buried red cube now always sits on the floor with grays touching its sides. The random drops above hide most of that, but it is a regularity a policy could learn.

Reset should average under 3 s of wall time on one CPU core, rejected attempts included. `scripts/mechsearch_smoke.py` measures it. If it's slower, report the number to dm1487 rather than cutting the cube count to hit it.

## `MechSearchEnv`, the 30 Hz env

A subclass of `SO101Env` that loads `search_scene.xml` and starts at `SEARCH_HOME`. Action and observation are the same as `SO101Env`: six joint targets in lerobot units in, and `observation.state` plus the front and wrist images out. Object poses never enter the observation.

Each `step` runs the parent step, then checks these in order:

1. **Cube left the bin.** If any cube's center is outside the inner footprint, the episode terminates. Reward is 0, and `info["failure"]` names the cube.
2. **Success.** The red cube counts as retrieved on a frame where both jaws touch it and its lowest corner is at least 1 cm above the rim. Changed after the final review: the lowest corner must also be 2 cm above where a top-layer cube rests. With 5 cm walls, rim + 1 cm is exactly that resting height, so a grasp and a 2 mm lift used to count. A primitive pick holds the cube's lowest corner at about floor + 8.8 cm, against the new line at floor + 8 cm. After 5 such frames in a row the episode terminates with reward 1.0. A frame that fails the test resets the count.
3. **Time limit.** The episode truncates after `max_steps` frames. The default is 1800, which is 60 s, and `None` turns it off.

Every other step returns reward 0.

`info` carries:

- `target`: the red cube's position and quaternion.
- `buried`: how the episode started.
- `target_pixels`: the red pixel count in each camera at reset.
- `n_in_bin`, `is_success`, `steps`.
- `failure`: `None`, or the reason the episode failed.
- `task`: always "retrieve the red cube from the bin", because lerobot datasets store a task string per frame.

Constructor arguments: `scene_path`, `max_steps=1800`, `buried_prob=0.5`, plus `SO101Env`'s `height` and `width`.

## `MechSearchPrimitiveEnv`, the primitive env

This env wraps a `MechSearchEnv` with `max_steps=None`, the same way `BinPickEnv` wraps `SO101Env`. One `step` runs one whole primitive.

The action is a dict:

- `kind`: `Discrete(3)`, one of pick, place or push.
- `xyz`: a `Box` covering the bin's inner footprint pulled in by one cube half-width on every side. z runs from f + 1.5 cm to f + 7.5 cm, layer 1 to layer 3.
- `angle`: a `Box` from −π to π, with the same meaning as in `BinPickEnv`. For pick it's jaw yaw, for place the held cube's yaw, and for push the heading.

z is always the jaw midpoint's height. For pick that's the grasp height, and for place it's where the held cube's center sits at release. For push it's the height of the closed jaw tips during the stroke. z is part of the action because one xy can hold a cube on any of three layers, and the policy has to say which. It reads that from the images.

A primitive calls the existing functions:

- Pick calls `primitives.pick(model, xy, yaw, z=z, hover_z=TRAVEL_Z)`.
- Place calls `primitives.place(model, xy, yaw, hover_z=TRAVEL_Z, release_z=z)`.
- Push calls `primitives.push(model, xy, angle, stroke=0.05, hover_z=TRAVEL_Z, push_z=z)`.
- `primitives.bridge(..., hover_z=TRAVEL_Z)` links each primitive to the previous one.

Pick lifts to `TRAVEL_Z`, which puts a lifted cube's bottom at least 1.5 cm over the rim. So a clean pick of the red cube ends the episode.

The primitive env feeds every trajectory row to the inner env's `step`. The inner env's checks run on every frame, and the primitive stops the moment the inner env terminates. After the trajectory it holds the last row for up to 60 frames to let things settle, with the checks still running. This matches `BinPickEnv._settle`.

Sometimes a primitive can't run. IK may raise `Unreachable`, or a push's end point may land outside the pulled-in footprint. The arm then doesn't move, the step still counts, and `info["refused"]` gives the reason. `info["failure"]` is kept for failures that end the episode.

- Reward is 1.0 if success happened during the primitive.
- The episode truncates after `max_primitives` primitives, default 15. A red cube under a two-cube stack needs 5, and jammed neighbors add pushes.
- An `on_frame` hook sees every 30 Hz frame, the same as `BinPickEnv`'s.

## Other code changes

- `so101sim/grasp.py`, new. It holds `jaw_geom_ids(model)` and `is_grasped(model, data, geom_id, jaw_ids)`, moved out of `BinPickEnv`. `BinPickEnv.is_grasped` calls them, with its behavior unchanged.
- `primitives.pick` gains a `z` argument that defaults to `CUBE_HALF`, so existing callers behave exactly as before.
- `so101sim/__init__.py` gains `SEARCH_SCENE_PATH`.
- `so101sim/mechsearch.py` holds both envs, `SEARCH_HOME` and `TRAVEL_Z`, and imports `bin_geometry` and `cube_names` from `binpick`.
- `scripts/mechsearch_smoke.py` resets N episodes. It writes both camera views of each starting pile side by side to an mp4 and prints reset time, acceptance rate, and red pixel counts.
- CLAUDE.md records that the task is decided, lists the new files and commands, records the riser height as a hardware requirement, and adds any traps found while building.

## Tests

`tests/test_mechsearch.py`, CPU only:

- `scene.xml` still compiles after the stage split. The existing suite passing covers this.
- `search_scene.xml` loads, and its timestep divides 1/30 s. `SO101Env` raises if it doesn't.
- `SEARCH_HOME` keeps every arm geom out of the drop column, and the wrist camera sees at least 50% of the empty floor.
- The front camera sees at least 70% of the empty floor.
- The same seed gives the same pile.
- A buried reset has zero red pixels in both cameras. A visible reset has at least 100 in one.
- After reset every cube is inside the bin and settled, there are at most three layers, and there's a free floor spot.
- Setting a gray cube's pose outside the bin makes the next step terminate, with the cube's name in `info["failure"]`.
- The success count needs 5 frames. With the held-and-raised predicate stubbed, 4 true frames followed by a false one give no success, and 5 true frames do.
- `max_steps` truncates the episode.
- No object pose appears in the observation.
- The primitive env's action bounds match the pulled-in bin footprint and the layer-1 to layer-3 heights. A push whose end point leaves the bin is refused, and the arm doesn't move.
- Solvability, on one fixed buried seed. Primitives chosen from `info` poses clear every cube stacked on the red one, placing each at a free floor spot, then pick the red cube. The env must report success with every cube still in the bin. If this can't pass, the task can't be done with this bin, and work stops until dm1487 decides what changes.

## Risks

- **The forearm near the close wall.** Reaching the floor right next to the wall closest to the base is where an arm link is most likely to hit the rim. Check 4 in the sizing rule is there for it. If it fails, the bin moves out or gets narrower.
- **The top layer rides above the rim.** With walls of 6 cm, the third layer of a pile sits about 3 cm over the rim. Reset rejects piles that spill while settling, and a push that knocks a top-layer cube over the wall ends the episode. Both show up in the acceptance rate and in how often the dig-out test succeeds.
- **The riser becomes a hardware requirement.** The real arm needs a riser D high, or a bin set D into the table. CLAUDE.md lists the table height relative to the base as unknown, so this spec fixes it rather than matching an existing setup.
- **Deep walls hide the pile from the side.** The front camera sees less of a deep bin, and the floor-visibility rule decides its height. The lower layers will mostly be seen by the wrist camera, which fits a search task.
- **Reset time grows with the cube count.** 20 to 30 cubes in waves is slower than 8 dropped one at a time. The smoke script reports it.
- **The success check uses privileged state.** Contact and pose checks decide success, and segmentation decides the buried label. The real robot has neither. That's fine for labels and rewards in sim, and it matters again only for evaluating on hardware.
