# Tilting the gripper a few degrees down for a low sweep: what it would take

Not built. Numbers below are from this study's `run.py` (outputs/reach.csv and outputs/slide_topple_runs.csv) and from study 04 (2026-11-21).

## Why it comes up

- With the gripper level, the lowest push height the env allows is 2.2 to 2.4 cm (board to jaw midpoint). The
  path check refuses anything lower because the arm comes within 6 mm of the board. Level IK itself solves down to
  at least 0.5 cm; the limit is clearance, not reach.
- What comes closest is the **gripper body** (the wrist-roll follower housing, 43 mm behind the jaw midpoint, at
  the same height) for pushes near the front, and the **forearm** (a collision box about 170 mm behind the jaw
  midpoint) for pushes deep in the bay. They would touch the board at about 1.6 and 1.8 cm.
- The theory h < r/μ asks for 2.14 cm at μ = 0.7, below that floor.

But part 2 says height is not the main problem in this sim. On a lone cylinder the r/μ curve doesn't separate
slide from topple: below the curve 6 of 69 runs toppled and 43 leaned 5 to 20°; above it 5 of 50 slid cleanly.
Tilt varies non-monotonically with height. The push is an impact, not a quasi-static push: the trajectory runs at
the 2°/frame joint cap the whole way (`primitives._retime`), the jaw arrives at full speed, the cylinder leaves at
about 0.2 m/s, and jaw and cylinder are in contact for only about 30 physics substeps (50 ms) of a 5 cm stroke.
Also the jaw-cylinder friction is 1.0 whatever μ is set to (MuJoCo takes the larger of the two geoms'
frictions, and the gripper geoms are 1.0), which the r/μ model ignores. So **before tilting, it's worth trying a
slower push**: slow down for the last few mm before contact and through the stroke, or start in contact.

## What a tilt buys

Pitching the tool down by θ about the jaw midpoint raises a point L behind it by L·sin θ:

| θ | gripper body (L = 43 mm) | wrist and forearm (L ≳ 100 mm, not yet measured) |
|---|---|---|
| 3° | +2.3 mm | ≳ +5 mm |
| 5° | +3.7 mm | ≳ +9 mm |
| 8° | +6.0 mm | ≳ +14 mm |

So 5° would let the jaw midpoint go about 4 mm lower near the front (about 1.9 cm instead of 2.3) and more deep in
the bay. Side effects to check: the jaw tip, not the midpoint, becomes the lowest point and the contact point;
the contact face is inclined, so the push gains a vertical component (down is good for stability, up is bad); and
for sideways pushes the contact height then varies along the jaw.

## Code that would change (proposal)

The arm is planar once pan is set: lift + elbow + wrist flex sum to the tool's pitch, and roll keeps the camera
mount up. A pitched tool is still 3 position + 2 orientation constraints on 5 joints, so the IK stays square.

1. `so101sim/ik.py`, `solve_side` (line 221) and `_side_from` (line 258). Add `pitch=0.0`. Today the orientation
   error is `np.cross(up, UP)[:2]` (line 276), which pulls the site's y axis to vertical and relies on heading
   being free. For a pitch, the target up vector tilts toward the ray direction:
   `up_des = cos θ · UP + sin θ · ray_dir`. Because `up_des` now has a horizontal part, rotation about vertical
   changes the error, so use the error's two components orthogonal to `up_des` (not just rows x and y), with the
   matching rows of the rotational Jacobian. `info["tilt"]` should report the angle from `up_des`.
2. `so101sim/side_primitives.py`: `_ok` (line 81) checks `tilt < TILT_TOL` against level; it must check against
   the requested pitch. `_trace` and `push` (line 171) take `pitch` and pass it down. Probably a new `sweep`
   primitive rather than changing `push`, so existing demos replay. Change pitch only outside the bay, during the
   drop at the front edge (`push` already drops from grasp height to push height there), never mid-stroke.
3. `bridge` and `tool_point` (line 200) treat a gripper within `LEVEL_DEG = 5` of level as level (line 33). Above
   5° the bridge falls back to a joint ramp, which inside the bay knocks things over. Either keep θ ≤ 5° or make
   `tool_point` and `bridge` pitch-aware.
4. `so101sim/shelf.py`, `ShelfPrimitiveEnv._plan` (line 578): call the new primitive and, if it becomes a policy
   action, add it to `PRIMITIVES` and the action space. `_path_hit` needs no change; it checks whatever trajectory
   it gets.
5. `so101sim/shelf_checks.py`, `level_pose` (line 96) and `sizing_checks` (line 70): the "push" sizing check is
   level-only; it needs the pitch for `scripts/size_shelf.py` to keep vouching for the layout.
6. Tests: `tests/test_side_ik.py` (pitched solves reach the target with the right pitch), and
   `tests/test_side_primitives.py` (a pitched sweep passes the path check at the new height).
7. Spec: an amendment to `docs/superpowers/specs/2026-09-26-shelf-search-design.md` under Primitives, and the
   push-height trap in `.claude/rules/shelf-task.md`.

## Questions for the discussion

- Is the goal a lower contact or a gentler push? The data points at speed more than height.
- Real arm: wrist flex has ±95° of range, so a few degrees is easy, but the real arm sags under load, and a
  planned 1.9 cm push may land lower than planned.
- How much tilt before the wrist camera stops seeing along the ray? `look` keeps the gripper level today.
