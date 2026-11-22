# 07 Retrieval check against physics

**Question.** Does `common/retrieval_check.can_retrieve(env)`, a kinematic "can the red be retrieved now?" check,
agree with what the repo's pick + pull actually do in physics when grays stand near the red?

**The check** (no physics stepped; the repo's own planners). In order, the first failure is returned: the red is in
the action box; level IK solves pick and pull; then for pick, then pull, the earliest frame where the arm is within
6 mm of the shelf or table (the env's own path check), any arm geom is within `ARM_GRAY_MARGIN_M` = 5 mm of a gray,
or, once the jaws have closed, the held red is within `HELD_RED_GRAY_MARGIN_M` = 5 mm of a gray or
`HELD_RED_SHELF_MARGIN_M` = 3 mm of a wall or the top board. The held red is placed upright at the jaw midpoint
with the gripper at its holding value (19.4), within 1.4 mm of where physics holds it. Example reason:
`pick: arm (gripper#25) within 5 mm of cyl_gray_0 at frame 64`.

**Method.** 300 scenes, each seeded by its index: the red anywhere in the env's action box, 2 grays upright on the
board, no two closer than the env's 3.5 cm centre spacing. Where the first gray goes: anywhere on the board (30%), in
the approach corridor (45%: between the red and the front edge, within 5 cm of the red's approach ray), or beside the
red (25%: centres at most 6 cm apart). The second gray goes anywhere. A scene is drawn again if the cylinders don't
come to rest, touch the arm or tilt over 2°. Then the check, then pick and pull in physics (friction 0.55, density
900, every env check on). Real success is the env's `is_success`. As a diagnostic, `min_gray_clearance` gives the
plan's smallest distance from the arm or held red to a gray (negative: the plan overlaps it).

**Modes.** `can_retrieve(env, mode=...)`: `"hybrid"` (the default, for search), `"exact"` (always runs pick + pull
in physics from the current state and puts the state back; for verifying scenes), `"kinematic"` (no physics, the
check above).

**Hybrid mode** (`can_retrieve(env, mode="hybrid")`). The exact refusals first (action box, IK, the env's path check
against the shelf, the held red within 3 mm of a wall): these are the env's own refusals. Then the plan's smallest
distance from the arm or held red to a gray: −5 mm or more (`HYBRID_CLEAR_M`) is yes; below −33 mm
(`HYBRID_DEEP_M`; −25 mm in the tuning and held-out runs below, moved after the held-out run) is no; in between, pick + pull run in physics from the current state and the state is put back
exactly (`simulation_copy`: the whole MjData and the env's bookkeeping), and that result is the answer.

**Run.** `experiments/sahal/.venv/bin/python experiments/sahal/studies/07_retrieval_check/run.py [--quick]
[--workers 10] [--mode kinematic|hybrid|exact] [--parts scenes margins otherwise] [--seeds tuning|heldout|test]`.
Kinematic scenes about 1 min on 10 workers, hybrid about 1.5 min, exact about 1.6 min. `margins` and `otherwise`
read the kinematic run's scenes.csv. Seed sets: tuning (seed 7, the default), heldout (70), test (700). Outputs:
`outputs/` (kinematic, tuning), `outputs/hybrid/` (hybrid, tuning), `outputs/<mode>_<seeds>/` otherwise.

## Kinematic check

**Results** (outputs/summary.json, scenes.csv, agreement.png, clearance_vs_outcome.png, examples/).

|  | really succeeded | really failed |
|---|---|---|
| **check says yes** | 62 | 0 |
| **check says no** | 86 | 152 |

- **Agreement 214 of 300 (0.713). The check never said yes to a failure**; every disagreement is a refusal of a
  scene that really worked: 86 of the 148 real successes.
- By scene kind (yes/succeeded, yes/failed, no/succeeded, no/failed): anywhere 47 / 0 / 21 / 26; approach corridor
  0 / 0 / 37 / 93; beside the red 15 / 0 / 28 / 33.
- Real failures: a gray left the shelf 71, a gray toppled 55, pick refused by the path check 23, pull refused because
  nothing was held 3. No glitches (peak arm-red force 253.8 N, under the 1000 N grasp limit; red at most 0.617 m/s).
- The check's reasons: arm near a gray 235, arm near the shelf 3, ok 62. All 86 wrong refusals were "arm near a
  gray"; in 76 of them the arm really touched the gray, which slid aside and stayed up.
- **Why it is too strict.** The plan's smallest distance to a gray (clearance_vs_outcome.png): real successes
  −27.0 to 50 mm (median −0.9; 50 is the cap); scenes where a gray toppled or left the shelf −42.0 to −7.9 mm
  (median −25.3). Pick + pull survive planned overlaps of up to 27 mm by shoving the gray.
- **Time per check**: median 162.7 ms, p90 205.7, max 301.7 with 10 workers sharing the CPU; median 123.8 ms (max
  179.6) for 20 scenes one at a time in one process. Pick + pull in physics: median 1.27 s.
- **Examples** (examples/, check said no and it really succeeded; for each kind the scene whose plan overlaps a gray
  most): scene 171 (anywhere, −21.7 mm), 212 (approach corridor, −27.0 mm), 150 (beside the red, −24.9 mm). In all
  three the gripper body (`gripper#25`) passes through where a gray stands and pushes it aside. Each picture shows the
  scene from above and the front and wrist cameras of the real run at the frame the check flagged. There is no
  example of the other disagreement (check yes, really failed): it never happened.

## Choosing thresholds (outputs/margin_table.csv)

From the kinematic run: answer yes if the plan's smallest gray clearance is at least the threshold, no if below.
The 23 scenes the env refused before moving are left out (the check refuses them exactly), so 277 are judged.

| threshold | yes | false yes | no | missed successes |
|---|---|---|---|---|
| +5 mm (the kinematic check) | 62 | 0 | 215 | 86 |
| 0 mm | 72 | 0 | 205 | 76 |
| −5 mm | 89 | 0 | 188 | 59 |
| −7 mm | 94 | 0 | 183 | 54 |
| −8 mm | 98 | 1 | 179 | 51 |
| −10 mm | 105 | 2 | 172 | 45 |
| −15 mm | 125 | 6 | 152 | 29 |
| −20 mm | 148 | 14 | 129 | 14 |
| −25 mm | 211 | 64 | 66 | 1 |
| −27 mm | 220 | 72 | 57 | 0 |
| −30 mm | 244 | 96 | 33 | 0 |

The fast yes stays right down to −7 mm (the shallowest gray failure was −7.9 mm); −5 mm keeps about 3 mm in hand.
A fast no below −25 mm is wrong once (one success at −27.0 mm); going to −27 mm would cost 9 more simulations to
save that one. Every 1 mm step is in the CSV.

## Hybrid check

**Results** (outputs/hybrid/summary.json, scenes.csv, agreement.png).

|  | really succeeded | really failed |
|---|---|---|
| **check says yes** | 147 | 0 |
| **check says no** | 1 | 152 |

- **Agreement 299 of 300 (0.997)**, no false yes. The one miss is the fast no on scene 212 (approach corridor, plan
  overlap 27.0 mm, really retrieved), the deepest-overlap success in the data.
- Routes: fast yes 89, fast no 67, exact refusal (arm near the shelf) 23, simulated 121. All 121 simulated answers
  matched the real run. The real outcomes were identical to the kinematic run's (the simulation copy leaves no
  trace).
- **Time per check**: mean 759.4 ms, median 284.0, p90 1891.1, max 2452.0 with 10 workers sharing the CPU; by
  route: simulated 1549.8 ms, fast yes 229.9, fast no 262.3, exact refusal 98.0. One at a time in one process (20
  scenes): mean 406.2 ms, median 227.8.

## Hybrid check on held-out scenes (outputs/hybrid_heldout/, fast-no cutoff −25 mm)

300 new scenes from `HELDOUT_STUDY_SEED` = 70 (`--mode hybrid --heldout`), drawn the same way, the thresholds
unchanged.

|  | really succeeded | really failed |
|---|---|---|
| **check says yes** | 132 | 0 |
| **check says no** | 8 | 160 |

- **Agreement 292 of 300 (0.973), still no false yes.** Real successes 140.
- Routes: fast yes 76 (0 wrong), fast no 84 (**8 wrong**), exact refusal (arm near the shelf) 21, simulated 119 (all
  matched the real run).
- **All 8 misses are fast no's on scenes that really succeeded**, 7 of them with a gray in the approach corridor:
  planned overlaps 25.3 to 35.2 mm (scenes 83, 113, 136, 137, 160, 221, 258, 264). On the tuning scenes the deepest
  success was 27.0 mm; here it is 35.2 mm. The shallowest gray failure is 8.5 mm (7.9 before), so the fast yes's
  margin held.
- How the fast no would do at other thresholds on these scenes (for information; the thresholds were not changed on
  held-out data): below −25 mm 84 fast, 8 wrong; −28 mm 64 fast, 3 wrong; −30 mm 52, 3 wrong; −33 mm 31, 1 wrong;
  −36 mm 17, 0 wrong. A deeper threshold sends more scenes to simulation.
- **Time per check**: mean 685.9 ms, median 272.4, p90 1809.8, max 2304.5 (10 workers); by route: simulated
  1402.7 ms, fast yes 215.6, fast no 245.0, exact refusal 90.0. One at a time in one process (20 scenes): mean
  541.5 ms, median 235.7.
- Real failures: gray toppled 69, gray left the shelf 67, refused by the path check 21, red toppled 2, nothing held 1.
  One scene (108) passes the glitch rule's 1 m/s: the red toppled during pick at 1.49 m/s with no arm contact (no arm
  force recorded), so likely knocked over by a shoved gray rather than a solver blow-up; the simulation predicted it.
  It is kept in the counts.

## Hybrid (cutoff −33 mm) and exact on a third set (outputs/hybrid_test/, outputs/exact_test/)

300 scenes from `TEST_STUDY_SEED` = 700, used for nothing before; the cutoff was moved to −33 mm before this run.

| hybrid | really succeeded | really failed |
|---|---|---|
| **check says yes** | 142 | 0 |
| **check says no** | 0 | 158 |

| exact | really succeeded | really failed |
|---|---|---|
| **check says yes** | 142 | 0 |
| **check says no** | 0 | 158 |

- **Both 300 of 300.** Hybrid routes: fast yes 78 (0 wrong), fast no 35 (0 wrong), exact refusal (arm near the shelf)
  23, simulated 164 (all matched). The deepest success here was a 29.3 mm overlap and the shallowest gray failure
  8.8 mm, both inside the thresholds. The real outcomes were identical in the two runs.
- **Time per check** (10 workers sharing the CPU): hybrid mean 884.0 ms (median 788.8, p90 1844.1, max 2457.2);
  exact mean 1104.3 ms (median 1229.0, p90 1653.1, max 2140.7). Hybrid by route: simulated 1442.0 ms, fast yes
  220.4, fast no 271.3, exact refusal 88.5. One at a time in one process (20 scenes): hybrid mean 525.4 ms (median
  198.5), exact 839.3 ms (median 1003.4).
- So hybrid saves about a fifth of the time here: 164 of 300 scenes still need a simulation, and its fast answers
  cost about 0.2 s because they scan every arm geom against every gray over every frame of the plan.
- Real failures: gray toppled 66, gray left the shelf 64, refused by the path check 23, nothing held 3, red toppled 1,
  red left the shelf 1. Scenes 211 and 215 pass the glitch rule's 1 m/s (a gray toppled with the red at 3.7 m/s
  and 196 N; the red toppled at 1.5 m/s with no arm force); both are failures the simulation predicted, kept in the
  counts.

## Scenes that failed for another reason (outputs/failed_otherwise.csv, failed_otherwise.json, otherwise/)

26 of the 300 failed without a gray toppling or leaving the shelf.

- **23 refused by the env's path check before moving**, right wall 14, left wall 9. The red was 5.3 to 22.3 mm from
  the nearer side wall (y −0.1397 to 0.1378 m). This is study 01's wall band (pull-out works for y from −0.115 to
  +0.125 m): the grays play no part, and both checks refuse these exactly.
- **The 3 with a positive gap** (scenes 25, 49, 76; plan gray clearance 37.1, 50 and 18.9 mm) are three of those
  23: the red at the front right (x 0.359 to 0.381 m, y −0.1306 to −0.134 m), refused because the arm would come
  within 6 mm of the right wall. The gripper opens toward the robot's right and needs room there. The kinematic check
  already gave this reason for them ("arm within 6 mm of shelf_right").
- **3 where pull was refused because nothing was held** (scenes 65, 102, 203; plan overlaps 23.0, 23.3 and 39.4 mm):
  in each the arm touched a gray, the red moved 18.7 to 21.3 mm before the jaws closed, and the jaws closed empty
  (gripper reading 10.0, no jaw contacts, red upright). Pushing the gray moved the red out of the grasp. All three
  are deep overlaps: the hybrid says no to 203 fast and simulates the other two.

**Caveats.** One friction (0.55), one density, two grays. Overlap depths for mesh-cylinder pairs come from
`mj_geomDistance`'s penetration estimate, so read them as "how far the gray would have to move", not exact geometry.
Whether a shoved gray survives depends on friction and on the chaos noted in earlier studies. The hybrid's
thresholds were chosen on the first 300 scenes, so its 299 of 300 there is in-sample; the held-out 292 of 300 tested the
first cutoff, and the test set's 300 of 300 tests the moved one. That is still 300 scenes at one friction: a fast
answer wrong about 1 time in 100 would often go unseen.
Its simulated answers are right by construction here, because the "real" run is the same simulator from the same
state; on a real arm, or with friction and pose uncertain, the simulation is a prediction.
