# 11 Wider, shorter cylinders (Dhruv's simplification)

**Question.** How wide can a shelf cylinder be and still be grasped, and which wider, shorter size never topples
under the sweep skill? Then: does the rest still work at that size, and how do random single sweeps do compared
with study 10?

**How the size is changed** (`common/cylinder_size.py`, no repo file changes). `use(size)` wraps the repo's
`load_model` so every `cyl_*` geom is resized in the MjSpec before compiling (mass, inertia and collision bounds
follow), and sets in the repo's modules, for that process only: radius and half-height, grasp height (half the
height) and carry height (grasp + 1 cm, the repo's lift), the env's spacing (diameter + 5 mm), the gripper values IK
plans with (`GRIP_PLAN`: jaw gap = diameter + the repo's 4.3 mm slack; `GRIP_HOLDING`: jaw gap = diameter) and the
side opening (diameter + 13 mm, as 43 mm is for 3 cm). At the repo's size every value is unchanged (opening 28,
plan 22, hold 19.4). At r 2 cm the gripper stopped at 25.9 on a held cylinder against 25.8 planned.

**Run.** `experiments/sahal/.venv/bin/python experiments/sahal/studies/11_wide_cylinders/run.py [--quick]
[--parts gripper sizes table reach check scenes trials analyse compare glitches videos] [--workers 10]`.
`videos` writes 6 demo videos to `experiments/sahal/videos/study11/` (see `videos/README.md`). On 10 workers: gripper 32 s,
sizes 279 s (1260 sweeps in 267 s), reach and check about 1 min together, scenes 10 s, trials 52 min (3097 s,
median 6.08 s per trial), glitches (replays) about 3 min.

## Step 1: the widest cylinder the gripper takes (outputs/gripper_table.csv, gripper_runs.csv)

The repo's pick + pull on a lone cylinder 5 cm tall (grasped at 2.5 cm), radius 1.5 to 4 cm, plus the repo's size: at
the centre (0.4075, 0), 2.5 cm (surface to wall) from the left, right and back walls, and 5 mm inside the front of the
action box; friction 0.4 and 0.7, density 900; 3 repeats with ±2 mm seeded jitter. 30 runs per size.

- **Every size from 1.5 to 4 cm radius grasped and pulled out in 30 of 30 runs**, at every spot. The side opening
  grows to lerobot 64 at 4 cm; the jaw tips open to 132 mm at 100. So the limit is above 4 cm (8 cm wide), and the
  candidates below are far inside it. Not tested: grays standing next to the cylinder, which the wider jaw swing
  may knock (the repo saw 55 mm of swing knock a neighbour 2 cm away).

## Step 2: choosing the size (outputs/size_table.csv, size_sweeps.csv, size_grasps.csv, size_summary.json)

Candidates radius 2, 2.25, 2.5 cm by height 4, 5, 6 cm (grasp at half the height), and the repo's size as a baseline.
A lone cylinder at (0.4075, 0) swept with `skills/sweep.py` unchanged (heights 23 to 26 mm, lowest that passes), in its
7 directions (0, 45, 90, 135, 225, 270, 315 deg), 3, 6 and 10 cm, friction 0.4 and 0.7, 3 seeded jittered starts:
126 sweeps per size. Plus grasp + pull at the five spots above, once per friction. A size passes with no topple, a
largest tilt under 5 deg over every executed sweep except glitches (jammed sweeps included: a cylinder tilting while
pinned against a wall is the near-topple to rule out), and every grasp working.

| size | topples (μ 0.4 / 0.7) | largest tilt, deg (μ 0.4 / 0.7) | largest tilt incl. jammed | jammed | glitches | grasps | passes |
|---|---|---|---|---|---|---|---|
| repo r1.5 h8 | 4 / 1 | 7.41 / 5.8 | 136.84 | 12 | 10 | 10/10 | no |
| r2 h4 | 0 / 1 | 7.53 / 23.84 | 23.84 | 9 | 0 | 0/10 | no |
| r2 h5 | 0 / 1 | 7.44 / 5.91 | 26.85 | 10 | 0 | 10/10 | no |
| r2 h6 | 0 / 0 | 9.69 / 4.48 | 9.69 | 6 | 0 | 10/10 | no |
| r2.25 h4 | 0 / 0 | 1.18 / 2.16 | 4.04 | 5 | 0 | 0/10 | no |
| **r2.25 h5** | **0 / 0** | **1.29 / 2.9** | **2.9** | 4 | 0 | 10/10 | **yes, chosen** |
| r2.25 h6 | 0 / 0 | 0.95 / 2.59 | 12.11 | 11 | 0 | 10/10 | no |
| r2.5 h4 | 0 / 0 | 1.07 / 1.39 | 1.39 | 0 | 0 | 0/10 | no |
| r2.5 h5 | 0 / 0 | 0.73 / 1.48 | 1.48 | 1 | 0 | 10/10 | yes |
| r2.5 h6 | 0 / 0 | 1.13 / 3.56 | 3.56 | 1 | 0 | 10/10 | yes |

("largest tilt" leaves out glitches and jammed sweeps; 120 executed per size; the 6 refused per size are all the 0 deg
10 cm sweeps, into the back wall: off the board for the repo's size, the path check at the wider ones.)

- **Chosen: radius 2.25 cm, height 5 cm, grasp at 2.5 cm.** The smallest radius that passes; at 2.25 cm, 5 cm is the
  tallest height that passes. No topple in 126 sweeps, largest tilt 2.9 deg (jammed included), and every grasp works.
- **2 cm is not enough.** Every height toppled once or tilted over 5 deg, at 315 deg 10 cm or 45 deg, mostly at μ 0.7
  (the push 23 to 24 mm up against r/μ = 2.9 cm).
- **4 cm tall can't be grasped at half its height.** At 2 cm the env's path check refuses every pick (the arm comes
  too close to the board), the same floor the repo found for pushes at 2 cm.
- **6 cm tall at 2.25 cm passes when not jammed (2.59 deg) but tilted 12.11 deg while jammed** at 315 deg 10 cm, μ 0.7.
- 2.5 cm passes with more margin (largest 1.48 deg at 5 cm tall); it is the fallback if clutter tilts 2.25 cm more.
- The repo's size, for comparison: 5 topples (4 at μ 0.4) and 10 glitches in the same sweeps.

**Caveats.** One spot (the centre) and one cylinder alone: clutter, pushing a cylinder into another, and wall contact
elsewhere are not covered here (step 4 measures them). Sweep heights were left at the skill's 23 to 26 mm, about half
of 5 cm.

## Step 3: regression at r 2.25 cm, 5 cm tall (outputs/reach_summary.json, reach.png, check_summary.json)

**Reach** (reach_runs.csv, reach.png). Pick + pull of a lone cylinder on a 2 cm grid, x 0.33 to 0.45 and y -0.13 to
0.13 m (98 points), at friction 0.4 and 0.7, for the chosen size and the repo's.

| size | points in the action box | worked at both frictions, in the box | pick refused near the left / right wall (runs) |
|---|---|---|---|
| repo r1.5 h8 | 70 | 64 | 2 / 12 |
| r2.25 h5 | 70 | 60 | 10 / 12 |

- The graspable region is nearly the same: the action box minus its side columns. The wider cylinder loses the
  points next to the left wall too (the arm would hit shelf_left), as the repo's size already did on the right.
- The grid's front row (x 0.33) is outside the action box at both sizes, and x 0.35 sits on its front edge (pick
  accepts some of those points within its tolerance).
- The 60 points that worked inside the box are the chosen size's graspable zone for step 4's scenes (reach_grid.csv;
  the red must have all four surrounding grid points graspable, as with study 01's 1 cm grid for study 10).

**Retrieval check** (check_scenes.csv). 50 random scenes at the chosen size (seed 111, the red in that zone, 3 grays
anywhere, blocked or not), friction 0.55: `can_retrieve` in hybrid (default) and kinematic mode against physics
(exact mode: the pick + pull run).

| check | yes, physics yes | yes, physics no | no, physics yes | no, physics no | agree |
|---|---|---|---|---|---|
| hybrid | 38 | 0 | 3 | 9 | 47 / 50 |
| kinematic | 20 | 0 | 21 | 9 | 29 / 50 |

- **No false yes** from either. The hybrid's 3 misses are all its fast no (the plan overlaps a gray by 33 mm or
  more: 33.1, 33.4 and 40.0 mm) where physics retrieved the red. The -33 mm cutoff was set on 3 cm cylinders
  (study 07); with 4.5 cm ones the same shove gives a deeper planned overlap, so the cutoff is too tight here. Step 4
  uses the exact mode, so it is unaffected. 16 of 50 hybrid answers were simulated. Median times: hybrid 0.40 s,
  kinematic 0.22 s, exact 2.30 s.

## Step 4: random single sweeps at r 2.25 cm, 5 cm tall, against study 10 (outputs/summary.json, scene_table.csv, by_cylinder.csv, trials.csv, sweeps/, comparison.csv, comparison.png, glitch_summary*.json)

Study 10's method unchanged (`common/random_sweeps.py`, the same code; seed 11): 10 scenes of the red and 3 grays,
the red in the chosen size's graspable zone (step 3), blocked at the start by the exact check (10 of 28 candidates;
the other 18 were retrievable), saved in `scenes/wide4/`; 500 trials each, friction 0.55.

| outcome | study 10 (r1.5 h8) | study 11 (r2.25 h5) |
|---|---|---|
| success | 1315 / 4860 = 27.1% (25.8 to 28.3) | 964 / 4709 = 20.5% (19.3 to 21.6) |
| still blocked | 55.6% | 70.5% |
| toppled | 514 = 10.6% (9.7 to 11.5) | 26 = 0.55% (0.4 to 0.8) |
| toppled, counting glitch-flagged trials that toppled (of all 5000) | 647 = 12.9% (12.0 to 13.9) | 221 = 4.4% (3.9 to 5.0) |
| jammed | 291 = 6.0% (5.4 to 6.7) | 391 = 8.3% (7.5 to 9.1) |
| gray left the board | 0.74% | 0.17% |
| arm hit the shelf | 0 | 2 |
| glitch (of all 5000) | 140 = 2.8% | 291 = 5.8% |
| refused before moving | 45.0% | 32.1% |
| mean progress | 12.0 mm | 7.3 mm |

(Rates out of non-glitch trials unless marked; 95% intervals in brackets.)

- **Toppling is much rarer but not gone.** As classified, topples fell from 10.6% to 0.55%. But most trials flagged
  as glitches were topples too: replaying them (same seed, same sweep; `--parts glitches`), 195 of study 11's 291 and
  133 of study 10's 140 ended with a cylinder toppled (the env's own check). In study 11, 140 of the 291 had arm forces
  of 100 N or less and were flagged only for a cylinder over 1 m/s; 130 of those 140 had a cylinder past the topple
  angle (median largest tilt 37.2 deg). The fastest cylinder was the swept one in only 88 of 291: mostly a neighbour,
  pushed or pinched by the swept cylinder, fell. Counting those, topples go from 12.9% to 4.4% of all trials, a third.
  The lone-cylinder test of step 2 could not see cylinder-on-cylinder pushes.
- **The glitch rule hides topples in clutter.** It was set from lone-cylinder pushes (study 05: 1 m/s, 100 N), and is
  checked before the topple outcome, so a cylinder knocked over fast enough counts as a glitch and drops out of the
  rates. Study 10's rates have the same bias (133 of its 140 glitches toppled; 3 of those 140 are in random4_0000,
  where the friction bug may change the replay).
- **Success fell from 27.1% to 20.5%.** The wider cylinders crowd the same board: the starting gap in the plan
  (arm or held red to the nearest gray) is -30.8 to -49.2 mm against study 10's -16.0 to -38.2, so more scenes start
  deeply blocked, and more sweeps leave them still blocked (70.5%). Jams rose from 6.0% to 8.3%.
- **The same patterns as study 10.** Sweeping the red succeeded 28.7% (25.9 to 31.7), a gray 18.5% (17.2 to 19.7);
  in 8 of 10 scenes one gray gives 40 to 75% (by_cylinder.csv), and in wide4_0018 no cylinder gives more than 13.6%.
  Longer strokes help: 2 to 5 cm 14.5%, 5 to 8 cm 19.3%, 8 to 12 cm 27.6%. Progress: successes mean 25.2 mm (94.1%
  positive), still blocked 2.1 mm (53.0% positive).

| scene | red (x, y) m | gap before, mm | success | 95% interval | toppled | jammed | glitch |
|---|---|---|---|---|---|---|---|
| wide4_0000 | 0.4458, 0.0323 | -30.9 | 119 / 480 | 0.211 to 0.288 | 5 | 42 | 20 |
| wide4_0002 | 0.4392, -0.0275 | -34.8 | 109 / 457 | 0.202 to 0.28 | 3 | 53 | 43 |
| wide4_0007 | 0.3942, -0.0177 | -37.5 | 186 / 459 | 0.361 to 0.451 | 4 | 32 | 41 |
| wide4_0015 | 0.4488, 0.023 | -30.8 | 93 / 457 | 0.169 to 0.243 | 3 | 25 | 43 |
| wide4_0016 | 0.4035, 0.0795 | -49.2 | 100 / 492 | 0.17 to 0.241 | 0 | 20 | 8 |
| wide4_0018 | 0.4483, -0.0326 | -44.5 | 28 / 446 | 0.044 to 0.089 | 5 | 53 | 54 |
| wide4_0021 | 0.4413, 0.0572 | -49.2 | 123 / 468 | 0.225 to 0.305 | 4 | 86 | 32 |
| wide4_0024 | 0.4338, -0.0945 | -31.2 | 52 / 485 | 0.083 to 0.138 | 1 | 44 | 15 |
| wide4_0025 | 0.4148, 0.0994 | -32.0 | 59 / 488 | 0.095 to 0.153 | 0 | 16 | 12 |
| wide4_0027 | 0.4254, 0.0452 | -40.0 | 95 / 477 | 0.166 to 0.237 | 1 | 20 | 23 |

Known issues for these numbers (the glitch rule, the friction-order fix): HANDOFF.md section 14.

**Caveats.** One friction (0.55), 10 scenes per study, different scenes in the two studies (same rules, different
sizes, so not paired). Study 10 has the friction bug in up to 250 of random4_0000's trials (its README). The size was
chosen on a lone cylinder; whether 2.5 cm, or a lower sweep height, removes the clutter topples is not tested.

