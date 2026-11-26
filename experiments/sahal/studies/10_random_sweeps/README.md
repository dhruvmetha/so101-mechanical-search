# 10 Random single sweeps (Dhruv's idea)

**Question.** In a random scene where the red can't be retrieved, how often does one random but sensible sweep make
it retrievable, and what else happens?

**Scenes** (`--parts scenes`; `experiments/sahal/scenes/random4/`, a JSON and a picture each). Seeded candidates: the
red uniform over the board, kept only inside study 01's graspable zone (all four 1 cm cells around it graspable,
counting only cells inside the env's own action box: 296 of study 01's 396, because pick refuses a red outside the
box), then 3 grays uniform over the board, 5 mm or more apart. A candidate is placed (it must settle upright, within
2 mm) and kept only if `can_retrieve(mode="exact")` says no. The first 10 that pass are used (10 of 24 candidates;
the other 14 were retrievable at the start). No hand-tuning. At the start, 9 of the 10 fail by a gray toppling
during pick and 1 by a gray leaving the shelf during pull.

**Trials** (`--parts trials`). 500 per scene, each from the same settled start (`simulation_copy`). A sweep: one of
the 4 cylinders at random, a direction uniform within ±157.5° of "away" (never toward the robot), a stroke of 2 to
12 cm, starting at the skill's start for that cylinder (5 mm short of it) moved up to ±1 cm sideways and 0 to 1 cm
further back (never closer). A sweep refused before moving (IK, path check, off the board) is logged and a new one
drawn; nothing that ran is redrawn. Outcomes, in order: jammed against a wall; glitch (arm force on any cylinder over
100 N or a cylinder over 1 m/s; excluded from rates); toppled; a gray or the red off the board; arm hit the shelf;
otherwise `can_retrieve(mode="exact")` says success or still blocked. Progress: the plan's smallest gap from the arm
or held red to any gray (`min_gray_clearance`, capped at 50 mm), after minus before. Friction 0.55, density 900.

**Run.** `experiments/sahal/.venv/bin/python experiments/sahal/studies/10_random_sweeps/run.py [--quick]
[--parts scenes trials analyse] [--workers 10]`. Trials: 48 min on 10 workers (median 5.6 s per trial). `analyse`
redoes the tables and plots from trials.csv.

**Results** (outputs/summary.json, scene_table.csv, by_cylinder.csv, trials.csv, sweeps/).

- **One random sweep made the red retrievable in 27.1% of trials** (1315 of 4860 non-glitch; 95% interval 0.258 to
  0.283). Outcomes of all 5000: success 1315, still blocked 2704, toppled 514, jammed 291, gray left the board 36,
  glitch 140. 45.0% of sampled sweeps were refused before moving (4092 refusals: off the board 2244, path check 1245,
  IK 603); every trial found an executable sweep.

| scene | red (x, y) m | gap before, mm | success | 95% interval | still blocked | toppled | jammed | gray off | refused | mean progress, mm |
|---|---|---|---|---|---|---|---|---|---|---|
| random4_0000 | 0.4536, −0.0818 | −18.5 | 157 / 497 | 0.277–0.358 | 234 | 38 | 68 | 0 | 57.1% | 13.4 |
| random4_0003 | 0.4001, 0.0081 | −17.6 | 110 / 480 | 0.194–0.269 | 250 | 92 | 17 | 11 | 39.5% | 5.0 |
| random4_0004 | 0.4293, 0.0158 | −29.5 | 128 / 454 | 0.243–0.325 | 270 | 22 | 34 | 0 | 28.3% | 7.8 |
| random4_0006 | 0.3755, −0.0996 | −16.0 | 94 / 494 | 0.158–0.227 | 348 | 38 | 9 | 5 | 46.8% | 11.5 |
| random4_0009 | 0.358, −0.1042 | −21.9 | 112 / 488 | 0.194–0.269 | 288 | 51 | 32 | 5 | 28.1% | 9.8 |
| random4_0015 | 0.3749, −0.0534 | −34.5 | 142 / 495 | 0.249–0.328 | 290 | 40 | 19 | 4 | 30.9% | 13.2 |
| random4_0016 | 0.4349, 0.1006 | −23.8 | 116 / 499 | 0.198–0.271 | 286 | 52 | 45 | 0 | 55.4% | 9.0 |
| random4_0018 | 0.4395, 0.076 | −38.2 | 141 / 491 | 0.249–0.329 | 258 | 57 | 34 | 1 | 56.4% | 17.9 |
| random4_0019 | 0.3685, −0.0018 | −30.0 | 178 / 475 | 0.332–0.419 | 221 | 62 | 14 | 0 | 20.3% | 17.5 |
| random4_0023 | 0.4124, 0.0434 | −23.0 | 137 / 487 | 0.243–0.323 | 259 | 62 | 19 | 10 | 56.5% | 14.3 |

  ("success" is out of the non-glitch trials; "refused" is refusals / (refusals + executed sweeps).)

- **Which cylinder matters.** Over all scenes, sweeping the red succeeded 34.8% (0.323 to 0.374) and sweeping a
  gray 24.0% (0.226 to 0.254). Within a scene it is sharper (by_cylinder.csv): in every scene exactly one gray gives
  42 to 79% success and the other two give 0 to 22% (both 0 in 5 scenes, each over 30 or more executed sweeps).
  Sweeping the red gives 15 to 58%. The executed count per cylinder is uneven because a
  refused sweep is redrawn, so cylinders that are hard to sweep are under-represented.
- **Longer strokes help and cost more.** Success by stroke 2 to 5 cm 0.19 (0.172 to 0.209), 5 to 8 cm 0.276, 8 to
  12 cm 0.359 (0.335 to 0.383); toppled and jammed together 231, 294 and 280 trials of 1739, 1627 and 1494.
- **Progress** (gap after minus before): successes mean 27.9 mm (median 24.3, 97.2% positive); still blocked mean
  4.3 mm (median 3.7, 58.8% positive). The overall mean is 12.0 mm.
- **Glitches** (140, excluded): 99 are clear blow-ups (over 1000 N or 5 m/s). 34 are moderate (at most 500 N and
  2 m/s) and may be real pinches of a cylinder against another, which the wall-jam rule does not cover; counting them
  as failures would move the overall rate from 0.271 to 0.269.

**Videos** (`--parts videos`, outputs/videos/, mp4s not in git; videos.csv lists them). Six replays of saved trials
of random4_0019, each the first trial in trials.csv that fits: success sweeping the key gray `cyl_gray_0` (trial 38),
success sweeping the red (12), success with a long stroke (0, 10.0 cm), still blocked after sweeping the key gray
(4), toppled (5) and jammed (9). Front | wrist camera, 1x in free space and 0.5x from 6 frames before to 6 after any
arm-cylinder touch; after a sweep that didn't fail, the pick + pull the exact check simulates. Each replay's
outcome, refusal count, direction and peak force match trials.csv. 4 to 18 s each.

**Caveats.** One friction, 10 scenes. The success test simulates pick + pull with the same simulator, so it is the
env's own verdict. A random sweep may also leave a gray worse placed for later moves; only the next retrieval is
judged. Directions are continuous here, while study 05 measured the skill at 7 fixed directions. **A bug made each worker's first job run at different physics**
(found and fixed 2026-11-25, `RedAndGraysShelf.reset_scene`): friction and density were set before the scene's grays
joined the task, so a gray kept whatever it last had. In a new env's first scene that is the model's stock friction 1.0
and density 1000, not 0.55 and 900. In the trials run each worker's first job came from scene random4_0000's blocks
(re-running its trials 0 to 24 reproduced all 25 saved outcomes with the bug, and 9 differed without it), so up to 250
of random4_0000's trials had grays at friction 1.0. The scene search had the same problem for each worker's first
candidate. The other nine scenes' trials are unaffected. These results have not been rerun.
