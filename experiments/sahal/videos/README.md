# Demo videos: one random sweep, then grasp + pull

Replays of saved trials from studies 10 and 11 (MuJoCo, the same seed, so the same sweep). Every replay ended the
same way as its saved trial (`replay_matches` in each folder's videos.csv). Normal speed in free space, 0.5x from 6
frames before to 6 after any arm-cylinder contact. The label gives the scene, trial, swept cylinder, sweep direction
(0 deg = straight away from the robot, positive = toward the robot's left) and stroke length (the jaw's travel from
its start a few cm behind the cylinder, so the cylinder itself moves less), and the outcome.

## study10/: repo cylinders (r 1.5 cm, 8 cm tall), scene random4_0019

Front camera | wrist camera. Made by `studies/10_random_sweeps/run.py --parts videos` and copied here.

| video | trial | what it shows | length | size |
|---|---|---|---|---|
| success_key_gray.mp4 | 38 | sweep gray_0 (this scene's key gray) +68 deg, 5.2 cm; grasp + pull take the red out | 15.5 s | 2.38 MB |
| success_red.mp4 | 12 | sweep the red itself +34 deg, 3.3 cm; then the red comes out | 15.9 s | 2.37 MB |
| success_long.mp4 | 0 | long sweep of gray_0 +60 deg, 10.0 cm; then the red comes out | 18.2 s | 2.91 MB |
| fail_still_blocked.mp4 | 4 | sweep gray_0 -111 deg, 6.0 cm; the pull drags gray_0 off the board | 15.3 s | 2.40 MB |
| fail_toppled.mp4 | 5 | sweep gray_2 +119 deg, 8.1 cm; gray_2 topples | 4.4 s | 0.77 MB |
| fail_jammed.mp4 | 9 | sweep gray_2 +125 deg, 5.4 cm; gray_2 is pinned against the wall | 7.0 s | 1.09 MB |

## study11/: wide cylinders (r 2.25 cm, 5 cm tall), scenes wide4_0007 and wide4_0021

Top-down view of the shelf floor (the shelf's top board hidden and the arm see-through, for the picture only),
framed on the cylinders, with the sweep drawn as a yellow arrow (jaw start to planned end) and a yellow ring under the
swept cylinder while it sweeps | wrist camera. After a failure the physics runs 2 s more (video only; the outcome was
decided before). Made by `studies/11_wide_cylinders/run.py --parts videos`.

| video | scene, trial | what it shows | length | size |
|---|---|---|---|---|
| success_key_gray.mp4 | wide4_0007, 8 | sweep gray_2 (this scene's key gray) +59 deg, 5.9 cm; grasp + pull take the red out | 17.9 s | 0.70 MB |
| success_red.mp4 | wide4_0007, 0 | sweep the red itself +52 deg, 6.2 cm; then the red comes out | 22.3 s | 0.93 MB |
| success_long.mp4 | wide4_0007, 19 | long sweep of the red -56 deg, 11.7 cm; then the red comes out | 24.4 s | 0.99 MB |
| fail_still_blocked.mp4 | wide4_0007, 3 | short sweep of gray_2 +106 deg, 2.8 cm; the retrieve fails, the red leaving the board during the pull without being held | 20.9 s | 0.76 MB |
| fail_toppled.mp4 | wide4_0007, 102 | sweep gray_1 +47 deg, 7.5 cm; a cylinder topples | 13.6 s | 0.50 MB |
| fail_jammed.mp4 | wide4_0021, 3 | sweep gray_2 -111 deg, 8.8 cm; gray_2 is pinned against the wall | 23.5 s | 0.71 MB |

Each trial is the first one in that study's trials.csv with that outcome (and, for the first four, the cylinder and
stroke named), so none was hand-picked for looks.
