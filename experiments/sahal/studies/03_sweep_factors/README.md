# 03 What reduces tilt: direction, height, arm friction, speed, board friction

**Question.** Which factor changes a pushed cylinder's tilt most?

**Method.** The study 02 push, every cell repeated 5 times with the push start jittered up to 1 mm (seeded by
repetition, height, friction and direction, so any worker gives the same run).

- `base`: heights 2.25 to 3.0 cm (what the level gripper is allowed), 3 directions, friction 0.4 to 0.7.
- `arm_friction`: the same with every arm geom's sliding friction set to the cylinder's μ for the run (normally 1.0;
  MuJoCo uses the larger of a contact's two frictions). A diagnostic only.
- `speed`: 2.5 cm at 0.5 and 0.25 of normal speed (the push trajectory resampled); normal speed comes from `base`.

Glitch runs (arm-cylinder force over 100 N or the red faster than 1 m/s) are excluded from every rate and mean.

**Run.** `experiments/sahal/.venv/bin/python experiments/sahal/studies/03_sweep_factors/run.py [--quick]
[--workers 10]`. Full: 750 runs, about 2 minutes.

**Results** (outputs/summary.json, cells.csv, runs.csv). Mean max tilt, topple rate; all non-glitch runs.

| factor | values |
|---|---|
| direction (base) | away 14.2°, 26%; left 7.3°, 8%; right 7.8°, 9% |
| height (base) | 2.25 cm 11.2°, 21%; 2.5 cm 9.9°, 12%; 2.75 cm 8.9°, 10%; 3.0 cm 9.3°, 16% |
| board friction μ (base) | 0.4 6.8°, 3%; 0.5 7.9°, 7%; 0.55 9.1°, 12%; 0.6 11.7°, 20%; 0.7 13.7°, 31% |
| arm friction = μ | away 14.2° → 8.0°, 26% → 3%; left 7.3° → 7.0°; right 7.8° → 7.2° |
| speed (2.5 cm) | away 13.9° / 7.4° / 1.0°; left 7.1° / 3.4° / 0.9°; right 8.8° / 3.6° / 0.8° (full / half / quarter) |

Topples at half and quarter speed: 0. First contact: away and right always a jaw (box or mesh), left always the
gripper body; the jaw tip spheres never.

Glitches: 8 of 750 (base 6, arm_friction 0, speed 2). Before the glitch rule, 6 base glitches counted as topples or
leans, which is why left and right read 8.8° and 10.5° in the earlier write-up.

**Caveats.** The `speed` runs differ from the earlier part4 script's: that script slowed the trajectory in float32
and this one in float64; 78 of those 150 runs changed. Pushes are chaotic enough for the last bit to matter, so read
speed effects as averages, not single runs.
