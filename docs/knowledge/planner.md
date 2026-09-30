# The planner

The expert as a program (`demonstrator/planner.py`): what the pilots measured, and what counts as honest play. Moved
here from `mechanical-search-learnings.md` on 2026-09-30, when the section passed 40 lines. General lessons about
objects and containers stay in that file.

Measured 2026-09-30 on 13 cube-bin seeds (7 played by agents overnight, 6 new), `demonstrator/planner.py`, records in
`outputs/planner_pilot_p0*`.

- Perception proposes, the simulator judges. Candidate moves come from camera data only (a pick on the target on both
  grasp axes, slid along its faces; long sweeps across the target's spot when objects cover it; relocations of covers
  and grasp-axis blockers to spots away from the target and its neighbours). Each is run on a snapshot; true poses only
  score the outcome. This is object-agnostic: nothing in the loop needs cube geometry except the candidate generator.
- Its numbers are not honest play (dm1487, 2026-09-30 14:15). It judged candidates with the simulator's true state, so it leaked
  hidden information; see the last bullet. With that caveat, p0.3 solved 13 of 13, buried starts in 2 to 6 moves and partial ones in
  1 or 2, 5 to 43 s of robot time, no tokens, 1 to 14 minutes of wall clock per seed on one core. The agents took 7 to
  11 moves on the same seeds and failed seed 1000, which the planner solved in 6.
- A long sweep across the target's spot was the first move of most buried solutions: it cleared the covers and often
  loosened the flush neighbours at once. Judged in the simulator first, no sweep in a kept line spilled.
- Never commit the least bad move. p0 did, and put a spill into a kept line. Commit only moves that score above zero;
  the one exception is setting down an object already held.
- Reject moves that look like progress but aren't: a pick that lifts nothing (it can knock covers off by luck), and a
  target pick that lifts a neighbour instead.
- Score progress per grasp axis: count the blocked ends of the target's better axis. Counting all neighbours hid real
  progress on one axis.
- One-step scoring is myopic. On seed 1004 a relocation beat a sweep by 0.2 points and stalled, where the sweep led to
  success. Scoring the three best first moves by the best second move each allows (p0.3) solved all four seeds p0.2
  stuck on, for 2 to 4 times the wall clock.
- Lookahead on the true world is cheating, in two ways (dm1487's question, 2026-09-30 14:15). Scoring with true poses reads hidden
  state: on seed 1038 the cameras saw no red at all and the score still knew where it was. And trying several moves on
  the real hidden pile, then keeping the one that worked, picks moves by cubes nobody saw; the demo looks as if the
  robot knew where to dig. Mechanical search is about acting to reveal what is hidden, so the expert has to plan with
  what it believes: judge candidates in worlds rebuilt from perception (seen objects at their perceived poses, unseen
  space filled with plausible objects, the target where the evidence allows), several of them, and let the real world
  answer only the move actually made. Moves that uncover the target then score well because they shrink the doubt,
  and a look becomes a move worth making. Scoring under aim errors (the build order) is a separate, smaller check.
- Undo on the real pile leaks the same way (found 2026-09-30 14:41). Taking a move back and trying another keeps the one
  that worked on cubes nobody saw. The 26 player demos hold 13 undone attempts in 10 episodes, none of them spills:
  10 missed picks and 3 cubes dropped on the way to a place (corrected 14:58; it said 12 misses). So those demos show
  an expert that never misses. Honest play keeps a miss and the recovery from it.
- Privileged steps are allowed only where dm1487 grants them, and each is marked and counted. Granted on 2026-09-30:
  reset a spill, so one spill doesn't cost a whole demo; and reset a missed pick with probability eps, which sets how
  often demos miss. Reset every miss and the demos show an arm that never misses; reset none and every miss stays and
  demos run longer. Both are discouraged. A chance reset has to be drawn by
  the tool, not chosen: a player who chooses resets the misses that hurt, and that choice is the leak again.
- A twin rebuilt from perception may not pay (open, 2026-09-30 14:41). Misses come from perception error, which a twin
  built from the same perception can't foresee. The spills p0.3's simulator caught (11 of 348 candidates) came from
  its own untargeted candidates; players using camera checks and the rulebook spilled in none of 26 solved episodes.
  And for realistic objects a twin needs shapes and masses of objects only partly seen. Measure before building one.
