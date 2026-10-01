# State

Where the project stands now. The boot hook loads this file at startup, resume, `/clear` and after every compaction.
Edit it in place when anything starts, finishes, gets decided or gets blocked, and log the same event in `journal.md`.
Keep it under 6,000 characters: finished work leaves this file for the journal. Last updated 2026-09-30 20:13 by Claude.

## Goal right now

An expert demonstrator that produces good demonstrations fast (dm1487, 2026-09-30). Good: solved, few wasted moves,
sensible kept moves. Fast: wall-clock and token cost per demo, and robot time. The student that learns from the demos
is out of scope until dm1487 says otherwise.

The cube bin is the proof-of-concept rig. The expert should later work with a family of containers: deeper bins up to
sink-like basins, bins set into the table with the rim flush (which is how the current "bin 10 cm below the base"
would be built for real), and contents from cubes to YCB objects and toys (dm1487, 2026-09-30).

## What exists

- Sim tasks: the bin (`so101sim/mechsearch.py`) and the shelf (`so101sim/shelf.py`), specs in `docs/superpowers/specs/`.
- The expert's tools, `demonstrator/tools.py` (`codebase.md` has the parts): observe, check (picks and strokes), act
  with calibrated refusals, undo (spills), places. Rulebook `demonstrator/instructions.md` v10.4; brief AGENT_BRIEF.md.
- Demonstrations v1, superseded (redo, dm1487): 26 episodes in `outputs/mechsearch_demos_v1` (LeRobot, 43,392
  frames); records in `demonstrator/episodes/`, notes in `demonstrator/notes/`.
- Analysis: `demonstrator/REPORT.md` (overnight results, what would make demos cheaper),
  `demonstrator/RELATED_WORK.md` (sweeping, pushing and scripted planners), `outputs/postmortem_2026-09-30.json`,
  and the Fable consult `demonstrator/consult/fable-2026-09-30.md`.
- Sweep pilot and miss study: `demonstrator/experiments/`, outputs in `outputs/`.
- The knowledge system: `docs/knowledge/README.md`; `.claude/hooks/knowledge.sh` boots it, traps in `.claude/rules/`.
- The expert as a program: `demonstrator/planner.py` p0.3 (9859fed) solved 13 of 13 pilot seeds in 1 to 6 moves, but
  it judged moves with the true hidden state, so it cheats (`planner.md`). Its demos are not usable yet.
- Realistic objects: `docs/reports/2026-09-30-realistic-objects.md` (Google Scanned Objects in MuJoCo, about 80
  fit the gripper; stay on MuJoCo CPU), `scripts/objects_*.py`, `models/objects_test_*`. Bin depth, partial: tip reach
  z -0.225 to +0.085 m (journal).
- Work is committed on `mechanical-search`, from b0455b2 on; p0.3 is 9859fed.

## Running now

- Nothing. Pushes first-class in the tools (19:40). Push studies done (journal, 20:00 to 20:13): one stroke opened the red
  in 25 of 29 piles, but my common-sense choice found it in only 8. Reading list sent. Demos so far: b5000, b5002,
  shelf 307 (videos sent).

## Next

1. What predicts a working push: the red's pose quality matters most (trusted 3 of 4, rough 5 of 25); test look-first,
   verify pickable verdicts with real picks; look at pictures (`outputs/push_goal_2026-09-30`)
   before encoding cues; then push against pick-and-place of the same blockers on the same piles.
2. The expert as a program (belief-space TAMP: plan on the belief, one move, look, replan), using the calibrated checks
   as its models; Claude only when it finds no plan. Then redo the demos (dm1487) and measure moves and cost.
3. Smaller: rerun the pick calibration (it predates the third-layer contact fix); shelf sweeps with a calibration;
   wall height with the sizing script; the bin-depth study; the container family and the object set.

## Decisions in force

- 2026-09-25: the task is mechanical search in a bin; spec amended twice since.
- 2026-09-29: noslip on in `models/stage.xml`; resets hand over piles at rest.
- 2026-09-30: primitives take a wrist `flip`; check chooses the roll that keeps the arm clear of itself.
- 2026-09-30: the expert runs only in sim, deciding from camera data (looks are real motions); a policy distilled
  from its demos runs on the real arm. The expert may use ideal perception of what the student's sensors see (exact
  masks included), no other channel: front camera depth and exact masks, wrist colour and exact masks, no wrist
  depth. No SAM for now; maybe later (dm1487).
- 2026-09-30: the goal is the expert, not the student; cubes are a proof of concept; knowledge lives in `docs/knowledge/`.
- 2026-09-30: pushes and sweeps are first-class expert moves; act refuses the strokes calibration showed spill (dm1487).
- 2026-09-30: the knowledge system is in place: boot hook, journal, path-scoped traps, a trimmed CLAUDE.md. The
  journal logs every decision, dm1487's and Claude's, in enough detail to replicate the demos (dm1487).
- 2026-09-30: planning must not use hidden state (dm1487). No trying moves on the real pile, no undo of misses.
- 2026-09-30: friction U(0.4, 0.7) and density U(600, 1200) drawn per episode from the seed, hidden from players
  (`demonstrator/episodes/.hidden/`); stiff contacts at 1/600 s (dm1487: realistic, no penetration).
- 2026-09-30: redo all demos; `outputs/mechsearch_demos_v1` used wrist depth and miss undos (dm1487).
- 2026-09-30: resets are privileged, allowed for now to get decent demos, discouraged and counted (dm1487): undo takes
  back a spill; a missed pick resets only by a fixed draw below eps = 0.1 (rulebook v10.1).

## Open questions

- Raise the 15-move cap, or cap robot time instead? d1000 needed about 16 moves. It's a spec change.
- Awaiting dm1487: wall height (sizing script); a stateless Claude player (fresh call per move with a compact memory
  doc) as a bridge, since a Claude-played demo costs about 5M cached tokens.
- How often will an honest planner still need Claude?
- Perception for realistic objects: grasps from each object's points instead of fitted cube poses.
- When do YCB or toy objects arrive? If within weeks, codify less of the cube rulebook and build the heightmap
  grasp sampler sooner (Fable's last point).
