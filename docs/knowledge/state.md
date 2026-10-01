# State

Where the project stands now. The boot hook loads this file at startup, resume, `/clear` and after every compaction.
Edit it in place when anything starts, finishes, gets decided or gets blocked, and log the same event in `journal.md`.
Keep it under 6,000 characters: finished work leaves this file for the journal. Last updated 2026-10-01 by Claude.

## Goal right now

An expert demonstrator that produces good demonstrations fast (dm1487, 2026-09-30). Good: solved, few wasted moves,
sensible kept moves. Fast: wall-clock and token cost per demo, and robot time. The later aim (dm1487, 20:40): distill
how Claude reasons through a task into a smaller model, implicitly (20:55): the student learns from what the cameras
show and what the arm does, not from Claude's written beliefs. So every move must follow from what the cameras showed.

The cube bin is the proof-of-concept rig. The expert should later work with a family of containers: deeper bins up to
sink-like basins, bins set into the table with the rim flush (which is how the current "bin 10 cm below the base"
would be built for real), and contents from cubes to YCB objects and toys (dm1487, 2026-09-30).

## What exists

- Sim tasks: the bin (`so101sim/mechsearch.py`) and the shelf (`so101sim/shelf.py`), specs in `docs/superpowers/specs/`.
- The expert's tools, `demonstrator/tools.py` (`codebase.md` has the parts): observe, check (picks and strokes), act
  with calibrated refusals, undo (spills), places. Rulebook `demonstrator/instructions.md` v10.4; brief AGENT_BRIEF.md.
- Demonstrations v1, superseded (redo, dm1487): 26 episodes in `outputs/mechsearch_demos_v1` (LeRobot, 43,392
  frames); records in `demonstrator/episodes/`, notes in `demonstrator/notes/`.
- Analysis: `demonstrator/REPORT.md`, `RELATED_WORK.md`, `consult/fable-2026-09-30.md`; studies in
  `demonstrator/experiments/`, outputs in `outputs/`.
- The knowledge system: `docs/knowledge/README.md`; `.claude/hooks/knowledge.sh` boots it, traps in `.claude/rules/`.
- The expert as a program: `demonstrator/planner.py` p0.3 (9859fed) solved 13 of 13 pilot seeds in 1 to 6 moves, but
  it judged moves with the true hidden state, so it cheats (`planner.md`). Its demos are not usable yet.
- Realistic objects: `docs/reports/2026-09-30-realistic-objects.md` (about 80 scanned objects fit the gripper).
  Bin depth, partial: tip reach z -0.225 to +0.085 m (journal).
- Work is committed on `mechanical-search`, from b0455b2 on; p0.3 is 9859fed.

## Running now

- The teacher driver (`demonstrator/driver.py`, journal 10-01 00:40): one Claude call per move. b5142 solved in 6
  moves for about 35k tokens, $0.40 (01:05). Planner design report done (01:20).

## Next

1. The bench (planner-design report, step 1 and 2): the teacher driver on the 29 look_first piles and 10 buried
   seeds; solved, moves, spills, tokens per demo against the rules' 3, 9, 11 (14 together). Awaiting dm1487's go.
2. Grasp and spill models from simulated picks and strokes on other seeds, camera features only (report step 3).
3. Planner p1 on the belief (phases, expected moves c + c'(1/p - 1), spill veto), Claude as arbiter on triggers and
   as strategy author between batches (`docs/reports/2026-10-01-planner-design.md`).
4. Smaller: rerun the pick calibration; shelf sweeps; wall height; bin depth; the container family and object set.

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
- 2026-09-30: the journal logs every decision, dm1487's and Claude's, enough to replicate the demos (dm1487).
- 2026-09-30: planning must not use hidden state (dm1487). No trying moves on the real pile, no undo of misses.
- 2026-09-30: friction U(0.4, 0.7) and density U(600, 1200) drawn per episode from the seed, hidden from players
  (`demonstrator/episodes/.hidden/`); stiff contacts at 1/600 s (dm1487: realistic, no penetration).
- 2026-09-30: redo all demos; `outputs/mechsearch_demos_v1` used wrist depth and miss undos (dm1487).
- 2026-09-30: resets are privileged, allowed for now to get decent demos, discouraged and counted (dm1487): undo takes
  back a spill; a missed pick resets only by a fixed draw below eps = 0.1 (rulebook v10.1).
- 2026-10-01, IMPORTANT (dm1487): Claude plays as the teacher with its own belief, by the five rules at the top of
  the learnings file. The student's design is not Claude's concern.

## Open questions

- Raise the 15-move cap, or cap robot time instead? d1000 needed about 16 moves. It's a spec change.
- Awaiting dm1487: wall height (sizing script); a stateless Claude player (fresh call per move with a compact memory
  doc) as a bridge, since a Claude-played demo costs about 5M cached tokens.
- How often will an honest planner still need Claude?
- Perception for realistic objects: grasps from each object's points instead of fitted cube poses.
- When do YCB or toy objects arrive? If within weeks, codify less of the cube rulebook and build the heightmap
  grasp sampler sooner (Fable's last point).
