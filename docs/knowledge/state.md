# State

Where the project stands now. The boot hook loads this file at startup, resume, `/clear` and after every compaction.
Edit it in place when anything starts, finishes, gets decided or gets blocked, and log the same event in `journal.md`.
Keep it under 6,000 characters: finished work leaves this file for the journal. Last updated 2026-09-30 14:14 by Claude.

## Goal right now

An expert demonstrator that produces good demonstrations fast (dm1487, 2026-09-30). Good: solved, few wasted moves,
sensible kept moves. Fast: wall-clock and token cost per demo, and robot time. The student that learns from the demos
is out of scope until dm1487 says otherwise.

The cube bin is the proof-of-concept rig. The expert should later work with a family of containers: deeper bins up to
sink-like basins, bins set into the table with the rim flush (which is how the current "bin 10 cm below the base"
would be built for real), and contents from cubes to YCB objects and toys (dm1487, 2026-09-30).

## What exists

- Sim tasks: the bin (`so101sim/mechsearch.py`) and the shelf (`so101sim/shelf.py`), specs in `docs/superpowers/specs/`.
- The expert's tools, `demonstrator/tools.py`: observe (front and wrist cameras fused, target pixels in 3D, what
  blocks each of the target's jaw axes), check (both wrist rolls, dense finger sweep against a height map, walls, arm
  clearance, which side the fixed finger lands), act, undo, places, list, status. Perception in `perception.py`, arm
  clearance in `reach.py`.
- The rulebook the players follow: `demonstrator/instructions.md` v9. The agent brief: `demonstrator/AGENT_BRIEF.md`.
- Demonstrations: 26 solved episodes exported to `outputs/mechsearch_demos_v1` (LeRobot format, 43,392 frames at 30 Hz,
  front and wrist images at 224). Episode files in `demonstrator/episodes/`, notes in `demonstrator/notes/`.
- Unfinished: d1000 failed (out of moves); d1002, d1014, d1020, d1022, d1025, d1029, d1032, d1034 are half played,
  stopped by the account's session limit at 02:33 on 2026-09-30. Their moves are saved and they can be finished.
- Analysis: `demonstrator/REPORT.md` (overnight results and what would make demos cheaper),
  `demonstrator/RELATED_WORK.md` (sweeping, pushing and scripted planners), `outputs/postmortem_2026-09-30.json`,
  and the Fable consult `demonstrator/consult/fable-2026-09-30.md`.
- Sweep pilot: `demonstrator/experiments/sweep_test.py` and `render_sweep.py`, results and pictures in
  `outputs/sweep_pilot_2026-09-30/`. Numbers in `mechanical-search-learnings.md`, "Strategy".
- The knowledge system (2026-09-30): `.claude/hooks/knowledge.sh` boots this file, the index and the latest journal
  headings; traps load from `.claude/rules/`. `docs/knowledge/README.md` explains it.
- The expert as a program: `demonstrator/planner.py` p0.3 (9859fed) solved 13 of 13 pilot seeds in 1 to 6 moves, but
  it judged moves with the true hidden state, so it cheats (learnings, "The planner"). Its demos are not usable yet.
- Realistic objects: `docs/reports/2026-09-30-realistic-objects.md` recommends Google Scanned Objects (MuJoCo port,
  about 80 fit the gripper, about 20 toys) and staying on MuJoCo CPU; loading scripts `scripts/objects_*.py`, test scenes
  `models/objects_test_*`. Bin depth, partial: tip reach z -0.225 to +0.085 m (journal, 2026-09-30).
- Commits on `mechanical-search`: b0455b2, f5790cd (p0), 477b345 (memory system), 9859fed (p0.3), and the
  compaction write-back after them.

## Running now

Nothing. The session is between steps.

## Next

1. Make the planner honest: judge candidates in several worlds rebuilt from perception (seen cubes at perceived poses,
   unseen space filled by dropping cubes into it, the red sampled where pixels and covers allow), never the true one.
   Looks become candidate moves. Then rerun the 13 pilot seeds and compare with p0.3.
2. The rest of the build order: aim-error scoring, cost logging with an export gate, a capped Claude rescue, looking
   from hover instead of parking; export the planner's solved episodes as demos.
3. Finish the bin-depth study from its partial files; then the container family and the object set.

## Decisions in force

- 2026-09-25: the task is mechanical search in a bin; spec amended twice since.
- 2026-09-29: noslip on in `models/stage.xml`; resets hand over piles at rest.
- 2026-09-29: the expert decides from camera data only; looks are real motions. Undo for spills and most bad misses
  came with rulebook v4 on 2026-09-30 (corrected 2026-09-30: this line had dated undo 2026-09-29).
- 2026-09-30: primitives take a wrist `flip`; check chooses the roll that keeps the arm clear of itself.
- 2026-09-30: observe fuses the wrist camera with the front camera.
- 2026-09-30: the goal is the expert, not the student; cubes are a proof of concept; knowledge lives in `docs/knowledge/`.
- 2026-09-30: build in the Fable order, as a pilot on a few seeds first (dm1487).
- 2026-09-30: the knowledge system is in place: boot hook, journal, path-scoped traps, a trimmed CLAUDE.md. The
  journal logs every decision, dm1487's and Claude's, in enough detail to replicate the demos (dm1487).
- 2026-09-30: the planner commits only moves that help, rejects empty picks and wrong-cube picks, scores the target's
  better grasp axis, and looks two moves ahead over its three best first moves; no full search tree.
- 2026-09-30: planning must not use hidden state; lookahead runs in worlds rebuilt from perception (dm1487).

## Open questions

- Raise the 15-move cap, or cap robot time instead? d1000 needed about 16 moves. It's a spec change.
- Keep the little-used 5 cm push? The sweep is now its own move and does most of the planner's work.
- How often will an honest planner still need Claude?
- Perception for realistic objects: grasps from each object's points instead of fitted cube poses.
- Finish the 8 half-played episodes? Offered, not decided.
- When do YCB or toy objects arrive? If within weeks, codify less of the cube rulebook and build the heightmap
  grasp sampler sooner (Fable's last point).
