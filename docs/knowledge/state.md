# State

Where the project stands now. The boot hook loads this file at startup, resume, `/clear` and after every compaction.
Edit it in place when anything starts, finishes, gets decided or gets blocked, and log the same event in `journal.md`.
Keep it under 6,000 characters: finished work leaves this file for the journal. Last updated 2026-09-30 14:59 by Claude.

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
- The rulebook the players follow: `demonstrator/instructions.md` v10.1. The agent brief: `demonstrator/AGENT_BRIEF.md`.
- Demonstrations: 26 solved episodes exported to `outputs/mechsearch_demos_v1` (LeRobot format, 43,392 frames at 30 Hz,
  front and wrist images at 224). Episode files in `demonstrator/episodes/`, notes in `demonstrator/notes/`.
- Unfinished: d1000 failed (out of moves); d1002, d1014, d1020, d1022, d1025, d1029, d1032, d1034 are half played,
  stopped by the account's session limit at 02:33 on 2026-09-30. Their moves are saved and they can be finished.
- Analysis: `demonstrator/REPORT.md` (overnight results, what would make demos cheaper),
  `demonstrator/RELATED_WORK.md` (sweeping, pushing and scripted planners), `outputs/postmortem_2026-09-30.json`,
  and the Fable consult `demonstrator/consult/fable-2026-09-30.md`.
- Sweep pilot: `demonstrator/experiments/`, `outputs/sweep_pilot_2026-09-30/`.
- The knowledge system: `docs/knowledge/README.md`; `.claude/hooks/knowledge.sh` boots it, traps in `.claude/rules/`.
- The expert as a program: `demonstrator/planner.py` p0.3 (9859fed) solved 13 of 13 pilot seeds in 1 to 6 moves, but
  it judged moves with the true hidden state, so it cheats (learnings, "The planner"). Its demos are not usable yet.
- Realistic objects: `docs/reports/2026-09-30-realistic-objects.md` (Google Scanned Objects in MuJoCo, about 80
  fit the gripper; stay on MuJoCo CPU), `scripts/objects_*.py`, `models/objects_test_*`. Bin depth, partial: tip reach
  z -0.225 to +0.085 m (journal).
- Work is committed on `mechanical-search`, from b0455b2 on; p0.3 is 9859fed.

## Running now

- Since 14:56: a subagent replays the 22 missed picks to find why (`outputs/miss_study_2026-09-30/`).

## Next

1. Make the planner honest (Claude's design, 14:41): a per-episode memory of what the cameras saw (space seen empty,
   where moved objects went, the target's last sighting, a map of where it could be), camera checks and rules as
   code, looks as moves. No twin unless a measured failure needs one. Rerun the 13 pilot seeds against p0.3.
2. The rest of the build order: aim-error scoring, cost logging with an export gate, a capped Claude rescue, looking
   from hover instead of parking; export the planner's solved episodes as demos.
3. Finish the bin-depth study from its partial files; then the container family and the object set.

## Decisions in force

- 2026-09-25: the task is mechanical search in a bin; spec amended twice since.
- 2026-09-29: noslip on in `models/stage.xml`; resets hand over piles at rest.
- 2026-09-29: the expert decides from camera data only; looks are real motions.
- 2026-09-30: primitives take a wrist `flip`; check chooses the roll that keeps the arm clear of itself.
- 2026-09-30: observe fuses the wrist camera with the front camera.
- 2026-09-30: the goal is the expert, not the student; cubes are a proof of concept; knowledge lives in `docs/knowledge/`.
- 2026-09-30: build in the Fable order, as a pilot on a few seeds first (dm1487).
- 2026-09-30: the knowledge system is in place: boot hook, journal, path-scoped traps, a trimmed CLAUDE.md. The
  journal logs every decision, dm1487's and Claude's, in enough detail to replicate the demos (dm1487).
- 2026-09-30: the planner commits only moves that help, rejects empty picks and wrong-cube picks, scores the target's
  better grasp axis, and looks two moves ahead over its three best first moves; no full search tree.
- 2026-09-30: planning must not use hidden state (dm1487). No trying moves on the real pile, no undo of misses.
- 2026-09-30: resets are privileged, allowed for now to get decent demos, discouraged and counted (dm1487): undo takes
  back a spill; a missed pick resets only by a fixed draw below eps = 0.5 (rulebook v10.1).

## Open questions

- Raise the 15-move cap, or cap robot time instead? d1000 needed about 16 moves. It's a spec change.
- Keep the little-used 5 cm push? The sweep is now its own move and does most of the planner's work.
- How often will an honest planner still need Claude?
- Perception for realistic objects: grasps from each object's points instead of fitted cube poses.
- Finish the 8 half-played episodes? And the 10 demos with pre-v10 miss undos: keep, mark or redo?
- When do YCB or toy objects arrive? If within weeks, codify less of the cube rulebook and build the heightmap
  grasp sampler sooner (Fable's last point).
