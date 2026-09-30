# State

Where the project stands now. The boot hook loads this file at startup, resume, `/clear` and after every compaction.
Edit it in place when anything starts, finishes, gets decided or gets blocked, and log the same event in `journal.md`.
Keep it under 6,000 characters: finished work leaves this file for the journal. Last updated 2026-09-30 19:39 by Claude.

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

- Nothing. Pushes are first-class since 19:39 (sweep, push report, calibrated refusal; journal). Reading list sent:
  `docs/reports/2026-09-30-reading-list.md`. Demos under the new rules so far: b5000, b5002, shelf 307.

## Next

1. Finish the tools (Running now). Then the honest planner (Claude's design, 14:41): a per-episode memory of what the
   cameras saw and a map of where the target could be, rules as code, looks as moves, no twin unless a measured
   failure needs one; rerun the 13 pilot seeds against p0.3. Then redo all demos (dm1487).
2. The rest of the build order: cost logging with an export gate, a capped Claude rescue, looking from hover.
3. Finish the bin-depth study from its partial files; then the container family and the object set.

## Decisions in force

- 2026-09-25: the task is mechanical search in a bin; spec amended twice since.
- 2026-09-29: noslip on in `models/stage.xml`; resets hand over piles at rest.
- 2026-09-30: primitives take a wrist `flip`; check chooses the roll that keeps the arm clear of itself.
- 2026-09-30: the expert runs only in sim, deciding from camera data (looks are real motions); a policy distilled
  from its demos runs on the real arm. The expert may use ideal perception of what the student's sensors see (exact
  masks included), no other channel: front camera depth and exact masks, wrist colour and exact masks, no wrist
  depth. No SAM for now; maybe later (dm1487).
- 2026-09-30: the goal is the expert, not the student; cubes are a proof of concept; knowledge lives in `docs/knowledge/`.
- 2026-09-30: build in the Fable order, as a pilot on a few seeds first (dm1487).
- 2026-09-30: the knowledge system is in place: boot hook, journal, path-scoped traps, a trimmed CLAUDE.md. The
  journal logs every decision, dm1487's and Claude's, in enough detail to replicate the demos (dm1487).
- 2026-09-30: planning must not use hidden state (dm1487). No trying moves on the real pile, no undo of misses.
- 2026-09-30: redo all demos; `outputs/mechsearch_demos_v1` used wrist depth and miss undos (dm1487).
- 2026-09-30: resets are privileged, allowed for now to get decent demos, discouraged and counted (dm1487): undo takes
  back a spill; a missed pick resets only by a fixed draw below eps = 0.1 (rulebook v10.1).

## Open questions

- Raise the 15-move cap, or cap robot time instead? d1000 needed about 16 moves. It's a spec change.
- Keep the little-used 5 cm push? The sweep is now its own move and does most of the planner's work.
- Awaiting dm1487 (19:10): a push with a chosen stroke (spec amendment); wall height (measure with the sizing script);
  the expert as belief-space TAMP (plan on the belief, one move, look, replan); a stateless Claude player (fresh call
  per move with a compact memory doc) as a cheap bridge. Measured: a Claude demo costs about 5M cached tokens.
- How often will an honest planner still need Claude?
- Perception for realistic objects: grasps from each object's points instead of fitted cube poses.
- When do YCB or toy objects arrive? If within weeks, codify less of the cube rulebook and build the heightmap
  grasp sampler sooner (Fable's last point).
