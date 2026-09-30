# Journal

The project's log: what happened and when, oldest first, newest at the bottom. Append only. Never edit or delete an
entry; to correct one, append a new entry that names the old one and says what was wrong. `state.md` says where things
stand now; this file says how they got there.

Each entry starts with a heading in one fixed form, so `grep '^## \[' docs/knowledge/journal.md | tail -5` lists the
last five (the form follows Karpathy's llm-wiki log). Under it go one to four lines: what happened, the numbers, who
decided, and the path to the evidence. Use local time from `date`.

    ## [YYYY-MM-DD HH:MM] kind | title

Kinds: `run` (a batch or job started or stopped), `result` (a measurement or outcome, with numbers), `decision`
(with who made it), `direction` (what dm1487 asked for, quoted where possible), `mistake` (ours, and what it cost),
`compaction` (a write-back after one), `consolidation` (what moved where). When this file passes 400 lines, move every
month but the current one to `journal/YYYY-MM.md`, unchanged.

## [2026-09-30 11:22] note | Entries before this one were reconstructed from existing files

The entries dated 2026-09-18 to 2026-09-30 11:13 were written on 2026-09-30 from the specs, CLAUDE.md, state.md,
demonstrator/REPORT.md, the rulebook's change log and Claude's memory files. Times appear only where a file records
one. They are in date order below this note, not in the order they were written.

## [2026-09-18] decision | Sim design spec

`docs/superpowers/specs/2026-09-18-so101-sim-design.md`: the SO-101 in MuJoCo on arrakis, headless.

## [2026-09-22] result | Hardware measurements from the `so101` session

Sag, backlash and stiffness fits as of this date, in `docs/knowledge/real-arm.md`.

## [2026-09-25] decision | The task is mechanical search in a bin

A 22 cm bin sunk 10 cm below the base, 29 gray cubes and one red. Spec:
`docs/superpowers/specs/2026-09-25-mechanical-search-design.md`, amended twice after measurements. The same day the
Warp renderer drew the scene's cameras at about 7,000 images/s at 128x128 for 1024 worlds (`codebase.md`), and
`primitives.bridge` switched from a fixed z = 0.03 to `hover_z`.

## [2026-09-26] decision | Second task: shelf search

14 upright cylinders on a shelf, one red, reached sideways. Spec:
`docs/superpowers/specs/2026-09-26-shelf-search-design.md`. Claude played shelf seeds 300 and 301 from images and
failed both (`outputs/shelf_play_30*.md`).

## [2026-09-28] result | Hand-played bin seeds 301 to 310, all solved

Claude as planner, 2026-09-26 to 09-28: 5.8 moves on average (buried 7.6, partial 4.0), 24 of 37 grasps missed.
Lessons in Claude's memory file `mechsearch-planner-playbook.md`. The weekly advisor log artifact was first published
on 2026-09-28.

## [2026-09-29] decision | noslip on; piles handed over at rest

`models/stage.xml` sets `noslip_iterations="10"`: 10 s of idle drift fell to 0.85 mm median. The scripted dig-out
went from 12 to 15 of seeds 0 to 19. Resets now hand over piles only once they hold still (`so101sim/rest.py`).

## [2026-09-29] decision | The expert decides from camera data only

Undo covers spills and most bad misses, and looks are real arm motions. Rulebook `demonstrator/instructions.md` v1 and
v2 written the same day.

## [2026-09-29] direction | Weekly advisor log paused

dm1487: no need for the advisor log for now; focus on the demonstrator. Memory file `weekly-advisor-log.md`.

## [2026-09-30 02:33] run | Overnight agent run stopped by the account's session limit

Eight agents played seeds 1000 to 1034 in parallel; the limit stopped all of them mid-episode with no recovery until
morning. 24 agent episodes and 2 of Claude's (e400, e402) were solved. The rulebook went from v4 to v9 overnight.
`MUJOCO_LOG.TXT` recorded "Insufficient arena memory" at 00:02. Report: `demonstrator/REPORT.md`.

## [2026-09-30 08:22] result | 26 solved demos exported

`outputs/mechsearch_demos_v1`: LeRobot format, 43,392 frames at 30 Hz, front and wrist images at 224. Every episode
replayed exactly against its record before export.

## [2026-09-30] decision | Wrist flip and camera fusion

Primitives take a wrist `flip`, and check picks the roll that keeps the arm clear of itself: 54 of 55 empty-bin
picks landed within 6 mm. Observe fuses the wrist camera with the front camera: unseen floor went from about 60% to
38%, pose error from 1.4 mm to 0.6 mm median.

## [2026-09-30] mistake | The fixed-finger shift rule was backwards

It was copied from a note instead of derived, and the tools printed it for hours. Fixed in rulebook v9.

## [2026-09-30] direction | The goal is the expert, and cubes are a proof of concept

dm1487: "It's not about the student. I want the expert to give me good demos and fast." The expert should later work
with deeper bins up to sink-like basins, bins set into the table, and YCB objects or toys.

## [2026-09-30] direction | Common sense every turn

dm1487: "I really like that you have common sense, but you seem to not activate it every turn." Led to
`operating-principles.md` and to keeping `docs/knowledge/` current.

## [2026-09-30 10:57] result | Related work and the sweep pilot

`demonstrator/RELATED_WORK.md` covers sweeping, pushing and scripted planners. The pilot
(`outputs/sweep_pilot_2026-09-30/`): one long stroke cleared the target in 8 of 16 tries on two piles, and 2 of 16
pushed an object out, both toward a nearby wall.

## [2026-09-30 11:12] result | Fable consult, and the cost of lookahead

`demonstrator/consult/fable-2026-09-30.md`: build a depth-1 planner first, no search tree. Checked the same day: a
pick costs 1.9 s of CPU and a place 0.7 s, saving the physics state 2 microseconds, copying the full MjData 35 ms;
about 25 to 45 episodes per core-hour at 8 candidates per move.

## [2026-09-30 11:13] decision | Knowledge moves into docs/knowledge/

README, state, operating principles and mechanical-search learnings created. dm1487 asked that they always be kept
current so the common sense survives compaction.

## [2026-09-30 11:22] consolidation | Boot hook, journal and path-scoped traps

dm1487 asked for a memory system along the lines of Karpathy's LLM OS. CLAUDE.md went from 311 to about 110 lines:
the layout section moved to `codebase.md`, the hardware notes to `real-arm.md`, and the traps to five files in
`.claude/rules/`. `.claude/hooks/knowledge.sh` now loads state, the index and this journal's tail at every boot and
after compaction, and flags stale state when a prompt is sent. The simulator lessons in the learnings file were
deduplicated against the traps; the noslip numbers there had mixed a 30 s and a 10 s window and are now correct.
The design is in `docs/knowledge/README.md`.
