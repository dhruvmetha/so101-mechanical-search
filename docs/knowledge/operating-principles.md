# Operating principles

How to work on this project, distilled from what went right and wrong. Each principle names the incident behind it,
so it can be judged rather than followed blindly. Started 2026-09-30 after dm1487 said: "I really like that you have
common sense, but you seem to not activate it every turn."

## Before deciding anything

- Rebuild the picture first. The boot hook puts `state.md`, the index and the journal's last entries in context; also
  read the relevant rulebook, topic files and memory, not only the last message. After a compaction or restart this
  is mandatory: summaries drop details, and decisions made from a summary repeat old mistakes.
- Look at the available skills and use the ones that fit: `recall` to rebuild context, `compute-resources` before using
  the machines, `vla-finetune` for training, `superpowers:systematic-debugging` for anything unexpected,
  `superpowers:verification-before-completion` before calling something done. `show-me-your-work` keeps a decision log
  but only dm1487 can start it. On 2026-09-29 and 30 most of these sat unused while their problems happened.
- Know the goal and judge each step by it. As of 2026-09-30 the goal is an expert that makes good demonstrations fast;
  the student is out of scope until dm1487 says otherwise. Cubes are the test rig for realistic objects.
- Answer what was asked. Don't ask dm1487 for what the code, the files or a quick measurement can tell you.

## Measure before asserting

- A claim that drives a decision needs a number and the place it came from. Three confident claims were later
  overturned by measurement: the near-base "dead strip" (it was the wrist roll, not reach), `mj_geomDistance` clearances
  (wrong for box pairs), and the wrist camera's pixels (not square).
- Check a rule against the geometry or physics before writing it down. The fixed-finger shift rule was written
  backwards and printed by the tools for hours because it was copied from a note, not derived.
- When a check disagrees with what happens, find out why before adjusting a threshold. The finger check missed a flush
  neighbour because the meshes had 60 mm faces with no vertices inside; the fix was denser sampling, not a margin.
- Score tools against real outcomes. The finger check was scored against 76 real picks, which showed hits mean less
  than assumed.
- Check what agents report, too. On 2026-09-30 a docs-lookup agent said SessionStart hook output had no size cap and
  that CLAUDE.md is not re-read after compaction. The Claude Code docs say the cap is 10,000 characters and that the
  project CLAUDE.md is re-read. Read the primary source before building on a claim.

## Long or parallel runs

- Budget before launching: requests and tokens per unit of work, the account's session limit, and what happens when
  it hits. On 2026-09-30 eight agents used up the limit in a few hours and the run sat dead from 02:33 until morning.
  Pace agents, and schedule a restart after a limit resets.
- Freeze tools and rules for the length of a batch, and record their version in the output. Changing them under
  running agents made early and late demonstrations differ.
- One unit of work per agent context. Agents that played three episodes in a row paid for the first two on every call
  of the third.
- Keep progress on disk as it happens (episodes save after every move), so a crash loses only the move in flight.
- Keep the main context for decisions. Send bulk reading and broad searches to subagents, and bring back summaries.

## Keeping knowledge

`README.md` in this folder has the whole system: where each kind of fact goes, the boot, the write-back and the
consolidation. The habits:

- Write back as you go. When anything starts, finishes, gets decided or gets blocked, edit `state.md` and append a
  journal entry. Don't wait for the end of a session, or for a compaction that may come first.
- After anything is learned, update `mechanical-search-learnings.md` in general form. Cube-only detail goes in
  `demonstrator/instructions.md`, and a code or simulator trap in the matching `.claude/rules/` file.
- One home per fact. Before adding one, grep for it; if it exists, correct that copy and point to it from elsewhere.
- A rule that recurs or that players keep getting wrong belongs in the tools as a check, not in more prose.
- Write lessons so they carry over to other objects: say what is general and what is cube-specific.
- Date entries and cite evidence. Correct a wrong entry in place and say when it changed. Take times from `date`, not
  from memory: on 2026-09-30 `state.md` was stamped 11:45 at 11:13 by the clock.

## Reporting

- Plain language and numbers. Say what is uncertain and what was not tested.
- Show spatial results as pictures (before and after), not only tables.
- Report failures as plainly as successes, including your own.
