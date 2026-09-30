# Knowledge

What this project has learned and how to work on it, kept so it survives context compaction, restarts and new
sessions. Chat history is not a record. If something matters later, it goes in one of these files.

| File | What it holds | Read it | Update it |
|---|---|---|---|
| `state.md` | Where things stand: what exists, what is running, what is next, decisions made, open questions | At the start of every session, and after any compaction or restart, before doing anything | When anything finishes, starts, or gets decided |
| `operating-principles.md` | How to work here: deciding, measuring, running long jobs, keeping knowledge, reporting | At the start of a session, and before any long or parallel run | When a mistake teaches a new habit, or dm1487 corrects how the work is done |
| `mechanical-search-learnings.md` | Domain lessons, written for any objects in a bin, with the evidence behind each | Before designing or changing the expert, perception, or skills | Whenever something is learned about the task, the arm, perception or the simulator |

Other records, with narrower scope:

- `demonstrator/instructions.md`: the rulebook players follow in the cube bin. Cube-specific detail goes there; the
  general lesson behind it goes in `mechanical-search-learnings.md`.
- `demonstrator/notes/`: one file per played episode.
- `demonstrator/REPORT.md`, `demonstrator/RELATED_WORK.md`, `demonstrator/consult/`: dated reports.
- `CLAUDE.md`: commands, layout and traps. It carries a short version of the operating principles because it is
  reloaded automatically after every compaction.
- Claude's own memory directory (outside the repo) points here.

Rules for these files: date every entry that can go stale, say where the evidence is, write plainly, and prefer
correcting an entry to adding a second one that contradicts it.
