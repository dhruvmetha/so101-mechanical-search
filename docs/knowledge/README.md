# Knowledge

This folder is the project's long-term memory. Claude's context window is its working memory, and it doesn't last:
compaction squeezes it into a lossy summary, a restart wipes it, and a new session starts empty. Anything that
matters later goes in a file, in one of the places the index below names. Chat history is not a record.

The part between the boot markers is loaded at every session start and after every compaction, so keep it under
2,000 characters. The design and its reasons come after it.

<!-- boot:start -->
## Index

Files named without a folder are in `docs/knowledge/`.

| Where | What goes there | Loads |
|---|---|---|
| `CLAUDE.md` | What every session needs: commands, map, traps that break any script | always |
| `state.md` | Goal, what exists, what runs, what's next, decisions in force, open questions | boot |
| `journal.md` | Every decision, dm1487's and Claude's, with why and evidence; runs, results, mistakes | boot, headings |
| `operating-principles.md` | Working habits, each with the incident behind it | before long runs |
| `mechanical-search-learnings.md` | Lessons for any objects and containers, with evidence | before changing the expert |
| `codebase.md` | What each module does, its conventions and numbers | before changing a module |
| `real-arm.md` | The physical arm: measurements, what blocks transfer | before hardware work |
| `.claude/rules/*.md` | Code and simulator traps, one file per subsystem | on Read of a covered file |
| `demonstrator/instructions.md` | Cube-only rules for the players | players read it |
| Cold storage | Episode notes and records, reports, consults, specs, plans, `outputs/` | when cited |
| Claude's memory directory | How dm1487 wants the work done; pointers here | `MEMORY.md` always |

## Write-back

- When anything starts, finishes, gets decided or gets blocked: edit `state.md` and append to `journal.md`, then.
- Every decision gets a journal entry as it's made: what, why, who, evidence. A run's entry names its commit.
- A lesson goes to its home above, in general form; cube-only detail to the rulebook, a code trap to its rule file.
- One home per fact. Correct it there and date the correction; anywhere else, point to it.
- After a compaction, write back what the summary holds and `state.md` lacks, before new work.
- Subagents write only their own files; the main session merges. Consolidate when the boot checks warn.
<!-- boot:end -->

## Where the idea comes from

Andrej Karpathy has described the LLM as an operating system's kernel rather than a chatbot. In September 2023 he wrote
of "LLMs not as a chatbot, but the kernel process of a new Operating System" ([X, 2023-09-28][k1]). His "LLM OS" diagram
([X, 2023-11-11][k2], and slide 31 of the [Intro to Large Language Models][k3] talk) draws the LLM as the CPU with "RAM"
labelled "context window", a "Disk" labelled "File system (+embeddings)", "Software 1.0 tools" such as a calculator,
Python interpreter and terminal, "Peripheral devices I/O" for video and audio, and a browser and "Other LLMs" over
"Ethernet". In the talk he says (from the captions): "this context window is your finite precious resource of your
working memory of your language model and you can imagine the kernel process this llm trying to page relevant
information in an out of its context window to perform your task".

He returned to it in the 2025 [YC talk "Software Is Changing (Again)"][k4]: "context windows are really kind of like
working memory and you have to sort of program the working memory quite directly" (auto-captions, about 17:06). The same
month he backed the term context engineering, "the delicate art and science of filling the context window with just the
right information for the next step", adding: "Too much or too irrelevant and the LLM costs might go up and performance
might come down" ([X, 2025-06-25][k5]). On memory, he wrote that "LLMs are quite literally like the guy in Memento,
except we haven't given them their scratchpad yet", and that learning could look like "the LLM writing a book for itself
on how to solve problems" ([X, 2025-05-11][k6]). Dates of posts on X are in UTC.

His [LLM Wiki gist][k7] (2026-04-04) is the closest match to this folder. The LLM maintains a folder of markdown pages
between immutable raw sources and a schema file such as CLAUDE.md. It runs three operations: ingest, query and lint.
It keeps two special files: `index.md`, a catalogue it reads first, and `log.md`, an "append-only record of what
happened and when" whose entries share a prefix so `grep` can list them. Here, the README index is his `index.md`, the
journal is his `log.md` in his entry format, cold storage is his raw sources, and consolidation is his lint.

The rows below marked "ours" are this project's extension of the analogy, not Karpathy's.

| OS part | Karpathy | Here |
|---|---|---|
| Kernel | "the kernel process" | Claude, in one session |
| RAM | "RAM" = "context window" | The context window. Compaction swaps it out to a lossy summary; a restart wipes it |
| Resident kernel image | ours | `CLAUDE.md`, which Claude Code loads at start and re-reads after compaction, and `MEMORY.md` |
| Boot loader | ours | `.claude/hooks/knowledge.sh boot`, run by the SessionStart hook |
| Process table | ours | `state.md` |
| System log | `log.md` in the LLM Wiki | `journal.md` |
| Disk, paged in on demand | "page relevant information in an out"; "File system" | the topic files in this folder |
| Demand paging on access | ours | `.claude/rules/*.md`, which Claude Code loads when Claude reads a matching file |
| Archive | "raw sources", immutable | cold storage |
| Other processes | "Other LLMs" | subagents, each with its own context window |
| Peripherals and tools | browser, Python interpreter, terminal | the simulator, `demonstrator/tools.py`, the GPUs, the web |
| fsck | "Lint" | consolidation |

## Tiers and budgets

| Tier | Files | Budget | Loads |
|---|---|---|---|
| Resident | `CLAUDE.md`; `MEMORY.md` in Claude's memory directory | 150 lines; 200 lines or 25 KB, which Claude Code enforces | every session; `CLAUDE.md` also in every subagent except Explore and Plan, and again after compaction |
| Boot | `state.md`, the index section above, the headings of the journal's last 8 entries, the checks | 6,000 + 2,000 characters, and the hook keeps its whole output under 9,800 | SessionStart: startup, resume, `/clear`, compaction, fork |
| On demand | topic files here, `.claude/rules/` | none fixed; split a section past about 40 lines into its own file | when a task needs them; a rule file when Claude reads a file it covers |
| Cold | `demonstrator/notes/`, `demonstrator/episodes/`, `REPORT.md`, `RELATED_WORK.md`, `consult/`, `docs/superpowers/`, `outputs/`, old journal months, session transcripts | none | when something cites them |

Why these sizes. Claude Code replaces hook output over 10,000 characters with a file path and a 2,000-character
preview, and doesn't ask Claude to read the file ([hooks reference][cc1]). Its docs advise keeping a CLAUDE.md under
200 lines, because longer files cost context and adherence ([memory docs][cc2]), and every subagent pays for
CLAUDE.md again: before 2026-09-30 it was 311 lines, loaded into each of the overnight run's agents.

Where each kind of fact goes:

- A command, a map line, or a rule whose absence breaks any script: `CLAUDE.md`.
- Where things stand: `state.md`. That something happened: `journal.md`.
- A habit, with the incident behind it: `operating-principles.md`.
- A lesson about searching, grasping, seeing or containers that would hold for other objects: the learnings file.
- A cube-only rule the players need: `demonstrator/instructions.md`, through its own change log.
- What a module does, or a number measured about it: `codebase.md`. A trap in one subsystem: its `.claude/rules/` file.
- Anything about the physical arm: `real-arm.md`. One episode's story: `demonstrator/notes/`.
- How dm1487 wants the work done, and corrections to how Claude works: Claude's memory directory, one file per memory
  with a line in `MEMORY.md`. Project facts stay out of it: it isn't versioned, and other sessions on other hosts
  can't see it.

## Boot

`.claude/settings.json` registers `.claude/hooks/knowledge.sh boot` as a SessionStart hook with no matcher, so it runs
for all five sources: startup, resume, `/clear`, compaction and fork. For SessionStart, Claude Code adds a hook's plain
stdout to Claude's context ([hooks reference][cc1]). The script prints:

1. One line saying where the text came from, plus an instruction for the source. After a compaction it asks Claude to
   write back what the summary holds and the files lack, before new work. On resume it warns that the files may be
   newer than the conversation.
2. `state.md`, headed by the file's own write time. The file's time is used rather than the stamp inside it, because
   the stamp is typed by hand: on 2026-09-30 it read 11:45 at 11:13.
3. The index section of this README, between the boot markers.
4. The headings of the journal's last 8 entries, which is Karpathy's `grep "^## \[" log.md | tail` tip.
5. The checks: sizes against budgets, entries since the last consolidation, and project files changed after
   `state.md` was last written.

If the whole is over 9,800 characters it drops journal headings first, then the index, and last cuts `state.md` and
says so. `CLAUDE.md` needs no hook: Claude Code loads it at start and re-reads it after compaction ([memory
docs][cc2]).

Tested on 2026-09-30. The script ran for all five sources in 0.06 to 0.08 s, with output of 9,100 to 9,300 characters. A
fake project checked each fallback: `state.md` at 7,000 characters dropped the journal headings and warned; at 12,000 it
was cut with a notice; with no boot markers it took the README's first 40 lines; with no `docs/knowledge/` it printed
nothing and exited 0. The nudge stayed silent when `state.md` was fresh or only episode files had changed. In a real
`claude -p` session Claude quoted the checks line and the last journal heading from its context without using a tool.
Another session, after one Read of `so101sim/ik.py`, quoted the first trap from `arm-and-motion.md`, and a third found
no rule file in its context at start.

## Write-back

Nothing here works if the files go stale, so writing back is part of the work, not a closing chore.

- `state.md` is edited in place. When something starts, it goes under "Running now" with what it is and where its output
  lands. When it finishes, it leaves "Running now", what it produced goes under "What exists", and a journal entry
  records it. A decision goes under "Decisions in force"; a superseded one comes out, with a journal entry that says so.
- `journal.md` only grows. Write an entry when a run starts or stops, when a result comes in (with numbers and a path),
  when something is decided and by whom, when dm1487 gives a direction, and when a mistake is found. Its header gives
  the entry format.
- Lessons go to their topic file as soon as they have evidence, not at the end of a session.
- In every file: date anything that can go stale, say where the evidence is, write plainly, and correct an entry
  rather than add a second one that contradicts it.
- A second hook, `knowledge.sh nudge` on UserPromptSubmit, prints one line when `state.md` is more than 20 minutes old
  and code, scripts, tests, docs, models or outputs changed after it. Otherwise it prints nothing. Episode files and
  notes don't count, since agents write them all through a batch.

## The journal as a replication record

dm1487, 2026-09-30: "maintain a log of every decision we make and also you make to maintain memory, because that is
what helps another person replicate the demos (information to instruct an LLM)". So the journal records every
decision, dm1487's and Claude's, with what was decided, why, who decided and where the evidence or code is. A run's
entry records what a replication needs: the commit and whether the tree was dirty, the MuJoCo version and physics
settings, the seeds and start kinds, the player and how many ran in parallel, the rulebook version, the exact commands,
and which directions came from dm1487. Per-move choices are the one exception, because every act in
`demonstrator/episodes/*.json` stores its own "why". Someone replicating starts from the journal, checks out the
commit a run names, hands their LLM `demonstrator/AGENT_BRIEF.md` and the rulebook at that commit, and plays the same
seeds. Entries written after the fact carry "(reconstructed)" in the title and name the files they came from.

## Consolidation

Karpathy's lint, for this folder. Run it when the boot checks warn (`state.md` over 6,000 characters, 25 journal entries
since the last consolidation, `CLAUDE.md` over 150 lines, the journal over 800 lines with entries from an earlier
month), at the end of a batch or milestone, before handing work to another session, or when dm1487 asks.

1. Run `bash .claude/hooks/knowledge.sh check`. It lists the budgets and the files changed after `state.md`.
2. Read the journal entries since the last `consolidation` entry. For each, check that its lasting part reached its
   home: a result into the learnings file or the rulebook, a trap into `.claude/rules/`, a change in a module into
   `codebase.md`, a decision into `state.md` and, if it changes a spec, into the spec's amendments.
3. Deduplicate. Grep for the key number or phrase across this folder, `.claude/rules/`, `CLAUDE.md` and the memory
   directory. Keep one home and replace the other copies with pointers. Copies drift: the learnings file's noslip
   line had mixed a 30 s and a 10 s window until 2026-09-30.
4. Correct contradictions in place, with "(corrected DATE: was X)".
5. Retire stale state. Finished work leaves "Running now"; answered questions leave "Open questions", with the answer in
   the journal; superseded decisions leave "Decisions in force".
6. Prune. Cube-only detail moves from the learnings file to the rulebook, and one episode's detail to its note. A
   section past about 40 lines becomes its own file, added to the index. When the journal passes 800 lines, entries
   from earlier months move unchanged to `journal/YYYY-MM.md`. The journal is never pruned otherwise.
7. Run the check again, then append a `consolidation` entry that says what moved where. That entry resets the count.

## Subagents

Subagents are separate processes with their own context windows. They load `CLAUDE.md`, except Explore and Plan,
but not `MEMORY.md` and not the boot hook's output. A fork inherits the parent's whole context ([subagent
docs][cc3]).

- The parent passes what the agent needs in its prompt: the task, the files to read, where to write, and a limit on
  the report's length. `demonstrator/AGENT_BRIEF.md` is the pattern.
- An agent writes only files it owns: a player writes its episode and its note, a research agent writes the report
  file it was given. It returns a short summary, not the raw material.
- Only the main session writes `state.md`, the journal, the topic files and the rule files. One writer avoids
  conflicting edits, and promoting a lesson stays a deliberate step: the parent reads the notes or report, moves
  general lessons into the learnings file and cube rules into the rulebook, and logs a journal entry.
- A batch's start entry in the journal records the rulebook and tool versions, which stay frozen for the batch.
- The parent checks what agents report before building on it. On 2026-09-30 a docs-lookup agent reported two things
  about hooks that the docs contradict (`operating-principles.md`, "Measure before asserting").

## Keeping it general

The cube bin is the proof-of-concept rig. Deeper bins up to sink-like basins, bins set into the table and YCB objects
or toys come later, with the same expert. So the learnings file states each lesson for any objects and containers and
names what is cube-only; cube numbers live in the rulebook and in the learnings file's last section. When a new object
set or container arrives, its specifics get their own section or rule file, and lessons that hold across sets move
up into the learnings file.

## Left out, and why

- **A PreCompact hook to write state back before compaction.** A PreCompact hook can block compaction, but it can't
  put text in front of Claude, and blocking a compaction triggered by a context-limit error fails the request ([hooks
  reference][cc1]). The write-back happens right after compaction instead, from the summary, prompted by the boot hook.
  Writing as things happen means a compaction loses little.
- **A PostCompact hook.** It receives the summary but can't add context, and the session transcript already stores it.
- **A Stop hook that forces a write-back.** It can refuse to let Claude stop, but it would fire after every turn,
  needs a loop guard, and invites filler entries. The one-line prompt-time nudge covers the same gap.
- **Injecting `state.md` into every subagent** with SubagentStart. Players and research agents need their brief, not
  the project's state, and each agent's context should hold one unit of work.
- **`@` imports in `CLAUDE.md`.** Imported files load in full at launch, which would put everything back in the
  resident tier ([memory docs][cc2]).
- **Search or embeddings over the docs.** At this size the index and `grep` find things; the LLM Wiki gist says the
  same up to hundreds of pages.

[k1]: https://x.com/karpathy/status/1707437820045062561
[k2]: https://x.com/karpathy/status/1723140519554105733
[k3]: https://www.youtube.com/watch?v=zjkBMFhNj_g
[k4]: https://www.youtube.com/watch?v=LCEmiRjPEtQ
[k5]: https://x.com/karpathy/status/1937902205765607626
[k6]: https://x.com/karpathy/status/1921368644069765486
[k7]: https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f
[cc1]: https://code.claude.com/docs/en/hooks
[cc2]: https://code.claude.com/docs/en/memory
[cc3]: https://code.claude.com/docs/en/sub-agents
