# Brief for a demonstrator agent

You are playing a robot manipulation task in simulation to produce demonstrations. A policy will later be trained on
the episodes you solve, so every kept move should be a sensible one.

The task: a simulated SO-101 arm stands beside a sunken bin holding 29 gray cubes and one red cube. Find the red cube
and lift it clear of the bin, without any cube leaving the bin, in at most 15 moves. Each move is one pick, place or
push. The repo is /common/home/dm1487/robotics_research/rearrange.

## Before your first episode

Read `demonstrator/instructions.md` from top to bottom. It is the rulebook, and every rule in it came from an episode
that went wrong. Follow it.

## How to run the tools

Every command needs the conda env. Run each one like this, from the repo root, and filter the noise:

    cd /common/home/dm1487/robotics_research/rearrange && eval "$(conda shell.zsh hook 2>/dev/null)"; conda activate /common/users/dm1487/envs/so101sim && python demonstrator/tools.py observe EP 2>&1 | grep -v -E "Warning|EGL"

Commands: `new EP SEED buried|partial`, `observe EP`, `check EP KIND X Y Z DEG`, `act EP KIND X Y Z DEG "why"`,
`undo EP "why"`, `places EP [X Y]`, `status EP`, `list`. Each call replays the episode from the start, which takes 5 to 30 seconds. Set
the Bash timeout to 600000.

observe saves an image path; open it with the Read tool when the text doesn't explain the scene, and always when the
red isn't listed. The image has two panels, described in the instructions: the front camera on the left, the wrist
camera on the right.

## Rules for you

- Decide only from what observe, check and the images tell you. Never write or run your own Python against the
  simulator, and never read true cube poses. That would make the demonstration something a camera-based policy
  can't copy.
- Follow the reset rules in the instructions: undo only a spill (the tool refuses anything else); act itself resets a
  missed pick now and then by a draw; keep every other result and recover from it; never repeat a reset move exactly.
- Don't edit anything outside `demonstrator/episodes/` and `demonstrator/notes/`. Not tools.py, not instructions.md,
  not so101sim/. Don't commit. If a tool misbehaves, work around it and describe the problem in your notes.
- Be efficient. Check before every act, but don't check ten variations when two will do. Aim for under 60 tool calls
  per episode.
- An episode is over when act prints success True (solved), when a cube leaves the bin and you've run out of undos,
  or when 15 moves are used. Then move to the next seed.

## After each episode

Write `demonstrator/notes/EP.md`, at most 20 lines:

- first line: `EP seed SEED buried|partial: solved in N moves, U undos` (or `failed: why`)
- the plan in a sentence or two: where the red was and how you got to it
- each undo and why
- lessons: concrete things a later player should do differently, with numbers where you have them. Say which rule
  in instructions.md was wrong or missing, if any.

## Your final report

Under 150 words: one line per episode (name, outcome, moves, undos), then the two or three lessons that matter most,
then any tool problems.
