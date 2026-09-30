#!/bin/bash
# Project memory hooks for the rearrange repo. docs/knowledge/README.md explains the system.
#
#   knowledge.sh boot    SessionStart: print state.md, the index and the journal's last entries, plus budget checks.
#                        Claude Code adds this stdout to Claude's context at startup, resume, /clear and compaction.
#   knowledge.sh nudge   UserPromptSubmit: print one line when state.md looks stale, nothing otherwise.
#   knowledge.sh check   By hand, during consolidation: budgets and the files changed since state.md was written.
#
# Hook stdout over 10,000 characters is swapped for a file path and a 2,000-character preview, so boot keeps its
# output under BOOT_CAP by dropping journal entries first, then the index, then the tail of state.md.
# Never blocks: every path exits 0.
set -u

MODE="${1:-boot}"
ROOT="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." 2>/dev/null && pwd)}"
K="$ROOT/docs/knowledge"
STATE="$K/state.md"
INDEX="$K/README.md"
JOURNAL="$K/journal.md"

BOOT_CAP=9500          # characters (counted as bytes, which is stricter)
STATE_BUDGET=6000      # characters
INDEX_BUDGET=2500      # characters in README's boot section
CLAUDE_MD_BUDGET=150   # lines
JOURNAL_ROTATE=400     # lines
CONSOLIDATE_AFTER=25   # journal entries since the last consolidation
NUDGE_AFTER=1200       # seconds since state.md was written before nudge speaks
JOURNAL_ENTRIES=4      # entries shown at boot

[ -d "$K" ] || exit 0
NOW=$(date +%s)

mtime() { stat -c %Y "$1" 2>/dev/null || date -r "$1" +%s 2>/dev/null || echo 0; }
stamp() { date -d "@$1" '+%Y-%m-%d %H:%M' 2>/dev/null || date -r "$1" '+%Y-%m-%d %H:%M' 2>/dev/null; }
age() {
  local s=$1
  if [ "$s" -lt 3600 ]; then echo "$((s / 60)) min"
  elif [ "$s" -lt 172800 ]; then echo "$((s / 3600)) h $(((s % 3600) / 60)) min"
  else echo "$((s / 86400)) days"; fi
}
nbytes() { wc -c < "$1" 2>/dev/null | tr -d ' ' || echo 0; }

# Files that record work, changed after state.md was last written. Episode files and notes are left out: agents
# write them continuously during a batch, and the batch itself is what state.md tracks.
changed_since_state() {
  [ -f "$STATE" ] || return 0
  cd "$ROOT" || return 0
  {
    find so101sim scripts tests demonstrator docs .claude \
      \( -path demonstrator/episodes -o -path demonstrator/notes -o -name '__pycache__' \) -prune -o \
      -type f -newer "$STATE" ! -name '*.pyc' ! -path docs/knowledge/state.md ! -path docs/knowledge/journal.md \
      -print 2>/dev/null
    find models outputs -maxdepth 1 -newer "$STATE" ! -name models ! -name outputs ! -name '.candidate_*' -print 2>/dev/null
    [ CLAUDE.md -nt "$STATE" ] && echo CLAUDE.md
  } | sort -u
}

journal_count_since_consolidation() {
  [ -f "$JOURNAL" ] || { echo 0; return; }
  awk '/^## \[/ { n++ } /^## \[[^]]*\] consolidation \|/ { n = 0 } END { print n + 0 }' "$JOURNAL"
}

checks() {
  local warn="" s_bytes i_bytes c_lines j_lines j_since changed n_changed
  s_bytes=$(nbytes "$STATE")
  i_bytes=$(index_section | wc -c | tr -d ' ')
  c_lines=$(wc -l < "$ROOT/CLAUDE.md" 2>/dev/null | tr -d ' ' || echo 0)
  j_lines=$(wc -l < "$JOURNAL" 2>/dev/null | tr -d ' ' || echo 0)
  j_since=$(journal_count_since_consolidation)
  changed=$(changed_since_state)
  n_changed=$(printf '%s' "$changed" | grep -c . )
  echo "state.md ${s_bytes}/${STATE_BUDGET} chars; index section ${i_bytes}/${INDEX_BUDGET}; CLAUDE.md ${c_lines}/${CLAUDE_MD_BUDGET} lines; journal ${j_lines} lines, ${j_since} entries since the last consolidation (consolidate at ${CONSOLIDATE_AFTER})."
  [ "$s_bytes" -gt "$STATE_BUDGET" ] && warn="${warn}- state.md is over budget: move finished work to the journal and history out of state.md.\n"
  [ "$i_bytes" -gt "$INDEX_BUDGET" ] && warn="${warn}- The README index section is over budget: shorten it; detail belongs below the boot markers.\n"
  [ "$c_lines" -gt "$CLAUDE_MD_BUDGET" ] && warn="${warn}- CLAUDE.md is over budget: move material into docs/knowledge/ or .claude/rules/ and leave a pointer.\n"
  [ "$j_lines" -gt "$JOURNAL_ROTATE" ] && warn="${warn}- journal.md passed ${JOURNAL_ROTATE} lines: move earlier months to docs/knowledge/journal/YYYY-MM.md.\n"
  [ "$j_since" -ge "$CONSOLIDATE_AFTER" ] && warn="${warn}- ${j_since} journal entries since the last consolidation: run the consolidation in docs/knowledge/README.md.\n"
  [ "$n_changed" -gt 0 ] && warn="${warn}- ${n_changed} project files changed after state.md was last written, e.g. $(printf '%s' "$changed" | head -3 | paste -sd, - | sed 's/,/, /g'). Check that state.md covers that work.\n"
  if [ -n "$warn" ]; then printf 'Warnings:\n%b' "$warn"; else echo "No warnings."; fi
  if [ "$MODE" = check ] && [ "$n_changed" -gt 0 ]; then
    echo "Changed after state.md:"; printf '%s\n' "$changed" | sed 's/^/  /'
  fi
}

index_section() {
  [ -f "$INDEX" ] || return 0
  if grep -q '<!-- boot:start -->' "$INDEX"; then
    awk '/<!-- boot:end -->/ { exit } f { print } /<!-- boot:start -->/ { f = 1 }' "$INDEX"
  else
    head -n 40 "$INDEX"
  fi
}

journal_tail() {  # last $1 entries of the journal
  [ -f "$JOURNAL" ] && [ "$1" -gt 0 ] || return 0
  awk -v n="$1" '
    /^## \[/ { k++ }
    { line[NR] = $0; entry[NR] = k }
    END { for (i = 1; i <= NR; i++) if (entry[i] > k - n) print line[i] }' "$JOURNAL"
}

cap_lines() {  # keep whole lines up to $1 bytes
  head -c "$1" | awk -v max="$1" '{ total += length($0) + 1; if (total <= max) print }'
}

boot() {
  local src="" input
  if [ ! -t 0 ]; then
    input=$(head -c 65536)
    src=$(printf '%s' "$input" | sed -n 's/.*"source"[[:space:]]*:[[:space:]]*"\([a-z]*\)".*/\1/p' | head -1)
  fi
  [ -n "$src" ] || src=startup

  local intro
  case "$src" in
    compact) intro="The conversation was just compacted, and the summary above drops detail. Before new work, compare the summary with the state below and write back to docs/knowledge/state.md and journal.md whatever they lack: work started or finished, results with their numbers, decisions, dm1487's instructions. Then carry on." ;;
    resume|fork) intro="This conversation is being resumed. The files below may be newer than anything in it: where they disagree, check the journal entries dated after the conversation's last message before acting." ;;
    *) intro="Read the state below before acting. It says what the goal is, what exists, what is running and what is waiting on dm1487." ;;
  esac

  local s_mt state_hdr state_body idx jn checks_out head body total n
  s_mt=$(mtime "$STATE")
  state_hdr="=== docs/knowledge/state.md (file written $(stamp "$s_mt"), $(age $((NOW - s_mt))) ago) ==="
  state_body=$(cat "$STATE" 2>/dev/null || echo "(state.md is missing)")
  idx=$(index_section)
  checks_out=$(checks)
  head="Project memory for rearrange, loaded by .claude/hooks/knowledge.sh (SessionStart, source: ${src}). ${intro}"

  assemble() {  # $1 = number of journal entries, $2 = include index (1/0)
    printf '%s\n\n%s\n%s\n\n' "$head" "$state_hdr" "$state_body"
    if [ "$2" = 1 ]; then printf '=== docs/knowledge/README.md, index section ===\n%s\n\n' "$idx"; fi
    if [ "$1" -gt 0 ]; then
      printf '=== docs/knowledge/journal.md, last %s entries ===\n%s\n\n' "$1" "$(journal_tail "$1")"
    fi
    printf '=== checks ===\n%s\n' "$checks_out"
  }

  for n in $JOURNAL_ENTRIES 2 0; do
    body=$(assemble "$n" 1)
    total=$(printf '%s' "$body" | wc -c | tr -d ' ')
    [ "$total" -le "$BOOT_CAP" ] && { printf '%s\n' "$body"; return 0; }
  done
  body=$(assemble 0 0)
  total=$(printf '%s' "$body" | wc -c | tr -d ' ')
  if [ "$total" -le "$BOOT_CAP" ]; then
    printf '%s\n(The index was left out to stay under the hook output cap. Read docs/knowledge/README.md.)\n' "$body"
    return 0
  fi
  printf '%s\n\n%s\n' "$head" "$state_hdr"
  printf '%s\n' "$state_body" | cap_lines $((BOOT_CAP - ${#head} - ${#state_hdr} - 400))
  printf '\n(state.md was cut off here to stay under the hook output cap. Read docs/knowledge/state.md in full, then consolidate it: it is over budget.)\n'
}

nudge() {
  [ -f "$STATE" ] || exit 0
  cat > /dev/null 2>&1 || true   # drain stdin
  local s_mt since changed n
  s_mt=$(mtime "$STATE")
  since=$((NOW - s_mt))
  [ "$since" -lt "$NUDGE_AFTER" ] && exit 0
  changed=$(changed_since_state)
  n=$(printf '%s' "$changed" | grep -c . )
  [ "$n" -eq 0 ] && exit 0
  echo "[project memory] docs/knowledge/state.md was last written $(age "$since") ago ($(stamp "$s_mt")), and ${n} project files changed since, e.g. $(printf '%s' "$changed" | head -3 | paste -sd, - | sed 's/,/, /g'). If that work started, finished or decided anything, update state.md and append to journal.md."
}

case "$MODE" in
  boot) boot ;;
  nudge) nudge ;;
  check) checks ;;
  *) echo "usage: knowledge.sh boot|nudge|check" >&2 ;;
esac
exit 0
