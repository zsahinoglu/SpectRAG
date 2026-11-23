#!/usr/bin/env bash
# Local task board for the SpectRAG agent team. Replaces GitHub issues, pull requests,
# labels and review comments with plain files under <spectrag-home>/board/items/<ID>/:
#   state  title  owner  deps  release  done  who  branch  sha  verdict  kind
#   log.md (append-only discussion)   submission.md (the "PR description")
#
# States: todo → in-progress → review → approved → merged
#         review/approved → blocked (fix and resubmit) · any → needs-human · any → deferred
#
# Agents:  claim ID · note ID [text|-] · ask ID "question" · submit ID [--release] < body
#          file "title" [--tag t] < body · show ID · list [state...] · deps-met ID
# Reviewer/evaluator/red-team:  verdict ID PASS|BLOCK < body · block ID "reason"
# Human:   answer ID "text" · defer ID "reason" · add ID "title" OWNER [deps] [release]
# Setup:   init   (creates one item per backlog row; safe to re-run)
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
ORIGIN="$(git -C "$ROOT" remote get-url origin 2>/dev/null || true)"
BOARD="${SPECTRAG_BOARD:-$(dirname "${ORIGIN:-$ROOT/x}")/board}"
WHO="${SPECTRAG_AGENT_NAME:-human}"
ZERO=0000000000000000000000000000000000000000

trim()  { sed 's/^[[:space:]]*//;s/[[:space:]]*$//'; }
die()   { echo "board: $*" >&2; exit 1; }
now()   { date '+%Y-%m-%d %H:%M'; }
idir()  { [ -d "$BOARD/items/$1" ] || die "no item '$1' (try: scripts/board.sh list)"; echo "$BOARD/items/$1"; }
get()   { cat "$BOARD/items/$1/$2" 2>/dev/null || true; }
put()   { printf '%s\n' "$3" > "$(idir "$1")/$2"; }
log()   { printf '\n### %s — %s\n%s\n' "$(now)" "$WHO" "$2" >> "$(idir "$1")/log.md"; }
state() { put "$1" state "$2"; log "$1" "**state → $2**${3:+ — $3}"; }
body()  { if [ "${1:-}" = "-" ] || [ -z "${1:-}" ]; then [ -t 0 ] && die "text expected on stdin"; cat; else printf '%s' "$1"; fi; }

new_item() {  # id title owner deps release done
  local d="$BOARD/items/$1"; [ -d "$d" ] && return 0
  mkdir -p "$d"
  printf '%s\n' "$2" > "$d/title"; printf '%s\n' "$3" > "$d/owner"; printf '%s\n' "${4:-—}" > "$d/deps"
  printf '%s\n' "${5:-}" > "$d/release"; printf '%s\n' "${6:-}" > "$d/done"; echo todo > "$d/state"
  printf '# %s: %s\n' "$1" "$2" > "$d/log.md"
  echo "created $1"
}

cmd="${1:-list}"; shift || true
case "$cmd" in
init)
  mkdir -p "$BOARD/items"
  while IFS='|' read -r _ id title rel owner deps done _; do
    id="$(trim <<< "$id")"; case "$id" in ""|ID|---*) continue ;; esac
    new_item "$id" "$(trim <<< "$title")" "$(trim <<< "$owner")" \
      "$(trim <<< "$deps")" "$(trim <<< "$rel")" "$(trim <<< "$done")"
  done < <(grep '^|' "$ROOT/docs/agents/BACKLOG.md")
  echo "board: $BOARD" ;;
add)
  [ $# -ge 3 ] || die "usage: add ID \"title\" OWNER [deps] [release]"
  new_item "$1" "$2" "$3" "${4:-—}" "${5:-}" "(see parent item)"; log "$1" "added by $WHO" ;;
file)
  [ $# -ge 1 ] || die "usage: file \"title\" [--tag t] < body"
  title="$1"; tag="${3:-}"; n=1; while [ -d "$BOARD/items/F$n" ]; do n=$((n+1)); done
  new_item "F$n" "$title" triage "—" "" "Triage: fix, defer, or turn into a backlog item"
  [ -n "$tag" ] && put "F$n" kind "$tag"
  log "F$n" "$(body -)" ;;
list|ls)
  [ -d "$BOARD/items" ] || die "board not initialised (scripts/board.sh init)"
  printf '%-8s %-12s %-16s %-9s %s\n' ID STATE OWNER WHO TITLE
  for d in "$BOARD"/items/*/; do
    id="$(basename "$d")"; s="$(get "$id" state)"
    if [ $# -gt 0 ] && ! printf '%s\n' "$@" | grep -qx "$s"; then continue; fi
    printf '%-8s %-12s %-16s %-9s %s\n' "$id" "$s" "$(get "$id" owner)" "$(get "$id" who)" "$(get "$id" title | cut -c1-50)"
  done ;;
ids)   # machine-readable: IDs in the given states
  for d in "$BOARD"/items/*/; do id="$(basename "$d")"
    printf '%s\n' "$@" | grep -qx "$(get "$id" state)" && echo "$id"; done; true ;;
get)   get "$(basename "$(idir "$1")")" "$2" ;;
show)
  d="$(idir "$1")"
  for f in title state owner who deps release branch sha verdict kind done; do
    [ -s "$d/$f" ] && printf '%-8s %s\n' "$f:" "$(cat "$d/$f")"; done
  [ -s "$d/submission.md" ] && { echo; echo "--- submission"; cat "$d/submission.md"; }
  echo; echo "--- log (last 60 lines)"; tail -n 60 "$d/log.md" ;;
claim)  idir "$1" >/dev/null; put "$1" who "$WHO"; state "$1" in-progress "claimed by $WHO" ;;
note)   log "$1" "$(body "${2:-}")" ;;
ask)    [ $# -ge 2 ] || die "usage: ask ID \"question\""; log "$1" "**QUESTION FOR HUMAN:** $2"
        put "$1" prev "$(get "$1" state)"; state "$1" needs-human ;;
answer) [ $# -ge 2 ] || die "usage: answer ID \"text\""; WHO=human; log "$1" "**ANSWER:** $2"
        p="$(get "$1" prev)"; state "$1" "${p:-todo}" "answered; re-run the agent" ;;
defer)  WHO=human; state "$1" deferred "${2:-}" ;;
submit)
  id="$1"; idir "$id" >/dev/null; kind=""; [ "${2:-}" = "--release" ] && kind=release
  cur="$(git branch --show-current)"
  case "$cur" in ""|main|master) die "submit from your item branch, not '${cur:-detached HEAD}'" ;; esac
  [ -z "$(git status --porcelain)" ] || die "commit your changes before submitting"
  br="$(get "$id" branch)"; br="${br:-$cur}"
  git push -q origin "HEAD:refs/heads/$br" || die "push failed. Never rebase or force-push a submitted branch; merge origin/main into it instead."
  put "$id" branch "$br"; put "$id" sha "$(git rev-parse HEAD)"; put "$id" verdict ""
  [ -n "$kind" ] && put "$id" kind "$kind"
  if [ -t 0 ]; then cp "$ROOT/docs/agents/SUBMISSION_TEMPLATE.md" "$(idir "$id")/submission.md"
  else cat > "$(idir "$id")/submission.md"; fi
  state "$id" review "branch $br @ $(git rev-parse --short HEAD)" ;;
verdict)
  [ $# -ge 2 ] || die "usage: verdict ID PASS|BLOCK < findings"
  id="$1"; v="$2"; sha="$(get "$id" sha)"; [ -n "$sha" ] || die "$id has no submission"
  text="$(body -)"; log "$id" "VERDICT: $v (for ${sha:0:9})"$'\n'"$text"
  put "$id" verdict "$v $sha"
  if [ "$v" = PASS ]; then state "$id" approved; elif [ "$v" = BLOCK ]; then state "$id" blocked "see review"; else die "PASS or BLOCK"; fi ;;
block)  [ $# -ge 2 ] || die "usage: block ID \"reason\""; put "$1" verdict ""; state "$1" blocked "$2" ;;
deps-met)
  unmet=()
  for d in $(get "$1" deps | tr ',' ' '); do
    case "$d" in "—"|"-"|decision|D[0-9]) continue ;; esac
    [ "$(get "$d" state)" = merged ] || unmet+=("$d")
  done
  if [ ${#unmet[@]} -eq 0 ]; then exit 0; else echo "${unmet[*]}"; exit 1; fi ;;
*) sed -n '2,16p' "$0"; exit 1 ;;
esac
