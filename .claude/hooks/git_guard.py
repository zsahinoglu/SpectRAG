#!/usr/bin/env python3
"""PreToolUse hook for Bash: block git operations and tooling no agent should run.

Blocks: pushes to main/master, pushes while on main/master, force/mirror/delete pushes,
merging or pulling into main, the human-only scripts (merge gate, release, setup), any
reference to the merge token, the local origin repository or the board's files (agents use
scripts/board.sh), and rm -rf aimed outside the working tree. Exit 2 = block.

These are tripwires against mistakes, not a security boundary: agents run as your OS user.
The local origin's pre-receive hook is the second line of defense for main.
"""
import json
import re
import subprocess
import sys

data = json.load(sys.stdin)
cmd = (data.get("tool_input") or {}).get("command") or ""
cwd = data.get("cwd") or "."


def block(reason: str) -> None:
    print(f"Blocked by git guard: {reason}", file=sys.stderr)
    sys.exit(2)


def current_branch() -> str:
    try:
        out = subprocess.run(["git", "-C", cwd, "branch", "--show-current"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return ""


HUMAN_ONLY = r"(merge-gate|release-local|setup-local|merge-token|SPECTRAG_MERGE_TOKEN|origin\.git|pre-receive|board/items|receive\.deny)"
if re.search(HUMAN_ONLY, cmd):
    block("that touches human-only tooling (merge gate, release, origin repo, merge token or "
          "raw board files). Use scripts/board.sh for the board; merging is done for you.")

for part in re.split(r"&&|\|\||;|\||\n", cmd):
    p = part.strip()
    if not p:
        continue

    if re.search(r"\bgit\b.*\bpush\b", p):
        if re.search(r"(\s--force(-with-lease)?\b|\s-f\b|\s--mirror\b|\s--delete\b|\s-d\b)", p):
            block("force, mirror and delete pushes are not allowed. Push a new branch instead.")
        after_push = p.split("push", 1)[1].split()
        refspecs = [t for t in after_push if not t.startswith("-")][1:]  # drop the remote name
        protected = {"main", "master", "refs/heads/main", "refs/heads/master"}
        for ref in refspecs:
            src, _, dst = ref.partition(":")
            if ref.startswith(":"):
                block("deleting remote branches is not allowed.")
            if src.lstrip("+") in protected or dst in protected or ref.startswith("+"):
                block("pushing to main (or forcing with +) is not allowed. Submit with scripts/board.sh submit.")
            if src == "HEAD" and not dst and current_branch() in ("main", "master"):
                block("you are on main. Create a branch (git switch -c <role>/<item>-<slug>) first.")
        if not refspecs and current_branch() in ("main", "master"):
            block("you are on main. Create a branch (git switch -c <role>/<item>-<slug>) first.")

    if re.search(r"\bgit\b.*\b(merge|pull|cherry-pick|reset)\b", p) and current_branch() in ("main", "master"):
        block("agents never change main. Work on your item branch; the merge gate merges for you.")
    if re.search(r"\bgit\b.*\b(update-ref|branch\s+(-f|-D|-M|--force))\b.*\b(main|master)\b", p):
        block("agents never move or delete main.")
    if re.search(r"\bgit\b.*\bremote\s+(add|set-url|remove|rename)\b", p):
        block("changing remotes is human-only.")
    if re.search(r"\brm\s+-[a-zA-Z]*r[a-zA-Z]*f?\b.*\s(/|~|\.\.)", p) or re.search(r"\brm\s+-rf\s*$", p):
        block("rm -rf outside the working tree is not allowed.")

sys.exit(0)
