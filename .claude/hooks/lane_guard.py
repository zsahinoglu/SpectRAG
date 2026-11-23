#!/usr/bin/env python3
"""PreToolUse hook: keep an agent inside its lane.

Usage (from an agent's frontmatter hooks):
    lane_guard.py deny  eval/ docs/agents/ scripts/     # block edits under these paths
    lane_guard.py allow docs/theory/ experiments/       # allow edits only under these paths

Reads the hook JSON from stdin. Exit 0 = allow, exit 2 = block (stderr goes back to Claude).
Paths are compared relative to the git repository root of the hook's `cwd`, so the guard works
inside worktrees too.
"""
import json
import os
import subprocess
import sys


def repo_root(cwd: str) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", cwd, "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, check=True,
        )
        return os.path.realpath(out.stdout.strip())
    except Exception:
        return os.path.realpath(cwd)


def matches(rel: str, prefix: str) -> bool:
    return rel == prefix.rstrip("/") or rel.startswith(prefix)


def main() -> int:
    if len(sys.argv) < 3 or sys.argv[1] not in ("allow", "deny"):
        print("lane_guard.py: usage: lane_guard.py allow|deny PREFIX...", file=sys.stderr)
        return 2
    mode, prefixes = sys.argv[1], sys.argv[2:]

    data = json.load(sys.stdin)
    tool_input = data.get("tool_input") or {}
    path = tool_input.get("file_path") or tool_input.get("notebook_path")
    if not path:
        return 0

    cwd = data.get("cwd") or os.getcwd()
    abspath = os.path.realpath(path if os.path.isabs(path) else os.path.join(cwd, path))
    rel = os.path.relpath(abspath, repo_root(cwd))

    if rel.startswith(".."):
        print(f"Blocked by lane guard: {path} is outside the repository.", file=sys.stderr)
        return 2

    hit = any(matches(rel, p) for p in prefixes)
    if mode == "deny" and hit:
        print(
            f"Blocked by lane guard: your role may not edit {rel}. "
            f"Forbidden paths: {', '.join(prefixes)}. Stop and report if this edit is needed.",
            file=sys.stderr,
        )
        return 2
    if mode == "allow" and not hit:
        print(
            f"Blocked by lane guard: your role may only edit {', '.join(prefixes)}; "
            f"{rel} is outside that. Open an issue for the right role instead.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
