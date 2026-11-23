#!/usr/bin/env python3
"""PreToolUse hook: keep non-evaluator agents away from the frozen test split.

Blocks any tool call whose input mentions the private evaluation directory. This is a
best-effort guard on top of the main protection, which is that the test split lives outside
the repository (default ~/spectrag-private/) and only the evaluator's scripted runs point at it.
"""
import json
import os
import sys

raw = sys.stdin.read()
try:
    data = json.loads(raw)
except json.JSONDecodeError:
    sys.exit(0)

needles = {"spectrag-private"}
test_dir = os.environ.get("SPECTRAG_TEST_DIR")
if test_dir:
    needles.add(os.path.basename(os.path.normpath(test_dir)))

text = json.dumps(data.get("tool_input") or {}).lower()
for needle in needles:
    if needle and needle.lower() in text:
        print(
            "Blocked by private guard: the frozen test split is evaluator-only. "
            "Use dev splits (python -m eval run --dev) instead.",
            file=sys.stderr,
        )
        sys.exit(2)
sys.exit(0)
