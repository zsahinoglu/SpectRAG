"""Helpers for pasting untrusted document text into prompts (H7 / A13).

Document text is attacker-controlled. Two defences live here, both independent of
the model resisting an injected instruction:

* `fence()` wraps a passage in a tag carrying a fresh random token, so the passage
  cannot reproduce the closing tag (it cannot know the token). The prompts tell the
  model that everything inside the tag is data.
* `clean_concepts()` / `clean_questions()` hold model output to the contract stated
  in the prompt before it becomes a retrieval key.
"""
from __future__ import annotations

import re
import secrets

MAX_CONCEPTS = 10          # the concept prompt asks for 5-10
MAX_CONCEPT_CHARS = 64     # "2-4 words each"
MAX_QUESTION_CHARS = 300
_QUESTION_ENDINGS = ("?", "？", "؟")


def fence(text: str, tag: str = "untrusted_text") -> str:
    """`text` between an opening and closing tag that include a per-call random token."""
    token = secrets.token_hex(8)
    return f"<{tag} id={token}>\n{text}\n</{tag} id={token}>"


def clean_concepts(raw: list[str]) -> list[str]:
    """Normalised, de-duplicated concept keys: at most MAX_CONCEPTS, one short line each.

    Over-long or multi-line entries are dropped (not truncated). The model's own
    order decides which MAX_CONCEPTS survive.
    """
    out: list[str] = []
    for c in raw:
        c = c.strip().lower()
        if not c or "\n" in c or "\r" in c or len(c) > MAX_CONCEPT_CHARS or c in out:
            continue
        out.append(c)
        if len(out) == MAX_CONCEPTS:
            break
    return sorted(out)


def clean_questions(raw: str, n: int) -> list[str]:
    """Up to `n` questions from a model reply: one per line, list markers stripped.

    Lines that are not questions (preamble, injected instructions) or are too long
    are dropped.
    """
    out: list[str] = []
    for ln in raw.strip().splitlines():
        ln = re.sub(r"^\s*(?:\d+[.)]|[-*•])\s*", "", ln.strip())
        if ln and len(ln) <= MAX_QUESTION_CHARS and ln.endswith(_QUESTION_ENDINGS):
            out.append(ln)
    return out[:n]
