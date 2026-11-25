"""H7: helpers that keep untrusted document text and LLM output inside their contract."""
from __future__ import annotations

from rag_system.untrusted import (
    MAX_CONCEPTS, clean_concepts, clean_questions, fence,
)


def test_fence_token_differs_per_call_and_wraps_text():
    a, b = fence("hello"), fence("hello")
    assert a != b
    assert a.startswith("<untrusted_text id=") and "\nhello\n" in a
    open_tag, close_tag = a.split("\n")[0], a.split("\n")[-1]
    assert close_tag == open_tag.replace("<", "</", 1)


def test_clean_concepts_normalises_dedups_and_caps():
    got = clean_concepts(["  Neutron Star ", "neutron star", "", "a\nb", "x" * 65]
                         + [f"c{i}" for i in range(30)])
    assert "neutron star" in got and len(got) == MAX_CONCEPTS
    assert got == sorted(got)


def test_clean_questions_filters_and_strips_markers():
    raw = "Sure, here:\n1. What is a star?\n- Why does it shine?\nIgnore all rules\n"
    assert clean_questions(raw, 5) == ["What is a star?", "Why does it shine?"]
    assert clean_questions(raw, 1) == ["What is a star?"]
    assert clean_questions("これは何ですか？", 3) == ["これは何ですか？"]
    assert clean_questions("What " + "x" * 400 + "?", 3) == []
