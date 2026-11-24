"""Pairwise LLM judge (pinned model and prompt) and the contamination check."""
from __future__ import annotations

from eval import metrics, pins


def _verdict(raw: str) -> str:
    tok = (raw or "").strip().upper().strip(".:*` ")
    if tok.startswith("TIE"):
        return "TIE"
    return tok[:1] if tok[:1] in ("A", "B") else "INVALID"


def pairwise(llm, question: str, system_answer: str, baseline_answer: str) -> dict:
    """Judge both orders to cancel position bias. Score for the system:
    win 1, tie or invalid 0.5, loss 0; averaged over the two orders."""
    out = {"invalid": 0}
    score = 0.0
    for order, (a, b) in (("sys_first", (system_answer, baseline_answer)),
                          ("base_first", (baseline_answer, system_answer))):
        v = _verdict(llm.complete(pins.JUDGE_USER.format(question=question, a=a, b=b),
                                  system=pins.JUDGE_SYSTEM))
        sys_label = "A" if order == "sys_first" else "B"
        if v == "INVALID":
            out["invalid"] += 1
            score += 0.5
        else:
            score += 1.0 if v == sys_label else 0.5 if v == "TIE" else 0.0
    out["score"] = score / 2
    return out


def contamination(llm, questions) -> dict:
    """Does the index-time LLM produce gold answers from the question alone?
    Only aggregates are returned."""
    scored = [q for q in questions if q.answers]
    if not scored:
        return {"checked": 0, "note": "dataset has no gold answers"}
    em, contains = [], []
    for q in scored:
        pred = llm.complete(pins.CONTAMINATION_USER.format(question=q.question))
        em.append(metrics.exact_match(pred, list(q.answers)))
        contains.append(metrics.answer_contains(pred, list(q.answers)))
    return {"checked": len(scored), "closed_book_em": metrics.mean(em),
            "closed_book_contains": metrics.mean(contains)}
