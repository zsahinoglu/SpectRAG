"""Pure metric functions. Evidence is labelled at document level (doc_id)."""
from __future__ import annotations

import re
import string
from collections import Counter

import numpy as np


def evidence_recall(selected_doc_ids: list[str], evidence: list[str]) -> float:
    """Share of gold evidence documents present in the token-budgeted context."""
    gold = set(evidence)
    return len(gold & set(selected_doc_ids)) / len(gold) if gold else 0.0


def precision_at_k(ranked_doc_ids: list[str], evidence: list[str], k: int = 5) -> float:
    """Share of the top-k ranked chunks that come from an evidence document (denominator k)."""
    gold = set(evidence)
    return sum(d in gold for d in ranked_doc_ids[:k]) / k


def reciprocal_rank(ranked_doc_ids: list[str], evidence: list[str]) -> float:
    gold = set(evidence)
    for i, d in enumerate(ranked_doc_ids, 1):
        if d in gold:
            return 1.0 / i
    return 0.0


def normalize_answer(s: str) -> str:
    """SQuAD/HotpotQA normalisation."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(prediction: str, golds: list[str]) -> float:
    p = normalize_answer(prediction)
    return float(any(p == normalize_answer(g) for g in golds)) if golds else 0.0


def _f1(prediction: str, gold: str) -> float:
    p, g = normalize_answer(prediction).split(), normalize_answer(gold).split()
    common = sum((Counter(p) & Counter(g)).values())
    if not p or not g or common == 0:
        return float(p == g)
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def f1(prediction: str, golds: list[str]) -> float:
    return max((_f1(prediction, g) for g in golds), default=0.0)


def answer_contains(prediction: str, golds: list[str]) -> float:
    """Gold answer appears inside the prediction (used for the contamination check)."""
    p = normalize_answer(prediction)
    return float(any(normalize_answer(g) and normalize_answer(g) in p for g in golds))


def topk_overlap(a: list[str], b: list[str], k: int = 10) -> float:
    """|top-k(a) ∩ top-k(b)| / k' where k' = max list length capped at k (1.0 for two empties)."""
    a, b = a[:k], b[:k]
    denom = max(len(a), len(b))
    return len(set(a) & set(b)) / denom if denom else 1.0


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q)) if values else 0.0


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0
