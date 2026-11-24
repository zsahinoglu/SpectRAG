"""Dataset loading. Layout per split root: `<root>/<dataset>/{corpus,questions}.jsonl`.

corpus.jsonl:    {"doc_id": str, "text": str, "title": str (optional)}
questions.jsonl: {"qid": str, "question": str, "answers": [str] (may be empty),
                  "evidence": [doc_id, ...]}

Error messages name files and line numbers only, never question or answer text, so
they are safe to print for the frozen test split.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


class DataError(RuntimeError):
    pass


@dataclass(frozen=True)
class Doc:
    doc_id: str
    text: str
    title: str = ""


@dataclass(frozen=True)
class Question:
    qid: str
    question: str
    answers: tuple[str, ...]
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class Dataset:
    name: str
    docs: tuple[Doc, ...]
    questions: tuple[Question, ...]

    @property
    def has_answers(self) -> bool:
        return any(q.answers for q in self.questions)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise DataError(f"missing file: {path}")
    rows = []
    with path.open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    raise DataError(f"{path}:{n}: not valid JSON") from None
    return rows


def load_dataset(root: Path, name: str, limit: int | None = None) -> Dataset:
    d = root / name
    docs = []
    for n, r in enumerate(_read_jsonl(d / "corpus.jsonl"), 1):
        if not isinstance(r.get("doc_id"), str) or not isinstance(r.get("text"), str):
            raise DataError(f"{d / 'corpus.jsonl'}:{n}: needs string 'doc_id' and 'text'")
        docs.append(Doc(r["doc_id"], r["text"], r.get("title", "") or ""))
    ids = {doc.doc_id for doc in docs}
    if len(ids) != len(docs):
        raise DataError(f"{d / 'corpus.jsonl'}: duplicate doc_id")
    qs = []
    for n, r in enumerate(_read_jsonl(d / "questions.jsonl"), 1):
        ok = (isinstance(r.get("qid"), str) and isinstance(r.get("question"), str)
              and isinstance(r.get("evidence"), list) and r["evidence"]
              and all(isinstance(a, str) for a in r.get("answers", [])))
        if not ok:
            raise DataError(f"{d / 'questions.jsonl'}:{n}: needs 'qid', 'question', "
                            "non-empty 'evidence' list, string 'answers'")
        if not set(r["evidence"]) <= ids:
            raise DataError(f"{d / 'questions.jsonl'}:{n}: evidence doc_id not in corpus")
        qs.append(Question(r["qid"], r["question"], tuple(r.get("answers", [])),
                           tuple(r["evidence"])))
    if not qs:
        raise DataError(f"{d / 'questions.jsonl'}: no questions")
    qs.sort(key=lambda q: q.qid)  # order-independent of file layout
    if limit:
        qs = qs[:limit]
    return Dataset(name, tuple(docs), tuple(qs))


def available(root: Path, names) -> list[str]:
    return [n for n in names if (root / n / "corpus.jsonl").is_file()]
