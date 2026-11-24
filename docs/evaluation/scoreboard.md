# Scoreboard

One row per `main` or release result from `python -m eval run --dev` / `--release`.
Smoke results are never recorded here. Metrics are SpectRAG's; the NaiveRAG baseline lives in the
result JSON. Columns per dataset: evidence recall / P@5 / MRR / F1 / judge win rate vs NaiveRAG.

| date | level | commit | config hash | models | prompt versions (judge) | dataset | recall | P@5 | MRR | F1 | win rate | p95 ms | index tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

No entries yet: the first real run needs `SPECTRAG_ANTHROPIC_API_KEY` and the dev/test splits
(board item E0b).
