# Reproducing the paper's numbers

**Status: not yet run.** No `SPECTRAG_ANTHROPIC_API_KEY` and no frozen test split were available
when the harness was built (E0a). This reproduction is follow-up item E0b.

## Plan

1. Pre-flight: contamination check per dataset (closed-book EM / contains, recorded in the result file).
2. `python -m eval run --release` on the six datasets with Claude Haiku (`claude-haiku-5-5`) as
   index-time LLM, generator and judge.
3. Compare with the paper's published table, which used gpt-4o-mini, two ways:
   - **Absolute gap**: our SpectRAG score minus the paper's SpectRAG score, per dataset.
   - **Model-independent gap**: our (SpectRAG − NaiveRAG) delta minus the paper's
     (SpectRAG − NaiveRAG) delta. Both systems share the generator and judge, so a model change
     moves both; the part of the absolute gap that disappears in the delta comparison is
     attributed to the model change; what remains is a reproduction gap in the retrieval code.
   - Retrieval-only metrics (evidence recall, P@5, MRR) depend on the index-time LLM only through
     concept and utility-question edges, so a gap there that survives the delta comparison
     points at code, not at the generator.
4. Never call OpenAI to close the gap; if an exact gpt-4o-mini comparison is needed, that is a
   human decision.

## Open inputs (needed from a human)

- `SPECTRAG_ANTHROPIC_API_KEY` for the evaluator.
- The dev and frozen test splits in the documented layout (`docs/evaluation/README.md`).
- The paper's per-dataset numbers for SpectRAG and NaiveRAG (not present in this repository),
  with the judge prompt and token budget it used.
