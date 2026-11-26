# Shared ablation protocol for T1–T4, T6, T7

Status: proposal support document (T-props). Author: Fourier (theory).

Each proposal's §4 names its arms. This page fixes what all of them share, so each proposal
need not repeat it.

## Splits and commands

- **Dev splits only** (`python -m eval run --dev`). Nobody uses the frozen test split for any
  ablation here. A winning arm reaches `--release` only as a preset, through the tuner (I4/I1),
  after the item is approved.
- Datasets: `multihoprag`, `ultradomain`, `musique`, `legal`, `hotpotqa`, `2wikimultihopqa`.
- **Needed from the evaluator (not available today).** `python -m eval run` has no
  config-override flag (`eval/__main__.py` accepts only `--datasets`, `--limit`, `--save`). Each
  arm below is a `Config` variant. The evaluator decides how arms are passed in (for example a
  `--config-override key=value` flag or a named-variant registry in `eval/`). Theory does not
  edit `eval/`. Several arms also need a few lines of prototype code. That code lives under
  `experiments/T<n>/` until the item is approved.

## Statistics

- **Paired comparison against the baseline arm on the same questions.** The unit is the
  question. Report the mean difference and a 95% paired-bootstrap CI (10 000 resamples, seed
  from `eval/pins.py`).
- **"Within noise"** means the CI contains 0. **"Moves"** means the CI excludes 0 in the
  predicted direction.
- **Multiple arms.** When a proposal tests more than three arms, apply Holm's correction across
  arms within that proposal. That keeps a grid search from producing a "win" by chance.
- **Cross-fitting.** Any arm that fits a parameter on dev labels (T7 calibration, any
  data-chosen grid point) is scored with 5-fold cross-fitting by question: fit on 4 folds,
  score the 5th, rotate. Picking the best grid point on the same questions it is scored on
  is not allowed.

## Metrics used across proposals

The following are already defined in `docs/evaluation/README.md`: `evidence_recall`,
`precision_at_5`, `mrr`, `f1`, `judge_win_rate_vs_naiverag`, `top10_stability`, `query_p50_ms`,
`query_p95_ms`, `build_time_s`.

Additional diagnostics are requested below. None of them needs an LLM call; each can be
computed from a built index.

| Diagnostic | Used by | Definition |
| --- | --- | --- |
| `context_tokens` | T4, T7 | mean and p95 tiktoken count of the selected context per question |
| `spectrum` | T1, T3, T6 | the `k_max + 2` smallest eigenvalues of the Laplacian actually used |
| `ipr` | T6 | inverse participation ratio `Σ_i u_l(i)^4` per stored eigenvector (high = localized) |
| `deg_hist` | T6 | degree histogram of the combined graph, and per layer in multiplex mode |
| `n_selected_docs` | T4 | distinct `doc_id`s in the selected context |

## Cost discipline

Run the LLM-free diagnostics first. Several proposals give a **pre-check that can reject the
idea before any paid run** (T3 §4 step 0, T6 §4 step 0). Paid `--dev` runs use `--limit` for
the grid stage, and run the full dev split only for the best arm against the baseline.
