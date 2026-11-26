# T4 — Selection objective with guarantees

Status: proposal (T-props, written in Phase 1; prototype and ablation belong to T4 in v1.2).
Author: Fourier (theory). Red-team review pending (§7).
Shared evaluation rules: `docs/theory/ablation-protocol.md`. Related: T2 (score scale), T7
(when to stop).

> **Verification status.** WebSearch and Python were unavailable. Citations come from memory
> (**[unverified]** where unsure). I could **not** search the 2023–2026 RAG literature, and I
> expect submodular context selection for RAG has been published. §6 says what to check
> before anyone calls this new.

## 1. Question and what the code does today

**Question.** Stage 4 picks the context under a token budget. Can it instead maximize a stated
objective with a provable approximation guarantee, and does that retrieve more evidence?

**Today** (`rag_system/query/cluster_selection.py`). Six modes; none optimizes a stated
objective.

- `token_budget` (the default, `_token_budget_select`, lines 244–315):
  - sort by diffused score;
  - reorder near-ties to prefer uncovered documents (`_coverage_order`,
    `token_budget_tie_band = 0.02`);
  - fill to `token_budget = 12000` tokens with a lenient head floor (0.30 × top) for the first
    `token_budget_head_k = 5` chunks and a strict tail floor (0.55 × top) after that;
  - backfill gated-out chunks above the head floor.
  - That is five knobs, chosen by A/B on Legal (config comments).
- `mmr`, `protected_mmr`: greedy `λ·rel − (1−λ)·max cos`, with a fixed count `final_context_k`
  and no token budget.
- `greedy`: a weighted sum of relevance, redundancy, connectivity and cluster novelty, with a
  fixed count.
- `topk_diffused`, `anchor_expand_ce`: ranking-based.

## 2. Derivation

### 2.1 The objective

`C` is the candidate set. `s_i` is the Stage 3 score, `c_i` the tiktoken count of chunk `i`,
`B` the budget, and `e_i` the L2-normalized embeddings. Define

    F(S) = Σ_{i∈C} w_i · max_{j∈S} κ(i,j)              (relevance-weighted facility location)
    κ(i,j) = max(0, ⟨e_i, e_j⟩)^p,  κ(i,i) = 1
    w_i = exp((s_i − s_max)/T),  T = θ·(s_max − s_med)

Maximize `F(S)` subject to `Σ_{j∈S} c_j ≤ B`.

**Reading.** Each candidate `i` stands for a piece of query-relevant content worth `w_i`. A
context `S` "covers" it to the degree of its best similar member. A near-duplicate of an
already selected chunk adds almost nothing. A distinct relevant chunk adds about its own
weight.

**Parameters.** `p` (similarity sharpness) and `θ` (relevance temperature). `w` is invariant to
positive affine maps of `s` (`s ↦ λs + c` leaves `(s_i − s_max)/(s_max − s_med)` unchanged), so
`θ` transfers across embedders. This matters for the same reason as T2 D1. If
`s_max = s_med`, set `w ≡ 1`.

### 2.2 Properties (proved)

**P1 (monotone submodular).** For each `i`, `g_i(S) = max_{j∈S} κ(i,j)` (with `g_i(∅) = 0`) has
marginal gain `(κ(i,x) − g_i(S))_+` for adding `x`. That gain is nonnegative and nonincreasing
in `S`. A nonnegative sum of such functions keeps both properties. ∎

**P2 (nests today's core ordering).** Assume distinct embeddings (`⟨e_i, e_j⟩ < 1` for
`i ≠ j`). As `p → ∞`, `κ(i,j) → 𝟙[i = j]`, so `F(S) → Σ_{j∈S} w_j` (modular). Cost-benefit
greedy then orders by `w_j/c_j`. When chunk sizes are equal, that is the order of `s`, which
is `token_budget` with its floors, tie-band and backfill removed. ∎

**Theorem 1 (cost-benefit greedy, small items).** Greedy repeatedly picks
`argmax_j Δ_j(G)/c_j` over **all** remaining candidates. Let `G_l` be its prefix before the
first pick that would exceed `B`, and let `ε = max_j c_j / B`. Then

    F(G_l) ≥ (1 − e^{−(1−ε)}) · OPT.

Continuing greedily with only the items that still fit can only increase `F` (P1).

*Proof.* Let `G_{t−1}` be the greedy set before step `t`, which picks `j_t`. By P1,
`OPT − F(G_{t−1}) ≤ Σ_{j∈S*} Δ_j(G_{t−1})`. That sum is
`≤ (Δ_{j_t}/c_{j_t})·c(S*) ≤ (Δ_{j_t}/c_{j_t})·B`, because `j_t` has the best ratio. Hence
`OPT − F(G_t) ≤ (1 − c_{j_t}/B)(OPT − F(G_{t−1}))`, and

`OPT − F(G_l) ≤ Π_t(1 − c_{j_t}/B)·OPT ≤ e^{−c(G_l)/B}·OPT`.

The next pick does not fit, so `c(G_l) > B − c_{max} ≥ (1−ε)B`. ∎

This is the argument of Khuller, Moss & Naor (1999) and Krause & Guestrin (2005), specialized
to small items.

**Numbers.** `chunk_size = 1200` and `token_budget = 12000` give `ε ≤ 0.1`, so the guarantee is
`F ≥ 0.593·OPT`. The prototype computes `ε` from the actual `c_max` per query, because table
chunks may exceed `chunk_size` [unchecked], and reports it.

**Cost.** Lazy greedy (Minoux 1978) with the `|C| × |C|` similarity matrix (`|C| ≤ 200`). That
is one `200 × 768` matrix product plus `O(|C|²)` updates, negligible next to Stage 3.

**What the guarantee is not.** It bounds `F` against the best feasible set **for the proxy
`F`**. It says nothing about evidence recall or answer quality. Those are empirical (§3). A
guarantee on a proxy is worth having only if the proxy correlates with recall, and Step 0
tests that.

### 2.3 Optional document-coverage term (still submodular)

    F_doc(S) = Σ_{docs d} √( Σ_{j∈S∩d} w_j )

This is the diversity reward of Lin & Bilmes (2011): a concave function of a modular function,
hence monotone submodular. `F + γF_doc` keeps Theorem 1. It replaces the ad-hoc
`token_budget_tie_band` document promotion with a term whose trade-off is explicit (`γ`).

### 2.4 Why it might raise evidence recall (CONJECTURE M)

On multi-evidence questions, the second-hop evidence document is usually less similar to the
query than further chunks of the first-hop document. Score-ordered filling therefore spends
budget on near-duplicates of hop 1 before reaching hop 2. Facility location discounts those
near-duplicates (P1), so hop 2 enters earlier. This is a conjecture about the data, not a
theorem. Step 0 measures its premise.

## 3. Predicted effect

All comparisons are at **equal `context_tokens`**, because F always fills the budget.

| Metric | Direction | Datasets | Basis |
| --- | --- | --- | --- |
| `n_selected_docs` | up | all | P1 (redundancy discount) |
| `evidence_recall` | up (≥ 1 pt) | multihoprag, musique, 2wikimultihopqa, hotpotqa | CONJECTURE M |
| `precision_at_5` | within noise or down ≤ 1 pt | all | the first greedy picks are the highest-weight chunks |
| `evidence_recall` | within noise | ultradomain, legal | fewer multi-document questions [assumption about these datasets] |
| number of Stage-4 knobs | 5 → 2 (`p`, `θ`), +1 with `γ` | — | construction |
| `query_p95_ms` | ≈ unchanged | all | §2.2 cost |

**Refutation.**

- Step 0 shows a crowding rate below 5% of questions: no target, skip the paid runs and
  reject.
- At equal tokens, recall does not move up on at least two of the four multi-hop datasets:
  reject M.
- P@5 drops more than 2 points on any dataset: reject that arm.

## 4. Ablation plan (dev splits)

- **Step 0 (no paid calls beyond the baseline run).** From the baseline's selected contexts
  and the dev evidence labels, compute the **crowding rate**: the share of questions where an
  evidence document is among the candidates but not selected, while two or more selected
  chunks come from one non-evidence or already-covered document. Also compute the rank
  correlation of `F(S)` with per-question evidence recall over random feasible `S`. This checks
  that the proxy is meaningful.
- **Step 1 (paid, `--limit 100`).** The arms:
  - S0: `token_budget` (today).
  - S1: F with `p ∈ {1, 4}` and `θ ∈ {0.5, 2}`.
  - S2: the best S1 plus `F_doc` with `γ ∈ {0.5, 1}`.
  - S3 (control for P2): F with `p = 64` (≈ modular). It should match S0 without gates. If S3
    alone beats S0, the gain comes from removing the gates, not from coverage.
  - Metrics: recall, P@5, MRR, `n_selected_docs`, `context_tokens`. Holm correction across
    arms.
- **Step 2 (paid, full dev).** The best S1/S2 against S0, adding F1 and the judge win rate.
- **Interaction.** Repeat the best arm with T2's diffusion if T2 is approved, since `w` depends
  on `s`.

## 5. Prototype

None in this submission. Planned: `experiments/T4/facility_location.py`, a lazy-greedy
selector with the same signature as `ClusterAwareSelector.select`, plus the Step 0 crowding
script. The backlog allows "or a paper draft" for T4. That choice comes after the ablation.

## 6. Prior art (nothing here is claimed as new)

- Nemhauser, Wolsey & Fisher (1978), Math. Prog. 14: `(1 − 1/e)` for cardinality-constrained
  monotone submodular maximization. Minoux (1978): lazy greedy.
- Khuller, Moss & Naor (1999), IPL 70: budgeted maximum coverage. Sviridenko (2004), ORL 32:
  `(1 − 1/e)` under a knapsack constraint with partial enumeration. Krause & Guestrin (2005),
  CMU tech report: the cost-benefit lemma used in Theorem 1. Leskovec et al. (2007), KDD
  (CELF).
- Lin & Bilmes (2010, NAACL; 2011, ACL): budgeted submodular extractive summarization,
  facility-location coverage plus concave diversity reward. T4 is essentially their objective
  applied to RAG context selection.
- Carbonell & Goldstein (1998), SIGIR: MMR. To my knowledge it has no approximation guarantee
  for a stated set objective.
- Krause & Golovin (2014), *Submodular function maximization* (survey).
- **To check before claiming anything** (not searchable this session): recent work on
  submodular, coverage-based or diversity-aware context selection for RAG (2023–2026). The
  red team or a later session should search this. If it exists, T4 is an application of it.

## 7. Open questions and red-team log

- Q1: should the similarity `κ` use embeddings (proposed) or spectral rows? Embeddings are
  available for every chunk. Spectral rows would couple T4 to T1/T6.
- Q2: how does F stop before the budget? That is T7's question. T4 always fills the budget.

Red-team objections: pending.
