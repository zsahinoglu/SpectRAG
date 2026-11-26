# T2 — One parameterized diffusion replacing the 50/50 hybrid

Status: proposal (T-props, written in Phase 1; prototype and ablation belong to T2 in v1.1).
Author: Fourier (theory). Red-team review pending (§7).
Shared evaluation rules: `docs/theory/ablation-protocol.md`. Related: T1 (heat filter), T6
(normalization), T7 (score scale).

> **Verification status.** WebSearch and Python were unavailable. Citations come from memory
> (**[unverified]** where unsure). Every claim marked "proved" has its proof here.
> **CONJECTURE** marks the rest.

## 1. Question and what the code does today

**Question.** Stage 3 mixes three signals with fixed weights. Can one diffusion operator with
one parameter replace them, and is it better?

**Today** (`rag_system/query/diffusion.py::RelevanceDiffuser.diffuse`, lines 105–145):

    s_c = (1 − α)·r_c + α·( ½·spec_c + ½·ppr_c / max_v ppr_v ),   α = 0.5, damping a = 0.85.

- `r = all_scores`. In `seed_retrieval.py:237–249` this is dense cosine, or with
  `hybrid_lexical_weight = 0.3` it is `0.7·cos/max cos + 0.3·bm25_norm`. It can be negative.
- `spec_c ∈ [−1, 1]` is the cosine of spectral rows, aggregated by `max` over seeds (T1).
- `ppr = nx.pagerank(subgraph, alpha=a, personalization=uniform over seeds)` runs on the
  candidate subgraph returned by `neo4j_client.subgraph` (`query/pipeline.py:65`).
- There are four knobs: `α`, `a`, the fixed ½/½ mix and the seed aggregator.

## 2. Derivation

**Notation.** `C` is the candidate set, `W` the weighted adjacency of the induced subgraph, and
`d = W𝟙`. Let `T = D^{-1}W` (row-stochastic), with the **lazy convention** `T_ii = 1` when
`d_i = 0`. `r ∈ ℝ^C` is the relevance restricted to candidates.

### 2.1 Three defects of the current blend (proved)

**D1: the effective α depends on the spread of relevance.** For `λ > 0` the ranking by `s` is
the ranking by `r + (α/(1−α))·m`, where `m` is the proximity term. If the candidate relevances
are `r′ = λr + c𝟙` (another query, embedder or `hybrid_lexical_weight`), the ranking equals the
one obtained with `α_eff = α/(α + (1−α)λ)`.
*Proof:* `(1−α)(λr + c) + αm` has the same ordering as `(1−α)λr + αm`. ∎

*Illustration (not measured).* If the candidate relevances span 0.07 for one query and 0.7 for
another, the effective weight of the graph term differs 10× between the two queries, and no
single `α` can tune for both. Step 0 of §4 measures the real spread.

**D2: the PPR term is proportional to degree.** Let `M_a = (1−a)(I − aT)^{-1}`. Row `v` of `M_a`
is the PPR vector of a walk restarted at `v`. NetworkX returns `π_x = xᵀM_a` for
personalization `x`. With `S = D^{-1/2}WD^{-1/2}` we have `M_a = D^{-1/2}ND^{1/2}` for the
symmetric `N = (1−a)(I − aS)^{-1}`. Hence `M_a(s,v)/d_v = M_a(v,s)/d_s` (the symmetry of
Andersen, Chung & Lang 2006), and

    π_x(v) = Σ_s x_s M_a(s,v) = d_v · ( M_a (x ⊘ d) )(v)      (every candidate has d > 0).

(NetworkX's dangling-node rule only matters for candidates with `d = 0`. Those are excluded
here.)

So today's PPR score is **`d_v` times** a degree-free diffusion of `x/d`. Within the candidate
subgraph, high-degree chunks (hubs) are boosted linearly in degree. That is the hub effect R1
tests for (A13), here coming from the scoring formula rather than from the data.

**D3: the restart ignores relevance.** `personalization = {sid: 1/|present|}` (line 92). A seed
admitted by BM25 or the utility-question list at rank 20 restarts with the same mass as the
top dense hit.

### 2.2 The proposal: relevance diffusion with a row-stochastic filter

    y = M r,   with  M = h(I − T)  and  h ∈ { PPR: h_a(μ) = (1−a)/(1 − a + aμ),  heat: h_t(μ) = e^{−tμ} }.

For PPR, `M = M_a = (1−a)Σ_m a^m T^m`. For heat, `M = e^{−t(I−T)} = e^{−t}Σ_m (t^m/m!) T^m`.
The final Stage 3 score is `s = y`. There is **one parameter** (`a` or `t`).

**Proposition 1 (proved).** For both choices of `h`:

- (a) **Fixed point / interpretation (PPR).** `y = (1−a)·r + a·Ty`. The score is
  `(1−a)·own relevance + a·(weighted mean of the neighbors' scores)`. This is today's
  "(1−α)·relevance + α·proximity" with `α = a`, but the proximity is in the same units as `r`.
- (b) **Row-stochastic.** `M ≥ 0` entrywise and `M𝟙 = 𝟙`. *Proof:* `T ≥ 0` and `T𝟙 = 𝟙` (the
  lazy convention covers `d_i = 0`). Both series have nonnegative coefficients summing to 1. ∎
- (c) **Affine equivariance.** `M(λr + c𝟙) = λMr + c𝟙`. The ranking is invariant under every
  positive affine map of `r`, so D1 disappears and negative relevances need no special
  handling (compare F1).
- (d) **Max principle.** `min r ≤ y_i ≤ max r`. No max-normalization is needed, and
  ratio-of-top floors in Stage 4 keep their meaning (T7).
- (e) **Degree neutrality.** If `r = c𝟙` then `y = c𝟙`. No chunk gains score from degree
  alone. This removes D2.
- (f) **Same family as T1.** `T = D^{-1/2}SD^{1/2}` and `I − S = L_sym` on the subgraph, so
  `M = D^{-1/2}h(L_sym)D^{1/2}`. `h` is a low-pass spectral filter, exactly as in T1.

**Proposition 2 (global low-rank variant, proved).** Let `U_r = [u_0..u_k]` be the stored
**full-graph** window **including** `u_0 = D^{1/2}𝟙/√vol` (T5 R1). Define
`M_g = D^{-1/2}U_r f(Λ_r)U_rᵀD^{1/2}` with `f(0) = 1`, for example `f = e^{−tλ}`. Then
`M_g𝟙 = 𝟙`, so (c) affine equivariance holds.
*Proof:* `U_rᵀD^{1/2}𝟙 = √vol·e_0`, because `u_l ⊥ u_0` for `l ≥ 1`. Then
`D^{-1/2}u_0·f(0)·√vol = 𝟙`. ∎

`M_g` can have **negative entries** (truncation), so (b) and (d) can fail; that is the price of
truncation. `y_g = M_g r` uses `all_scores` over **all** `N` chunks: two `N × (k+1)` matrix
products, `O(Nk)` vectorized per query, with no Python loop (H6 rule A10). It sees graph
structure outside the 2-hop candidate subgraph, which the local variant cannot.

Relation to today's spectral term: `spec_c` is a cosine-max of rows of `U`, a nonlinear,
scale-free statistic. `y_g` is the linear diffusion of the relevance field through the same
eigenbasis. The two share the subspace and differ in what they compute.

### 2.3 What is conjectured

- **CONJECTURE R1.** Removing D2 lowers the share of hub chunks among the top 5 and raises
  P@5 on multi-hop corpora, where seeds touch broad "overview" chunks.
- **CONJECTURE R2.** Because of (c), the dev-optimal `a` varies less across the six datasets
  than the dev-optimal `α` does today. This matters for I4 (presets chosen by worst-domain
  score).
- **CONJECTURE R3.** Local and global diffusion are complementary. If both win alone, a
  composition `M_local M_g` is the next thing to test. It is still one filter family, but it
  has two parameters.

## 3. Predicted effect

| Metric | Direction | Datasets | Basis |
| --- | --- | --- | --- |
| number of Stage-3 knobs | 4 → 1 (local) or 2 (local + global) | — | construction |
| `query_p95_ms` | down | all | a dense `|C|×|C|` solve (`|C| ≤ 200`) replaces `nx.pagerank`'s Python iteration; measured, size unknown |
| `precision_at_5`, `mrr` | up | multihoprag, musique, 2wikimultihopqa, hotpotqa | R1 |
| worst-domain score at one shared parameter | up | all six | R2 |
| `evidence_recall` | within noise | all | Stage 2 candidates unchanged; only reordering |

**Refutation.**

- The best single-parameter arm loses more than 1 point of `evidence_recall` or P@5 on two or
  more datasets against B0: reject.
- The best arm ties B0 everywhere **and** the cross-dataset spread of the optimal parameter is
  no smaller than today's `α` spread: reject R2 and keep the change only if latency improves.

## 4. Ablation plan (dev splits)

- **Step 0 (no LLM; embeddings are cached).** On the baseline, log the per-query spread
  `max_C r − min_C r` and the Spearman correlation between degree and `ppr` on the candidates.
  This measures how large D1 and D2 are in practice. If the spread varies by less than 2×
  across queries **and** the correlation is below 0.2, D1 and D2 are negligible; run only
  B1–B3.
- **Step 1 (paid, `--limit 100`).** The arms:
  - B0: baseline (`hybrid`, `α = 0.5`, `a = 0.85`).
  - B1: local PPR diffusion, `a ∈ {0.3, 0.5, 0.7, 0.85}`.
  - B2: local heat diffusion, `t ∈ {1, 2}`.
  - B3: global low-rank diffusion, `τ ∈ {1, 2, τ*}` (T1 notation). Needs stored eigenvalues,
    `u_0` and degrees (T5 R1).
  - B4 and B5 (attribution only): today's formula with relevance personalization `x ∝ r_+`
    (fixes D3 only), and with `ppr/d` (fixes D2 only).
  - Metrics: recall, P@5, MRR, F1, `query_p95_ms`, and the mean degree of the top-5 chunks.
  - Holm correction across arms.
- **Step 2 (paid, full dev).** The best B1/B2 arm and the best B3 arm against B0, adding
  `judge_win_rate_vs_naiverag`. Report the per-dataset optimum of the parameter (for R2).
- **Interaction.** Run B-best with `selection_mode = token_budget` (the default) and
  `topk_diffused`. Floors in `token_budget` are relative to the top score, and Proposition 1(d)
  changes the score distribution they act on.

## 5. Prototype

None in this submission. Planned for v1.1: `experiments/T2/relevance_diffusion.py`, a drop-in
`diffuse()` with the local (dense solve) and global (low-rank) variants, plus the Step 0
diagnostics.

**API note for the architect.** `ChunkExplanation.score_parts` (I5) has the keys
`spectral_proximity` and `ppr_proximity`. A unified diffuser would report something like
`diffused_local` / `diffused_global`. Naming is the architect's decision before ARCH3; T2 does
not choose.

## 6. Prior art (nothing here is claimed as new)

- Page, Brin, Motwani & Winograd (1999); Jeh & Widom (2003); Haveliwala (2002): personalized
  PageRank.
- Andersen, Chung & Lang (2006), *Local graph partitioning using PageRank vectors*, FOCS: the
  symmetry `pr_u(v)/d_v = pr_v(u)/d_u` and degree-normalized sweeps. D2 is a direct
  consequence.
- Zhou, Bousquet, Lal, Weston & Schölkopf (2004), *Learning with local and global consistency*,
  NIPS: the `(I − αS)^{-1}y` label spreading.
- Zhu, Ghahramani & Lafferty (2003), ICML: harmonic functions. These are the averaging
  (row-stochastic) view used in Proposition 1.
- Chung (2007), PNAS: heat-kernel PageRank `e^{−t(I−T)}`. Kloster & Gleich (2014), KDD: its
  local computation.
- Gasteiger (Klicpera), Weißenberger & Günnemann (2019), *Diffusion improves graph learning*,
  NeurIPS: generalized graph diffusion `Σ θ_m T^m`, with PPR and heat as its two standard
  members. T2's family is a special case. Klicpera, Bojchevski & Günnemann (2019, ICLR, APPNP)
  propagate predictions with PPR in the same way.
- Proposition 2 (keeping `u_0` makes the truncated operator preserve constants) is a one-line
  observation. I know no specific reference and do not claim it as a contribution.

## 7. Open questions and red-team log

- Q1: the `T` normalization (row-stochastic vs symmetric `S`). Only the row-stochastic choice
  has (b)–(e). The symmetric `S` gives `y ∝ d^{1/2}` on constant input. T6 owns the general
  question; T2 fixes the row-stochastic choice and says why.

Red-team objections: pending.
