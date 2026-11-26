# T3 — Eigengap choice of k

Status: proposal (T-props, written in Phase 1; prototype and ablation belong to T3 in v1.1).
Author: Fourier (theory). Red-team review pending (§7).
Shared evaluation rules: `docs/theory/ablation-protocol.md`. Builds on T5 (drift bound) and T1.

> **Verification status.** WebSearch and Python were unavailable. Citations come from memory
> (**[unverified]** where unsure). Claims marked **CONJECTURE** are not proved.

## 1. Question and what the code does today

**Question.** SpectRAG fixes two spectral sizes by hand. Should either be chosen from the
spectrum at the largest eigengap?

1. **`k` = `IndexingConfig.n_spectral_components = 32`.** This is the dimension of
   `chunk.spectral_coords`. `rag_system/indexing/spectral.py::_solve` requests `k + 1`
   eigenpairs and drops the bottom one.
2. **`K` = `IndexingConfig.n_offline_clusters = 5`.** This is the KMeans cluster count in
   `rag_system/indexing/cluster_assigner.py::SpectralClusterAssigner.assign`. There is also a
   query-time fallback `RetrievalConfig.n_clusters = 5`.

**Fact that limits K's relevance.** `chunk.cluster_label` is read at query time only in
`cluster_selection.py:415–418`, which is the `selection_mode == "greedy"` branch. The default
`token_budget` mode, and the `mmr`, `protected_mmr`, `topk_diffused` and `anchor_expand_ce`
modes, never read it. Outside the index file (`spectrag/formats/files.py:130`), no other
consumer exists in `rag_system/` or `spectrag/`. So **choosing K has no effect on retrieval in
the default configuration.** The K part of T3 matters only if `greedy` survives I4 (presets).

## 2. Derivation

Notation as in T5 §2.0, on a connected graph. The eigenvalues are
`0 = λ_0 < λ_1 ≤ λ_2 ≤ …` of `L_sym`. The window for dimension `k` is `{λ_0..λ_k}`, and its
**boundary gap** is `δ_k = λ_{k+1} − λ_k`.

### 2.1 Why the boundary gap is the right criterion for k (proved, via T5)

**P1 (update drift).** T5 Theorem 1 (Davis–Kahan) bounds the subspace drift under an update
`E` by `‖sinΘ‖ ≤ ‖EU‖/(δ̂_k − ‖E‖_2)`. The T5 rebuild trigger fires when `e₂ ≥ δ̂_k`. For the
same corpus and the same update, a larger `δ_k` gives a proportionally smaller bound and a
larger update size before H4 must re-solve or rebuild.

**P2 (solver determinism).** If a solver returns `X` with residual `R = LX − XΘ`, then by the
residual form of Davis–Kahan (T5 §2.4), `‖sinΘ(X, U_k)‖ ≤ ‖R‖/δ̃_k` with
`δ̃_k ≈ δ_k`. With `spectral_lobpcg_tol = 1e-4`, two runs that differ only in start vector,
BLAS thread order or machine can return subspaces that differ by up to `~1e-4/δ_k` per
direction. That is negligible at `δ_k = 0.05` and of order 1 at `δ_k = 1e-4`.

**P3 (cutting through a degenerate cluster).** If `λ_k = λ_{k+1}` exactly, the window is not
unique. Every `(k+1)`-dimensional subspace that contains `span{u_0..u_{k−1}}` and lies inside
`span{u_0..u_{k+1}}` is an equally valid output, and `ZZᵀ`, and therefore every cosine in
Stage 3, depends on which one the solver returns. *Proof:* rotate `u_k` and `u_{k+1}` inside
their common eigenspace; both rotated vectors are eigenvectors. ∎ Near-degeneracy is the
continuous version (P2).

Convergence speed points the same way. The rate at which Lanczos/LOBPCG converge on the
boundary eigenpairs is governed by the relative gap near `λ_k` (Saad, *Numerical Methods for
Large Eigenvalue Problems*; Knyazev 2001 [chapter/theorem unverified]). So a cut at a large gap
also costs fewer iterations for the same tolerance.

### 2.2 Rule A — choose k at the largest gap in a range

    k* = argmax_{k_min ≤ k ≤ k_max} δ_k        (ties → larger k);  default k_min = 16, k_max = 48.

- This needs `k_max + 2` eigenpairs, one solve. The cost grows roughly linearly to
  quadratically in the number of pairs, from Lanczos matvecs plus reorthogonalization.
  **CONJECTURE C:** `build_time_s` of the spectral step at most doubles (measured in §4).
- **Absolute, not relative, gap.** The update size `‖E‖_2` that the gap must dominate is
  absolute. `L_sym` has spectrum in `[0, 2]` on every corpus, so absolute gaps are comparable
  across corpora.
- **With T1 (heat weighting) on.** `k` must also make the truncation tail small (T1 P3):
  `k ≥ k_ε = min{k : e^{−t(λ_{k+1}−λ_1)} ≤ ε}`. Rule A then searches
  `[max(k_min, k_ε), k_max]`.
- The bound `k_min` exists because the gap criterion knows nothing about retrieval quality.
  Spectra of graphs with community structure often have their largest gap at very small `k`,
  and a 3-dimensional embedding would be stable but useless.

**Pre-check that can refute Rule A before any paid run.** Gap salience
`g = max_{k∈[k_min,k_max]} δ_k / median_{k∈[k_min,k_max]} δ_k`. If `g < 3`, there is no
distinguished gap in the range, and `k*` is the noisiest of near-equal gaps. **CONJECTURE G
(my prior, stated so it can be checked):** on the larger dev corpora (`n ≳ 10⁴`), sparse
kNN-plus-section graphs have a dense spectrum around `λ_16..λ_48`, `g < 3`, and Rule A will be
rejected there. It has a better chance on small corpora and on single-domain corpora with
clear topical blocks.

### 2.3 Rule B — choose K by the eigengap heuristic (only matters for `greedy`)

    K* = argmax_{2 ≤ K ≤ K_max} λ_K / λ_{K−1}       (K_max = 10).

Justification, from published results:

- **Higher-order Cheeger** (Lee, Oveis Gharan & Trevisan 2014): `λ_{K−1}/2 ≤ ρ(K)` and
  `ρ(K) ≤ O(K²)·√λ_{K−1}`, where `ρ(K)` is the best `K`-way expansion. Here I use 0-based
  indexing, so `λ_{K−1}` is the K-th smallest eigenvalue. The constants are as I remember
  them, **[unverified]**.
- **Peng, Sun & Zanetti (2015/2017).** If `Υ(K) = λ_K/ρ(K)` is large, spectral clustering with
  `K` clusters recovers the planted partition with misclassified volume `O(K³/Υ)·vol`
  [exponent unverified].
- From the upper Cheeger bound, `Υ(K) ≥ λ_K / (O(K²)·√λ_{K−1})`. So a large ratio
  `λ_K/λ_{K−1}` is necessary for, and with the Cheeger upper bound partially certifies, a
  clusterable `K`-way structure.

Rule B, with the ratio in place of `Υ`, is the **heuristic** von Luxburg (2007, §8.3)
describes. It is not a theorem: the `O(K²)` slack means a moderate ratio certifies nothing.
Report the ratio. If the best ratio is below 2, keep `K = 5`. **This is a heuristic with
partial theoretical support, not a proved rule.**

## 3. Predicted effect

| Metric | Direction | Datasets | Basis |
| --- | --- | --- | --- |
| update drift `‖sinΘ‖` in 1% replay; share of updates passing T5 Thm 1(a) | better at `k*` than at `k = 32`, by the ratio `δ_{k*}/δ_32` | datasets with `g ≥ 3` | P1 (bound) |
| `top10_stability` | up or equal | datasets with `g ≥ 3` and `δ_32` small | P2 |
| `build_time_s` (spectral step) | up, ≤ 2× | all | CONJECTURE C |
| recall, P@5, MRR | within noise (T1 off); unknown sign (T1 on) | all | `k` changes the coordinate dimension. No theory predicts the sign. |
| `greedy`-mode recall (Rule B) | unknown | only if `greedy` survives I4 | — |

**Refutation.**

- Step 0 gives `g < 3` on all six datasets: reject Rule A outright (no paid run).
- `k*` loses more than 1 point on any quality metric on any dataset, against both `k = 32` and
  the matched-size control (C2 below): reject.
- `k*` is no more stable than a fixed `k` of similar size (C2): the gap is not what matters;
  reject.

## 4. Ablation plan (dev splits)

- **Step 0 (no LLM).** Compute `λ_0..λ_49` for each dev dataset's combined graph. Report
  `δ_k`, `g`, `k*`, `δ_32`, the ratio `δ_{k*}/δ_32`, and the Rule B ratios `λ_K/λ_{K−1}` for
  `K ≤ 10`.
- **Step 1 (no LLM, drift replay; the same script as T1 Step 1).** For `k ∈ {32, k*, 16, 24, 48}`
  measure:
  - `‖sinΘ‖_F` under 1% random document removal (5 seeds);
  - the share of simulated updates with `e₂ < δ̂_k`;
  - top-10 proximity overlap between two solves started from different random seeds
    (LOBPCG) and between ARPACK and LOBPCG.
- **Step 2 (paid, `--limit 100`; only datasets with `g ≥ 3`).** The arms:
  - C0: `k = 32`.
  - C1: `k = k*`.
  - C2: the fixed `k ∈ {16, 24, 48}` nearest to `k*` that is **not** at a large gap. This is
    the control that separates "a different `k`" from "a `k` at a gap".
  - Metrics: recall, P@5, MRR, `top10_stability`, `build_time_s`.
  - Then repeat C0/C1 with T1's `τ*`, if T1 is approved.
- **Step 3 (Rule B, only if `greedy` is kept after I4).** `selection_mode = greedy` with
  `K = 5` against `K*`; recall, P@5, `n_selected_docs`.

## 5. Prototype

None in this submission. Planned: `experiments/T3/spectrum_report.py` for Steps 0 and 1. It
needs only a built index and no LLM.

## 6. Prior art (nothing here is claimed as new)

- The eigengap heuristic: von Luxburg (2007), *A tutorial on spectral clustering*, Stat.
  Comput. 17, §8.3. Ng, Jordan & Weiss (2002), NIPS. The scree test (Cattell 1966) is the
  PCA analogue.
- Zelnik-Manor & Perona (2004), *Self-tuning spectral clustering*, NIPS: argue that the
  eigengap is unreliable on noisy data and choose K by an eigenvector-alignment cost instead.
  This is relevant to CONJECTURE G, and an alternative for Rule B if the ratio test is
  inconclusive.
- Lee, Oveis Gharan & Trevisan (2014), JACM (STOC 2012): higher-order Cheeger inequalities.
  Peng, Sun & Zanetti (2015, COLT; SICOMP 2017): *Partitioning well-clustered graphs:
  spectral clustering works!*
- Davis & Kahan (1970). Stewart & Sun, *Matrix Perturbation Theory* (1990): gap-dependence of
  invariant-subspace perturbation, the basis of P1 and P2.
- Saad (2011, 2nd ed.); Knyazev (2001), SIAM J. Sci. Comput. 23: convergence of Lanczos and
  LOBPCG in terms of gaps.
- Applying the stability criterion to the dimension of a *retrieval* embedding, as opposed to
  the cluster count, is not something I have seen stated. It is a direct application of the
  standard results above and is not claimed as a contribution.

## 7. Open questions and red-team log

- Q1: should `k*` be recomputed on every rebuild (and stored in the manifest), or fixed per
  corpus at first build? A changing `k` changes stored-row dimension, which H4 must handle.
  The recommendation is to fix it at the first full build and re-choose only on full rebuilds.
  That is an on-disk-format consequence for the architect.

Red-team objections: pending.
