# T6 — Normalization effects on the Laplacian

Status: proposal (T-props, written in Phase 1; prototype and ablation belong to T6 in v1.1).
Author: Fourier (theory). Red-team review pending (§7).
Shared evaluation rules: `docs/theory/ablation-protocol.md`. Related: T5 (R1, R2, A1), F6
(multiplex dropped column), T1, T3.

> **Verification status.** WebSearch and Python were unavailable. Citations come from memory
> (**[unverified]** where unsure). Every "proved" claim has its proof here. **CONJECTURE**
> marks the rest.

## 1. Question and what the code does today

**Question.** Which Laplacian normalization should SpectRAG use, and which choices actually
change retrieval?

**Today.**

- **Combined mode (default).** `rag_system/indexing/spectral.py::compute` uses
  `L_sym = I − D^{-1/2}AD^{-1/2}` (`scipy.sparse.csgraph.laplacian(A, normed=True)`) on the
  combined graph from `graph_builder.py::GraphBuilder.build`.
- **Multiplex mode.** `compute_multiplex` builds `L = Σ_s β_s L_sym,s`, one normalized
  Laplacian per signal. A node with no edge in layer `s` gets a **zero** row in `L_sym,s`.
  That is scipy's convention for isolated nodes (T5 A1; T5's ablation Step 0 checks it).
- **Consumers.** Stage 3 takes the cosine of rows (`diffusion.py:45–64`). KMeans runs on the
  **raw** rows (`cluster_assigner.py:37–42`). The KMeans labels are read only in `greedy`
  mode (T3 §1).

## 2. Derivation

### 2.1 P1 — For Stage 3, `L_sym` versus `L_rw` makes no difference (proved)

`L_rw = D^{-1}L = I − D^{-1}A` and the generalized problem `Lv = λDv` (Shi & Malik) have the same
eigenvalues as `L_sym`, with eigenvectors `v_l = D^{-1/2}u_l`. So the row of chunk `i` changes
by the positive scalar `d_i^{-1/2}`, and cosines are invariant to positive row scaling. ∎

**Consequence.** For Stage 3 proximity, the only normalization choices that matter are those
that change the eigenvectors themselves: unnormalized vs normalized (§2.3), regularization
(§2.4) and multiplex aggregation (§2.5). Debating `sym` versus `rw` for Stage 3 is moot.

### 2.2 P2 — KMeans on raw `L_sym` rows is degree-biased; row normalization fixes it (proved, ideal case)

Take a graph with `K` connected components `C_1..C_K` and the bottom `K` eigenvectors. The null
space of `L_sym` is spanned by `D^{1/2}𝟙_{C_j}/√vol(C_j)`, so every orthonormal basis of it is
this matrix times an orthogonal `Q`. Row `i ∈ C_j` is therefore `√(d_i/vol(C_j))·Q[j,:]`.

The rows of one cluster lie on a ray, at distance `∝ √d_i` from the origin. Low-degree chunks
of **different** clusters all sit near 0, and KMeans (Euclidean) can merge them.

After row normalization, every row of `C_j` equals `Q[j,:]` exactly, and the clusters are `K`
orthonormal points. ∎ (This is the Ng–Jordan–Weiss argument; von Luxburg 2007, §7.)

**Recommendation N3:** run KMeans on L2-normalized rows. That is the same geometry Stage 3
already uses (cosine). It affects `greedy` mode and the visualization only.

### 2.3 P3 — Do not switch to the unnormalized `L = D − A` (proved bound + cited theory)

**Proved.** Let `S` be a set of `m` chunks that are pairwise non-adjacent and have degree
`≤ d*`. The principal submatrix `L[S,S]` is diagonal with entries `d_i ≤ d*`. Cauchy interlacing
then gives `λ_{m−1}(L) ≤ d*`. If `m ≥ k + 1`, the whole stored window has eigenvalues `≤ d*`.
It is pinned by low-degree chunks, whatever the community structure.

Von Luxburg, Belkin & Bousquet (2008, Ann. Stat. 36) prove that unnormalized spectral
clustering can be inconsistent when the relevant eigenvalues are not below `min d`. Normalized
clustering is consistent under general conditions. The recommendation is to keep the
normalized Laplacian.

### 2.4 P4 — Regularization against small and weakly attached sets (main proposal)

**Dangling sets pin the window (proved).** For a set `S`, take the test vector
`x = D^{1/2}𝟙_S`. Then `xᵀL_sym x = cut(S)` and `‖x‖² = vol(S)`, so the Rayleigh quotient
equals the conductance `φ(S)`. If `S_1..S_m` are disjoint and pairwise non-adjacent, the
quadratic form decouples on `span{x_j}`, and Courant–Fischer gives
`λ_{m−1}(L_sym) ≤ max_j φ(S_j)`.

In SpectRAG, a document whose chunks are bound by adjacency and section edges but have few
cross-document edges is exactly such a set. With 33 or more such documents, the entire stored
window has eigenvalues `≤` the conductance of the most isolated documents.

**CONJECTURE P4-loc** (supported by Zhang & Rohe 2018, not proved here): the corresponding
eigenvectors localize on those documents. The Stage 3 spectral term then mostly encodes
"same weakly attached document", not cross-document topic structure.

Two published regularizers act on this:

**(a) Rank-one regularization** (Amini, Chen, Bickel & Levina 2013):
`A_τ = A + (τ/n)𝟙𝟙ᵀ`, followed by the usual `L_sym` of `A_τ`.

- *Proved:* `cut_τ(S) = cut(S) + τ|S|(n−|S|)/n` and `vol_τ(S) = vol(S) + τ|S|`. So with
  `d̄_S = vol(S)/|S|`,

      φ_τ(S) ≥ τ(1 − |S|/n) / (d̄_S + τ).

  A small set can no longer have small conductance, however isolated it is.
- For a balanced split (`|S| = n/2`, `cut = c·vol`), the same algebra gives
  `φ_τ = (c·d̄ + τ/2)/(d̄ + τ)`. At `τ = d̄` that is `≈ 1/4 + c/2`, against `≈ 1/2` for a small
  isolated set. **Large-scale structure now sits below small sets in the Rayleigh-quotient
  ordering** (Zhang & Rohe's CoreCut view).
- *Proved, engineering:* `A_τ` is the adjacency of a connected graph. So
  `λ_0 = 0` is simple with `u_0 = D_τ^{1/2}𝟙/‖·‖` in closed form (T5 R1 works unchanged), and
  T5's A1 failure modes (components, isolated nodes) disappear.
- The eigensolver never materializes the dense matrix:
  `A_τx = Ax + (τ/n)𝟙(𝟙ᵀx)`, one extra `O(n)` term per matvec.
- **Conflict to resolve.** Isolated chunks get nonzero rows, which collides with H5's "zero
  spectral credit". Keep H5's behavior by masking their rows to zero after the solve (T5 R2).

**(b) Degree-shift regularization** (Chaudhuri, Chung & Tsiatas 2012; Qin & Rohe 2013):
`L_τ = I − D_τ^{-1/2}AD_τ^{-1/2}` with `D_τ = D + τI`.

- *Lemma (proved).* Let `S` be any set and `θ_S = max_{i∈S} d_i/(d_i+τ)`. Write `N_τ` for
  `D_τ^{-1/2}AD_τ^{-1/2}`. Two bounds follow from the Schur test with weights `√(d_i+τ)`:
  `‖N_τ[S,S]‖ ≤ θ_S` and `‖N_τ[S,S̄]‖ ≤ √θ_S`. Therefore:
  - (i) every unit `x` supported on `S` has `R_{L_τ}(x) ≥ 1 − θ_S ≥ τ/(d*_S + τ)`, where
    `d*_S = max_{i∈S} d_i`;
  - (ii) every eigenvector with eigenvalue `λ < 1 − θ_S` satisfies
    `‖x_S‖ ≤ (√θ_S/(1 − λ − θ_S))·‖x_{S̄}‖`. Low-degree sets cannot carry most of a
    bottom eigenvector's mass.

  *Proof of (ii):* restrict `N_τx = (1−λ)x` to the rows of `S`. ∎

  At `τ = 0`, `θ_S` can be 1, and no such bound holds.
- **Limitation.** (b) penalizes **low degree**, not **smallness**. A weakly attached document
  whose chunks have ordinary degree (`d_i ≈ τ`) is lifted only to `≈ 1/2`, and so is the core.
  Option (b) helps only if the periphery has lower degree than the core, which `deg_hist`
  measures.
- **Engineering cost:** `L_τ` has no exact zero eigenvalue. The bottom vector is a
  non-trivial global mode, the same issue as F6. For both reasons, (a) is the primary arm and
  (b) the secondary.

### 2.5 P5 — Multiplex mode under-weights the diagonal of sparsely linked chunks (proved)

With the zero-row convention, `L[i,i] = Σ_{s: i has an edge in s} β_s =: b_i ≤ 1`.

If `m ≥ k + 1` chunks that are pairwise non-adjacent in every layer have `b_i ≤ b`, then Cauchy
interlacing gives `λ_k(L) ≤ b`. Example: a chunk with only adjacency edges has `b_i = 0.1`
with the default weights. The window is then pinned below `b` by chunks that have **few
signals**, not by structure.

**Fix N4:** `L_mx = I − Σ_s β_s N_s`, with `N_s` the normalized adjacency of layer `s` (zero rows
for absent nodes). This equals `Σβ_s L_sym,s` when every node is present in every layer
(`Σβ_s = 1` is enforced by `EdgeWeights`). Every diagonal entry is then 1. This does not fix F6:
`L_mx` still has no exact zero eigenvalue.

## 3. Predicted effect

| Metric | Direction | Datasets | Basis |
| --- | --- | --- | --- |
| `ipr` of stored eigenvectors; mass share of the top document per eigenvector | down with (a) | datasets where Step 0 shows localization | P4 (bound), measured |
| P@5, MRR | up | multi-document datasets (multihoprag, musique, 2wikimultihopqa, hotpotqa) **if** Step 0 shows localization | CONJECTURE P4-loc → cross-document proximity |
| `evidence_recall` | within noise or up | same | reordering only |
| `top10_stability`; T5 gap `δ` | up | same | (a) connects the graph and removes near-zero dangling eigenvalues |
| `greedy`-mode metrics (N3) | up | only if `greedy` survives I4 | P2 |
| multiplex-mode metrics (N4) | up | only if multiplex survives I4 | P5 |

**Refutation.**

- Step 0 shows delocalized eigenvectors: median `ipr ≤ 5/n` and top-document mass share
  below 0.3 for most of the 32 vectors. Then P4 has no target; skip the paid runs and reject (a)
  and (b).
- The best (a)/(b) arm is within noise of N0 on P@5 and MRR on all multi-document datasets:
  reject.
- Any arm loses more than 1 point of recall on two or more datasets: reject that arm.

## 4. Ablation plan (dev splits)

- **Step 0 (no LLM).** For each dev dataset's combined graph, report:
  - `deg_hist`, and the number of connected components and isolated chunks;
  - `ipr` and the top-document mass share for `u_1..u_32` under N0, (a) with `τ = d̄` and
    (b) with `τ = d̄`;
  - the conductance of each document's chunk set (its distribution).
- **Step 1 (paid, `--limit 100`; combined mode; only datasets that pass Step 0).** The arms:
  - N0: today.
  - N2 = (a) with `τ ∈ {0.25, 1, 4}·d̄`.
  - N1 = (b) with `τ ∈ {0.25, 1, 4}·d̄`.
  - Metrics: recall, P@5, MRR, F1, `top10_stability`, `build_time_s`. Holm correction across
    the 6 arms.
- **Step 2 (paid, full dev).** The best N1/N2 against N0, adding the judge win rate.
- **Side arms (only if the mode survives I4).** N3 = row-normalized KMeans, in `greedy` mode.
  N4 = the fixed multiplex aggregate, in `spectral_mode = multiplex`.
- **Prerequisites.** T5 R1 (stored eigenvalues, explicit `u_0`) for all arms. For (b), an
  explicit decision on the non-trivial bottom vector: drop it (today's behavior) or keep it.
  The two are separate arms if (b) survives Step 1.

## 5. Prototype

None in this submission. Planned: `experiments/T6/normalization_report.py` (Step 0) and a
`SpectralComputer` subclass with the `A_τ` matvec as a `LinearOperator`.

## 6. Prior art (nothing here is claimed as new)

- Shi & Malik (2000), TPAMI: `L_rw` / normalized cut. Ng, Jordan & Weiss (2002), NIPS:
  row normalization. von Luxburg (2007), Stat. Comput. 17: survey, §7–8.
- von Luxburg, Belkin & Bousquet (2008), *Consistency of spectral clustering*, Ann. Stat. 36.
- Chaudhuri, Chung & Tsiatas (2012), COLT; Qin & Rohe (2013), NIPS: degree-shift
  regularization `D + τI`.
- Amini, Chen, Bickel & Levina (2013), Ann. Stat. 41: rank-one regularization
  `A + (τ/n)𝟙𝟙ᵀ`. Joseph & Yu (2016), Ann. Stat. 44: the impact of regularization.
- Zhang & Rohe (2018), *Understanding regularized spectral clustering via graph conductance*,
  NeurIPS: dangling sets, CoreCut. P4 restates their mechanism for SpectRAG's document-shaped
  sets.
- Multiplex aggregation of normalized Laplacians: Dong, Frossard, Vandergheynst & Nefedov
  (2012/2014, IEEE TSP), *Clustering on multi-layer graphs via subspace analysis on Grassmann
  manifolds* [title unverified]. P5's diagonal observation follows directly from scipy's
  isolated-node convention and is not claimed as new.

## 7. Open questions and red-team log

- Q1: does (a) conflict with H5's zero-credit rule beyond the masking described in §2.4?
  Builder or architect decision, not T6's.
- Q2: edge-weight normalization (per-signal min-max in `graph_builder.py::_minmax_normalize`)
  is not a Laplacian normalization. It is out of scope here; T5 A2 covers its update behavior.

Red-team objections: pending.
