# T5 — Drift bound for incremental spectral indexing

Status: proposal (T5p, v0.3). Consumer: H4 (incremental indexing, v0.4).
Author: Fourier (theory). Red-team objections are answered in §8.

> **Verification status.** WebSearch and local Python execution were both unavailable in the
> session that wrote this document. Citations in §7 come from memory and are marked
> **[unverified]** where I am not sure of the venue or numbering. No numerical prototype was
> run. Every result below is either a textbook theorem (cited) or a short proof given here.
> Steps I could not prove are marked **CONJECTURE**.

---

## 1. Question and what the code does today

**Question.** When H4 adds or removes documents incrementally, how far can the stored spectral
coordinates move? Which cheap, computable quantity tells H4 (a) whether it can skip the
eigensolve, (b) whether a warm-started solve can be trusted, and (c) when it must rebuild the
global index state from scratch?

**Today (full rebuild only).**

- `rag_system/indexing/spectral.py::SpectralComputer.compute` builds the symmetric normalized
  Laplacian `L = I − D^{-1/2} A D^{-1/2}` (`scipy.sparse.csgraph.laplacian(A, normed=True)`) of
  the combined graph from `rag_system/indexing/graph_builder.py::GraphBuilder.build`.
  `compute_multiplex` instead builds `L = Σ_s β_s L_s`, a weighted sum of per-signal normalized
  Laplacians.
- `_solve` asks for `r = k + 1` eigenpairs (`k = n_spectral_components = 32`), using ARPACK
  shift-invert (`_arpack`) or, for `n > 50 000`, LOBPCG (`_lobpcg`, `tol = 1e-4`,
  `maxiter = 300`, seed 42). It **drops the eigenvector with the smallest eigenvalue**
  (`order[1:]`) and stores the remaining `k` columns as float32 `chunk.spectral_coords`.
- Nothing stores the eigenvalues, the `(r+1)`-th eigenvalue, or the residuals (H5 will add
  residual norms to the manifest).
- `rag_system/indexing/cluster_assigner.py::SpectralClusterAssigner.assign` runs KMeans (K = 5)
  on the rows and stores `chunk.cluster_label`.
- Query-time consumers:
  - `rag_system/query/diffusion.py::RelevanceDiffuser._spectral_proximity` computes the cosine
    between a candidate's row and each seed's row, aggregated by `max`, `relevance_weighted` or
    `topk_mean`. It is blended into the score as `(1 − α)·cos(q, c) + α·prox(c)` (in `hybrid`
    mode, the proximity is the mean of the spectral and PPR terms).
  - `rag_system/query/cluster_selection.py` uses `cluster_label` (novelty bonus) and, in
    `_cluster_at_query_time`, the rows themselves.
- Edge scores are min-max normalized **per signal, across the whole corpus**
  (`graph_builder.py::_minmax_normalize`), then thresholded at `edge_sparsify_threshold`. In
  `calibrate` and `fit` modes the per-signal weights β also depend on the whole corpus
  (`weight_calibrator.py`).

## 2. Derivation

### 2.0 Notation and assumptions

- `L` is a real symmetric `n × n` matrix with ascending eigenvalues `λ_0 ≤ λ_1 ≤ … ≤ λ_{n−1}`.
  The **window** is the bottom `r = k + 1` eigenpairs, the ones `_solve` computes.
  `U ∈ ℝ^{n×r}` is an orthonormal basis of their span, `Λ = diag(λ_0..λ_{r−1})`, and
  `P = U Uᵀ`.
- **Boundary gap** `δ = λ_r − λ_{r−1}`. This is the gap between the last eigenvalue in the
  window and the first one outside it. Gaps *inside* the window do not matter (§2.1).
- `Z ∈ ℝ^{n×k}` is the stored coordinate matrix. Its row `z_i` belongs to chunk `i`.
- `sinΘ(U, U′)` is the diagonal matrix of sines of the principal angles between `span U` and
  `span U′`. Standard identities: `‖P − P′‖_2 = ‖sinΘ‖_2` and `‖P − P′‖_F = √2‖sinΘ‖_F`.
  Also, `min_{Q orthogonal} ‖U′Q − U‖_F ≤ √2 ‖sinΘ‖_F` (Cape, Tang & Priebe 2019,
  Lemma 6.7 [numbering unverified]; also used in Yu, Wang & Samworth 2015, Thm 2).
- `⫴·⫴` is any unitarily invariant norm. Below it is always `‖·‖_2` or `‖·‖_F`.

Assumptions:

- **A1 (well-posed window).** `δ > 0`. If the graph has `c` connected components plus `n_iso`
  isolated nodes, eigenvalue 0 has multiplicity `c + n_iso`, so A1 requires
  `c + n_iso ≤ r − 1`.
  - For an isolated node, scipy sets the diagonal to `1 − isolated = 0`, giving a zero row and
    eigenvector `e_i`. (This is from my recollection of the scipy source; ablation step 0
    checks it.)
  - **Requirement on H4/H5:** solve on the subgraph of non-isolated nodes (H5 already gives
    isolated chunks zero spectral credit).
- **A2 (frozen normalizers).** During incremental updates, the per-signal `(lo, hi)` used by
  `_minmax_normalize` and the weights β are frozen. They are stored in the manifest and refreshed
  only by a full rebuild. New raw scores are normalized with the frozen `(lo, hi)` and clipped
  to `[0, 1]`. Without A2, one added document can rescale every edge in the corpus, and every
  bound in §2.3 becomes vacuous (§2.6 quantifies this).
- **A3 (padding constant below the window).** `λ_{r−1} < 1` in both the old and the new
  matrix. On graphs with community structure, the bottom 33 eigenvalues of a normalized
  Laplacian are typically far below 1. H4 must check this at run time; it is free once the
  eigenvalues are stored.

### 2.1 Retrieval depends only on the subspace (rotation invariance)

**Lemma 1.** Replace `Z` by `ZQ` for any orthogonal `k × k` matrix `Q`. Then
`_spectral_proximity`, the KMeans objective in `SpectralClusterAssigner`, and
`_cluster_at_query_time` are unchanged. The only exceptions are floating-point effects and the
KMeans initialization, which is equivariant given the same RNG stream. 2-D visualization of the
raw first coordinates is *not* invariant.

*Proof.* `cos(z_i, z_j) = (ZZᵀ)_{ij} / √((ZZᵀ)_{ii}(ZZᵀ)_{jj})` and
`‖z_i − z_j‖² = (ZZᵀ)_{ii} + (ZZᵀ)_{jj} − 2(ZZᵀ)_{ij}`. Both depend only on the Gram matrix
`ZZᵀ`, and `(ZQ)(ZQ)ᵀ = ZZᵀ`. KMeans and its k-means++ seeding depend only on pairwise
distances. ∎

Consequences:

1. Individual eigenvectors are ill-posed objects: signs are arbitrary, and near-degenerate
   pairs inside the window rotate freely. Only `span U` matters. So drift must be measured with
   principal angles (`sinΘ`), not with per-eigenvector differences.
2. Procrustes alignment in H4 serves only two purposes: reusing frozen KMeans centroids (§2.5)
   and keeping the visualization stable. It does **not** change retrieval.
3. Only the boundary gap `δ` enters the bounds. Small gaps inside the window are harmless.

**The dropped vector.** Assume the combined-mode graph is connected (`c = 1`, A1). Then
`λ_0 = 0` is simple with `u_0 = D^{1/2}𝟙 / √vol(G)`, so the dropped column is `±u_0` up to
solver accuracy and `ZZᵀ = P − u_0u_0ᵀ`. (The code comment says "constant eigenvector". For
the normalized Laplacian the vector is `D^{1/2}𝟙`, not the constant vector. The code is
unaffected.)

With `c > 1`, the dropped column is an arbitrary vector of the null space, chosen by ARPACK, and
`Z` is not well-defined. **Recommendation R1 (for H4/H5):**

- Store all `r` columns plus their eigenvalues.
- Define `Z = (I − u_0u_0ᵀ)U` with `u_0 = D^{1/2}𝟙/‖·‖` computed in closed form.
- On connected graphs this is identical to today's output up to rotation, so by Lemma 1
  retrieval is unchanged. On disconnected graphs it removes solver arbitrariness, which
  changes retrieval, so it needs an eval run.

In multiplex mode, `L = Σ β_s L_s` generally has **no** exact zero eigenvalue: the per-layer
null vectors `D_s^{1/2}𝟙` differ. The dropped column is therefore a genuine, non-trivial
eigenvector. This is outside T5's scope and is filed on the board as a separate observation.

### 2.2 Modelling an update as a perturbation on a common node set

Let `V = V_old ∪ V_new`, with added nodes `𝒜 = V_new \ V_old` and removed nodes
`ℛ = V_old \ V_new`.

**Padding.**

- `L̂_old` is `L_old` on `V_old`, with identity rows and columns on `𝒜`.
- `L̂_new` is `L_new` on `V_new`, with identity on `ℛ`.

**Lemma 2.** Under A3, the bottom `r` eigenpairs of `L̂_old` are those of `L_old`, with the
eigenvectors padded by zeros on `𝒜`. The boundary gap becomes
`δ̂ = min(λ_r, 1) − λ_{r−1} > 0`.

*Proof.* `L̂_old` is block diagonal, `L_old ⊕ I_{|𝒜|}`, and the identity block contributes only
eigenvalue 1, which is greater than `λ_{r−1}`. ∎

The padding value 1 is a free choice: any `c > λ_{r−1}` works. It affects only the Weyl term
below, never `‖EU‖`, because `U` is zero on `𝒜`.

**Perturbation.** `E = L̂_new − L̂_old`. Under A2, `E` is supported on the rows and columns of
`T = 𝒜 ∪ ℛ ∪ {endpoints of changed edges} ∪ N(those nodes)`. The reason is that
`N_ij = A_ij / √(d_i d_j)` changes only if `A_ij`, `d_i` or `d_j` changes, and a degree changes
only when an incident edge changes. For a 1% update, `|T| ≪ n`, and `E` is built in
`O(nnz(E))` from the changed rows. In multiplex mode, `E = Σ_s β_s E_s`.

Cheap, **rigorous** upper bounds on `‖E‖_2` (for symmetric `E`):

`e₂ := min(‖E‖_F, ‖E‖_∞)`, where `‖E‖_∞` is the maximum absolute row sum.

Lanczos estimates of `‖E‖_2` approach from below, so they must not be used where an upper bound
is needed.

### 2.3 A-priori drift bound (before any eigensolve)

**Theorem 1.** Let `U` be the padded old basis, `Λ` its eigenvalues, `δ̂` as in Lemma 2, and
`U′` the bottom-`r` basis of `L̂_new`.

(a) If `e₂ < δ̂`, then for every unitarily invariant norm

    ⫴sinΘ(U, U′)⫴  ≤  ⫴E U⫴ / (δ̂ − e₂).

(b) Unconditionally,

    ‖sinΘ(U, U′)‖_F  ≤  2 · min(√r · e₂, ‖E‖_F) / δ̂.

*Proof of (a).* This is the Davis–Kahan sinΘ theorem in residual form (Davis & Kahan 1970;
Stewart & Sun 1990, Thm V.3.6). Its statement: if `X` has orthonormal columns,
`L̂_new X − XΛ = R`, the eigenvalues of `Λ` lie in `[a, b]`, and the eigenvalues of `L̂_new`
belonging to the complement of the target subspace lie outside `(a − δ′, b + δ′)`, then
`⫴sinΘ⫴ ≤ ⫴R⫴/δ′`.

- Take `X = U`, so `R = (L̂_old + E)U − UΛ = EU`.
- `[a, b] = [λ_0, λ_{r−1}]`.
- The complement eigenvalues are `λ′_j` for `j ≥ r`. By Weyl's inequality,
  `λ′_r ≥ λ̂_r − ‖E‖_2 ≥ λ̂_r − e₂`.
- So `δ′ = δ̂ − e₂ > 0`. ∎

*Proof of (b).* This is Yu, Wang & Samworth (2015), Theorem 1, applied to `−L̂`. For a window at
the end of the spectrum only one gap enters, and their bound depends only on the *old* gap. ∎

**The a-priori drift estimate** is the smaller of (a) and (b):

    η_prior = min( ‖EU‖_F / (δ̂ − e₂)  [only if e₂ < δ̂],   2·min(√r·e₂, ‖E‖_F) / δ̂ ).

Cost: `O(nnz(E)·r)` for `EU`, and `O(nnz(E))` for the norms. It needs the old eigenvalues
`λ_{r−1}` and `λ_r`, so **H4/H5 must compute at least `r + 1` eigenpairs** (one guard). I
recommend `r + p` with `p = 4` guards, and storing the eigenvalues and residual norms in the
manifest.

**Why `‖EU‖` matters, and where the bound fails.** `EU` sees only the part of the change that
touches the current subspace. Its rows on `𝒜` are `−N′_{𝒜,old}U`, and its rows on old nodes
are `(N_old − N′_old)U`. If the new chunks attach to regions where the low eigenvectors are
small, `‖EU‖` is small even when `E` is large.

The `e₂` term is different. Suppose the added documents form a **cohesive new topic**, with most
of their degree internal. Then `‖N′_{𝒜𝒜}‖_2 ≈ 1`, so `e₂ ≳ 1 > δ̂` and (a) does not apply,
while (b) gives a bound of at least `2√r·e₂/δ̂ ≫ 1`, which says nothing. **This is not an
artifact of the bound.** A new cohesive topic can create a new small eigenvalue that enters the
window and pushes out `λ_{r−1}`, a real change of the subspace. Theorem 1 is a *screen* for the
cheapest path (skip the solve), not a certificate for every update.

### 2.4 A-posteriori certificate for a warm-started solve

H4 runs LOBPCG on `L̂_new`. It starts from `[U padded; out-of-sample rows for 𝒜]` plus `p`
guard columns, which are random vectors supported on `T`. The result is orthonormal Ritz vectors
`Ũ` with Ritz values `θ_0 ≤ … ≤ θ_{r−1} ≤ θ_r ≤ …` and residual `R̃ = L̂_new Ũ − ŨΘ` (restricted
to the first `r` columns).

**Theorem 2.** If `ℓ` is a rigorous lower bound on `λ′_r` (the first eigenvalue of `L̂_new`
outside the window) and `ℓ > θ_{r−1}`, then

    ⫴sinΘ(Ũ, U′_true)⫴  ≤  ⫴R̃⫴ / (ℓ − θ_{r−1})  =:  ε_solve.

*Proof.* Same theorem as Theorem 1(a), with `X = Ũ`, `Λ = Θ` and the exact matrix `L̂_new`. ∎

Sources for `ℓ`:

| Source | Rigorous? | Cost | When it is useful |
| --- | --- | --- | --- |
| Weyl: `ℓ = λ̂_r(old) − e₂` | yes | free | local updates; vacuous for a cohesive new topic (§2.3) |
| Sylvester inertia of `L̂_new − σI` (sparse LDLᵀ). If the count below `σ` is `r`, then `ℓ = σ` | yes | one sparse indefinite factorization; needs a solver that reports inertia (MUMPS/PARDISO class, not in scipy) | always, but adds a dependency (architect decision, §8 Q1) |
| Guard Ritz value: `ℓ ≈ θ_r − ‖r̃_r‖` | **no** | free | heuristic; an eigenvector nearly orthogonal to the search space can be missed. Ritz values are *upper* bounds on eigenvalues (Courant–Fischer), so this is not a lower bound |

**Solver tolerance.** Using the residual tolerance as `⫴R̃⫴`, LOBPCG with `tol = 1e-4` on `r`
columns gives `ε_solve ≈ √r·1e-4/δ`. With `r = 33`, any `δ < 6e-3` makes `ε_solve > 0.1`.
**CONJECTURE (P5):** on large corpora, the boundary gap at `k = 32` is of order `1e-3`. If so,
today's cold LOBPCG path (`n > 50 000`) is itself not converged in subspace terms, and incremental
and full builds would disagree mostly because of solver tolerance, not because of the update.

### 2.5 From subspace drift to retrieval scores and labels

Align the new coordinates to the reference by orthogonal Procrustes (Schönemann 1966):
`Q* = argmin_Q ‖Z′Q − Z‖_F`. Define the row displacement `Δ_i = z′_iQ* − z_i` and the
**row angle** `φ_i = ∠(z_i, z′_iQ*)`. If `‖Δ_i‖ < ‖z_i‖`, then `sin φ_i ≤ ‖Δ_i‖/‖z_i‖`.
After a solve, `φ_i` is **computed exactly** in `O(n k²)`: no bound is needed a posteriori.
The bounds below say what a given `φ` means for retrieval.

**Lemma 3.** For non-zero vectors `a, b, a′, b′` with `∠(a, a′) ≤ φ_a` and `∠(b, b′) ≤ φ_b`:
`|cos∠(a′, b′) − cos∠(a, b)| ≤ φ_a + φ_b`.

*Proof.* Geodesic distance on the sphere satisfies the triangle inequality, so
`|∠(a′, b′) − ∠(a, b)| ≤ φ_a + φ_b`, and cos is 1-Lipschitz. By Lemma 1, the result does not
depend on which `Q` is used. ∎

**Proposition 4 (Stage-3 score perturbation).** Let `S` be the seeds and `φ_S = max_{s∈S} φ_s`.

- For each aggregator in `_spectral_proximity` (`max`, `relevance_weighted` with fixed weights,
  `topk_mean`), `|Δ prox(c)| ≤ φ_c + φ_S`.
- Hence in `spectral` mode `|Δ score(c)| ≤ α(φ_c + φ_S)`, and in `hybrid` mode
  `|Δ score(c)| ≤ (α/2)(φ_c + φ_S)`.
- The PPR term uses the graph itself, not `Z`, so spectral drift does not move it.

*Proof.* Each aggregator is 1-Lipschitz in `ℓ∞` over the vector of seed similarities:

- `max`: immediate.
- Convex combination with weights fixed by `all_scores`, which do not depend on `Z`: immediate.
- Mean of the top-`m` order statistics: immediate.

Then apply Lemma 3 to each entry. ∎

**Corollary 5 (top-m stability of Stage 3).** If the score gap between rank `m` and rank `m+1`
exceeds `4α·φ_max`, the top-`m` set after Stage 3 is unchanged. Here `φ_max` is the largest
`φ_i` over candidates and seeds. This is a sufficient condition, not a necessary one.

Stage 4 (`cluster_selection.py`: MMR, novelty bonus, token budget, cross-encoder) is
discontinuous in its inputs. **CONJECTURE:** Stage-4 top-10 overlap tracks Stage-3 overlap
within a few points. Ablation B measures this.

**Lemma 6 (frozen-centroid labels).** Keep the KMeans centroids `{c_b}` from the last full
rebuild, mapped into the new frame by the same `Q*`, and label every chunk by nearest centroid.
Let `a` be chunk `i`'s current label and

`μ_i = min_{b≠a} ‖z_i − c_b‖ − ‖z_i − c_a‖`

its margin. If `‖Δ_i‖ < μ_i / 2`, the label does not change.

*Proof.* `‖z′ − c_a‖ ≤ ‖z − c_a‖ + ‖Δ‖` and `‖z′ − c_b‖ ≥ ‖z − c_b‖ − ‖Δ‖`. ∎

So the number of label flips is at most `F_lab := #{i : ‖Δ_i‖ ≥ μ_i/2}`, which is exact and
cheap to compute. A-priori version: summing `‖Δ_i‖²` gives
`F_lab ≤ 8‖sinΘ_Z‖_F² / μ_min²` (Markov counting with
`Σ‖Δ_i‖² ≤ 2‖sinΘ_Z‖_F²`). Compare Lei & Rinaldo (2015), Lemma 5.3, which gives the analogous
count for re-run KMeans against a planted partition.

**Link from the window to `Z`.** `P_Z = P_U − u_0u_0ᵀ` (R1). So
`‖P_Z − P′_Z‖_F ≤ √2‖sinΘ(U, U′)‖_F + √2 sin∠(u_0, u′_0)`. The last term is closed-form,
because `u_0 ∝ D^{1/2}𝟙` is known for both degree vectors.

### 2.6 Normalizer drift (why A2 is needed, and when to refresh)

A full rebuild recomputes `(lo′, hi′)` per signal. For a raw score `x ∈ [lo, hi]`, the
difference between the frozen normalized score `ŝ` and the rebuilt one `ŝ′` is affine in `x`,
so its maximum is at the endpoints:

    ρ_s := max_x |ŝ − ŝ′| = max(|lo − lo′|, |hi − hi′|) / (hi′ − lo′)      (range grown)

The combined weight of each edge then moves by at most `ρ = Σ_s β_s ρ_s`, on **every** edge:
a dense perturbation, plus edges crossing `edge_sparsify_threshold`.

Two remarks:

1. In multiplex mode, the per-layer normalized Laplacian is invariant to scaling `A_s → cA_s`.
   A change of `hi` alone (with `lo′ = lo`) is a pure scaling of that layer and therefore has
   **no effect**. Only `lo` shifts matter.
2. In combined mode, a change of `hi` re-mixes the layers. It is equivalent to changing `β_s`
   by the factor `(hi − lo)/(hi′ − lo)`.

I do not have a sharp bound from `ρ` to `sinΘ` (relative changes of small weights can be
large). The refresh threshold `ρ_max` below is a **heuristic** to be calibrated by Ablation C.
The `calibrate` and `fit` weight modes have the same issue: freeze β, and refresh at rebuild.

## 3. Rebuild-trigger formula (deliverable for H4)

Every check is relative to the **last solved basis** `U_t`, cumulatively. `E` is always
`L̂_current − L̂_at_last_solve`, so effects of skipped solves accumulate correctly. Label and
normalizer checks are relative to the **last full rebuild** `R`.

    Inputs per update batch (all O(nnz(E)·r) or O(n·k²)):
      δ̂, e₂ = min(‖E‖_F, ‖E‖_∞), ‖EU‖_F, ‖E‖_F               (§2.2–2.3)
      η_prior = min( ‖EU‖_F/(δ̂−e₂) if e₂<δ̂ else ∞ ,  2·min(√r·e₂, ‖E‖_F)/δ̂ )
      ρ = Σ_s β_s ρ_s   (frozen vs. current corpus range; §2.6)

    Level 0  EXTEND (no eigensolve) if   η_prior ≤ ε_0  and  ρ ≤ ρ_max
             new rows: z_{i,l} = (Σ_j N′_ij u_{j,l}) / (1 − λ_l)   (out-of-sample extension)
    Level 1  WARM SOLVE otherwise; accept if   ℓ > θ_{r−1}  and  ε_solve = ‖R̃‖_F/(ℓ − θ_{r−1}) ≤ ε_1
             (ℓ rigorous ⇒ "certified"; ℓ from guard Ritz ⇒ "uncertified", count it)
    Level 2  COLD SOLVE (same graph, fresh start) if Level 1 is not accepted,
             or uncertified_count ≥ N_u, or guard gap (θ_r − θ_{r−1}) < g_min·δ̂_R
    Level 3  FULL REBUILD of global state (normalizers, β, graph, cold solve, KMeans),
             reusing cached per-chunk LLM/embedding outputs, if any of:
               ρ > ρ_max                             (normalizer drift, §2.6)
               F_lab / n > f_lab                     (label flips, Lemma 6)
               F_nov / |𝒜_since_R| > f_nov           (new chunks farther from every centroid than the
                                                      99th percentile of old nearest-centroid distances)
               churn_since_R / n_R > c_max           (backstop)

Proposed starting values (all to be tuned on dev splits, §5):

| Threshold | Value | Notes |
| --- | --- | --- |
| `ε_0` | 0.05 | |
| `ε_1` | 0.10 | |
| `ρ_max` | 0.05 | heuristic |
| `N_u` | 5 | |
| `g_min` | 0.25 | |
| `f_lab` | 0.02 | |
| `f_nov` | 0.20 | |
| `c_max` | 0.20 | |

The exact post-solve quantities `‖sinΘ‖_F`, `φ_95`, `φ_max` and `F_lab` are logged and written
to the manifest for every update.

Requirements on H4 implied by the above:

1. Store `r + p` eigenpairs, the eigenvalues, the residual norms, the frozen `(lo, hi, β)`, the
   KMeans centroids and the basis of the last full rebuild, in float32. Float32 error is about
   `1e-7` relative, far below every `ε` above.
2. Solve on non-isolated nodes only (A1). Check `λ_{r−1} < 1` (A3).
3. Treat `u_0` explicitly (R1).
4. Function names are left to the architect. This document does not define public API.

## 4. Predicted effect

T5 itself does not change retrieval output for a full build; R1 changes it only on disconnected
graphs. Predictions for H4 and for the ablation:

| # | Prediction | Metric | Refuted by |
| --- | --- | --- | --- |
| P1 | Theorem 1 holds: measured `‖sinΘ‖_F ≤ η_prior` on every update where `η_prior` is finite | theory log | any violation (that means a bug or a broken A2/A3; the theorems themselves are textbook) |
| P2 | For 1% additions to existing topics, Theorem 1(a) applies and `η_prior / measured ≤ 30` in the median (**CONJECTURE**) | ratio | median ratio > 100: then Level 0 is useless and should be dropped (H4 always warm-solves) |
| P3 | A cohesive new topic (one ultradomain domain added at once) makes `e₂ ≥ δ̂`. Theorem 1 does not apply, the Weyl `ℓ` fails, and the trigger escalates to Level 2 or Level 3 | trigger log | Theorem 1(a) applying in this scenario (would mean the new topic is not cohesive in the graph) |
| P4 | Incremental (Levels 0–2, frozen normalizers and centroids) vs. full rebuild on the same final corpus: Stage-3 top-10 overlap ≥ 0.97, end-to-end top-10 overlap ≥ 0.95, `evidence_recall` within ±0.5 pt, on all six dev datasets, for 1% updates | `top10_stability` (cross-index variant), `evidence_recall`, `mrr` | overlap < 0.95 with certified `ε_solve ≤ 0.1`: Stage-4 discontinuities (labels) dominate and Lemma 6's policy is wrong |
| P5 | Today's LOBPCG path has `ε_solve > 0.1` at `k = 32`. Two cold LOBPCG builds with different `X0` seeds show `top10_stability < 1` on the largest dev corpus | `top10_stability` across seeds | `ε_solve ≤ 0.01` on all dev corpora, or perfect stability across seeds |
| P6 | Drift per 1% update is largest on ultradomain (multi-domain, a new-topic risk) and legal (long cohesive documents), and smallest on hotpotqa, musique and 2wikimultihopqa (many short Wikipedia paragraphs, local attachment) (**CONJECTURE**) | `‖sinΘ‖_F`, `φ_95` | the reverse order |

## 5. Ablation plan (dev splits only)

**Step 0 (no LLM, smoke fixture plus fake embedder).** Confirm on a tiny graph that an isolated
node gets a zero Laplacian row (A1). Confirm that the dropped column equals `±D^{1/2}𝟙/‖·‖` on a
connected graph (R1).

**Scenarios.** For each dev dataset (`multihoprag`, `ultradomain`, `musique`, `legal`,
`hotpotqa`, `2wikimultihopqa`), with corpus `C`:

- **U1:** hold out a random 1% / 5% / 10% of documents `H`. Build on `C \ H`, then add `H`.
- **U2:** build on `C`, then remove a random 1%.
- **U3 (new topic):** on ultradomain, hold out one whole domain, then add it. On the other
  datasets, use the documents of one KMeans cluster from the full build.
- **U4:** a sequence of 10 updates of 1% each, to exercise cumulative drift and the counters.

**Configurations.** Reference **A0** is a full rebuild on the final corpus with today's settings.

| Config | Description | What it isolates |
| --- | --- | --- |
| A1 | Level 1 only (always warm-solve), frozen normalizers, frozen centroids | warm-start vs. cold |
| A2 | A1 + Level 0 with `ε_0 ∈ {0.01, 0.05, 0.1}` | value of skipping solves |
| A3 | A1 with normalizers recomputed each update (A2 off) | contribution of normalizer freezing |
| A4 | A1 + KMeans re-run each update | contribution of stale labels |
| A5 | A0 with `spectral_large_graph_threshold = 0` (force LOBPCG), seeds {42, 43} | P5, solver tolerance |

**Metrics.**

- **Theory log** (no LLM; `experiments/T5/` script): `δ̂`, `e₂`, `‖EU‖_F`, `η_prior`; measured
  `‖sinΘ‖_2` and `‖sinΘ‖_F` against A0; `φ_95` and `φ_max`; `F_lab`; `ε_solve`; LOBPCG
  iterations; wall time; the trigger level chosen.
- **Eval** (`python -m eval run --dev`, warm caches, so no extra index-time LLM cost):
  `evidence_recall`, `mrr`, `precision_at_5`, and a **cross-index `top10_stability`** (A_i vs.
  A0). The harness currently measures stability between two builds of the *same* config, so the
  cross-index variant is a request to the evaluator.

**Decision rules.**

- Keep Level 0 only if P2 holds.
- Set `ε_1` to the largest value with A1 end-to-end overlap ≥ 0.95 on the worst dev dataset.
- Set `ρ_max`, `f_lab` and `f_nov` from A3 and A4 the same way.

## 6. Prototype

Deferred. Python execution was unavailable in the session that wrote this proposal, and I will
not commit an unrun script. The `experiments/T5/` script described in §5 (theory log on a
stochastic block model plus the smoke fixture) is the first task once the proposal is approved.

## 7. Prior art

Nothing in §2 is a new theorem. The contribution is fitting known results to SpectRAG:

- the rotation-invariance observation (Lemma 1), which reduces drift to subspace angles;
- the zero-padding of added and removed nodes;
- the `‖EU‖` form of the a-priori bound;
- the explicit identification of global normalizers and the dropped vector `u_0` as the
  non-local parts of an "incremental" update;
- the leveled trigger.

References:

- **Davis–Kahan sinΘ theorem.** C. Davis, W. M. Kahan, "The rotation of eigenvectors by a
  perturbation. III", *SIAM J. Numer. Anal.* 7(1), 1970. Also G. W. Stewart, J.-G. Sun,
  *Matrix Perturbation Theory*, Academic Press 1990, Ch. V [theorem number unverified].
- **Weyl's inequality.** H. Weyl, 1912. Textbook (Stewart & Sun, Ch. IV).
- **Gap-only-on-the-old-matrix variant.** Y. Yu, T. Wang, R. J. Samworth, "A useful variant of the
  Davis–Kahan theorem for statisticians", *Biometrika* 102(2), 2015.
- **Row-wise (2→∞) and entrywise bounds.** These would sharpen §2.5 a priori. J. Cape, M. Tang,
  C. E. Priebe, "The two-to-infinity norm and singular subspace geometry…", *Ann. Statist.* 47(5),
  2019. E. Abbe, J. Fan, K. Wang, Y. Zhong, "Entrywise eigenvector analysis of random matrices with
  low expected rank", *Ann. Statist.* 2020. A. Damle, Y. Sun, "Uniform bounds for invariant
  subspace perturbations", *SIMAX* 2020 [venue unverified]. Not used here, because the
  a-posteriori `φ_i` are exact and cheap.
- **Spectral clustering stability.** J. Lei, A. Rinaldo, "Consistency of spectral clustering in
  stochastic block models", *Ann. Statist.* 43(1), 2015. U. von Luxburg, "A tutorial on spectral
  clustering", *Stat. Comput.* 17, 2007.
- **Incremental spectral clustering and embeddings (closest prior art).**
  - H. Ning, W. Xu, Y. Chi, Y. Gong, T. Huang, "Incremental spectral clustering by efficiently
    updating the eigen-system", *Pattern Recognition* 43(1), 2010.
  - C. Dhanjal, R. Gaudel, S. Clémençon, "Efficient eigen-updating for spectral graph
    clustering", *Neurocomputing* 2014. Uses perturbation arguments to decide when to recompute
    [details unverified].
  - V. Charisopoulos, A. R. Benson, A. Damle, "Incrementally updated spectral embeddings",
    arXiv:1909.01188 [venue unverified]. Warm-started subspace iteration with accuracy bounds;
    it may subsume Levels 0–1. **The red team should check whether their criterion makes §3
    redundant.**
- **Solvers and certificates.** A. V. Knyazev, "Toward the optimal preconditioned eigensolver:
  LOBPCG", *SISC* 23(2), 2001. B. N. Parlett, *The Symmetric Eigenvalue Problem*, SIAM 1998
  (residual bounds, Ritz values as upper bounds, spectrum slicing by Sylvester inertia).
- **Out-of-sample extension.** Y. Bengio et al., "Out-of-sample extensions for LLE, Isomap, MDS,
  Eigenmaps, and Spectral Clustering", NeurIPS 2003/2004.
- **Orthogonal Procrustes.** P. H. Schönemann, *Psychometrika* 31, 1966.

## 8. Open questions and red-team log

**Q1 (architect, not blocking T5p).** Is a sparse indefinite LDLᵀ with inertia (MUMPS, PARDISO
or similar) acceptable as an optional extra, so that Level 1 can be certified rigorously for
cohesive additions? Without it, those updates are "uncertified" and are bounded only by
`N_u` and Level 2.

**Q2 (evaluator).** Add a cross-index `top10_stability` (index A vs. index B on the same
questions) for Ablation B and for H4's ≥ 95% gate.

Red-team objections and answers: (none yet).
