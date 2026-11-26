# T1 — Eigenvalue-weighted (heat-kernel) spectral proximity

Status: proposal (T-props, written in Phase 1; prototype and ablation belong to T1 in v1.1).
Author: Fourier (theory). Red-team review pending (§7).
Shared evaluation rules: `docs/theory/ablation-protocol.md`. Builds on
`docs/theory/T5-drift-bound.md` (notation, R1, R2).

> **Verification status.** WebSearch and local Python execution were both unavailable in the
> session that wrote this. Citations come from memory; those marked **[unverified]** need a
> check. No numerical prototype was run. Steps not proved here are marked **CONJECTURE**.

## 1. Question and what the code does today

**Question.** Stage 3 compares chunks by the cosine of their rows of the bottom-`k` Laplacian
eigenvectors, and every eigenvector gets the same weight. Should eigenvector `l` instead be
weighted by a decreasing function of its eigenvalue `λ_l`, as in the heat kernel `e^{−tL}`?
If so, which `t`, and what does that change?

**Today.**

- `rag_system/indexing/spectral.py::SpectralComputer._solve` returns eigenvectors
  `u_1..u_k` of `L = I − D^{-1/2}AD^{-1/2}` (`k = n_spectral_components = 32`). The
  eigenvalues are **discarded**. They are needed here, and T5 R1 already asks to store them.
- `rag_system/query/diffusion.py::RelevanceDiffuser._spectral_proximity` (lines 45–71)
  normalizes rows `z_c = U[c,:]` and takes `cos(z_c, z_s)`, aggregated over seeds by `max`
  (the default `diffusion_seed_aggregation`).
- `diffuse` blends with the result
  `(1 − α)·r_c + α·(½·spectral + ½·ppr)` (default `hybrid`, `α = 0.5`). So the spectral term
  carries weight 0.25 today.

## 2. Derivation

### 2.1 Setup

Use T5 §2.0 notation, on a connected combined graph (T5 A1, with R1 applied). The eigenvalues
are `0 = λ_0 < λ_1 ≤ … ≤ λ_k < λ_{k+1}`, the stored basis is `U = [u_1..u_k]`, the boundary gap
is `δ = λ_{k+1} − λ_k`, and `P = Σ_{l=0}^{k} u_l u_lᵀ` is the window projector. For any filter
`f ≥ 0`, define

    z_i^{(f)} = U[i,:] · f(Λ)^{1/2},   so   ⟨z_i^{(f)}, z_j^{(f)}⟩ = K_f(i,j),   K_f = U f(Λ) Uᵀ.

Today's proximity is `f ≡ 1`: `K_1 = P − u_0u_0ᵀ` is the ideal low-pass projector. The
proposal is the heat filter `f_t(λ) = e^{−tλ}` with `K_t := K_{f_t}`, parameterized by the
dimensionless `τ = t·λ_k`, so the last kept eigenvector gets weight `e^{−τ}` relative to a
hypothetical `λ = 0` mode.

### 2.2 Exact properties (proved)

**P1 (nesting).** `τ = 0` reproduces today's scores exactly. *Proof:* `f_0 ≡ 1`. ∎

**P2 (basis independence; compatible with H4 alignment).** Let `X = UQ` for any orthogonal `Q`
(for example H4's Procrustes-aligned basis) and let `B = XᵀLX`. Then
`X f(B) Xᵀ = U f(Λ) Uᵀ`, so `z_i = X[i,:]·f(B)^{1/2}` gives the same Gram matrix and the same
cosines (T5 Lemma 1).
*Proof:* `B = QᵀΛQ`, so `f(B) = Qᵀf(Λ)Q` for any function applied spectrally. ∎

Consequence: **the weights must not be applied as `diag(f(λ))` to a rotated basis.** If H4
stores rotated coordinates, it must also store the `k × k` matrix `B` (or the Ritz pairs). If
the subspace is only approximately invariant, `B` is the Rayleigh–Ritz matrix, and its Ritz
values have error `O(‖R‖²/gap)` (Parlett, *The Symmetric Eigenvalue Problem*, Ch. 11
[section unverified]).

**P3 (truncation).** Let `H'_t = e^{−tL} − u_0u_0ᵀ` be the full deflated heat kernel. Then
`‖H'_t − K_t‖_2 = e^{−tλ_{k+1}}`, and relative to the leading weight `e^{−tλ_1}` the ignored tail
is `ρ(t) = e^{−t(λ_{k+1}−λ_1)}`. *Proof:* the difference is `Σ_{l>k} e^{−tλ_l}u_lu_lᵀ`. ∎

So for `t > 0` the stored coordinates approximate a well-defined full-graph object, with
controlled error. At `t = 0`, `ρ = 1`: today's proximity is defined by an arbitrary cut at
`k = 32` and changes as `k` crosses eigenvalue clusters (see T3).

**P4 (large-t degeneracy).** If `λ_1 < λ_2`, then as `t → ∞`,
`cos(z_i, z_j) → sign(u_1(i)·u_1(j))` whenever `u_1(i)u_1(j) ≠ 0`.
*Proof:* `e^{tλ_1/2} z_i → (u_1(i), 0, …, 0)`. ∎

So proximity collapses to the Fiedler bipartition (one bit). The useful `τ` is interior, and
the grid must stop well before this limit.

### 2.3 Stability (proved for the kernel; CONJECTURE for rankings)

**Lemma A (Frobenius-Lipschitz).** If `φ: ℝ → ℝ` is `ℓ`-Lipschitz on an interval containing
the spectra of the symmetric matrices `M` and `M′`, then `‖φ(M′) − φ(M)‖_F ≤ ℓ‖M′ − M‖_F`.
*Proof:* write `M = Σ a_i v_iv_iᵀ` and `M′ = Σ b_j w_jw_jᵀ`. Then
`v_iᵀ(φ(M) − φ(M′))w_j = (φ(a_i) − φ(b_j))·v_iᵀw_j` and `v_iᵀ(M − M′)w_j = (a_i − b_j)·v_iᵀw_j`.
Summing the squares over `i, j` gives the claim. ∎ (Standard; see Bhatia, *Matrix Analysis*,
Ch. VII/X [section unverified].)

**Theorem 1.** Let `L′ = L + E` (both Laplacians, spectra in `[0, 2]`), with the same window
size and `‖E‖_2 < δ`. Write `K^{win}_t = Σ_{l=0}^{k} e^{−tλ_l}u_lu_lᵀ` (the same with primes for
`L′`) and `δ′ = δ − ‖E‖_2`. Then

    ‖K′^{win}_t − K^{win}_t‖_F ≤ e^{−tλ_k}·‖P′ − P‖_F + 2t‖E‖_F
                              ≤ ( √2·e^{−tλ_k}/δ′ + 2t ) · ‖E‖_F.

*Proof.* Let `c = e^{−tλ_k}` and `φ(λ) = (e^{−tλ} − c)_+`. On the old window (`λ_l ≤ λ_k`)
we have `e^{−tλ_l} = c + φ(λ_l)`, and outside it `φ = 0`. So `K^{win}_t = cP + φ(L)` exactly.
For `L′`: `K′^{win}_t = cP′ + φ(L′) + Δ′`, where `Δ′` collects two kinds of terms:

- (i) window terms with `λ′_l > λ_k`. Each is `(e^{−tλ′_l} − c)u′_lu′_lᵀ`, of magnitude
  `≤ t(λ′_l − λ_k) ≤ t|λ′_l − λ_l|`, because `λ_l ≤ λ_k`.
- (ii) outside terms with `φ(λ′_l) > 0`. By Weyl, `λ′_{k+1} ≥ λ_{k+1} − ‖E‖_2 > λ_k`, so there
  are none.

Hence `‖Δ′‖_F ≤ t(Σ_l |λ′_l − λ_l|²)^{1/2} ≤ t‖E‖_F` (Hoffman–Wielandt/Mirsky). The map `φ` is
`t`-Lipschitz on `[0, ∞)`, so Lemma A gives `‖φ(L′) − φ(L)‖_F ≤ t‖E‖_F`. Finally,
`‖P′ − P‖_F = √2‖sinΘ‖_F ≤ √2‖E‖_F/δ′` (Davis–Kahan sinΘ; T5 §2.3). ∎

The stored kernel is `K_t = K^{win}_t − u_0u_0ᵀ`, and `u_0 = D^{1/2}𝟙/‖·‖` is closed-form, so its
drift `‖u′_0u′_0ᵀ − u_0u_0ᵀ‖_F` is computed exactly and simply added.

**Corollary (parameter-free τ).** The bound `B(t) = (√2e^{−tλ_k}/δ′ + 2t)‖E‖_F` is minimized at
`τ* = t*λ_k = ln(x)` with `x = λ_k/(√2δ′)`, provided `x > 1`. The ratio to today's bound is
`B(t*)/B(0) = (1 + ln x)/x`. Example: with `λ_k = 0.3` and `δ′ = 0.005`, `x = 42.4` and the
ratio is 0.11, so the worst-case kernel drift bound is about 9× smaller. When there is a big
gap at `k` (`x ≤ 1`), `τ* = 0`: weighting does not help stability.

**What this does and does not say.**

- (a) It is an upper-bound comparison, not a measured drift.
- (b) Cosines divide by `K_ii`, and `K_t,ii ∈ [e^{−τ}P_ii, P_ii]` (with `u_0` removed). A node
  whose mass sits on high-`λ` eigenvectors has its `K_ii` shrink by up to `e^{−τ}`, so its
  *relative* perturbation need not improve. To first order,
  `|Δcos_ij| ≤ 2·max|ΔK|/min(K_ii, K_jj)`.
- **CONJECTURE S:** for chunks whose mass is mostly on the low eigenvectors (community cores),
  top-10 proximity rankings are more stable under eigensolver tolerance and under 1% updates
  at `τ*` than at `τ = 0`.

### 2.4 Why it might help retrieval quality (CONJECTURE Q)

In a normalized Laplacian, high-index eigenvectors in the bottom 33 are fine-scale.
Davis–Kahan bounds each eigenvector's sensitivity by `1/gap_l`, and gaps are small where the
spectrum is dense. They are also often localized on small weakly-attached sets (T6 §2.4).
With equal weights, two unrelated chunks that both load on one localized eigenvector get a
high cosine. Heat weighting suppresses this; it is the smoothness prior of Smola & Kondor's
spectral transforms. Expected sign: fewer spurious graph-proximate candidates.

## 3. Predicted effect

| Metric | Direction | Where | Basis |
| --- | --- | --- | --- |
| kernel drift `‖ΔK_t‖_F/‖K_t‖_F` under 1% update replay | down at `τ*`, roughly by `(1+ln x)/x` | every dataset with `x > 1` | Theorem 1 (bound), measured |
| `top10_stability` | up or equal | datasets where it is < 1 today | CONJECTURE S |
| `precision_at_5`, `mrr` | small up (≤ 1–2 pts) | corpora with many low-degree chunks (high `ipr`) | CONJECTURE Q |
| `evidence_recall` | within noise | all | proximity reorders candidates; Stage 2 is unchanged |
| `query_p95_ms` | unchanged | all | `O(k)` extra work per candidate row |

The effect is diluted by the 0.25 weight of the spectral term in `hybrid`. Arm A3 below
isolates it.

**Refutation.** Each of these rejects part or all of the idea:

- (i) Every `τ > 0` arm is within noise of `τ = 0` on recall, P@5 and MRR on all six datasets,
  **and** neither the drift replay nor `top10_stability` improves. Reject: no effect.
- (ii) The `τ*` arm loses more than 1 point of `evidence_recall` on two or more datasets.
  Reject the parameter-free default.
- (iii) The measured drift at `τ*` is not smaller than at `τ = 0`. The theorem's mechanism is
  then irrelevant at this scale; reject S.

## 4. Ablation plan (dev splits)

- **Step 0 (no LLM).** For each dev dataset, record `spectrum` (`λ_1..λ_{k+1}`), `δ`, `x` and
  `τ*`. If `x ≤ 1` on every dataset, Theorem 1 predicts no stability gain. Run only the
  quality arms.
- **Step 1 (no LLM, drift replay).** Remove a random 1% of documents (5 seeds), rebuild, and
  compute `‖ΔK_t‖_F/‖K_t‖_F` and the mean top-10 proximity overlap for
  `τ ∈ {0, 0.5, 1, 2, 4, 8, τ*}`. Compare with the Theorem 1 bound.
- **Step 2 (paid, `--limit 100`).** The arms:
  - A0: baseline (`τ = 0`, `hybrid`).
  - A1: `τ ∈ {0.5, 1, 2, 4}` (`hybrid`).
  - A2: `τ*` (`hybrid`).
  - A3: A0 and A2 with `diffusion_mode = "spectral"` (isolates the term).
  - Metrics: recall, P@5, MRR, F1, `top10_stability`, `query_p95_ms`.
- **Step 3 (paid, full dev).** The best of A1/A2 against A0, adding `judge_win_rate_vs_naiverag`.
  Holm correction across the A1/A2 arms.
- **Prerequisites.** Eigenvalues stored (T5 R1) and rows of isolated chunks zeroed (T5 R2;
  weighting interpolated rows has no meaning). The combined mode only: in multiplex mode the
  dropped column is not trivial (F6), and T1 waits for that fix.

## 5. Prototype

None in this submission. Planned for v1.1: `experiments/T1/heat_proximity.py`, a drop-in
`_spectral_proximity` that takes the stored eigenvalues and `τ`, plus the Step 1 replay script.

## 6. Prior art (nothing here is claimed as new)

- Kondor & Lafferty (2002), *Diffusion kernels on graphs and other discrete input spaces*,
  ICML: the heat kernel `e^{−βL}` as a graph kernel.
- Smola & Kondor (2003), *Kernels and regularization on graphs*, COLT: the general family
  `U r(Λ) Uᵀ` (diffusion, regularized Laplacian, p-step walk). T1 picks one member and
  truncates it.
- Coifman & Lafon (2006), *Diffusion maps*, ACHA 21: eigenvalue-powered coordinates `λ_l^t ψ_l`
  of the random-walk operator, and truncation by the spectral tail. This is the closest
  analogue: diffusion-map coordinates with heat rather than power weights. Their coordinates
  use `D^{-1/2}u`, which is a per-row rescaling and leaves cosines unchanged (T6 P1).
- Chung (2007), *The heat kernel as the pagerank of a graph*, PNAS 104: heat-kernel PageRank,
  the operator view used in T2.
- Davis & Kahan (1970), SIAM J. Numer. Anal. 7; Mirsky (1960) / Hoffman & Wielandt (1953);
  Bhatia (1997): the ingredients of Theorem 1. I do not know a reference for Theorem 1 in
  exactly this form. It is a short combination of standard results, so treat it as folklore,
  not as a contribution.
- Commute-time / effective-resistance weights `f(λ) = 1/λ` are another member of the family.
  They are deferred to SWEEP (effective resistance), because von Luxburg, Radl & Hein (2014,
  JMLR) show that commute distances degenerate on large neighborhood graphs.

## 7. Open questions and red-team log

- Q1: should `τ` be per-index (`τ*`) or a global preset? The ablation decides.
- Q2: whether KMeans should also use weighted coordinates. That belongs to T6.

Red-team objections: pending.
