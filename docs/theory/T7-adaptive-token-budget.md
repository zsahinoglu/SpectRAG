# T7 — Adaptive token budget (optimal stopping)

Status: proposal (T-props, written in Phase 1; prototype and ablation belong to T7 in v1.2,
which also depends on **decision D4**).
Author: Fourier (theory). Red-team review pending (§7).
Shared evaluation rules: `docs/theory/ablation-protocol.md`. Related: T2 (score scale), T4
(objective).

> **Verification status.** WebSearch and Python were unavailable. Citations come from memory
> (**[unverified]** where unsure). **CONJECTURE** marks unproved claims.
>
> **About D4.** The backlog makes the T7 *item* depend on decision D4. The plan's text for
> D1–D5 is not in this repository, so I do not know what D4 decides. CLAUDE.md forbids
> guessing on open decisions. This proposal therefore assumes nothing about D4: every arm
> keeps today's `token_budget` as a hard upper cap, and the rule only ever selects **less**.
> If D4 changes the cap's role, only §4's arm list changes.

## 1. Question and what the code does today

**Question.** How much context should each query get? The backlog frames this as optimal
stopping. Is that the right model, and what rule follows?

**Today** (`rag_system/query/cluster_selection.py::_token_budget_select`, lines 244–315):

- Fill to a fixed `token_budget = 12000`.
- A head floor `top − 0.70·|top|` applies to the first 5 chunks and a tail floor
  `top − 0.45·|top|` after that. Gated-out chunks above the head floor are backfilled.
- The budget is used up whenever enough candidates clear the head floor. Config comments
  record that "the judge rewards" larger contexts, and that backfill was added because
  "shorter answers were losing".
- The evaluator does not report context tokens today (`docs/evaluation/README.md`), so the
  cost side of this question is not measured.

## 2. Derivation

### 2.1 P0 — With all scores known, this is not an optimal-stopping problem (proved)

Stage 3 scores **every** candidate before Stage 4 starts. Any "stop at position n" rule is then
a function of the fully observed score vector. The best such rule solves a **static** problem:
maximize expected utility over subsets. Secretary and prophet-inequality results apply when
values are revealed sequentially and decisions are irrevocable (Ferguson 1989). Neither holds
here. ∎

Optimal stopping does apply when **looking at a candidate costs something and reveals
information**: a cross-encoder call, an LLM relevance check, or fetching deeper pages of a
ranking. §2.5 treats that case.

### 2.2 Model M1 — additive expected utility (proved results under stated assumptions)

**Assumptions.**

- Chunk `i` has probability `p_i` of belonging to an evidence document, given the observed
  features.
- Including it costs `c_i` tokens at price `μ` per token.
- Including it yields expected value `g·p_i`, minus a distractor cost `h·(1−p_i)·c_i`. The sign
  of `h` is not known a priori (§2.6).

So `U(S) = Σ_{i∈S} u_i` with `u_i = g·p_i − (μ + h(1−p_i))·c_i`, subject to `c(S) ≤ B`.

**Proposition 1.**

- (a) Without a binding budget, the optimum is `S* = {i : u_i > 0} = {i : p_i > π_i}` with
  `π_i = c_i(μ+h)/(g + h·c_i)`. For equal chunk sizes this is **one threshold `p*`**, and the
  number of chunks `n*(q) = #{i : p_i > p*}` adapts per query.
- (b) With the budget binding, the problem is a 0/1 knapsack in `u_i`. Greedy by `u_i/c_i`
  solves the fractional relaxation exactly (Dantzig 1957), and its integer version is within
  `max_i u_i` of the optimum.

*Proof.* (a) `U` is separable. (b) Standard. ∎

**Consequence.** The rule is: *take chunks in order of `p` while `p_i > p*`, and stop at the
cap `B`*. That is one parameter, `p*`. It merges the ratios `g : μ : h`, which cannot be
identified separately from recall data, and is tuned on dev.

**M1 is a simplification.** Recall is measured per document, so a second chunk of a covered
evidence document adds content but no recall. The coverage-aware version is T4's `F` with a
price per token: maximize `F(S) − μ·c(S)`, that is, greedy while `Δ_j/c_j > μ`. For
monotone submodular `F` minus a modular cost, "distorted greedy" guarantees
`F(S) − μc(S) ≥ (1 − 1/e)F(OPT) − μc(OPT)` under a cardinality constraint (Harshaw, Feldman,
Ward & Karbasi 2019, ICML [statement and constraint type unverified]). Using it under T4's
knapsack is **CONJECTURE K**, and is used only as arm A3.

### 2.3 P1 — The features must carry per-query score geometry (proved)

If `p_i` depends only on the rank of `i`, then `n*(q)` is the same for every query, and the
rule degenerates to a fixed top-n. *Proof:* `p_{(1)} ≥ p_{(2)} ≥ …` is then a fixed sequence,
and `#{r : p_{(r)} > p*}` does not depend on `q`. ∎

So calibration must use features that vary across queries. Proposed features are
affine-invariant and computable from Stage 3 output:

- `z_i = (s_i − s_med)/(s_max − s_med)`, the normalized score;
- `rank_i`;
- the top gap `(s_(1) − s_(2))/(s_max − s_med)`;
- the number of seed lists that contain `i` (from the I5 trace).

### 2.4 P2 — Today's relative floors are shift-dependent (proved)

The floor `top − (1−ρ)·|top|` is not invariant to `s ↦ s + c`.

*Example:* scores `{1.0, 0.5}` with `ρ = 0.55`. The floor is 0.55, so 0.5 is excluded. After
shifting by +1, the scores are `{2.0, 1.5}`, the floor is 1.1, and 1.5 is included. ∎

The offset of Stage 3 scores depends on `α`, the embedder's cosine range and
`hybrid_lexical_weight` (T2 D1). So the gate admits different counts for equivalent rankings.
The `z_i` features above are invariant to this.

### 2.5 Where optimal stopping is real: costly inspection (Pandora's box)

Suppose that before including a candidate we may inspect it (cross-encoder or LLM check) at
cost `κ_i`. Inspection reveals the realized utility `U_i`, and we include `i` iff `U_i > 0`.

- With additive utility and no budget, the problem separates. Inspect `i` iff the value of
  information `E[(U_i)_+] − (E[U_i])_+ ≥ κ_i`. Proof: per-item comparison of the two options.
- With a single best item as the goal, this is Weitzman's (1979) Pandora's box. Reservation
  values `σ_i` solve `E[(X_i − σ_i)_+] = κ_i`. Inspect in decreasing `σ`, and stop when the best
  observed value is at least every remaining `σ`. That policy is optimal.
- With a budget coupling items: a Lagrangian version of the above (heuristic, not proved).

In SpectRAG this applies only to `anchor_expand_ce`, which today cross-encodes the whole
expansion pool. It is recorded for completeness. **No arm uses it unless that mode survives
I4.**

### 2.6 Sign of the distractor cost `h` (literature conflict)

- Liu et al. (2024, TACL, *Lost in the middle*) and Shi et al. (2023, ICML, *LLMs can be
  easily distracted by irrelevant context*) support `h > 0`.
- Cuconasu et al. (2024, SIGIR, *The power of noise*) report that adding random documents can
  *help*, which suggests `h ≤ 0` in some regimes.
- The judge's preference for longer contexts (§1) is a further, evaluation-side bias.
- T7 therefore does not assume a sign. `p*` is tuned on dev, and generation metrics are read
  separately from retrieval metrics.

## 3. Predicted effect

| Metric | Direction | Datasets | Basis |
| --- | --- | --- | --- |
| `context_tokens` (mean) | down ≥ 25% at `evidence_recall` within 1 pt | hotpotqa, 2wikimultihopqa (two evidence docs per question) | CONJECTURE A: many budget-filling chunks have low `p` |
| `context_tokens` | down less | multihoprag, musique (more evidence docs) | same |
| variance of `context_tokens` across queries | up | all | Prop 1(a): `n*(q)` adapts |
| F1 | within noise | all | M1 trades low-`p` tokens only |
| judge win rate | down possible | all | the judge's length preference (§1, §2.6); reported, not optimized |

**Refutation.**

- On the LLM-free recall/token Pareto curve (Step 1), no `p*` gives ≥ 15% token reduction at
  ≤ 1 pt recall loss on at least three datasets. There is no slack to adapt; reject.
- The calibrated `p` has cross-fitted ECE > 0.1 or is no better than rank-only calibration
  (P1 control). The features carry no per-query information; reject.
- At the chosen `p*`, F1 drops by more than 1 point on two or more datasets: reject that
  operating point.

## 4. Ablation plan (dev splits)

Steps 0 and 1 need **no generation**. Evidence recall and context tokens depend on retrieval
only.

- **Step 0.** On the baseline's Stage 3 outputs:
  - fit isotonic calibration of `p` on the §2.3 features with 5-fold cross-fitting;
  - report the reliability diagram and ECE;
  - fit the same with `rank` as the only feature (control for P1).
- **Step 1.** Pareto curve of `evidence_recall` against mean `context_tokens`:
  - A0: today, with `B ∈ {4k, 6k, 8k, 12k}` (the fixed-budget frontier).
  - A1: the `p*` rule with cap `B = 12k`, `p*` swept.
  - A2: rank-only `p` (fixed count).
  - A1 beats the fixed-budget frontier only if its curve lies above A0's.
- **Step 2 (paid, `--limit 100`).** Three A1 operating points (recall loss 0, 0.5 and 1 pt on
  the Step 1 curve) against A0 at 12k, measuring F1, EM, the judge win rate and
  `context_tokens`.
- **Step 3 (paid, full dev).** The best A1 point against A0. Holm correction across the three
  points.
- **A3 (if T4 is approved).** T4's `F` with the price stop `Δ_j/c_j > μ`, with `μ` swept on the
  Step 1 curve.

## 5. Prototype

None in this submission. Planned: `experiments/T7/` with the calibration fit, Pareto-curve
script (LLM-free) and the `p*` selector.

## 6. Prior art (nothing here is claimed as new)

- Robertson (1977), *The probability ranking principle in IR*, J. Doc. 33: ranking by `p` is
  optimal under additive utility. Prop 1 adds a cost and a cutoff.
- **Ranked-list truncation**, the established name for this problem: Arampatzis, Kamps &
  Robertson (2009), SIGIR, *Where to stop reading a ranked list?*; Bahri et al. (2020),
  SIGIR, *Choppy*; Wu et al. (2021), WWW, *AttnCut* [author lists and venues unverified]. T7
  applies this to RAG context with a token cost.
- Dantzig (1957): the greedy fractional knapsack. Weitzman (1979), Econometrica 47:
  Pandora's box. Ferguson (1989), Stat. Sci. 4: the secretary problem.
- Zadrozny & Elkan (2002), KDD: isotonic calibration. Platt (1999). Guo et al. (2017), ICML:
  ECE.
- Harshaw, Feldman, Ward & Karbasi (2019), ICML: submodular minus modular ("distorted
  greedy").
- Adaptive retrieval in RAG decides *whether* or *how often* to retrieve: Asai et al. (2024,
  ICLR, Self-RAG); Jeong et al. (2024, NAACL, Adaptive-RAG). I could not search for 2024–2026
  work on adaptive *context size*. It likely exists and must be checked before any novelty
  claim.

## 7. Open questions and red-team log

- Q1 (for the human; it may or may not fall under D4): is reducing generation cost a goal, or
  only quality? If only
  quality, Step 2's operating point is "recall loss 0", and T7's value is limited to cost and
  latency.

Red-team objections: pending.
