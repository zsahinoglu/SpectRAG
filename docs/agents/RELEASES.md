# Release gates

A release is tagged and exported (`scripts/release-local.sh`) only when every criterion below is MET in the
release manager's gate report (`docs/reports/gate-<version>.md`) and the human approves.
Week numbers are estimates from kickoff.

## Standing criteria (every release)

1. All tests and golden tests pass on `main`.
2. No dataset's frozen-test score drops more than 1 point vs the previous release.
3. Query p95 and index cost within 10% of the previous release, unless justified in the notes.
4. Top-10 retrieval stability ≥ 95% (from v0.4 on).
5. Results reproducible from a clean checkout: `python -m eval run --release`.
6. No open red-team finding of high severity.

## Phase 1

| Release | Weeks | Items | Release-specific gate |
| --- | --- | --- | --- |
| v0.2 Measurable | 1–3 | C1, E0, H1a, ARCH1, H2a | Harness reproduces the paper's numbers (or the gap is explained and accepted by the human) |
| v0.3 Safe | 4–6 | T5p, R1, H3, H5, ARCH2, H7 | No `pickle` in load paths; R1 tests pass |
| v0.4 Incremental | 7–10 | H8, H4, H6 | 1% update ≤ 10% of rebuild; ≥ 95% stability |
| v1.0-rc Stable API | 11–13 | I2, I4, I5, H1b, ARCH3, T-props | API frozen; release candidate tagged and published to TestPyPI only; not announced |
| v1.0 Pilot → final | 14–16 | P1, P2, P3, P4 | Pilot passes (below); human approves; then v1.0.0 to PyPI and announcement |

## v1.0 pilot criteria

- **P1 Real corpora:** 2–3 corpora outside the benchmarks, chosen by the human, index and query
  without errors; a person who knows each corpus rates answers on 20+ real questions and finds
  them at least as good as plain RAG.
- **P2 Install from the docs:** 1–2 outside developers install, index a folder and query using
  only the docs, in under an hour; every snag is filed on the board.
- **P3 Security review:** an outside reviewer checks the on-disk format and prompt-injection
  defenses; no high-severity finding left open.
- **P4 Fix findings:** every pilot issue fixed or deferred with a written reason.
- If the pilot finds a **design** problem (not just bugs), v1.0 waits until it is fixed.

## Phase 2

| Release | Weeks | Items | Release-specific gate |
| --- | --- | --- | --- |
| v1.1 Tuned | 17–21 | I1, I3, T1, T2, T3, T6 | Each change wins its own ablation; no domain down > 1 point |
| v1.2 Research | 22–27 | T4, T7, SWEEP | Only ablation-confirmed ideas ship; paper candidates written up |
