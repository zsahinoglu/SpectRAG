# SpectRAG backlog (agent-readable)

Source: "SpectRAG — Production Library Plan". One row per item. Agents claim an item on the
local board (`scripts/board.sh claim <ID>`), never by editing this file. Only the release manager
edits this file, and only to add sub-items or correct dependencies.

Column order is fixed; `scripts/run-item.sh` reads the Owner column.

## Phase 1 — production fixes and prep

| ID | Title | Release | Owner | Depends on | Done when |
| --- | --- | --- | --- | --- | --- |
| E0 | Evaluation harness in `eval/` | v0.2 | evaluator | C1 | `python -m eval smoke` runs in < 5 min with no paid API calls; `run --dev` and `run --release` work with Claude Haiku (`claude-haiku-5-5`) as index-time LLM, generator and judge; the paper's numbers (produced with gpt-4o-mini) are reproduced, or the gap and its share due to the model change are explained in `docs/evaluation/reproduction.md` |
| H1a | Golden tests with a deterministic fake embedder | v0.2 | builder | — | A fixed small corpus plus fake embedder gives byte-identical retrieval across runs; tests cover each query stage; no network needed |
| ARCH1 | Package skeleton and interface ADRs | v0.2 | architect | — | `spectrag/` package with `__init__`, `settings`, and stub `Protocol` interfaces (Embedder, IndexLLM, VectorStore, GraphStore, Reranker); ADRs 0001–0003 written; old imports still work via shims |
| H2a | Installable package | v0.2 | builder | ARCH1 | `pyproject.toml`; `pip install -e .` works in a clean venv; core install has no PyTorch; extras `[neo4j]`, `[gliner]`, `[anthropic]`, `[openai]`, `[ollama]`, `[rerank]`, `[dev]`; `anthropic` is in the default install |
| C1 | Claude Haiku backend for index-time LLM and generation | v0.2 | builder | — | `utility_question_llm_backend`, concept extraction and the generator accept `"anthropic"`; default model `claude-haiku-5-5`, pinned; key read from `SPECTRAG_ANTHROPIC_API_KEY` (fallback `ANTHROPIC_API_KEY`); same disk caching and retry behavior as today; JSON outputs schema-validated; unit tests use a mocked client; indexing and generation run with no OpenAI key; defaults in `config.py` switched to anthropic |
| T5p | Drift-bound proposal (perturbation theory) | v0.3 | theory | — | Proposal in `docs/theory/T5-drift-bound.md` with derivation, rebuild-trigger formula and ablation plan; red-team objections answered |
| R1 | Red-team baseline sweep | v0.3 | red-team | H1a | Failing tests exist for A3 (pickle load), A13 (injection, poisoning, hub concepts) |
| H3 | Safe on-disk format | v0.3 | builder | H2a, R1 | No `pickle` in any load path; Parquet/Arrow + `.npy` + JSON manifest with version and checksums; one-way migration tool; R1's A3 tests pass |
| H5 | Spectral robustness | v0.3 | builder | H1a | Isolated chunks get zero spectral credit; ARPACK non-convergence retries then raises; residual norms logged and written to the manifest |
| ARCH2 | Embedder and LLM interfaces wired in | v0.3 | architect | ARCH1 | Ollama, OpenAI and sentence-transformers embedders and Anthropic (default)/Ollama/No LLMs implement the Protocols; OpenAI LLM only as an optional extra; pipelines take them by injection |
| H7 | Untrusted-content defenses | v0.3 | builder | R1 | Schema-validated LLM outputs, delimited untrusted text, hub-concept cap; R1's A13 tests pass; demo app binds localhost with a token, no default passwords |
| H8 | One graph-store contract | v0.4 | builder | ARCH2 | `GraphStore` conformance suite; file store and Neo4j give identical retrievals on the golden corpus |
| H4 | Incremental indexing with aligned spectra | v0.4 | builder | H3, H8, T5p | `add`/`remove` process only changed docs; warm-started LOBPCG + Procrustes alignment; drift trigger from T5; 1% update on 100k chunks ≤ 10% of rebuild time; ≥ 95% top-10 stability |
| H6 | Scale and latency | v0.4 | builder | H3 | Benchmarks at 10⁴/10⁵/10⁶ chunks published; no per-query full-corpus Python loops (A10) |
| I2 | Cheaper index-time LLM calls | v1.0 | builder | E0, H7 | One validated JSON call for concepts + questions; LLM cost per chunk down ≥ 50% with evidence recall within 1 point |
| I4 | Presets replace mode sprawl | v1.0 | tuner | E0 | 4 presets (balanced, precision, recall, cheap) chosen on dev splits by worst-domain score; losing modes moved to `spectrag.experimental` |
| I5 | `ctx.explain()` | v1.0 | builder | ARCH2 | Every selected chunk carries seed source and score parts; tested |
| H1b | Coverage target | v1.0 | builder | H1a | Line coverage ≥ 80% on `index/` and `retrieve/`, enforced in CI |
| ARCH3 | API freeze and app split | v1.0 | architect | I4, I5 | Public API documented and pinned by `tests/api/`; web app moved to `spectrag-app`; diffusion and selection are pluggable components |
| T-props | Proposals for T1–T4, T6, T7 | v1.0 | theory | — | One proposal document per item in `docs/theory/`, each red-teamed; no library code |
| P1 | Pilot on 2–3 real corpora | v1.0 | evaluator | ARCH3 | Human-chosen corpora outside the benchmarks index and query without errors; answer ratings from a domain expert on 20+ real questions recorded in `docs/evaluation/pilot.md` |
| P2 | Install-from-docs test by outside developers | v1.0 | human | ARCH3 | 1–2 outside developers succeed in under an hour from the docs alone; every snag filed on the board (`scripts/board.sh file`) |
| P3 | External security review | v1.0 | human | ARCH3 | Review of on-disk format and injection defenses completed; findings filed on the board; no high-severity finding open |
| P4 | Fix pilot findings | v1.0 | builder | P1, P2, P3 | Every pilot issue fixed or deferred with a written reason; v1.0.0 released |

## Phase 2 — performance (starts after v1.0)

| ID | Title | Release | Owner | Depends on | Done when |
| --- | --- | --- | --- | --- | --- |
| I1 | Precision preset | v1.1 | tuner | ARCH3 | Precision@5 up with ≤ 2 points evidence-recall loss at 12k tokens |
| I3 | Per-domain tuning | v1.1 | tuner | ARCH3 | Worst-domain score improves; no domain down > 1 point |
| T1 | Eigenvalue-weighted (heat-kernel) spectral proximity | v1.1 | theory | T-props | Ablation confirms the predicted effect, then a builder implements it as a preset |
| T2 | One parameterized diffusion replacing the 50/50 hybrid | v1.1 | theory | T-props | Same as T1 |
| T3 | Eigengap choice of k | v1.1 | theory | T-props | Same as T1 |
| T6 | Normalization effects on the Laplacian | v1.1 | theory | T-props | Same as T1 |
| T4 | Selection objective with guarantees | v1.2 | theory | T-props | Same as T1; or a paper draft |
| T7 | Adaptive token budget (optimal stopping) | v1.2 | theory | T-props, decision D4 | Same as T1 |
| SWEEP | Physics and biology idea sweep | v1.2 | theory | ARCH3 | ~10 ranked candidates with predicted effects; top 3 ablated |
