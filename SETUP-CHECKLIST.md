# SpectRAG agent team — one-page setup checklist

Local mode: no GitHub account needed. Tick each box in order.

## A. Before you start (once, about 15 minutes)

- [ ] Claude Code v2.1.259+ (`claude --version`), git, Python 3.11,
      tmux 3.0+ (`brew install tmux` / `apt install tmux`),
      Ollama with `ollama pull nomic-embed-text`
- [ ] Your SpectRAG source folder on disk (a git clone keeps history; a plain folder works)
- [ ] Claude Console: create an API key with a monthly spend limit for SpectRAG evaluation
      (Claude Haiku). Agent sessions themselves use your normal Claude Code login.
- [ ] Co-authors told that agents will prepare releases for their review (decision D1)

## B. Create the workspace (about 5 minutes, terminal)

```bash
unzip spectrag-agent-kit.zip && cd spectrag-agent-kit
scripts/setup-local.sh /path/to/your/SpectRAG        # creates ~/spectrag/
```

- [ ] Script finished: `origin.git` protected, kit committed, board created with 33 items,
      `.venv` created

## C. Manual settings (about 5 minutes)

- [ ] `~/.config/spectrag/eval.env` created with
      `SPECTRAG_ANTHROPIC_API_KEY=<key>` and `SPECTRAG_TEST_DIR=$HOME/spectrag-private/eval-test`
- [ ] Frozen test splits placed in `~/spectrag-private/eval-test/`
- [ ] (If available) the paper's original `generation_eval/` folder in
      `~/spectrag-private/original-eval/`

## D. Verify (about 15 minutes)

```bash
cd ~/spectrag/spectrag-dev && source .venv/bin/activate
claude            # accept the trust prompt once
```

- [ ] `/agents` lists 8 roles; `/hooks` shows the git guard; `/permissions` shows the rules
- [ ] Exit Claude, run `scripts/check-guards.sh` → `passed: 3  failed: 0`
- [ ] `git push origin HEAD:main` from your terminal is **rejected** ("main is protected")
- [ ] Dry run of the pipeline: `scripts/board.sh add TEST "Pipeline check" builder`, then
      `git switch -c test/pipeline && echo >> README.md && git commit -am "TEST: check" &&
      scripts/board.sh submit TEST < /dev/null`, then in `claude`: `/review-pr TEST`, then
      `scripts/merge-gate.sh` → TEST is merged (or blocked with a clear reason).
      `git switch main`, then `scripts/board.sh defer TEST "pipeline check done"`.

**Do not continue until every box in D is ticked.**

## E. Start v0.2

- [ ] In `claude`: `/kickoff v0.2` → note the batches and any `needs-human` questions
- [ ] Check `docs/agents/TEAM.conf` matches the kickoff batches
- [ ] `scripts/launch-team.sh` → one tmux window per agent (see `docs/agents/TEAM.md`).
      Batch 1 (Atlas ARCH1, Forge C1, Rivet H1a) starts at once; batch 2 (Gauge E0, Anvil H2a)
      starts by itself when C1 / ARCH1 merge; Lens reviews and merges as submissions arrive
- [ ] Answer questions as they appear in the `control` window:
      `scripts/board.sh answer <ID> "..."`, then type `force` or re-run that agent's window
- [ ] Headless alternative: `scripts/run-batch.sh C1 H1a ARCH1`, later `scripts/run-batch.sh E0 H2a`

## F. Every week (about 30 minutes)

- [ ] `/digest` → answer the decisions on the board
- [ ] Re-run anything `blocked` that its agent has not picked up: `scripts/run-item.sh <ITEM>`
- [ ] Update `TEAM.conf` for the next batch from the `/kickoff` plan
- [ ] `scripts/cleanup-worktrees.sh`
- [ ] Check spend: Claude Console and the `total_cost_usd` lines in `logs/`

## G. At each release (1–2 hours)

- [ ] `/gate <version>` → read `docs/reports/gate-<version>.md` (on the REL-<version> branch)
- [ ] Approve → `scripts/merge-gate.sh --release REL-<version>` →
      `scripts/release-local.sh <version>` → share `~/spectrag/releases/spectrag-<version>.tar.gz`
- [ ] At v1.0-rc: run the pilot (P1 via `run-item.sh`; arrange P2 outside developers and
      P3 security reviewer yourself), then P4 fixes, then `/gate v1.0`
