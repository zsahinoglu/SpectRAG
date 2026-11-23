# SpectRAG agent team — roster

Each agent has a name, a role (its file in `.claude/agents/`) and its own tmux window.
`scripts/launch-team.sh` opens them all from `docs/agents/TEAM.conf`; edit that file at each
`/kickoff` to move agents onto the next release's items.

| Name | Role | Model | Lane (may edit) | v0.2 assignment | Then |
| --- | --- | --- | --- | --- | --- |
| **Atlas** | architect | opus | `spectrag/` structure, `docs/` | ARCH1 package skeleton + ADRs (starts now) | ARCH2 (v0.3), ARCH3 (v1.0) |
| **Forge** | builder | sonnet | files for its one item + tests | C1 Claude Haiku backend (starts now) | H3, H4 |
| **Rivet** | builder | sonnet | files for its one item + tests | H1a golden tests (starts now) | H5, H6, H1b |
| **Anvil** | builder | sonnet | files for its one item + tests | H2a installable package (after ARCH1) | H7, H8, I2, I5 |
| **Gauge** | evaluator | opus | `eval/`, `docs/evaluation/` | E0 eval harness (after C1) | scores PRs, P1 pilot |
| **Fourier** | theory | opus | `docs/theory/`, `experiments/` | standby — T5p drift bound (no deps) | T-props |
| **Raven** | red-team | opus | `tests/security/`, `tests/redteam/`, `docs/redteam/` | standby — R1 (after H1a) | attacks every theory PR |
| **Dial** | tuner | sonnet | presets, `experiments/`, `docs/tuning/` | standby — I4 (after E0) | I1, I3 (Phase 2) |
| **Harbor** | release-manager | sonnet | `CHANGELOG.md`, version, `docs/reports/`, `docs/agents/` | standby — `/digest`, `/gate v0.2` | every gate |
| **Lens** | pr-reviewer | opus | nothing (board verdicts) | reviews each submission, then runs the merge gate | every submission |

You are the only one who answers `needs-human` questions (`scripts/board.sh answer`), merges
release items (`scripts/merge-gate.sh --release`), and runs `scripts/release-local.sh`.

## Windows

- `control` — your `needs-human` decisions, items in flight, recent merges to main.
- One window per name above. `auto` agents start when every "Depends on" item has a merged
  PR; `hold` agents wait for Enter. Type `force` in a waiting window to start early.
- Default mode `watch` runs interactive sessions: follow each agent and answer permission
  prompts in its window. Interactive sessions have no USD cap;
  Lens's reviews are headless and capped by `SPECTRAG_REVIEW_BUDGET` (default $5). Set
  `SPECTRAG_AUTOMERGE=0` to have Lens review only and leave merging to you; use
  `SPECTRAG_TEAM_MODE=auto scripts/launch-team.sh` for headless runs capped by
  `SPECTRAG_ITEM_BUDGET` (default $20 per item).

tmux keys: `Ctrl-b w` window list · `Ctrl-b n` / `p` next / previous · `Ctrl-b d` detach ·
`tmux attach -t spectrag` to return.
