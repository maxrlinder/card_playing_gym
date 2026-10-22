# Published models

Each directory is a release written by `cardgym <game> export` (or, for
plump-bot checkpoints, `cardgym plump import-legacy`):

- `actor.pt`: actor weights and the `[model]` config section that builds them
- `critic.pt`: critic weights and entropy temperatures, so training can resume
- `info.json`: iteration, provenance, and file checksums

Load one with `--init-from <dir>` (training), `cardgym plump play <dir>`, or
`cardgym plump evaluate <dir>`.

| Release | Iteration | Notes |
| --- | --- | --- |
| `plump/rl-v6` | see `info.json` | 8.5M-parameter actor; see `docs/plump.md` |
| `piquet/p2` | 13600 | 9.1M-parameter single-deal actor (+13.9 points per deal against the heuristic); the base of `snapshots/piquet/partie1` and its score-blind opponent; see `docs/piquet.md` |

Training runs in progress live in `snapshots/` (see its README) with their optimizer state, so they can be resumed elsewhere.
