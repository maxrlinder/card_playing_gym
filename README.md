# Card Playing Gym

Self-play reinforcement learning for imperfect-information card games. Each
game supplies its rules, token schema, and model heads; the gym supplies the
rest: a causal-transformer trunk, KV-cached batched self-play, PPO with a
privileged critic (or the actor's own value head), run management, and a CLI.

| Game | Status |
| --- | --- |
| `plump` | Trained. The `rl-v6` model (8.5M parameters, iteration 71k) ships in `checkpoints/plump/rl-v6`. |
| `rummy` | Implemented (Scandinavian rules: 1-2 decks, 7 cards, deep discard pickups, 500 Rum scoring): engine, decision and opponent-turn tokens, card-grid model, candidate heads, continuous asynchronous self-play, fused Metal kernels ([Rummy](docs/rummy.md)). |
| `piquet` | Implemented (exchange, declarations with sinking, tricks, repique/pique/capot; single deals or six-deal Rubicon parties, each deal a fresh context with the score in its tokens): per-card ledger tokens, 4 x 8 card grid, 16-layer 9.1M actor with nine prediction heads and no critic, fused Metal grid ([Piquet](docs/piquet.md)). |

## Quick start

```bash
uv sync --dev
uv run pytest
uv run cardgym games
```

Play plump against the published model in the browser:

```bash
uv run cardgym plump play            # http://127.0.0.1:8765
```

Measure it against the rule-based heuristic on a fixed, paired deal bank:

```bash
uv run cardgym plump evaluate checkpoints/plump/rl-v6 --deals 32
```

Train. `smoke.toml` finishes in seconds on a CPU; `rl-v6.toml` is the
production preset, which continues the published model:

```bash
uv run cardgym plump train smoke --config configs/plump/smoke.toml
uv run cardgym rummy train smoke --config configs/rummy/smoke.toml
uv run cardgym piquet train smoke --config configs/piquet/smoke.toml

CARDGYM_METAL_TRUNK=1 uv run cardgym plump train rl-v7 \
  --config configs/plump/rl-v6.toml --init-from checkpoints/plump/rl-v6
```

Re-running the same command resumes. A run refuses a changed config unless
you pass `--reconfigure --reason "..."`, which archives the old config and
records the change. Override single values with `--set section.key=value`.
For long MPS runs, `tools/train_supervised.sh plump RUN CONFIG` restarts the
process every 1000 iterations to shed native heap MPS leaks.
`tools/bench.py` records a run's batches and times its update and rollout.
`tools/run_snapshot.py pack|unpack` carries a run (checkpoint, optimizer state,
metrics, dashboard) to another machine through git; see `snapshots/README.md`.

## Commands

```text
cardgym games
cardgym <game> train RUN --config FILE [--set k=v] [--init-from DIR] [--reconfigure --reason ...]
cardgym <game> dashboard RUN              render runs/<game>/RUN/dashboard.png
cardgym <game> export RUN --out DIR       publish actor/critic weights as a release
cardgym <game> grow NEW --from RUN --config FILE --fresh 4,5,...   fork a run into a deeper trunk
cardgym plump evaluate MODEL [--opponent heuristic|MODEL] [--deals N] [--mode argmax|sample]
cardgym plump play [MODEL]
cardgym plump import-legacy CHECKPOINT --out DIR    convert a plump-bot checkpoint
cardgym rummy evaluate MODEL [--opponent heuristic|random] [--players 2,3,4] [--decks 1,2] [--deals N]
cardgym rummy simulate [--policy heuristic|random]   rule-based statistics, no model
cardgym piquet evaluate MODEL [--opponent heuristic|random | --against MODEL] [--mode deal|rubicon] [--deals N]
cardgym piquet simulate [--policy heuristic|random] [--opponent heuristic|random]
```

## Layout

```text
cardgym/core/         run directories, TOML config, metrics, dashboard, devices
cardgym/nn/           GQA decoder trunk, KV cache, fused Metal trunk (MPS, any length, packed rows),
                      fused field lookups
cardgym/rl/           self-play collectors (batch, continuous, async), batches, PPO trainer, checkpoints, run loop
cardgym/games/base.py the contract a game implements
cardgym/games/plump/  engine, heuristic, tokens, labels, model, training hooks,
                      evaluation, browser GUI, legacy import
cardgym/games/rummy/  engine and meld search, ledger tokens, card-grid model,
                      candidate heads, training hooks, evaluation
cardgym/games/piquet/ engine and declaration tables, ledger tokens, card-grid model,
                      prediction heads, training hooks, heuristic, evaluation
configs/<game>/       presets
checkpoints/<game>/   published releases (actor.pt, critic.pt, info.json)
tests/                engine, trunk, PPO, plump, rummy and piquet streams, release regression, CLI
```

Run artifacts go to `runs/<game>/<run>/` (git-ignored; `CARDGYM_RUNS_DIR`
relocates them).

## Documentation

- [Architecture](docs/architecture.md): how self-play, batching, PPO, and runs fit together.
- [Plump](docs/plump.md): rules, token schema, model, auxiliary heads, and the rl-v6 lineage.
- [Rummy](docs/rummy.md): the state-representation design (card ledger, card grid, candidate heads).
- [Piquet](docs/piquet.md): rules, what sets it apart from plump and Rummy, the critic-less actor and its prediction heads.
- [Adding a game](docs/adding-a-game.md): the contract a game implements.

## Requirements

Python 3.11 or 3.12. The lock file pins a PyTorch nightly for Apple-silicon
macOS: the fused Metal trunk uses `torch.mps.compile_shader`. Everything else
runs on any PyTorch with `scaled_dot_product_attention(enable_gqa=...)`.

On Linux with an NVIDIA GPU the lock does not apply; build the environment by
hand:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv/bin/python -e . pytest ninja
```

CUDA runs compile the trunk (`torch.compile`), replay every rollout wave as a
CUDA graph, and build a small decode-attention extension on first use (nvcc
and ninja); see [Piquet on CUDA](docs/piquet.md#cuda).
