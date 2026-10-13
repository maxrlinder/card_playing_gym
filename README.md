# Card Playing Gym

Self-play reinforcement learning for imperfect-information card games. Each
game supplies its rules, token schema, and model heads; the gym supplies the
rest: a causal-transformer trunk, KV-cached batched self-play, PPO with a
privileged critic, run management, and a CLI.

| Game | Status |
| --- | --- |
| `plump` | Trained. The `rl-v6` model (8.5M parameters, iteration 71k) ships in `checkpoints/plump/rl-v6`. |
| `rummy` | Next: see [Adding a game](docs/adding-a-game.md). |

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

CARDGYM_METAL_TRUNK=1 uv run cardgym plump train rl-v7 \
  --config configs/plump/rl-v6.toml --init-from checkpoints/plump/rl-v6
```

Re-running the same command resumes. A run refuses a changed config unless
you pass `--reconfigure --reason "..."`, which archives the old config and
records the change. Override single values with `--set section.key=value`.
For long MPS runs, `tools/train_supervised.sh plump RUN CONFIG` restarts the
process every 1000 iterations to shed native heap MPS leaks.

## Commands

```text
cardgym games
cardgym <game> train RUN --config FILE [--set k=v] [--init-from DIR] [--reconfigure --reason ...]
cardgym <game> dashboard RUN              render runs/<game>/RUN/dashboard.png
cardgym <game> export RUN --out DIR       publish actor/critic weights as a release
cardgym plump evaluate MODEL [--opponent heuristic|MODEL] [--deals N] [--mode argmax|sample]
cardgym plump play [MODEL]
cardgym plump import-legacy CHECKPOINT --out DIR    convert a plump-bot checkpoint
```

## Layout

```text
cardgym/core/         run directories, TOML config, metrics, dashboard, devices
cardgym/nn/           GQA decoder trunk, KV cache, fused Metal trunk (MPS)
cardgym/rl/           self-play collector, batches, PPO trainer, checkpoints, run loop
cardgym/games/base.py the contract a game implements
cardgym/games/plump/  engine, heuristic, tokens, labels, model, training hooks,
                      evaluation, browser GUI, legacy import
configs/<game>/       presets
checkpoints/<game>/   published releases (actor.pt, critic.pt, info.json)
tests/                engine, trunk, PPO, plump streams, release regression, CLI
```

Run artifacts go to `runs/<game>/<run>/` (git-ignored; `CARDGYM_RUNS_DIR`
relocates them).

## Documentation

- [Architecture](docs/architecture.md): how self-play, batching, PPO, and runs fit together.
- [Plump](docs/plump.md): rules, token schema, model, auxiliary heads, and the rl-v6 lineage.
- [Adding a game](docs/adding-a-game.md): the contract, with notes for Rummy.

## Requirements

Python 3.11 or 3.12. The lock file pins a PyTorch nightly for Apple-silicon
macOS: the fused Metal trunk uses `torch.mps.compile_shader`. Everything else
runs on any PyTorch with `scaled_dot_product_attention(enable_gqa=...)`.
