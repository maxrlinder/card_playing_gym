# Run snapshots

A snapshot is a training run's latest state, packed by `tools/run_snapshot.py`
so that it fits in git (GitHub refuses files over 100 MB). It holds:

- `checkpoint.pt`: weights, entropy temperatures, RNG and iteration;
  everything except the optimizer.
- `optimizer.pt`: the actor's Adam moments.
- the run's `config.toml`, `metadata.json`, `metrics.csv`, `train.log` and
  `dashboard.png`.
- `snapshot.json`: checksums.

To continue a run on another machine:

```bash
.venv/bin/python tools/run_snapshot.py unpack snapshots/piquet/partie1   # rebuilds runs/piquet/partie1
CARDGYM_METAL_TRUNK=1 tools/train_supervised.sh piquet partie1 configs/piquet/partie.toml
```

`unpack` puts the checkpoint back together and checks the files' checksums.
The train command then resumes exactly where the run stopped: weights, Adam
state, entropy temperatures, RNG, iteration count, metrics and dashboard.
`CARDGYM_METAL_TRUNK=1` applies only on Apple silicon; elsewhere leave it
out. `[run] device = "auto"` picks CUDA, MPS or CPU.

A run resumes only under the config it recorded (`--reconfigure` records a
change). `partie1` last trained on CUDA, so its snapshot carries
`partie-cuda.toml`: fp16, two rollout workers, 8 GB microbatches. The
training itself is the same as `partie.toml`. On an NVIDIA GPU, supervise it as
it is:

```bash
tools/train_supervised.sh piquet partie1 configs/piquet/partie-cuda.toml
```

On Apple silicon, switch back to the MPS execution settings once, then
supervise with them:

```bash
.venv/bin/python -m cardgym piquet train partie1 --config configs/piquet/partie.toml \
  --reconfigure --reason "MPS execution settings" --prepare-only
CARDGYM_METAL_TRUNK=1 tools/train_supervised.sh piquet partie1 configs/piquet/partie.toml
```

To pack a run again afterwards:

```bash
.venv/bin/python tools/run_snapshot.py pack runs/piquet/partie1 snapshots/piquet/partie1
```

| Snapshot | Iteration | Notes |
| --- | --- | --- |
| `piquet/partie1` | 12700 | Rubicon parties, fine-tuned from p2 at iteration 13600 (`checkpoints/piquet/p2`); see `docs/piquet.md#parties`. Iterations 1-300 on the Mac, 301-12700 on an RTX 2070 Super |

Throughput on the Apple M5 Pro laptop (24 GB, MPS, bf16, the Metal trunk,
one asynchronous rollout worker); for the RTX 2070 Super see
`docs/piquet.md#cuda` (`partie1`: 3.6-3.9 s per update).

- **`partie1`:** 3.7-3.8 s per update of 192 parties (1152 deals, ~36k
  decisions). That is about 9.5k decisions per second, or roughly
  1.1M deals (180k parties) an hour. The update takes 2.8-3.2 s, and the
  rollout of the next batch (3.3 s) runs alongside it.
- **p2, single deals (`base.toml`):** 3.1-3.3 s per update of 1024 deals
  (~32k decisions), about 10k decisions per second.
- **Memory:** the trainer uses ~8 GB and the rollout worker ~2.5 GB.
