# Architecture

The gym trains one policy by self-play: every seat at the table is the policy
being trained, and every seat's decisions train it. A game plugs in its rules,
its token schema, and its model heads ([Adding a game](adding-a-game.md));
everything on this page is shared.

```text
game.new_games()        one update's schedule of fresh deals
   |
SelfPlayCollector       KV-cached decoding of every game at once (cardgym/rl/selfplay.py)
   |  Trajectory: each seat's token stream, decisions (behavior probs), returns
game.sequences()        actor stream per seat + one critic stream, with labels
   |
build_batch             bucketed groups, per-head decision rows, loss weights (rl/batch.py)
   |
PPOTrainer.update       critic baseline -> advantages -> actor step -> critic epochs (rl/trainer.py)
```

## Streams and the trunk

Each seat sees its own causal token stream: the deal it may see, then every
public event, from its own point of view (player fields are relative to the
observer). A token is a fixed-width row of categorical fields. The policy reads
the hidden state at the last token before a decision.

The game's embedder turns a row into the sum of one embedding per present
field. The shared `Trunk` (`cardgym/nn/trunk.py`) then applies:

* RMSNorm on the embedding sum;
* a stack of pre-LN decoder blocks with grouped-query attention (`n_kv_heads`
  < `n_heads` keeps the KV cache small);
* per-layer embeddings (PLE): a packed lookup of the same token fields plus a
  projection of the normalized input, added after every block;
* value embeddings: a packed lookup added to attention V (alternating or all
  blocks), before the cache write;
* a final LayerNorm.

PLE and value embeddings start at zero, so enabling them leaves the function
unchanged until they learn something.

## Self-play collection

`SelfPlayCollector` decodes the whole schedule in one continuously batched
loop. Each observer of each game owns one KV-cache row with its own length:

1. **Prefill.** Every observer's prefix is encoded; rows with equal prefix
   lengths share one forward.
2. **Wave.** Every live game makes one decision. The actor's logits come from
   its row's last hidden state (and, for heads that want raw fields, its
   last token); actions are sampled from the masked softmax with the game's
   own RNG, and recorded with the full behavior distribution.
3. **Append.** `game.step(action)` returns each observer's new tokens. Rows
   are padded to the wave's longest run and decoded in one ragged forward.
   Each row writes at its own offset and attends only to its own prefix. A
   padded token lands past the row's length, where it is overwritten later
   and never attended to meanwhile.

Live rows must stay a dense prefix of the pool, so attention can read rows
`0..B-1` as a zero-copy slice. Games are dealt longest-lived first, so they
retire from the tail. Out-of-order retirement triggers a compaction that
slides survivors down. Measured on plump, merging the 24 round shapes into
one loop cut collection 2.9x versus one lockstep loop per shape.

## Training batch

For each finished game the batch holds:

* one **actor stream** per learning seat, with the game's per-position labels;
* one **critic stream**. Plump's is a perfect-information "oracle" stream,
  with every hand visible and absolute seat ids, so its value columns name
  seats directly;
* every decision, addressed into both: its actor row and position, and the
  critic position of the same pre-action state.

Weights: each of G games carries mass 1/G, split evenly over its P learning
seats. Every policy row has weight `1 / (G P)`, so longer games contribute
more decisions. Dense per-position losses (values, auxiliaries) spread each
stream's mass evenly over its genuine positions, so they do not grow with
length. Streams are tail-padded into length buckets; causal attention makes
that exact, since padding follows every readout.

## The PPO update

1. **Baseline.** The frozen critic evaluates every critic stream. Its
   per-seat outputs are projected onto the zero-sum subspace (relative rewards
   sum to zero in every game). The baseline is the acting seat's value at the
   pre-action state. Advantages are Monte Carlo `R - V(s)` or GAE over the
   game's decision states, then weight-normalized. A critic that sees hidden
   cards is still a valid baseline: it conditions on the state, not on the
   action about to be sampled.
2. **Actor.** All microbatches accumulate into one Adam step:
   * clipped PPO loss per policy head, fp32 masked log-softmax;
   * minus `alpha_h * H_norm`, where `H_norm = H / log|legal|`. Each head's
     temperature adapts toward its target and the targets can follow a
     schedule;
   * the game's auxiliary losses, which differentiate through the shared trunk;
   * optionally, a per-seat value probe that never feeds advantages.

   With `auxiliary_grad_fraction > 0`, the auxiliary and policy gradients are
   taken separately, and the auxiliary gradient is rescaled to that fraction
   of the policy gradient's norm before the single global clip. Adam is
   invariant to a uniform rescale, so only the ratio matters. The fused Metal
   trunk serves both cotangents with one backward.
3. **Critic.** `critic_epochs` full passes, each one Adam step: dense per-seat
   value MSE plus the game's critic auxiliaries.

`behavior_kl` in the metrics compares the cached, reduced-precision rollout
policy with the full-sequence replay. It should stay at numerical noise; a
rise means rollout and training disagree about the inputs.

## Precision and speed

* bf16 autocast for compute; fp32 master weights, Adam state, and every
  probability-sensitive quantity. KV cache in fp16 (faster than bf16 for
  cached attention on MPS).
* `CARDGYM_METAL_TRUNK=1` on Apple silicon routes the trunk (forward,
  backward, and cached decode) through fused Metal kernels
  (`cardgym/nn/metal_trunk.py`) for sequences up to 128 tokens. It halved
  plump's iteration time and matches autocast within bf16 rounding.
* `microbatch_positions` is the update's memory lever; it does not change the
  accumulated loss. Attention activations grow with stream length, so long
  streams (Rummy's reach 400 tokens) need smaller microbatches.
* Gradients are given their parameter's layout before each optimizer step:
  fused MPS Adam rejects mismatched strides, which MPS Conv1d backward
  produces.
* Each iteration runs inside an Objective-C autorelease pool, and
  `--recycle-after` bounds the remaining MPS native-heap growth.

## Runs, checkpoints, releases

`runs/<game>/<run>/` holds the resolved `config.toml`, `metrics.csv`,
`train.log`, `dashboard.png`, and `checkpoints/iter_*.pt`. Checkpoints carry
the full training state: models, optimizers, entropy temperatures, RNGs. A
`latest.json` manifest with size and sha256 marks the resume point. A run
lock prevents two trainers from writing one run.

`cardgym <game> export` writes a **release**: `actor.pt` and `critic.pt`
(weights plus the `[model]` section) and `info.json` (provenance). Releases
are what gets published; `--init-from` starts a new run from one, with fresh
optimizer moments under the learning-rate warmup.
