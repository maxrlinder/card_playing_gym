# Plump

## Rules

Three to five players, three to ten cards, one round per game for training.
Each player bids how many tricks they will take; the last bidder may not make
the bids sum to the hand size. The highest bidder leads (earliest bidder on a
tie). Players must follow suit; there is no trump, so the highest card of the
led suit wins. Hitting a bid exactly scores 10 + bid (5 for a successful zero
bid); missing scores 0. The training reward is each seat's score minus the
mean opponent score, which is zero-sum.

The engine (`env.py`, `rules.py`, `state.py`) also supports multi-round games
and trump variants; the GUI uses multi-round games.

## Token stream

```text
[GAME] [HAND x N] [TURN] [BID] { [TURN] [BID] } x (P-1)  { [PLAY x P] [TRICK_WIN] } x N
```

* The TURN token ("seat r is about to bid") is the bid head's readout. A play
  is read off the token before it, whose next-actor/next-phase fields
  announce who acts.
* Bidding-state fields (bids made, bid sum, hand size minus bid sum, bids
  remaining) ride on GAME, HAND, TURN, and BID rows.
* TRICK_WIN rows also carry the observer's remaining hand.
* The last token field is the id of the row's pattern of absent fields. One
  learned row per pattern replaces per-field "not applicable" rows.

`tokens.py` builds streams two ways: incrementally per self-play step
(`wave_rows`) and from scratch (`seat_tokens`, used by evaluation and the
GUI). Tests require the two to agree exactly.

## Model

`PlumpActor` (8.5M parameters as configured in `rl-v6`): 16 layers,
d_model 240, 10 query heads over 2 KV heads, feed-forward width 720,
64-wide PLE, value embeddings on alternating blocks. A card is embedded and
scored as rank + suit only, so rank behavior is shared across suits and suit
behavior across ranks.

Heads:

* `bid` over 0..10 and `play` over 52 cards (masked to legal actions);
* a per-seat value probe (observer-relative, zero-sum), which never feeds
  advantages;
* auxiliary beliefs, trained densely from the opening bid onward:
  * opponent suit presence;
  * rank boundaries: does an opponent hold a card below or above my lowest
    or highest card of that suit?
  * every seat's final trick count, restricted to feasible totals;
  * the seat that takes the next trick;
  * bid hit, an MLP head because "won exactly the bid" is a bump, not a
    threshold.

`PlumpCritic` reads the oracle stream, which replaces the observer's hand
with all P x N dealt cards tagged by absolute owner. It predicts per-seat
values, final trick counts, the next trick winner, and each seat's remaining
play order (a Plackett-Luce likelihood within each hand).

## Training (`configs/plump/rl-v6.toml`)

Every update plays 116 deals for each of the 24 (players, hand size) shapes:
2784 rounds and roughly 72k decisions. Every seat learns. The production
settings:

* PPO with clip 0.1 and Monte Carlo advantages from the oracle critic;
* 25% auxiliary gradient budget;
* adaptive entropy floors of 0.05 (bid) and 0.10 (play);
* actor learning rate 9e-6, critic 3e-4 with two epochs;
* bf16 with an fp16 KV cache, and the fused Metal trunk.

On an M5 Pro this ran about 31 s per iteration (7 s collection, 24 s update).

### rl-v6 lineage

rl-v6 was trained in the predecessor repository (`plump-bot`) with the same
algorithm this package implements. Its 16-layer student was distilled from a
6-layer PPO teacher (v4-5), then trained with self-play PPO from iteration 0:

| Iterations | Change |
| --- | --- |
| 0 | entropy floors 0.25 (bid) / 0.25 (play) |
| 9.2k-14.9k | floors ramped to 0.10 / 0.15 |
| 23.6k | learning rate 3e-5 -> 4.5e-5 |
| 52.7k | fused Metal trunk; deals per shape 29 -> 58 |
| 56.2k-66k | cooldown: rate 4.5e-5 -> 4.5e-6, floors -> 0.05 / 0.10 |
| 71.1k | rate 9e-6; deals per shape 116 |

Against the heuristic (`cardgym plump evaluate`), the release scores about +2
relative points per round with a 0.60 bid-hit rate. The old pipeline measured
+2.09 (95% CI 1.96-2.22) at iteration 45k on 64 deals per shape.

The dropped NeuRD, sampled-mirror, counterfactual-branching, distillation,
league, and multi-actor code paths remain in the `plump-bot` history.

### Importing later plump-bot checkpoints

```bash
cardgym plump import-legacy ../plump-bot/runs/rl-v6/checkpoints/iter_NNNNNN.pt \
  --out checkpoints/plump/rl-v6
```

The importer drops the factored model's dummy exact-card embedding row,
merges the bidding-state table into the field table, and renames the trunk.
`tests/plump/test_plump_release.py` checks the published release against
logits and values the original code produced from the same checkpoint.

The original evaluator and GUI encoded the opening bidder's first decision
with a duplicated TURN token (seven tokens where training used six). The
reference builder here matches training.
