# Rummy

## Rules

Two 52-card decks without jokers (104 cards), 2-4 players, 13 cards each by
default (`[game] hand_sizes`), cyclic runs, a 30-point opening. A turn is: draw
from the stock or take the discard top, make any number of melds and
lay-offs, discard. Discarding your last card wins: everyone else loses the
value of their hand (Ace 11, faces 10, pips face value) and the winner gains
the sum, doubled for going Rummy. Drawing the last stock card ends the round
at the end of that turn; if nobody went out, everyone loses their hand value.
The training reward is a seat's score minus the mean opponent score.

Interpretations (from `rummy-bot`'s `docs/rules.md`, kept as written):

* a group is 3-4 cards of one rank in distinct suits, even with two decks;
* a run is 3-13 cards of one suit on a contiguous arc of the rank cycle
  (`Q-K-A`, `K-A-2`); in runs an Ace is 1 when it starts the run and 11 when
  a King precedes it (always 11 in a full 13-card run);
* the opening may span several melds; a meld or lay-off never takes the last
  card (you win by discarding it); `max_turns` bounds discard-take cycles and
  scores like a stock-out.

`engine.py` implements this; `melds.py` enumerates the 589 valid melds once.

## The representation problem

A Rummy decision depends on three things:

1. **the viewer's own hand**: up to 14 cards, changing every turn by draws,
   melds, lay-offs, and discards;
2. **the public table**: the melds, the discard pile and its top, hand sizes,
   who has opened;
3. **beliefs** about the hidden cards (opponents' hands, the stock), inferred
   from who took or passed on which discard, what they threw, what they melded.

Items 1 and 2 are deterministic functions of the history. A causal
transformer over the event history can rebuild them in principle (a card is
in my hand if the last event that moved it moved it to me), but that is set
algebra over the whole history, redone at every position for ~50 card types,
before the model can start on item 3, which is the actual inference problem.
In plump the hand only ever shrinks, so the history is enough. In Rummy,
cards also arrive, so item 1 and 2 must be handed to the model directly.

There are three places a per-position state can go:

| Placement | Extra tokens | Extra work per token (base preset) | Problem |
| --- | --- | --- | --- |
| state as tokens (one per card) | 30-75 per event | attention over 30-75x longer streams | sequence length explodes |
| cross-attention to the visible cards at every layer | none | K/V projections of every item in every layer: 1.3-4x the whole trunk for 30-75 cards | nothing is amortized; every position and layer re-projects its own memory |
| **state in the token's own embedding** | **none** | **one card-grid pass before the trunk: ~0.3x the trunk** | **must not lose information (below)** |

Two misconceptions are worth stating:

* **The KV cache is not the obstacle.** A position that cross-attends (or
  embeds) the state of its own moment never needs recomputing: later states
  do not change what was true then. The cost of cross-attention is the
  per-position, per-layer memory projection, not cache invalidation.
* **Summing cards does not collide by itself.** `s7 + h5 = s5 + h7` holds
  for a *factored* embedding (card = rank + suit, as in plump), whose 52
  vectors span only 17 dimensions. With one independent row per card type
  the sum of a multiset is injective (any card count is a linear readout once
  `d >= 52`). Rummy goes further: each card type is its own categorical field.

## Tokens: one token per own decision

A seat's stream holds exactly one token per decision that seat makes: the
observation it decides from.

```text
seat 0:  [obs before decision 1] [obs before decision 2] ...
```

Opponents' turns are never tokens of their own in it. A token states the
present and summarizes everything since the seat's previous token:

* event fields (`tokens.py`): the decision phase (draw or play), the seat's
  own previous action (kind, card, meld kind and size), per opponent whether
  it drew from the stock or took the discard top since then (and which card
  it took) and what it discarded, stock size, the four hand sizes, who has
  opened, opening points laid this turn, players, round;
* the **ledger**: 52 codes, one per card type: the locations of its two
  copies as an unordered pair over 8 locations (36 codes), times who moved the
  card type since the seat's previous token (nobody, the seat itself, or
  opponent 1-3), times whether a copy can be laid off on the table: 360
  codes. The seat's locations are: unseen, its hand, each opponent's publicly
  known holdings (cards it took from the discard pile and has not played
  since), table, discard pile, discard top.

So a game costs `decisions + seats` actor positions, not `seats x events`
(3.5x fewer at three players), streams are tens of tokens (the longest
possible is `2 * ceil(max_turns / 2) + 104`), every position is a decision,
and the KV cache only grows for the seat that acts. What is given up is the
order of an opponent's melds within its turn, which the policy has no use for.

The ledger is exactly the viewer's knowledge: a test rebuilds it, and the
mover codes and opponent fields, from the events the viewer saw
(`tests/rummy/test_rummy_streams.py`).

The critic reads a per-seat **oracle** stream with the same positions and
layout, except that location `1 + j` is seat `j`'s whole hand, "unseen" is
the stock, stock draws show their card, and one more field shows the next
stock card. Its value head is seat-relative like the actor's.

## The card grid

`CardGrid` (`model.py`) turns a ledger into per-card features and a token
summary. Rummy's rules are invariant to permuting suits and, because runs are
cyclic, to rotating ranks; only card points break the rank symmetry. So:

* card type `c` starts as `code_embedding[ledger[c]] + rank_embedding[rank(c)]`
  (width 32);
* each block mixes a circular convolution along the 13-rank cycle (runs: is
  the 6 in my hand next to a 5 and a 7, are the outs dead) with each rank's
  mean over suits (groups), then a residual MLP;
* a per-card linear map flattens the 52 x 32 grid into the token's d-wide
  embedding, added to the event-field sum. Keeping per-card weights (rather
  than pooling) preserves card identity.

It runs once per token and is part of the token's input, so rollout caches
it implicitly.

Execution is shaped for MPS, where reductions and broadcasts over a middle
axis are pathologically slow: the rank convolution runs as one
block-circulant GEMM per suit row (built from the Conv1d weights and cached
between optimizer steps), and suit pooling and its broadcast are [1, 4] and
[4, 1] matmuls. With `CARDGYM_METAL_TRUNK=1` the narrow per-card work
(embedding + RMSNorm, pooling, the across projection, GELU, the output
projection, residual, next norm) runs in fused kernels (`metal_grid.py`),
forward and backward.

## Actions as candidates

The engine enumerates every legal action; the policy scores the list:

* `draw` head: stock, or take the discard top;
* `play` head: every distinct meld the hand can form, every distinct lay-off
  (card, target meld), every distinct discard. Discarding ends the turn, so
  "stop melding" needs no action of its own.

The opening is laid one meld at a time. A meld is legal only if the rest of
the hand can still reach the threshold while keeping a card to discard
(`melds.opening_melds`, an exact search), and until it does the seat may only
continue melding. Copies of a card are interchangeable, so a discard is one
candidate per card type and identical target melds are one candidate.

`CandidateScorer` embeds a candidate as its kind, plus the sum of its cards
(their grid features at the decision point, plus an identity row), plus for a
lay-off the sum of the target meld's cards. It adds a projection of the
readout hidden state and scores the result with a small MLP (additive
attention). The grid features are the ones the embedding computed for the
readout token (`pop_context`), so the card-level view is exact even where the
token summary blurs, at no extra encoding.

## Training

Rummy uses the shared pipeline ([architecture](architecture.md)): PPO with
Monte Carlo advantages from the oracle critic, adaptive entropy floors per
head, the auxiliary gradient budget, and an actor value probe. Every position
is a decision row; its baseline is the critic's value at the same position of
the seat's oracle stream. Every position is also supervised by the
auxiliaries:

* `hands` (actor): how many copies of each card type each opponent holds,
  trained only where the viewer cannot already know (some copy unseen). This
  is the belief state; `hands_held_recall` tracks it on cards actually held;
* `winner` (actor and critic): who goes out, or nobody;
* `penalty` (actor and critic): each seat's final hand value / 100.

Returns are relative points (score minus mean opponent score) and
`value_reward_scale = 50`.

Collection is continuous and asynchronous (`[rollout]`): one lane per
(players, hand size) keeps `games_per_shape` games in flight, finished games
are replaced in the same wave, unfinished ones carry over, and a separate
process collects the next batch during the update. Decisions are therefore
1-2 policy versions old when trained (`policy_lag`); PPO's ratio is taken
against their recorded probabilities. If lag starts to hurt (rising
`behavior_kl` and `clip_fraction`), decoupled PPO (clip around the
pre-update policy, truncated importance weight to the behavior policy) is the
first lever.

## Evaluation

`cardgym rummy evaluate MODEL --opponent heuristic` plays a seeded deal bank
in both directions (one model seat against P - 1 heuristic seats, and one
heuristic seat against P - 1 model seats, rotating the lone seat), through
the continuous collector. `cardgym rummy simulate` prints rule-based statistics.
The heuristic (`heuristic.py`) takes a discard that makes a new meld, melds
and lays off at once, and discards its least connected, highest-value card.

## Measured (M5 Pro, base preset)

At the untrained start (games of ~250 events, the slowest regime), 144 games
per update:

| | per iteration |
| --- | --- |
| first design (one token per event per stream, batch collection) | 56 s (collect 24, update 33) |
| one token per own decision, continuous collection | 22 s |
| + grid as GEMMs (no middle-axis reductions) | ~19 s (update 8.1) |
| + fused trunk (tiled attention) and fused card grid | 7 s (collect 3, update 3.8-4.1) |
| + asynchronous rollout | **4.9 s** (update 4.7, rollout 3.9 hidden) |

| | |
| --- | --- |
| actor positions per update | 115k -> ~40k (critic the same) |
| collection | 4.8k -> 12.5k decisions/s; host dispatch per wave 22 -> 2.7 ms |
| host side per decision (engine, tokens, candidates) | ~30 us, now the rollout's bound |
| fused trunk backward, 65 x 160 positions, two cotangents | 198 -> 90 ms |
| card grid forward+backward, 16k positions | 429 -> 143 (GEMMs) -> 59 ms (fused) |
| memory | trainer ~5 GB peak, rollout process ~2 GB |
| `behavior_kl` (synchronous, iteration 1) | ~1e-6: cached rollout equals replay |

One unexplained glitch: a single iteration of one benchmark reported a
normalized draw entropy of 1.076 (at most 1 by construction), with every
gradient norm and loss matching an identical rerun. It did not reproduce; it
looks like a small-reduction flake on MPS (plump saw the same class below a
few hundred rows).

## Status and next steps

The pipeline is complete and tested (`tests/rummy`: meld search against brute
force, engine invariants, ledger privacy and mover codes against an
independent replay, oracle streams, continuous rollout against full replay,
fused kernels against PyTorch, end-to-end CLI including asynchronous
rollout). The first run on the earlier design (`r3`) reached -23 points per
seat against the heuristic after 200 iterations. Everything about the recipe
is a starting point:

* **Schedule.** 144 games and two PPO epochs per update. With the rollout
  hidden, iteration time is the update's; larger batches cost linearly.
* **Size.** d_model 256, 8 layers, head_dim 32 (GQA 8/2), card grid 2 x 32
  wide.
* **Advantages.** Monte Carlo; GAE is the first variance lever.
* **Lag.** Decoupled PPO if `policy_lag` starts to cost (see Training).
* **Beliefs.** `hands` is linear in the readout; a card-conditioned readout
  (hidden x per-card grid features) is the natural upgrade.
* **Symmetry.** Suit permutation is an exact symmetry of the rules; the grid
  respects it, the per-card summary and event-card rows do not. Random suit
  relabelling per game is free augmentation.
* **Speed.** The fused grid and attention kernels use scalar FMAs (~1
  TFLOPS); simdgroup-matrix versions and fused policy heads are the next
  kernel work, and the host engine (~30 us per decision) bounds the rollout.
