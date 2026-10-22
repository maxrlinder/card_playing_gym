# Rummy

## Rules

Scandinavian rummy with 500 Rum's discard pile and scoring. One 52-card deck
without jokers by default, two as an option (`[game] decks`); 2-4 players;
seven cards each (`[game] hand_sizes`); cyclic runs; no opening requirement.
A turn is:

1. **Draw**: the stock's top card, or any card in the discard pile together
   with every card above it. The top card alone may be kept or used, but
   not discarded in the same turn. A deeper
   card must be melded or laid off in the same turn (the cards above it go
   to the hand), so it is offered only when that is possible: on a table
   meld, in a meld from the hand, or at the end of a chain of lay-offs (with
   2-3-4 on the table and a 5 in hand, a 6 deep in the pile can be taken:
   lay off the 5, then the 6). Until it has been used the seat may only make
   the melds and lay-offs that use it, or that start such a chain.
2. **Meld and lay off**, any number, in any order, onto any meld on the
   table. A meld is laid three cards at a time and grown by lay-offs (4-5-6,
   then the 7), which scores the same as laying 4-5-6-7 at once.
3. **Discard** one card. Discarding your last card goes out and ends the
   round.

Scoring (500 Rum): every card a seat melds or lays off scores its points for
that seat; when the round ends, the cards left in each hand count against it
(Ace 11, faces 10, pips face value). The round ends when someone goes out,
or at the end of the turn that drew the last stock card. The training reward
is a seat's score minus the mean opponent score.

Interpretations:

* a group is 3-4 cards of one rank in distinct suits, even with two decks;
* a run is 3-13 cards of one suit on a contiguous arc of the rank cycle
  (`Q-K-A`, `K-A-2`); in runs an Ace is 1 when it starts the run and 11 when
  a King precedes it (always 11 in a full 13-card run). A meld scores these
  points; a lay-off scores what it adds to the meld;
* a meld or lay-off never takes the last card (you go out by discarding it);
* a top card taken from the pile cannot be discarded that turn; with two
  decks neither can its other copy (the rule is on the card type). The top
  is offered only if the hand holds another card type, and no meld or
  lay-off may leave only that type, so the turn can always end. (Allowing
  the throw-back, as `r7` did until iteration 1500, let self-play learn to
  pass: take the top and discard it straight back on a third of all turns,
  stalling a fifth of rounds into `max_turns`);
* `max_turns` bounds a round and scores like a stock-out (reported as
  `truncated`);
* `[game] opening_points > 0` restores an opening threshold (melds in one
  turn worth at least that much before a seat may lay off or discard).

Training mixes every (players, decks, hand size) combination of the `[game]`
lists, one rollout lane each: by default 2-4 players with one and two decks.

`engine.py` implements this; `melds.py` enumerates the 589 valid melds once.

## The representation problem

A Rummy decision depends on three things:

1. **the viewer's own hand**: a handful of cards, changing every turn by draws,
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

## Tokens: own decisions and opponent turns

A seat's stream interleaves two kinds of token, in table order:

```text
seat 0:  [decide: draw] [decide: meld] [decide: discard] [turn: opp 1] [turn: opp 2] [decide: draw] ...
```

Each token is a row of categorical fields plus 52 per-card codes, one per
card type (`tokens.py`).

* **Decision tokens**, one right before each decision the seat makes, are
  the readout positions. Their card codes are the explicit state: the
  locations of each card type's two copies as an unordered pair over 11
  locations, times whether a copy can be laid off on the table (462 codes).
  The locations are: unseen, the seat's hand, each opponent's publicly known
  holdings (cards it took from the discard pile and has not played since),
  the table split by who put each copy there (the seat or opponent 1-3), the
  discard pile by depth (0 is the top, 1-9 exactly, then "10 or deeper"),
  and "not in this game" (the second copy with one deck). The fields add the
  phase (draw or play), the seat's own previous action (kind, card, meld kind
  and size), a card it took from below the top and must still use, and the
  opening points laid this turn (when a threshold is set).
* **Turn tokens**, one per completed opponent turn, carry what that opponent
  did: who it was, whether it drew from the stock, took the discard top, or
  dug deeper (the deepest card and how many cards it took), what it
  discarded, how many melds and lay-offs it made, and per card type what the
  turn did with it (copies taken, melded, laid off, discarded: 108 codes
  after the state codes).

Both kinds carry the public counters: stock and pile sizes, the four hand
sizes, who has opened, players, decks, round. Every stream is observer-relative.

So the stream is the round as the seat saw it: every opponent's every turn is
its own position, in order, with its cards attributed (the "tells"), and
every decision restates the present exactly, so the trunk need not rebuild
the hand or the table from history. Order within an opponent's turn is the
rules' (draw, melds and lay-offs, discard); the order of its melds within the
turn is not kept, which the policy has no use for.

A game costs `decisions + opponent turns` positions per seat: about 40
tokens per stream (1.8x the earlier one token per own decision, which folded
up to three opponents' turns into each token), not `events` per seat. The
longest possible stream is `2 * ceil(max_turns / 2) + 104 + max_turns`.

Decision tokens are exactly the viewer's knowledge and turn tokens exactly
the public record: a test rebuilds both from the events the viewer saw
(`tests/rummy/test_rummy_streams.py`).

The critic reads a per-seat **oracle** stream with the same positions and
layout, except that location `1 + j` is seat `j`'s whole hand, "unseen" is
the stock, stock draws show their card (in turn tokens too), and one more
field shows the next stock card. Its value head is seat-relative like the
actor's.

## The card grid

`CardGrid` (`model.py`) turns a token's 52 card codes (a decision token's
state, or a turn token's moves) into per-card features and a token summary.
One embedding table holds both code ranges. Rummy's rules are invariant to permuting suits and, because runs are
cyclic, to rotating ranks; only card points break the rank symmetry. So:

* card type `c` starts as `code_embedding[code[c]] + rank_embedding[rank(c)]`
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

* `draw` head: stock, the discard top, or each deeper discard whose card the
  seat could then use (with every card above it);
* `play` head: every distinct three-card meld the hand can form, every
  distinct lay-off (card, target meld), every distinct discard. Discarding
  ends the turn, so "stop melding" needs no action of its own.

One route per outcome. Without the three-card rule, 4-5-6-7 could be laid
at once or as 4-5-6 and then the 7: two routes to one result, and a policy
that always took the long meld never faced "table the 7 or hold it". With
it, every card past the third is its own decision, and a turn that plays
some cards and keeps others is a sequence of "table one more" or "stop".
(Unopened seats under an opening threshold still see melds of every size;
the threshold search counts them whole.)

After a deeper pickup the play head offers only the melds and lay-offs that
use the obligated card, and the lay-offs that start a chain toward it on a
table run, until it is used. With an opening threshold, the
opening is laid one meld at a time; a meld is legal only if the rest of the
hand can still reach the threshold while keeping a card to discard
(`melds.opening_melds`, an exact search), and until it does the seat may only
continue melding. Copies of a card are interchangeable, so a discard is one
candidate per card type and identical target melds are one candidate.

`CandidateScorer` embeds a candidate as its kind, plus the sum of its cards
(their grid features at the decision point, plus an identity row), plus for a
lay-off the sum of the target meld's cards, plus for a pickup the sum of
every card it takes from the pile. It adds a projection of the
readout hidden state and scores the result with a small MLP (additive
attention). The grid features are the ones the embedding computed for the
readout token (`pop_context`), so the card-level view is exact even where the
token summary blurs, at no extra encoding.

## Training

Rummy uses the shared pipeline ([architecture](architecture.md)): PPO with
Monte Carlo advantages from the oracle critic, adaptive entropy floors per
head, the auxiliary gradient budget, and an actor value probe.

Hold or play. Self-play learned to table everything it could, every turn;
holding melds back (no information for opponents, nothing for them to lay
off on) never got explored. The `hold` gate (`RummyHooks.gates`) is the
binary split, at every play decision that could table a card and could also
end the turn, between ending it (P of all discards) and tabling one more card
(P of all melds and lay-offs). Its entropy in bits, averaged over the batch,
has its own adaptive floor (`entropy_targets.hold`). The play head's entropy
then measures only the rest: by the chain rule a decision's entropy is the
split's entropy plus the expected entropy within the chosen side, and
`entropy_play` is the second part (each side normalized by its own size,
weighted by its probability held constant), so `play` and `hold` never
count the same uncertainty. The deep preset targets 0.05 for draw, play and
hold. Logged: `entropy_hold`, `hold_probability` (the policy's P(hold) where
the gate applies), `hold_rate` and `hold_rate_endgame` (sampled holds;
endgame = an opponent down to two cards or fewer), `hold_eligible_fraction`.

With the gate (r7, 16100 -> 20600) the policy came to hold about 40% of
eligible decisions but about 10% in the endgame, mostly confidently (hold
entropy at its floor with a small alpha), and the argmax policy at 20600
beat 16200 by +4.4 +- 1.0 points per game. Every
decision token is a decision row; its baseline is the critic's value at the
same position of the seat's oracle stream. The auxiliaries:

* `hands` (actor, decision tokens): how many copies of each card type each
  opponent holds, trained only where the viewer cannot already know (some
  copy unseen). This is the belief state; `hands_held_recall` tracks it on
  cards actually held;
* `winner` (actor and critic, every position): who goes out, or nobody;
* `penalty` (actor and critic, every position): each seat's final hand
  value / 100.

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

These measurements predate turn tokens and the Scandinavian rules (two
decks, 13 cards, one token per own decision, ~21 per stream in trained
play). Now heuristic rounds take ~50 decisions with ~30 tokens per stream,
and the host costs ~50 us per decision (rules, tokens, candidates; draw
candidates check every pile depth).

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

### Deep preset (`configs/rummy/deep.toml`): 12 layers, 432 games

`r6` is `r5` (8 layers, iteration 22,200) grown to 12 layers with fresh
identity blocks at positions 4, 5, 10, 11 (`cardgym rummy grow`), with three
times the games per update. Measured on its own batches (trained play: ~21
tokens per stream, ~27.5k decisions per update), update alone, one process:

| | update |
| --- | --- |
| as `r5` ran (fused trunk, padded buckets of 16) | 4.04 s |
| attention kernels for short streams, packed rows | 3.04 s |
| non-blocking uploads | 2.41 s |
| one host read per phase for reporting scalars | 2.34 s |
| allocator cache kept between updates | 2.05 s |
| heads on packed rows; buckets of 64, one microbatch per epoch | 1.89 s |
| fused field lookups | 1.71 s |
| card-grid backward as GEMMs, bias gradients as ones columns | 1.63 s |
| candidate lists scored in width buckets (8, 16, rest) | 1.61 s |
| gradient norms as squares and sums | **1.49 s** |

The last five rows ran with a cache ceiling the one-microbatch update
exceeds, so the cache was released every update; at the preset's ceiling (0.6
of the recommended working set) it is released every other update, 1.50 s on
average (1.33 s never releasing, but the cache then grows past 15 GB).

End to end (asynchronous rollout, two workers sharing the GPU with the
update): **~2.0-2.1 s per iteration** (update ~1.7 s under contention, rollout
~1.8 s), the same wall time `r5` took for a third of the games at 8 layers.
Memory: trainer ~9.5 GB driver-allocated (12.9 GB peak), each rollout worker
~1.6 GB. A third worker was no faster (more contention) and left too little
memory free alongside other applications; `cohorts = 2` inside each worker
was slower (twice the waves, and each worker's host is already busy).

### Scandinavian rules on the deep preset

After the rules pivot (7 cards, deep pickups, turn tokens) untrained play is
long: ~60 turns and ~160k positions per update (~57k decisions). On those
batches, update alone:

| | update |
| --- | --- |
| starting point | 5.65 s |
| critic's first epoch is also the advantage baseline (no separate forward); hands head on decision rows only | 5.32 s |
| candidate scorer: one scatter, bmm and GEMM for all card slots | 5.24 s |
| card grid as fused tensor-op kernels (`metal_grid`), peak memory 14 -> 10 GB | **5.04 s** |

Rollout, one process collecting all 432 games: 5.06 -> 4.16 s with the
scorer as one Metal kernel per head (`metal_scorer`) and one upload per dtype
per wave, -> 3.99 s with everything after attention in a decode layer as one
kernel (`cardgym.nn.metal_decode`) and lane-parallel decode attention; host
dispatch 1.08 -> 0.26 s per collection. The host engine (~45 us per decision)
is now most of it, and the workers wait on the update.

End to end (two workers): ~7.7 -> ~7.0 s per iteration at these lengths
(48 -> 43 us per position); the update runs ~1.5 s slower than alone because
the rollout's decode waves share the GPU.

Kernel level: card grid forward at 26k positions 24 -> 11.5 ms, backward
~50 -> ~48 ms (the per-block weight-gradient tensor ops are bound by register
pressure); a decode step at 336 rows 2.70 -> 2.08 ms. Two levers measured
and not taken: microbatches of 64k positions (4.87 s, but 14 GB peak), and
larger decode tiles (each threadgroup streams the layer's weights; fewer
threadgroups leave cores idle).

The fused path agrees with the PyTorch path on everything but the policy
gradient to 1e-4..1e-3. The policy gradient, a sum of advantage-weighted
terms that mostly cancel, differs by 4% with clipping off and 16% with it on,
where the two paths' bf16 roundings also move decisions across the clip
boundary.
The fused path is the one closer to the rollout (`behavior_kl` 0.0027 against
0.0060 for PyTorch on the same batch), and packing and width buckets are exact
(1e-5 against the padded, unbucketed fused path).

One unexplained glitch: a single iteration of one benchmark reported a
normalized draw entropy of 1.076 (at most 1 by construction), with every
gradient norm and loss matching an identical rerun. It did not reproduce; it
looks like a small-reduction flake on MPS (plump saw the same class below a
few hundred rows).

## Status and next steps

The pipeline is complete and tested (`tests/rummy`: meld search against brute
force, engine invariants, decision-token state and turn-token records
against an independent replay, oracle streams, continuous rollout against full replay,
fused kernels against PyTorch, end-to-end CLI including asynchronous
rollout). The first run on the earlier design (`r3`) reached -23 points per
seat against the heuristic after 200 iterations. Everything about the recipe
is a starting point:

* **Schedule.** 1296 games and two PPO epochs per update (`deep.toml`;
  432 until r7 iteration 14300, when gains had slowed to ~1 point per
  2,600 iterations). Each epoch accumulates its ~6 microbatches into one
  optimizer step, so memory stays at the 32k-position microbatch.
* **Size.** d_model 256, 12 layers, head_dim 32 (GQA
  8/2), card grid 2 x 32 wide.
* **Advantages.** Monte Carlo; GAE is the first variance lever.
* **Lag.** Decoupled PPO if `policy_lag` starts to cost (see Training).
* **Beliefs.** `hands` is linear in the readout; a card-conditioned readout
  (hidden x per-card grid features) is the natural upgrade.
* **Symmetry.** Suit permutation is an exact symmetry of the rules; the grid
  respects it, the per-card summary and event-card rows do not. Random suit
  relabelling per game is free augmentation.
* **Speed.** The update is the bound (~5 s alone at untrained lengths,
  shorter as games shorten). What is left is mostly the trunk at MPS's GEMM
  rate and the exact two-stream backward the auxiliary gradient budget
  needs; next levers: one grid and field-lookup backward for both streams
  (~0.2-0.3 s), a fused training scorer (~0.25 s), split-key decode
  attention. The rollout's host engine (~45 us per decision) matters only
  once the update is faster than it.
