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

| Placement | Extra tokens | Extra work per position | Problem |
| --- | --- | --- | --- |
| state as tokens (one per card) | ~50 per event | attention over ~50x longer streams | sequence length explodes |
| cross-attention to a state memory at every layer | none | K/V projections of ~50 items in every layer, ~10x the trunk | nothing is amortized; the memory is rebuilt per position |
| **state in the token's own embedding** | **none** | **one card-grid pass per token, before the trunk** | **must not lose information (below)** |

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

## Tokens: one event, one decision

Every decision produces one event and every event is one token in every
stream:

```text
[DEAL] [event of decision 1] [event of decision 2] ...
```

A stream therefore has `1 + decisions` tokens, every stream of a game has the
same length, and the token an action produces is the readout of the next
decision. The longest stream is `1 + 2 * max_turns + 104` (a draw and a
discard per turn, plus at most one meld or lay-off per card that reaches the
table): 505 at `max_turns = 200`. Measured lengths are far shorter (below).

Each token is 17 event fields plus the **ledger**: 52 codes, one per card
type.

* Event fields (`tokens.py`): type (deal, stock draw, discard take, meld,
  lay-off, discard), actor, the card (hidden on someone else's stock draw),
  meld kind and size, next actor and phase, stock size, the four hand sizes,
  who has opened, opening points laid so far this turn, players, round. Seats
  are observer-relative.
* Ledger code per card type: the locations of its two copies as an unordered
  pair over 8 locations (36 codes), times two flags: *moved by this event*
  and *can be laid off on the table*. The actor's locations are: unseen, my
  hand, each opponent's publicly known holdings (cards it took from the
  discard pile and has not played since), table, discard pile, discard top.
  A meld token names its cards through the moved flags; the layable flag
  summarizes the table's structure in card space.

The ledger is exactly the viewer's knowledge: a test rebuilds it from the
events the viewer saw and compares (`tests/rummy/test_rummy_streams.py`).

The critic reads a perfect-information **oracle** stream with the same events
and the same ledger layout, except that location `1 + j` is absolute seat
`j`'s whole hand, "unseen" becomes the stock, every drawn card is visible,
and one more field shows the next stock card.

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

It runs once per token, so its cost is that of roughly one extra trunk
layer, and it is part of the token's input, so rollout caches it implicitly.

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
attention). The policy reads the readout token itself, so its card-level view
is exact even where the token summary blurs. The core passes readout tokens
to every policy head for this (`PolicyModel.policy_logits(..., tokens)`).

## Training

Rummy uses the shared pipeline unchanged ([architecture](architecture.md)):
KV-cached merged self-play, PPO with Monte Carlo advantages from the oracle
critic, adaptive entropy floors per head, the auxiliary gradient budget, and
an actor value probe. Each decision of a turn is its own PPO row; its baseline
is the critic's value at the token before it. Every stream position is
supervised by the auxiliaries:

* `hands` (actor): how many copies of each card type each opponent holds,
  trained only where the viewer cannot already know (some copy unseen). This
  is the belief state; `hands_held_recall` tracks it on cards actually held;
* `winner` (actor and critic): who goes out, or nobody;
* `penalty` (actor and critic): each seat's final hand value / 100.

Returns are relative points (score minus mean opponent score) and
`value_reward_scale = 50`.

## Evaluation

`cardgym rummy evaluate MODEL --opponent heuristic` plays a seeded deal bank
in both directions (one model seat against P - 1 heuristic seats, and one
heuristic seat against P - 1 model seats, rotating the lone seat), through
the batched collector. `cardgym rummy simulate` prints rule-based statistics.
The heuristic (`heuristic.py`) takes a discard that makes a new meld, melds
and lays off at once, and discards its least connected, highest-value card.
