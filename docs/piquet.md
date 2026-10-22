# Piquet

## Rules

Two players, a 32-card deck (7 to Ace in four suits, Ace high), 12 cards
each and an eight-card talon. The non-dealer is **Elder**, the dealer
**Younger**; the deal alternates. A deal has three phases:

1. **Exchange.** Elder discards 1 to 5 cards and draws as many from the top
   of the talon; then Younger discards from 0 up to every card left and draws
   the same way. A hand without K, Q, or J (**carte blanche**) is shown before
   its owner exchanges and scores 10.
2. **Declarations**, in the order point, sequence, sets. Elder declares its
   best combination of the category or says nothing; Younger then contests or
   concedes ("good"), or declares its own after Elder said nothing. Only the
   better combination scores; equal points or sequences score for neither.
   * Point: most cards in one suit, then the higher card total (A 11, K/Q/J/10
     10, pips face value); 1 per card.
   * Sequence: 3+ consecutive cards of a suit, longer then higher top; the
     winner scores every sequence it holds (3, 4, then 15-18 for 5-8 cards).
   * Sets: three or four of a kind from the 10 up, four beat three, then
     rank; the winner scores every set (3 or 14).
3. **Tricks.** Elder leads; follow suit if possible; no trumps. Leading a
   card scores 1, winning a trick the opponent led scores 1, the last trick 1
   more. Seven or more tricks score 10, all twelve 40 instead (capot).

**Repique** (+60): 30 points from declarations alone (carte blanche
included, counted in order) before the opponent scores anything. **Pique**
(+30, Elder only): 30 from declarations and trick play before Younger scores
anything.

Two modes ([game] `modes`, one rollout lane each):

* `deal`: one deal; the reward is the point difference.
* `rubicon`: six deals; the winner scores 100 plus the difference, or 100
  plus both totals when the loser stayed under 100 (rubiconed). Its rewards
  are several times a deal's, so mixing it with `deal` in one run lets it
  dominate the normalized advantages; train it on its own, after `deal`
  (see [Parties](#parties)).

Interpretations, where the classic rules leave a choice:

* Elder may look at the cards it left of its five (they are the first
  Younger would draw); cards Younger leaves stay face down for both.
* A winning declaration shows its cards (the point's suit, every scoring
  sequence or set). A losing or equal one reveals only what was announced
  (length and total, length and top card, size and rank).
* Saying nothing is how a combination is sunk: it scores nothing and hides
  the cards. Elder saying nothing and Elder having nothing look the same.
  Younger sees a choice only when it holds an equal or better combination
  (contest or concede) or Elder said nothing (declare or not).
* Carte blanche is always declared. The 10 or 40 for the cards come after
  the last trick and do not count toward pique; pique and repique never
  stack.
* A decision with a single legal action (a forced follow, the last trick, a
  sixth discard) is taken automatically and never reaches the policy.

`engine.py` implements this; `rules.py` holds the declaration tables over
suit bytes (a hand is a 32-bit mask, one byte per suit).

## What shapes the model

Plump and Rummy needed different architectures because of what their states
do; Piquet sits between them.

| | Plump | Rummy | Piquet |
| --- | --- | --- | --- |
| players | 3-5 | 2-4 | **exactly 2** |
| deck | 52, unique | 52 or 104 (duplicates) | **32, unique** |
| hand | only shrinks | grows and shrinks every turn | **one exchange, then only shrinks** |
| public cards picked up | none | the discard pile | **none** (talon draws are private) |
| actions | bid, card | variable candidate lists (melds, lay-offs, takes) | **a card, stop, or declare/quiet** |
| hidden information | opponents' hands | hands and stock | hands, **both discard piles**, the talon, sunk combinations |
| stream length | fixed by shape | bounded by a turn limit | **at most 64 tokens per deal** |

What follows from it:

* **No candidate lists.** Every action is a card (32), "stop", or a binary
  declaration, so all three heads are fixed-width with a legal mask. The
  combinatorial choice (1 585 possible Elder discards) is factorized: one
  card per decision, then "stop", each choice with the full trunk behind it.
* **A per-card ledger anyway.** The hand changes in both directions once
  (discard and draw), declarations reveal some opponent cards, carte blanche
  reveals a hand that then partly leaves, and Elder knows exactly which
  talon cards Younger drew. Rebuilding that from history is the set algebra
  Rummy's design avoids, so every token restates where each of the 32 cards
  is, as far as the viewer knows (`tokens.py`), as Rummy's decision tokens
  do. With 32 unique cards the ledger is one code per card (12 codes),
  not Rummy's copy pairs.
* **A 4 x 8 card grid** reads the ledger: a convolution along each suit's
  ranks (sequences; zero-padded, ranks do not wrap as in Rummy's runs) and a
  per-rank mean over suits (sets), equivariant to suit permutations, plus a
  rank embedding (values and the 10-and-up sets break rank symmetry). It
  runs once per token before the trunk, so the KV cache never goes stale.
* **Short streams, deep trunk.** A deal is at most 64 tokens per seat (about
  50 in practice), so depth is cheap: 16 layers of width 256.
* **Two seats, both learning, observer-relative.** Every seat reference is
  "me" or "the opponent".

## Tokens

Every event reaches both seats as one token, in order: the deal, a carte
blanche, each exchange (who, how many cards), each declaration (Elder's
announcement; Younger's answer with the category's winner and points), each
card played (with the trick's winner and points when it completes the
trick), and the end of the deal (the cards bonus or capot). The seat about to
decide also gets a **decision** token, its policy's readout. Choosing a card
for discard has no token; the next decision token shows it.

A token is 32 categorical fields plus 32 card codes. The fields describe
the event and restate the counters: role, phase, both seats' tricks and
points this deal, talon size, the suits the opponent failed to follow, and
in a partie the score (below). Decision tokens add the head and its
context: cards chosen so far and the most allowed; the category, the seat's
best combination in it, and how it compares with Elder's announcement; the
card led. The card codes are: unknown, mine (legal now, at a decision),
chosen for discard, my discards, seen in the talon, the opponent's (known),
the opponent's hand or discards (shown by carte blanche before it
exchanged), played by me or the opponent, and on the table in this trick.

`tests/piquet/test_piquet_streams.py` rebuilds each seat's knowledge from the
events and that seat's private information alone and checks every token
against it.

## Model

One model, the actor (`model.py`), 9.08M parameters:

| Part | Shape | Parameters |
| --- | --- | --- |
| trunk | 16 layers, d 256, 8 heads (2 KV), MLP 512, per-layer embeddings 64, alternating value embeddings | 7.11M |
| embedding | field tables (token, per-layer, value) and the card grid (width 32, 2 blocks) with its 32 x 32 -> 256 summary | 1.60M |
| heads | two card scorers, stop, declare, value, predictions (one 999-wide projection) | 0.37M |

Heads read the decision token's hidden state:

* `exchange`: 32 card logits plus "stop"; `play`: 32 card logits. Each card
  is scored by an MLP of the readout plus that card's grid features and an
  identity row (additive attention), so the head sees each card's context
  (its suit's length, its sequence, its guards), not just its identity.
* `declare`: 2 logits (say nothing or concede / declare or contest).
* `value`: both seats' expected return. **There is no critic.** The value
  head is the PPO baseline, trained with the policy through `[ppo]
  actor_value_coef`. With `actor_baseline = "rollout"` (the preset) the
  collector reads it at every sampled decision, next to the policy logits,
  and the trainer uses those values directly; `"value_pass"` instead reads it
  with a no-grad pass over the batch before the update (current weights,
  0.2 s). The recorded values are as old as the policy lag (~1 version).

Prediction heads (`training.py`), one packed projection, every position:

| Head | Target | Why |
| --- | --- | --- |
| `belief` | each card the viewer cannot place: opponent's hand, its discards, or the talon | card counting; the opponent's discards are the game's main hidden set |
| `tricks` | tricks the seat takes this deal (0-12) | plans the play: the cards (7+) and capot |
| `declarations` | who wins point, sequence, sets (until settled) | what the exchange is about |
| `points` | both seats' points this deal | the deal's outcome in parts |
| `bonus` | repique, pique, capot: none, me, the opponent | the large swings |
| `trick` | who wins the current trick (play phase) | the next step of the play |
| `opp_suits` | the opponent's length in each suit after the exchange (0-8) | its guards and long suit: what leads and discards depend on |
| `opp_next` | the next card the opponent plays (play phase) | opponent modelling |
| `fate` | for every card not yet played: the trick it falls in and whether it wins it, or never played | the play as a plan: which cards are winners, and when |

With `auxiliary_grad_fraction` their gradient is held to a fixed share of the
policy gradient (as in Plump and Rummy).

## Training

```bash
uv run cardgym piquet train smoke --config configs/piquet/smoke.toml
CARDGYM_METAL_TRUNK=1 tools/train_supervised.sh piquet p1 configs/piquet/base.toml
uv run cardgym piquet evaluate runs/piquet/p1/checkpoints/iter_001000.pt --deals 512
uv run cardgym piquet simulate --policy heuristic --opponent random
```

`base.toml`: 1024 deals per update (~32k decisions, ~100k tokens),
continuous asynchronous rollouts in one worker, two epochs of two
minibatches, Monte Carlo advantages against the value head, entropy floors
0.05 on all three heads (batch-level Lagrangian floors: each head's mean
normalized entropy over the update's non-forced decisions, with its own
temperature).

### Speed

Measured on the M5 Pro laptop (`tools/bench.py` on recorded batches, then
short real runs):

| Step | Update (bench) | Iteration (real) |
| --- | --- | --- |
| first version (PyTorch grid and field lookups, budget measured every step) | 5.8 s | 6.4 s |
| fused field lookups (`cardgym.nn.metal_bag`) | | |
| fused card grid: Rummy's kernels, generalized to 8 zero-padded ranks (`cardgym.nn.metal_grid`) | 2.3 s | |
| `auxiliary_grad_refresh = 4`: the gradient budget's scale measured once per update | 2.2 s | |
| prediction rows padded to 8192 (MPS compiled a graph per shape) | 2.2 s (16k microbatches: 3.4 -> 2.5 s) | |
| 16k microbatches, 6 GB allocator cache, one rollout worker | 2.5 s | 3.8 s |

Per 32k-token microbatch a forward and backward went from 437 to 247 ms
(grid 185 -> 56 ms, field lookups 85 -> ~25 ms). The real update is slower
than the bench because the rollout worker shares the GPU and CPU; a second
worker made it slower still (4.7 s per iteration). 32k microbatches run at
3.4 s per iteration, but the trainer's footprint then peaks near 15 GB
against ~10-11 GB at 16k.

## Parties

A partie is six deals scored by the Rubicon payoff. Playing for points is
nearly right, except near the end: the win is worth 100 on top of the margin,
and a loser under 100 adds their whole total to the winner's payoff. A
partie-aware player trades margin for those when the score makes it worth it.
For example, behind it chases a pique or repique, ahead it plays safe, and
near the line it pushes for, or blocks, the rubicon.

**What the policy sees.** Each deal is a fresh context. Earlier deals' cards
don't matter; what they decided is the score. Every token of a partie
carries:

* both running totals (earlier deals plus this deal's points so far), exact
  below 120 and in tens up to 320;
* the running lead, exact within +-60 and in tens out to +-200;
* the deal number and the mode.

In a single deal these fields are 0.

**How the fresh context works.** `SelfPlayGame.context_starts()` tells the
collectors where a new context begins in the tokens a step returned. They
reset that observer's cache length to 0: the new deal is written from
position 0, and causal attention never reaches the stale keys beyond it. The
recorded stream keeps every token. Training splits each seat's stream into
one sequence per deal, and decision positions count from the stream's
start. So no sequence is longer than a deal (59 tokens), the same as in
single-deal training.

**Starting from a single-deal model.** A checkpoint from before the score
fields loads with every row of those fields equal to its row 0, and the new
lead field set to 0 (`migrate_field_table`). It computes exactly what it
did, blind to the score, so:

* the fine-tune starts from its play;
* the old model is the score-blind baseline in head-to-head games.

**Training.** `configs/piquet/partie.toml`:

* 192 parties per update (1152 deals, ~36k decisions), all in flight
  together.
* GAE with lambda 0.95 over the partie's decisions. Each advantage then
  leans mostly on the value at the start of the next deal rather than on
  the noise of every later deal.
* `value_reward_scale = 100`.
* Half the single-deal learning rate.
* `policy_warmup_steps = 200`: for the first 50 updates only the value head
  and the prediction heads train. The value head learned deal margins and
  must first learn what a partie is worth.

```bash
uv run cardgym piquet train partie1 --config configs/piquet/partie.toml \
  --init-from checkpoints/piquet/p2 --reset-iteration --prepare-only
CARDGYM_METAL_TRUNK=1 tools/train_supervised.sh piquet partie1 configs/piquet/partie.toml
uv run cardgym piquet evaluate runs/piquet/partie1/checkpoints/iter_001000.pt --mode rubicon \
  --against checkpoints/piquet/p2 --deals 1024 --device mps
```

Prepare the run first, then supervise it. The supervisor passes its
arguments again on every restart, and `--init-from` only accepts a new run.

**Throughput** on the M5 Pro laptop (24 GB, MPS, bf16, Metal trunk, one
asynchronous rollout worker): 3.7-3.8 s per update of 192 parties (1152
deals, ~36k decisions). That is about 9.5k decisions per second, or 1.1M
deals an hour. The update takes 2.8-3.2 s, and the next batch's rollout
(3.3 s) runs alongside it, with a policy lag of 1.2-1.4.

**Moving a run to another machine.** `tools/run_snapshot.py pack` stores
the run's latest checkpoint as weights and optimizer state, both under
GitHub's 100 MB limit, plus its config, metrics, log and dashboard;
`unpack` restores it for an exact resume (`snapshots/README.md`).

**Measuring it.** `--against` plays the two models in duplicate parties
(`cardgym.rl.match`: each seat decoded by its own model, one shared cache).
Sampling uses each game's own random numbers, so p2 against itself scores
exactly 0. Starting point, p2 at iteration 13600: +13.9 +- 0.7 points per
deal against the heuristic, and +218 +- 6 per partie with 83% of parties
won.

Evaluation plays seeded deals in duplicate (each deal twice with the seats
swapped) against the rule-based heuristic (`heuristic.py`: Elder throws its
five least useful cards, Younger up to three weak ones, always declares,
cashes sure winners of its longest suit, follows with the cheapest winner).
The heuristic beats random play by about +29 points per deal; with heuristic
against heuristic Elder wins by about +16.
