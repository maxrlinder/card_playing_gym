# Adding a game

A game is a package under `cardgym/games/<name>/` with a `Game` subclass
(`cardgym/games/base.py`), registered in `cardgym/games/__init__.py`. Plump is
the worked example; this page lists what each piece must do.

## 1. Config sections

Three frozen dataclasses, parsed strictly from the run's TOML (unknown keys
fail):

* `ModelConfig` (`[model]`): trunk shape plus any vocabulary limits. A
  `trunk()` method returns the shared `TrunkConfig`.
* `GameConfig` (`[game]`): what one update plays (table sizes, deals).
* `LossConfig` (`[losses]`): auxiliary loss coefficients.

`[run]`, `[rollout]`, and `[ppo]` are shared (`cardgym/rl/config.py`).
`[ppo] entropy_targets` names your policy heads.

## 2. Self-play

`new_games(rng, iteration)` returns fresh `SelfPlayGame`s
(`cardgym/rl/selfplay.py`), each owning its engine and an RNG:

* `prefix_tokens()`: each observer's stream before the first decision.
* `decision()`: the acting seat, the policy head, and a legal mask. Pointer
  heads also pass `features`: arrays whose leading axis is the candidate
  list. The collector pads them and passes them to `policy_logits`. Cache the
  result until `step`.
* `step(action)`: apply the action (plus any forced follow-ups) and return
  every observer's new tokens. The next actor's stream must end at its
  readout token. Observers may receive different numbers of tokens, including
  none, so private actions can stay private.
* `done()`, `returns()` (zero-sum per absolute seat), and `lifetime()`
  (expected remaining decisions, used to order games longest-first).

The streams the collector records are exactly what the model saw. Build
training inputs from them rather than re-tokenizing, or test that the two
agree (`tests/plump/test_plump_streams.py`).

## 3. Models

Subclass `PolicyModel` and `CriticModel` (`cardgym/games/base.py`):

* `embed(tokens)` returns `(sum of field embeddings, PLE lookup or None,
  value-embedding lookup or None)`. The trunk does the rest.
* The actor implements `policy_logits(hidden, head, features, tokens)` and
  `player_values(hidden)`, which is `[..., max_players]`, observer-relative.
  `tokens` are the readout tokens themselves, so a head can read raw fields
  (Rummy re-encodes the card ledger there to score candidates card by card).
* The critic implements `player_values(hidden)` on whatever seat axis its
  stream uses. Absolute seats make value columns name seats directly.

## 4. Training streams and objectives

`sequences(trajectory)` returns `GameSequences`:

* one actor `TrainingSequence` per learning seat, plus one critic stream;
* `critic_positions[i]`, the critic-stream position of decision `i`'s
  pre-action state;
* labels as numpy arrays, declared in `actor_labels` / `critic_labels` as
  `LabelSpec(per_position, fill, dtype)` so batching can pad them.

`hooks(device, precision)` returns the auxiliary losses: `actor_auxiliary`,
`critic_auxiliary`, and `summarize`. Each receives the microbatch's hidden
states and its `DenseRows` (genuine positions and their weights) and returns
a scalar loss. Project the whole `[B*L, d]` hidden matrix and gather rows
afterwards; on MPS that is faster than gathering first.

## 5. CLI and tests

`add_commands` registers game-specific subcommands (evaluate, play, ...).
`train`, `dashboard`, and `export` come for free. Add a `configs/<name>/`
smoke preset and an end-to-end test that trains it for a couple of
iterations. `behavior_kl` should stay at numerical noise.

## Worked examples

* [Plump](plump.md): fixed heads (bid, card), a hand that only shrinks, so
  the event history alone determines the current state.
* [Rummy](rummy.md): pointer heads over engine-enumerated candidates, and a
  state that changes in both directions, carried on every token as a card
  ledger. A game whose state cannot be cheaply recovered from its history
  should follow it.
