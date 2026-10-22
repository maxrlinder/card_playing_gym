"""Streams carry exactly the viewer's knowledge, and rollout equals replay."""

from __future__ import annotations

import itertools
import random
from dataclasses import replace

import numpy as np
import pytest
import torch

from cardgym.games.rummy.cards import NO_CARD, NUM_CARDS
from cardgym.games.rummy.engine import (
    EVENT_DISCARD,
    EVENT_LAYOFF,
    EVENT_MELD,
    EVENT_STOCK,
    EVENT_TAKE,
    LOC_ABSENT,
    LOC_PILE,
    LOC_TABLE,
    LOC_UNSEEN,
    NUM_LOCATIONS,
    PILE_DEPTHS,
    NUM_PAIRS,
    PAIR,
    PAIR_LOCATIONS,
    RummyEngine,
)
from cardgym.games.rummy.game import RummyGame, RummyGameConfig
from cardgym.games.rummy.heuristic import HeuristicPolicy
from cardgym.games.rummy.melds import EXTENSIONS
from cardgym.games.rummy.model import RummyModelConfig
from cardgym.games.rummy.selfplay import RummySelfPlay
from cardgym.games.rummy.tokens import (
    CARD_START,
    CARD_VOCAB,
    DRAW_DEEP,
    DRAW_STOCK,
    DRAW_TOP,
    EVENT_BASE,
    EVENT_PLACE,
    F_ACTOR,
    F_DISCARD,
    F_DRAW,
    F_HAND,
    F_KIND,
    F_LAST_CARD,
    F_LAYOFFS,
    F_MELDS,
    F_OBLIGATION,
    F_PHASE,
    F_STOCK_TOP,
    F_TAKE_COUNT,
    F_TAKEN,
    KIND_DECISION,
    KIND_TURN,
    max_stream_length,
)
from cardgym.games.rummy.training import IGNORE, RummyLossConfig, training_sequences
from cardgym.rl.continuous import ContinuousCollector
from cardgym.rl.ppo import masked_probabilities
from tests.conftest import randomize

# Whether a pair code has a copy the viewer cannot place.
PAIR_UNSEEN = np.zeros(NUM_PAIRS, dtype=bool)
for _a in range(NUM_LOCATIONS):
    for _b in range(_a, NUM_LOCATIONS):
        PAIR_UNSEEN[PAIR[_a, _b]] = LOC_UNSEEN in (_a, _b)


def heuristic_round(
    players: int, seed: int, hand_size: int = 7, decks: int = 1
) -> tuple[RummySelfPlay, list[np.ndarray]]:
    game = RummySelfPlay(players, hand_size, seed=seed, decks=decks)
    policy = HeuristicPolicy()
    streams = [[block] for block in game.prefix_tokens()]
    while not game.done():
        for seat, block in enumerate(game.step(policy.act(game.engine))):
            streams[seat].append(block)
    return game, [np.concatenate(blocks) for blocks in streams]


class Replay:
    """One observer's view of the round, rebuilt from the events it saw (not from the engine)."""

    def __init__(self, game: RummySelfPlay, observer: int) -> None:
        engine = RummyEngine(game.num_players, game.hand_size, game.seed, decks=game.decks)  # the same deal
        self.players = game.num_players
        self.decks = game.decks
        self.obligation = NO_CARD
        self.observer = observer
        self.own = engine.hands[observer].copy()
        self.sizes = engine.hands.sum(axis=1)
        self.known = np.zeros((self.players, NUM_CARDS), dtype=np.int64)
        self.placed = np.zeros((self.players, NUM_CARDS), dtype=np.int64)
        self.table: list[int] = []
        self.pile = [engine.top]

    def pairs(self) -> np.ndarray:
        counts = np.zeros((NUM_LOCATIONS, NUM_CARDS), dtype=np.int64)
        counts[1] = self.own
        for offset in range(self.players):
            seat = (self.observer + offset) % self.players
            if offset:
                counts[1 + offset] = self.known[seat]
            counts[LOC_TABLE + offset] = self.placed[seat]
        for depth, card in enumerate(reversed(self.pile)):
            counts[LOC_PILE + min(depth, PILE_DEPTHS - 1), card] += 1
        counts[LOC_ABSENT] = 2 - self.decks
        counts[LOC_UNSEEN] = 2 - counts[1:].sum(axis=0)
        cumulative = counts.cumsum(axis=0)
        return PAIR[(cumulative >= 1).argmax(axis=0), (cumulative >= 2).argmax(axis=0)]

    def layable(self) -> np.ndarray:
        layable = np.zeros(NUM_CARDS, dtype=np.int64)
        for meld in self.table:
            for card, _ in EXTENSIONS[meld]:
                layable[card] = 1
        return layable

    def leave(self, actor: int, card: int) -> None:
        self.sizes[actor] -= 1
        if actor == self.observer:
            self.own[card] -= 1
        elif self.known[actor, card]:
            self.known[actor, card] -= 1

    def apply(self, event) -> None:
        actor = event.actor
        if event.kind == EVENT_STOCK:
            self.sizes[actor] += 1
            if actor == self.observer:
                self.own[event.card] += 1
        elif event.kind == EVENT_TAKE:
            for card in event.cards:  # top first
                assert self.pile.pop() == card
                self.sizes[actor] += 1
                if actor == self.observer:
                    self.own[card] += 1
                else:
                    self.known[actor, card] += 1
            assert event.card == event.cards[-1]
            if actor == self.observer and len(event.cards) > 1:
                self.obligation = event.card
        elif event.kind == EVENT_MELD:
            self.table.append(event.meld)
        elif event.kind == EVENT_LAYOFF:
            position = next(
                i for i, meld in enumerate(self.table) if (event.card, event.meld) in EXTENSIONS[meld]
            )
            self.table[position] = event.meld
        if event.kind in (EVENT_MELD, EVENT_LAYOFF):
            for card in event.cards:
                self.leave(actor, card)
                self.placed[actor, card] += 1
                if actor == self.observer and card == self.obligation:
                    self.obligation = NO_CARD
        elif event.kind == EVENT_DISCARD:
            self.leave(actor, event.card)
            self.pile.append(event.card)


def expected_tokens(game: RummySelfPlay, seat: int) -> list[tuple[int, int]]:
    """(kind, event index) of each token: a decision before each own event, a turn after each opponent's discard."""

    tokens = []
    for index, event in enumerate(game.events):
        if event.actor == seat:
            tokens.append((KIND_DECISION, index))
        elif event.kind == EVENT_DISCARD:
            tokens.append((KIND_TURN, index))
    return tokens


def turn_codes(events, *, oracle: bool) -> np.ndarray:
    counts = np.zeros((5, NUM_CARDS), dtype=np.int64)
    for event in events:
        if event.kind == EVENT_TAKE:
            for card in event.cards:
                counts[0, card] += 1
        elif event.kind == EVENT_STOCK and oracle:
            counts[1, event.card] += 1
        elif event.kind == EVENT_MELD:
            for card in event.cards:
                counts[2, card] += 1
        elif event.kind == EVENT_LAYOFF:
            counts[3, event.card] += 1
        elif event.kind == EVENT_DISCARD:
            counts[4, event.card] += 1
    return EVENT_BASE + np.asarray(EVENT_PLACE) @ counts


def test_actor_streams_hold_exactly_public_knowledge():
    deep = 0
    for players, decks, seed in itertools.product((2, 3, 4), (1, 2), range(4)):
        game, streams = heuristic_round(players, seed, decks=decks)
        deep += sum(event.kind == EVENT_TAKE and len(event.cards) > 1 for event in game.events)
        for seat in range(players):
            stream = streams[seat].astype(np.int64)
            expected = expected_tokens(game, seat)
            assert [kind for kind, _ in expected] == stream[:, F_KIND].tolist()
            assert len(stream) <= max_stream_length(game.engine.max_turns)
            replay = Replay(game, seat)
            applied = 0
            for token, (kind, index) in zip(stream, expected):
                if kind == KIND_DECISION:
                    while applied < index:
                        replay.apply(game.events[applied])
                        applied += 1
                    codes = token[CARD_START:]
                    assert np.array_equal(codes % NUM_PAIRS, replay.pairs())
                    assert np.array_equal(codes // NUM_PAIRS, replay.layable())
                    own = [e for e in game.events[:index] if e.actor == seat]
                    assert token[F_LAST_CARD] == (own[-1].card if own else NO_CARD)
                    assert token[F_OBLIGATION] == replay.obligation
                    assert token[F_PHASE] == int(game.events[index].kind not in (EVENT_STOCK, EVENT_TAKE))
                else:
                    actor = game.events[index].actor
                    start = max(i for i in range(index) if game.events[i].kind in (EVENT_STOCK, EVENT_TAKE))
                    turn = game.events[start : index + 1]
                    assert all(event.actor == actor for event in turn)
                    while applied <= index:
                        replay.apply(game.events[applied])
                        applied += 1
                    assert token[F_ACTOR] == (actor - seat) % players
                    draw = turn[0]
                    if draw.kind == EVENT_STOCK:
                        assert token[F_DRAW] == DRAW_STOCK and token[F_TAKEN] == NO_CARD
                    else:
                        assert token[F_DRAW] == (DRAW_TOP if len(draw.cards) == 1 else DRAW_DEEP)
                        assert token[F_TAKEN] == draw.card and token[F_TAKE_COUNT] == len(draw.cards)
                    assert token[F_DISCARD] == turn[-1].card
                    assert token[F_MELDS] == sum(e.kind == EVENT_MELD for e in turn)
                    assert token[F_LAYOFFS] == sum(e.kind == EVENT_LAYOFF for e in turn)
                    assert np.array_equal(token[CARD_START:], turn_codes(turn, oracle=False))
                sizes = [replay.sizes[(seat + j) % players] for j in range(players)]
                assert token[F_HAND : F_HAND + players].tolist() == sizes
                assert token[F_STOCK_TOP] == NO_CARD
    assert deep > 0  # the heuristic digs, so deep pickups are covered


def test_oracle_streams_see_every_hand():
    for players in (2, 3, 4):
        game, streams = heuristic_round(players, seed=5)
        for seat in range(players):
            oracle = np.stack(game.oracle[seat]).astype(np.int64)
            hands = np.stack(game.hands[seat]).astype(np.int64)  # [L, P, 52]
            stream = streams[seat].astype(np.int64)
            assert oracle.shape == stream.shape
            decision = stream[:, F_KIND] == KIND_DECISION
            # Same fields except the stock top and, on turn tokens, the card drawn.
            same = np.ones(F_STOCK_TOP, dtype=bool)
            assert np.array_equal(oracle[decision, :F_STOCK_TOP], stream[decision, :F_STOCK_TOP])
            same[F_TAKEN] = False
            assert np.array_equal(oracle[~decision, :F_STOCK_TOP][:, same], stream[~decision, :F_STOCK_TOP][:, same])
            locations = PAIR_LOCATIONS[oracle[decision, CARD_START:] % NUM_PAIRS]  # [D, 52, 2]
            for offset in range(players):
                held = (locations == 1 + offset).sum(axis=-1)
                assert np.array_equal(held, hands[decision][:, (seat + offset) % players])
            # Turn tokens: stock draws are visible to the oracle only.
            turns = ~decision
            assert np.all(oracle[turns, F_TAKEN] != NO_CARD)
            assert np.all(oracle[turns, CARD_START:] >= stream[turns, CARD_START:])


def test_training_sequences_and_belief_masks():
    game, streams = heuristic_round(3, seed=11)

    class Record:
        def __init__(self, seat, head, position):
            self.seat, self.head, self.position = seat, head, position

    decisions = []
    for event_index, event in enumerate(game.events):
        head = "draw" if event.kind in (EVENT_STOCK, EVENT_TAKE) else "play"
        position = expected_tokens(game, event.actor).index((KIND_DECISION, event_index))
        decisions.append(Record(event.actor, head, position))

    class Trajectory:
        pass

    trajectory = Trajectory()
    trajectory.game, trajectory.streams, trajectory.decisions = game, streams, decisions
    trajectory.returns = game.returns()
    sequences = training_sequences(trajectory)
    assert sequences.critic_positions == [record.position for record in decisions]
    assert set(sequences.critic) == set(sequences.actor) == {0, 1, 2}
    for seat, sequence in sequences.actor.items():
        hands = sequence.labels["hands"]
        stream = streams[seat].astype(np.int64)
        assert hands.shape == (len(stream), 3, NUM_CARDS)
        assert np.all(hands[:, 2] == IGNORE)  # no fourth seat at a 3-player table
        # Cards the viewer can place exactly are never supervised, nor are turn tokens.
        unseen = PAIR_UNSEEN[stream[:, CARD_START:] % NUM_PAIRS] & (stream[:, F_KIND, None] == KIND_DECISION)
        assert np.all((hands[:, :2] == IGNORE) == ~unseen[:, None, :])


def tiny_game(**game) -> RummyGame:
    settings = dict(player_counts=(2, 3, 4), hand_sizes=(7,), games_per_shape=3, max_turns=30)
    settings.update(game)
    model = RummyModelConfig(
        d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=64, per_layer_dim=8,
        ledger_width=8, ledger_hidden=16, ledger_layers=2, score_dim=16,
    )
    return RummyGame(model, RummyGameConfig(**settings), RummyLossConfig())


DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


@pytest.mark.parametrize("cohorts", [1, 2])
@pytest.mark.parametrize("device", DEVICES)
def test_continuous_rollout_matches_full_replay(device, cohorts):
    """Decoding (cached, gathered rows, refilled slots, carried over, cohorts) equals a full replay."""

    game = tiny_game()
    actor = randomize(game.build_actor().eval(), seed=3, scale=0.2).to(device)
    collector = ContinuousCollector(
        actor, device=torch.device(device), precision="fp32", kv_dtype="fp32", max_length=game.max_length,
        cohorts=cohorts,
    )
    lanes = [replace(lane, in_flight=lane.quota + 1) for lane in game.lanes(1.0)]
    rng = random.Random(0)
    trajectories = collector.collect(lanes, rng, version=1)
    carried = collector.in_flight
    assert len(trajectories) == sum(lane.quota for lane in lanes) and carried > 0
    trajectories += collector.collect(lanes, rng, version=2)
    lags = {record.version for trajectory in trajectories for record in trajectory.decisions}
    assert lags == {1, 2}  # some games straddled the two collections
    worst = 0.0
    with torch.no_grad():
        for trajectory in trajectories:
            for seat, stream in enumerate(trajectory.streams):
                if not len(stream):
                    continue
                tokens = torch.from_numpy(stream.astype(np.int64))[None].to(device)
                hidden = actor.forward_hidden(tokens)[0]
                actor.pop_context()
                for decision in trajectory.decisions:
                    if decision.seat != seat:
                        continue
                    features = {k: torch.from_numpy(v)[None].to(device) for k, v in decision.features.items()}
                    logits = actor.policy_logits(
                        hidden[decision.position][None], decision.head, features,
                        tokens[0, decision.position][None],
                    ).cpu().numpy()
                    probs = masked_probabilities(logits, np.ones_like(logits, dtype=bool))[0]
                    worst = max(worst, float(np.abs(probs - decision.probs).max()))
    assert worst < (1e-5 if device == "cpu" else 1e-3)


def test_card_grid_is_suit_equivariant():
    """Relabelling suits in the ledger relabels the per-card features the same way."""

    torch.manual_seed(0)
    grid = randomize(tiny_game().build_actor(), seed=1, scale=0.1).embedding.grid.eval()
    ledger = torch.randint(0, CARD_VOCAB, (6, NUM_CARDS))
    permutation = torch.tensor([2, 0, 3, 1])  # new suit of each old suit
    cards = torch.arange(NUM_CARDS)
    moved = permutation[cards // 13] * 13 + cards % 13
    relabelled = torch.empty_like(ledger)
    relabelled[:, moved] = ledger
    with torch.no_grad():
        original = grid.cards(ledger)
        permuted = grid.cards(relabelled)
    assert torch.allclose(permuted[:, moved], original, atol=1e-5)
