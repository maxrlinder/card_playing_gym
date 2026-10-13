"""Streams carry exactly the viewer's knowledge, and rollout equals replay."""

from __future__ import annotations

import random

import numpy as np
import torch

from cardgym.games.rummy.cards import NO_CARD, NUM_CARDS
from cardgym.games.rummy.engine import (
    EVENT_DISCARD,
    EVENT_LAYOFF,
    EVENT_MELD,
    EVENT_STOCK,
    EVENT_TAKE,
    LOC_PILE,
    LOC_TABLE,
    LOC_TOP,
    LOC_UNSEEN,
    NUM_LOCATIONS,
    NUM_PAIRS,
    PAIR,
)
from cardgym.games.rummy.game import RummyGame, RummyGameConfig
from cardgym.games.rummy.heuristic import HeuristicPolicy
from cardgym.games.rummy.model import RummyModelConfig
from cardgym.games.rummy.selfplay import RummySelfPlay
from cardgym.games.rummy.tokens import F_CARD, F_NEXT_ACTOR, F_NEXT_PHASE, LEDGER_START
from cardgym.games.rummy.training import IGNORE, RummyLossConfig, training_sequences
from cardgym.rl.ppo import masked_probabilities
from cardgym.rl.selfplay import SelfPlayCollector
from tests.conftest import randomize

# Whether a pair code has a copy the viewer cannot place.
PAIR_UNSEEN = np.zeros(NUM_PAIRS, dtype=bool)
for _a in range(NUM_LOCATIONS):
    for _b in range(_a, NUM_LOCATIONS):
        PAIR_UNSEEN[PAIR[_a, _b]] = LOC_UNSEEN in (_a, _b)


def heuristic_round(players: int, seed: int, hand_size: int = 13) -> tuple[RummySelfPlay, list[np.ndarray]]:
    game = RummySelfPlay(players, hand_size, seed=seed)
    policy = HeuristicPolicy()
    streams = [[block] for block in game.prefix_tokens()]
    while not game.done():
        for seat, block in enumerate(game.step(policy.act(game.engine))):
            streams[seat].append(block)
    return game, [np.concatenate(blocks) for blocks in streams]


def public_ledgers(game: RummySelfPlay, observer: int) -> np.ndarray:
    """The observer's copy-location pairs, rebuilt from what it saw (no engine state)."""

    players = game.num_players
    own = game.hands[0][observer].astype(np.int64).copy()
    known = np.zeros((players, NUM_CARDS), dtype=np.int64)
    table = np.zeros(NUM_CARDS, dtype=np.int64)
    pile = [game.first_discard]

    def snapshot() -> np.ndarray:
        counts = np.zeros((NUM_LOCATIONS, NUM_CARDS), dtype=np.int64)
        counts[1] = own
        for offset in range(1, players):
            counts[1 + offset] = known[(observer + offset) % players]
        counts[LOC_TABLE] = table
        for card in pile[:-1]:
            counts[LOC_PILE, card] += 1
        if pile:
            counts[LOC_TOP, pile[-1]] = 1
        counts[LOC_UNSEEN] = 2 - counts[1:].sum(axis=0)
        cumulative = counts.cumsum(axis=0)
        return PAIR[(cumulative >= 1).argmax(axis=0), (cumulative >= 2).argmax(axis=0)]

    def leave(actor: int, card: int) -> None:
        if actor == observer:
            own[card] -= 1
        elif known[actor, card]:
            known[actor, card] -= 1

    pairs = [snapshot()]
    for event in game.events:
        actor = event.actor
        if event.kind == EVENT_STOCK:
            if actor == observer:
                own[event.card] += 1
        elif event.kind == EVENT_TAKE:
            assert pile.pop() == event.card
            if actor == observer:
                own[event.card] += 1
            else:
                known[actor, event.card] += 1
        elif event.kind in (EVENT_MELD, EVENT_LAYOFF):
            for card in event.cards:
                leave(actor, card)
                table[card] += 1
        elif event.kind == EVENT_DISCARD:
            leave(actor, event.card)
            pile.append(event.card)
        pairs.append(snapshot())
    return np.stack(pairs)


def test_actor_streams_hold_exactly_public_knowledge():
    for players in (2, 3, 4):
        for seed in range(4):
            game, streams = heuristic_round(players, seed)
            length = 1 + len(game.events)
            for seat in range(players):
                stream = streams[seat]
                assert stream.shape[0] == length
                ledger = stream[:, LEDGER_START:].astype(np.int64)
                expected = public_ledgers(game, seat)
                assert np.array_equal(ledger % NUM_PAIRS, expected)
                moved = (ledger // NUM_PAIRS) % 2
                assert not moved[0].any()
                assert np.array_equal(moved[1:], (expected[1:] != expected[:-1]).astype(np.int64))
                for position, event in enumerate(game.events, start=1):
                    if event.kind == EVENT_STOCK and event.actor != seat:
                        assert stream[position, F_CARD] == NO_CARD


def test_training_sequences_and_belief_masks():
    game, streams = heuristic_round(3, seed=11)

    class Record:
        def __init__(self, seat, head, position):
            self.seat, self.head, self.position = seat, head, position

    # Rebuild decision addresses: decision i is read at position i.
    decisions = []
    oracle = np.stack(game.oracle)
    for position in range(len(game.events)):
        seat = int(oracle[position, F_NEXT_ACTOR])
        decisions.append(Record(seat, "draw" if oracle[position, F_NEXT_PHASE] == 0 else "play", position))

    class Trajectory:
        pass

    trajectory = Trajectory()
    trajectory.game, trajectory.streams, trajectory.decisions = game, streams, decisions
    trajectory.returns = game.returns()
    sequences = training_sequences(trajectory)
    assert sequences.critic_positions == list(range(len(game.events)))
    for seat, sequence in sequences.actor.items():
        hands = sequence.labels["hands"]
        assert hands.shape == (len(streams[seat]), 3, NUM_CARDS)
        assert np.all(hands[:, 2] == IGNORE)  # no fourth seat at a 3-player table
        # Cards the viewer can place exactly are never supervised.
        unseen = PAIR_UNSEEN[streams[seat][:, LEDGER_START:].astype(np.int64) % NUM_PAIRS]
        assert np.all((hands[:, :2] == IGNORE) == ~unseen[:, None, :])


def tiny_game(**game) -> RummyGame:
    settings = dict(player_counts=(2, 3, 4), hand_sizes=(7,), games_per_shape=3, max_turns=30)
    settings.update(game)
    model = RummyModelConfig(
        d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=64, per_layer_dim=8,
        ledger_width=8, ledger_hidden=16, ledger_layers=2, score_dim=16,
    )
    return RummyGame(model, RummyGameConfig(**settings), RummyLossConfig())


def test_cached_rollout_matches_full_replay():
    game = tiny_game()
    actor = randomize(game.build_actor().eval(), seed=3, scale=0.2)
    collector = SelfPlayCollector(
        actor, device=torch.device("cpu"), precision="fp32", kv_dtype="fp32", max_rows=64,
        max_length=game.max_length,
    )
    trajectories = collector.collect(game.new_games(random.Random(0), 0))
    assert collector.stats.compactions > 0  # games retire out of order
    worst = 0.0
    with torch.no_grad():
        for trajectory in trajectories:
            for seat, stream in enumerate(trajectory.streams):
                tokens = torch.from_numpy(stream.astype(np.int64))[None]
                hidden = actor.forward_hidden(tokens)[0]
                for decision in trajectory.decisions:
                    if decision.seat != seat:
                        continue
                    features = {k: torch.from_numpy(v)[None] for k, v in decision.features.items()}
                    logits = actor.policy_logits(
                        hidden[decision.position][None], decision.head, features,
                        tokens[0, decision.position][None],
                    )
                    probs = masked_probabilities(logits.numpy(), np.ones_like(logits.numpy(), dtype=bool))[0]
                    worst = max(worst, float(np.abs(probs - decision.probs).max()))
    assert worst < 1e-5


def test_card_grid_is_suit_equivariant():
    """Relabelling suits in the ledger relabels the per-card features the same way."""

    torch.manual_seed(0)
    grid = randomize(tiny_game().build_actor(), seed=1, scale=0.1).embedding.grid.eval()
    ledger = torch.randint(0, 144, (6, NUM_CARDS))
    permutation = torch.tensor([2, 0, 3, 1])  # new suit of each old suit
    cards = torch.arange(NUM_CARDS)
    moved = permutation[cards // 13] * 13 + cards % 13
    relabelled = torch.empty_like(ledger)
    relabelled[:, moved] = ledger
    with torch.no_grad():
        original = grid.cards(ledger)
        permuted = grid.cards(relabelled)
    assert torch.allclose(permuted[:, moved], original, atol=1e-5)
