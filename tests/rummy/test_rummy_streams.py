"""Streams carry exactly the viewer's knowledge, and rollout equals replay."""

from __future__ import annotations

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
    LOC_PILE,
    LOC_TABLE,
    LOC_TOP,
    LOC_UNSEEN,
    MAX_PLAYERS,
    NUM_LOCATIONS,
    NUM_MOVERS,
    NUM_PAIRS,
    PAIR,
    PAIR_LOCATIONS,
    RummyEngine,
)
from cardgym.games.rummy.game import RummyGame, RummyGameConfig
from cardgym.games.rummy.heuristic import HeuristicPolicy
from cardgym.games.rummy.model import RummyModelConfig
from cardgym.games.rummy.selfplay import RummySelfPlay
from cardgym.games.rummy.tokens import (
    DRAW_ABSENT,
    DRAW_NONE,
    DRAW_STOCK,
    DRAW_TAKE,
    F_OPP_DISCARD,
    F_OPP_DRAW,
    F_OPP_TAKE,
    F_PHASE,
    LEDGER_START,
    LEDGER_VOCAB,
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


def heuristic_round(players: int, seed: int, hand_size: int = 13) -> tuple[RummySelfPlay, list[np.ndarray]]:
    game = RummySelfPlay(players, hand_size, seed=seed)
    policy = HeuristicPolicy()
    streams = [[block] for block in game.prefix_tokens()]
    while not game.done():
        for seat, block in enumerate(game.step(policy.act(game.engine))):
            streams[seat].append(block)
    return game, [np.concatenate(blocks) for blocks in streams]


def token_events(game: RummySelfPlay, seat: int) -> list[int]:
    """Events seen before each of the seat's tokens: one token right before each of its own events."""

    return [index for index, event in enumerate(game.events) if event.actor == seat]


def public_pairs(game: RummySelfPlay, observer: int) -> np.ndarray:
    """The observer's copy-location pairs after 0..E events, rebuilt from what it saw."""

    players = game.num_players
    engine = RummyEngine(players, game.hand_size, game.seed)  # the same deal
    own = engine.hands[observer].copy()
    known = np.zeros((players, NUM_CARDS), dtype=np.int64)
    table = np.zeros(NUM_CARDS, dtype=np.int64)
    pile = [engine.top]

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
            for seat in range(players):
                stream = streams[seat].astype(np.int64)
                seen = token_events(game, seat)
                assert stream.shape[0] == len(seen)  # one token per own decision
                expected = public_pairs(game, seat)
                ledger = stream[:, LEDGER_START:]
                assert np.array_equal(ledger % NUM_PAIRS, expected[seen])
                # Mover: who last changed each card type (in this view) since the previous token.
                mover = (ledger // NUM_PAIRS) % NUM_MOVERS
                previous = 0
                for token, now in enumerate(seen):
                    want = np.zeros(NUM_CARDS, dtype=np.int64)
                    for count in range(previous + 1, now + 1):
                        changed = expected[count] != expected[count - 1]
                        want[changed] = 1 + (game.events[count - 1].actor - seat) % players
                    want[expected[now] == expected[previous]] = 0
                    assert np.array_equal(mover[token], want)
                    # The opponents' turns since the previous token.
                    for offset in range(1, MAX_PLAYERS):
                        draw = stream[token, F_OPP_DRAW + offset - 1]
                        taken = stream[token, F_OPP_TAKE + offset - 1]
                        discard = stream[token, F_OPP_DISCARD + offset - 1]
                        if offset >= players:
                            assert draw == DRAW_ABSENT
                            continue
                        moves = [e for e in game.events[previous:now] if e.actor == (seat + offset) % players]
                        draws = [e for e in moves if e.kind in (EVENT_STOCK, EVENT_TAKE)]
                        discards = [e for e in moves if e.kind == EVENT_DISCARD]
                        kind = DRAW_NONE if not draws else DRAW_STOCK if draws[-1].kind == EVENT_STOCK else DRAW_TAKE
                        assert draw == kind
                        assert taken == (draws[-1].card if draws and draws[-1].kind == EVENT_TAKE else NO_CARD)
                        assert discard == (discards[-1].card if discards else NO_CARD)
                    previous = now


def test_oracle_streams_see_every_hand():
    for players in (2, 3, 4):
        game, streams = heuristic_round(players, seed=5)
        for seat in range(players):
            oracle = np.stack(game.oracle[seat]).astype(np.int64)
            hands = np.stack(game.hands[seat]).astype(np.int64)  # [L, P, 52]
            assert oracle.shape == streams[seat].shape
            assert np.array_equal(oracle[:, F_PHASE], streams[seat][:, F_PHASE])
            locations = PAIR_LOCATIONS[oracle[:, LEDGER_START:] % NUM_PAIRS]  # [L, 52, 2]
            for offset in range(players):
                held = (locations == 1 + offset).sum(axis=-1)
                assert np.array_equal(held, hands[:, (seat + offset) % players])


def test_training_sequences_and_belief_masks():
    game, streams = heuristic_round(3, seed=11)

    class Record:
        def __init__(self, seat, head, position):
            self.seat, self.head, self.position = seat, head, position

    decisions = []
    seen = [0] * game.num_players
    for event in game.events:
        head = "draw" if event.kind in (EVENT_STOCK, EVENT_TAKE) else "play"
        decisions.append(Record(event.actor, head, seen[event.actor]))
        seen[event.actor] += 1

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


DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_continuous_rollout_matches_full_replay(device):
    """Decoding (cached, gathered rows, refilled slots, carried over) equals a full replay."""

    game = tiny_game()
    actor = randomize(game.build_actor().eval(), seed=3, scale=0.2).to(device)
    collector = ContinuousCollector(
        actor, device=torch.device(device), precision="fp32", kv_dtype="fp32", max_length=game.max_length,
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
    ledger = torch.randint(0, LEDGER_VOCAB, (6, NUM_CARDS))
    permutation = torch.tensor([2, 0, 3, 1])  # new suit of each old suit
    cards = torch.arange(NUM_CARDS)
    moved = permutation[cards // 13] * 13 + cards % 13
    relabelled = torch.empty_like(ledger)
    relabelled[:, moved] = ledger
    with torch.no_grad():
        original = grid.cards(ledger)
        permuted = grid.cards(relabelled)
    assert torch.allclose(permuted[:, moved], original, atol=1e-5)
