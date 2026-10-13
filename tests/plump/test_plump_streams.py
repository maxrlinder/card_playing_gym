"""Self-play streams, labels, and cached readouts against reference builders."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from cardgym.games.plump.labels import replay_labels
from cardgym.games.plump.tokens import (
    IGNORE,
    NA_PATTERN_MASKS,
    NEXT_BID,
    NEXT_PLAY,
    SLOT_NA_PATTERN,
    SLOT_NEXT_ACTOR,
    SLOT_NEXT_PHASE,
    SLOT_REL_PLAYER,
    SLOT_TYPE,
    TOKEN_BID,
    TOKEN_TURN,
    seat_tokens,
)
from cardgym.rl.selfplay import SelfPlayCollector
from tests.conftest import randomize, tiny_plump_game


@pytest.fixture(scope="module")
def collected():
    game = tiny_plump_game(hand_sizes=(3, 4, 7, 10), player_counts=(3, 4, 5))
    actor = randomize(game.build_actor()).eval()
    collector = SelfPlayCollector(
        actor, device=torch.device("cpu"), precision="fp32", kv_dtype="fp32",
        max_rows=40, max_length=game.max_length,
    )
    return game, actor, collector.collect(game.new_games(random.Random(11), 1))


def test_collected_streams_equal_the_reference_builder(collected):
    game, _, trajectories = collected
    for trajectory in trajectories:
        play = trajectory.game
        round_state = play.env.state.current_round
        for seat, stream in enumerate(trajectory.streams):
            reference = seat_tokens(
                game.schema, play.env.state.event_log, seat, play.num_players, play.hand_size,
                round_state.initial_hands[seat], play.bidding_start,
            )
            np.testing.assert_array_equal(stream, reference)
            assert stream.shape[0] == game.schema.seq_len(play.num_players, play.hand_size)
            assert set(stream[:, SLOT_NA_PATTERN]) <= set(range(1, len(NA_PATTERN_MASKS)))


def test_decisions_read_the_annotated_readout(collected):
    _, _, trajectories = collected
    for trajectory in trajectories:
        players, hand = trajectory.game.num_players, trajectory.game.hand_size
        assert len(trajectory.decisions) == players * hand  # last trick is forced
        for decision in trajectory.decisions:
            row = trajectory.streams[decision.seat][decision.position]
            assert row[SLOT_NEXT_ACTOR] == 0  # announces the observer itself
            assert row[SLOT_NEXT_PHASE] == (NEXT_BID if decision.head == "bid" else NEXT_PLAY)
            assert (row[SLOT_TYPE] == TOKEN_TURN) == (decision.head == "bid")
            assert decision.probs[decision.action] > 0


def test_cached_policy_equals_full_forward_replay(collected):
    _, actor, trajectories = collected
    for trajectory in trajectories[:12]:
        for seat in range(trajectory.game.num_players):
            stream = torch.from_numpy(trajectory.streams[seat])[None]
            with torch.no_grad():
                hidden = actor.forward_hidden(stream)[0]
            for decision in (d for d in trajectory.decisions if d.seat == seat):
                with torch.no_grad():
                    logits = actor.policy_logits(hidden[decision.position][None], decision.head)[0]
                legal = decision.probs > 0
                expected = torch.softmax(logits[: len(legal)].masked_fill(~torch.from_numpy(legal), -1e9), -1)
                np.testing.assert_allclose(expected.numpy(), decision.probs, atol=1e-5)


def test_returns_are_zero_sum(collected):
    _, _, trajectories = collected
    for trajectory in trajectories:
        assert trajectory.returns.sum() == pytest.approx(0.0, abs=1e-9)


def test_labels_are_consistent_with_the_round(collected):
    game, _, trajectories = collected
    trajectory = next(t for t in trajectories if t.game.hand_size == 10)
    play = trajectory.game
    round_state = play.env.state.current_round
    for seat in range(play.num_players):
        labels = replay_labels(
            game.schema, play.initial_hands, play.env.state.event_log, seat, play.num_players, play.hand_size
        )
        final = labels["trick_targets"][: play.num_players]
        expected = [round_state.tricks_won[(seat + rel) % play.num_players] for rel in range(play.num_players)]
        assert final.tolist() == expected
        # The final-trick target is always within the feasible mask.
        masks = labels["trick_masks"]
        for position in range(masks.shape[0]):
            assert all(masks[position, rel, final[rel]] for rel in range(play.num_players))
        # Next-winner labels exist exactly from the last bid's token through
        # the token before the final trick's winner.
        labeled = np.flatnonzero(labels["next_winner"] != IGNORE)
        assert labeled.size and labeled.max() < masks.shape[0] - 1
        assert (labels["suit"][:, play.num_players - 1 :] == IGNORE).all()
        stream = trajectory.streams[seat]
        assert stream[labeled[0], SLOT_TYPE] == TOKEN_BID  # the last bid
        assert stream[0, SLOT_REL_PLAYER] == game.schema.player_na
