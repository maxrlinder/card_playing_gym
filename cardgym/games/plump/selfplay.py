"""Plump rounds as self-play games for the shared collector."""

from __future__ import annotations

import random

import numpy as np

from cardgym.rl.selfplay import Decision, SelfPlayGame

from .env import PlumpEnv
from .rounds import RoundSpec, round_game_config
from .rules import relative_rewards
from .state import BidAction, Phase, PlayCardAction
from .tokens import (
    NEXT_BID,
    NEXT_PLAY,
    NUM_CARDS,
    Schema,
    card_from_id,
    card_id,
    prefix_tokens,
    round_events,
    wave_rows,
)


class PlumpSelfPlay(SelfPlayGame):
    """One round, every seat played by the policy being trained."""

    def __init__(
        self, schema: Schema, num_players: int, hand_size: int, *, bidding_start: int, seed: int
    ) -> None:
        self.schema = schema
        self.num_players = num_players
        self.hand_size = hand_size
        self.bidding_start = bidding_start
        self.env = PlumpEnv(
            round_game_config(RoundSpec(num_players, hand_size), bidding_start_player=bidding_start),
            seed=seed,
        )
        # Nothing reads per-step observations; building them dominates stepping.
        self.env.emit_observations = False
        self.env.reset(seed=seed)
        self.rng = random.Random(seed ^ 0x5EED)
        round_state = self.env.state.current_round
        self.initial_hands = {seat: list(hand) for seat, hand in round_state.initial_hands.items()}
        self._seen = len(self.env.state.event_log)
        self._decision: Decision | None = None

    def lifetime(self) -> int:
        return self.num_players * self.hand_size

    def prefix_tokens(self) -> list[np.ndarray]:
        return [
            prefix_tokens(
                self.schema, seat, self.num_players, self.hand_size,
                self.initial_hands[seat], self.bidding_start,
            )
            for seat in range(self.num_players)
        ]

    def decision(self) -> Decision:
        if self._decision is None:
            seat = self.env.current_player()
            if self.env.phase() == Phase.BIDDING:
                legal = np.zeros(self.schema.bid_count, dtype=bool)
                legal[self.env.legal_bid_values()] = True
                self._decision = Decision(seat=seat, head="bid", legal=legal)
            else:
                legal = np.zeros(NUM_CARDS, dtype=bool)
                legal[[card_id(card) for card in self.env.legal_card_values()]] = True
                self._decision = Decision(seat=seat, head="play", legal=legal)
        return self._decision

    def step(self, action: int) -> list[np.ndarray]:
        env = self.env
        seat = env.current_player()
        if env.phase() == Phase.BIDDING:
            env.step_unchecked(BidAction(seat, action))
        else:
            env.step_unchecked(PlayCardAction(seat, card_from_id(action)))
        # Once every hand is down to one card the rest of the round is forced.
        while not env.is_done() and len(env.state.current_round.current_hands[env.current_player()]) <= 1:
            player = env.current_player()
            env.step_unchecked(PlayCardAction(player, env.state.current_round.current_hands[player][0]))
        events = round_events(env.state.event_log[self._seen :])
        self._seen = len(env.state.event_log)
        self._decision = None
        round_state = env.state.current_round
        done = env.is_done()
        return wave_rows(
            self.schema,
            events,
            self.num_players,
            self.hand_size,
            bids=[bid.value for bid in round_state.bids],
            hands=round_state.current_hands,
            next_actor=None if done else env.current_player(),
            next_phase=NEXT_BID if env.phase() == Phase.BIDDING else NEXT_PLAY,
        )

    def done(self) -> bool:
        return self.env.is_done()

    def returns(self) -> np.ndarray:
        rewards = relative_rewards(self.env.state.current_round.round_scores)
        return np.asarray([rewards[seat] for seat in range(self.num_players)], dtype=np.float64)
