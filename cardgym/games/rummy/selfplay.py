"""Rummy rounds as self-play games for the shared collectors.

Each decision's legal actions go to the policy as a candidate list (a pointer
head): per candidate its kind, the cards it moves out of the hand (for a
discard take, the deepest card taken), and for a lay-off the target meld's
cards. Draw candidates also carry every card they would take from the pile.
"""

from __future__ import annotations

import random

import numpy as np

from cardgym.rl.ppo import relative_rewards
from cardgym.rl.selfplay import Decision, SelfPlayGame

from .cards import NO_CARD, NUM_CARDS
from .engine import DRAW, LAYOFF_GROUP, LAYOFF_RUN, MELD_GROUP, MELD_RUN, TAKE, Action, Event, RummyEngine
from .melds import MELD_CARDS
from .tokens import WIDTH, ObservationWriter

MAX_MELD_CARDS = 13
EMPTY = np.zeros((0, WIDTH), dtype=np.int16)


def candidate_features(engine: RummyEngine, actions: list[Action]) -> dict[str, np.ndarray]:
    count = len(actions)
    kind = np.empty(count, dtype=np.int8)
    cards = np.full((count, MAX_MELD_CARDS), NO_CARD, dtype=np.int8)
    target = np.full((count, MAX_MELD_CARDS), NO_CARD, dtype=np.int8)
    for index, action in enumerate(actions):
        kind[index] = action.kind
        if action.kind in (MELD_GROUP, MELD_RUN):
            meld = MELD_CARDS[action.meld]
            cards[index, : len(meld)] = meld
        elif action.card != NO_CARD:
            cards[index, 0] = action.card
            if action.kind in (LAYOFF_GROUP, LAYOFF_RUN):
                meld = MELD_CARDS[engine.table[action.meld]]
                target[index, : len(meld)] = meld
    features = {"kind": kind, "cards": cards, "target": target}
    if engine.phase == DRAW:
        # Copies of each card type each draw would take from the pile.
        taken = np.zeros((count, NUM_CARDS), dtype=np.int8)
        for index, action in enumerate(actions):
            if action.kind == TAKE:
                np.add.at(taken[index], engine.pile[len(engine.pile) - 1 - action.meld :], 1)
        features["taken"] = taken
    return features


def head_of(engine: RummyEngine) -> str:
    return "draw" if engine.phase == DRAW else "play"


class RummySelfPlay(SelfPlayGame):
    """One round, every seat played by the policy being trained.

    A step that ends a turn gives every other seat that turn's token; the seat
    about to decide then receives its decision token last (its readout).
    Besides the actor streams it records what training needs: each seat's
    oracle stream (same positions) and the true hands at each of its tokens.
    """

    def __init__(
        self,
        num_players: int,
        hand_size: int,
        *,
        seed: int,
        decks: int = 1,
        opening_points: int = 0,
        max_turns: int = 200,
    ) -> None:
        self.num_players = num_players
        self.hand_size = hand_size
        self.decks = decks
        self.seed = seed
        self.engine = RummyEngine(
            num_players, hand_size, seed, decks=decks, opening_points=opening_points, max_turns=max_turns
        )
        self.rng = random.Random(seed ^ 0x5EED)
        self.writer = ObservationWriter(self.engine)
        self.oracle: list[list[np.ndarray]] = [[] for _ in range(num_players)]
        self.hands: list[list[np.ndarray]] = [[] for _ in range(num_players)]
        self.events: list[Event] = []
        self._decision: Decision | None = None
        self._prefix = self._observe()

    def lifetime(self) -> int:
        return len(self.engine.stock)

    def _observe(self, turn: np.ndarray | None = None, actor: int = -1) -> list[np.ndarray]:
        engine = self.engine
        rows: list[list[np.ndarray]] = [[] for _ in range(self.num_players)]
        hands = engine.hands.astype(np.int8)
        if turn is not None:
            for seat in range(self.num_players):
                if seat != actor:
                    rows[seat].append(turn[seat, 0])
                    self.oracle[seat].append(turn[seat, 1])
                    self.hands[seat].append(hands)
        if not engine.done:
            seat = engine.current
            token, oracle = self.writer.observe(seat)
            rows[seat].append(token)
            self.oracle[seat].append(oracle)
            self.hands[seat].append(hands)
        return [np.stack(seat_rows) if seat_rows else EMPTY for seat_rows in rows]

    def prefix_tokens(self) -> list[np.ndarray]:
        return self._prefix

    def decision(self) -> Decision:
        if self._decision is None:
            engine = self.engine
            actions = engine.candidates()
            self._decision = Decision(
                seat=engine.current,
                head=head_of(engine),
                legal=np.ones(len(actions), dtype=bool),
                features=candidate_features(engine, actions),
            )
        return self._decision

    def step(self, action: int) -> list[np.ndarray]:
        engine = self.engine
        event = engine.apply(engine.candidates()[action])
        turn = self.writer.record(event)
        self.events.append(event)
        self._decision = None
        return self._observe(turn, event.actor)

    def done(self) -> bool:
        return self.engine.done

    def returns(self) -> np.ndarray:
        return relative_rewards(self.engine.scores())
