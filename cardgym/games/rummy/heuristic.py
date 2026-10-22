"""Rule-based baselines that read only their own hand and public state."""

from __future__ import annotations

import random

from .cards import CARD_VALUES, NUM_RANKS, card_id, rank_of, suit_of
from .engine import (
    DISCARD,
    DRAW,
    LAYOFF_GROUP,
    LAYOFF_RUN,
    MELD_GROUP,
    MELD_RUN,
    STOCK,
    TAKE,
    RummyEngine,
)
from .melds import MELD_CARDS, MELD_POINTS, hand_melds


class RandomPolicy:
    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def act(self, engine: RummyEngine) -> int:
        return self.rng.randrange(len(engine.candidates()))


class HeuristicPolicy:
    """Greedy: take the discard top if it makes a new meld, else dig at most
    ``max_dig`` cards deep for a card that melds (a deeper card is only offered
    when it can be used); meld and lay off at once, then throw the card with
    the least meld potential (highest value first)."""

    def __init__(self, max_dig: int = 3) -> None:
        self.max_dig = max_dig

    def act(self, engine: RummyEngine) -> int:
        candidates = engine.candidates()
        seat = engine.current
        hand = engine.hands[seat]
        if engine.phase == DRAW:
            for index, action in enumerate(candidates):
                if action.kind != TAKE:
                    continue
                if action.meld == 0 and self._wants(engine, seat, action.card):
                    return index
                if 0 < action.meld < self.max_dig:
                    return index  # the shallowest usable deeper card
            return next(i for i, action in enumerate(candidates) if action.kind == STOCK)
        melds = [i for i, a in enumerate(candidates) if a.kind in (MELD_GROUP, MELD_RUN)]
        if melds:
            return max(melds, key=lambda i: (len(MELD_CARDS[candidates[i].meld]), MELD_POINTS[candidates[i].meld]))
        layoffs = [i for i, a in enumerate(candidates) if a.kind in (LAYOFF_GROUP, LAYOFF_RUN)]
        if layoffs:
            return layoffs[0]
        discards = [i for i, a in enumerate(candidates) if a.kind == DISCARD]
        return min(discards, key=lambda i: (potential(hand, candidates[i].card), -CARD_VALUES[candidates[i].card]))

    @staticmethod
    def _wants(engine: RummyEngine, seat: int, card: int) -> bool:
        if engine.opened[seat] and engine.layable[card]:
            return True
        hand = engine.hands[seat].copy()
        before = {meld for meld in hand_melds(hand) if card in MELD_CARDS[meld]}
        hand[card] += 1
        return any(card in MELD_CARDS[meld] and meld not in before for meld in hand_melds(hand))


def potential(hand, card: int) -> int:
    """How much a card contributes to melds: 3 inside a meld, else partial links."""

    if any(card in MELD_CARDS[meld] for meld in hand_melds(hand)):
        return 3
    suit, rank = suit_of(card), rank_of(card)
    same_rank = sum(1 for s in range(4) if s != suit and hand[card_id(s, rank)])
    neighbours = sum(
        1 for step in (-2, -1, 1, 2) if hand[card_id(suit, (rank + step) % NUM_RANKS)]
    )
    return min(2, max(same_rank, neighbours))
