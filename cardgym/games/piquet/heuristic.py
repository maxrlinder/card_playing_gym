"""Rule-based baselines that read only their own hand and public state."""

from __future__ import annotations

import random

import numpy as np

from .cards import NUM_RANKS, NUM_SUITS, TEN, card_id, cards_of, rank_of, suit_bits, suit_mask, suit_of
from .engine import DECLARE, EXCHANGE, SPEAK, STOP, PiquetEngine
from .rules import RUN_BITS


class RandomPolicy:
    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def act(self, engine: PiquetEngine) -> int:
        return self.rng.choice(np.flatnonzero(engine.legal()).tolist())


# How much a card is worth keeping by rank alone (guards and winners).
RANK_KEEP = (0, 0, 0, 1, 1, 2, 4, 7)


def keep_score(hand: int, card: int) -> int:
    """Higher for cards in sequences, sets, the long suit, and high ranks."""

    suit, rank = suit_of(card), rank_of(card)
    bits = suit_bits(hand, suit)
    score = 2 * bin(bits).count("1") + RANK_KEEP[rank]
    if RUN_BITS[bits] >> rank & 1:
        score += 8
    if rank >= TEN and sum(hand >> card_id(s, rank) & 1 for s in range(NUM_SUITS)) >= 3:
        score += 8
    # Near-sequences: a neighbour or a one-gap neighbour in the suit.
    score += sum(1 for step in (-2, -1, 1, 2) if 0 <= rank + step < NUM_RANKS and bits >> (rank + step) & 1)
    return score


class HeuristicPolicy:
    """Elder throws its five least useful cards, Younger up to three weak ones; always declares;
    leads sure winners of its longest suit, else low from it; follows with the cheapest winner,
    else its lowest card; when void, throws the lowest card of its shortest suit."""

    def __init__(self, younger_threshold: int = 8) -> None:
        self.younger_threshold = younger_threshold

    def act(self, engine: PiquetEngine) -> int:
        if engine.phase == EXCHANGE:
            return self._exchange(engine)
        if engine.phase == DECLARE:
            return SPEAK
        return self._play(engine)

    def _exchange(self, engine: PiquetEngine) -> int:
        seat = engine.current
        hand = engine.hands[seat]
        ranked = sorted(cards_of(hand), key=lambda card: (keep_score(hand, card), rank_of(card)))
        if seat == engine.elder:
            plan = ranked[: engine.max_discards]
        else:
            weak = [card for card in ranked if keep_score(hand, card) < self.younger_threshold]
            plan = weak[: min(3, engine.max_discards)]
        for card in plan:
            if not engine.selected >> card & 1:
                return card
        return STOP

    @staticmethod
    def _play(engine: PiquetEngine) -> int:
        seat = engine.current
        hand = engine.hands[seat]
        legal = cards_of(engine.legal_mask())
        gone = engine.played[0] | engine.played[1] | hand
        if engine.trick:
            led = engine.trick[0][1]
            same = [card for card in legal if suit_of(card) == suit_of(led)]
            if same:
                winners = [card for card in same if rank_of(card) > rank_of(led)]
                return min(winners) if winners else min(same)
            lengths = {s: bin(suit_bits(hand, s)).count("1") for s in range(NUM_SUITS)}
            return min(legal, key=lambda card: (lengths[suit_of(card)], rank_of(card)))

        def sure(card: int) -> bool:
            higher = suit_mask(suit_of(card)) & ~((2 << card) - 1)
            return higher & ~gone == 0

        lengths = {s: bin(suit_bits(hand, s)).count("1") for s in range(NUM_SUITS)}
        winners = [card for card in legal if sure(card)]
        if winners:
            return max(winners, key=lambda card: (lengths[suit_of(card)], rank_of(card)))
        longest = max(range(NUM_SUITS), key=lambda s: (lengths[s], s))
        return min(card for card in legal if suit_of(card) == longest)
