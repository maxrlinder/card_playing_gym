"""Melds: the universe of valid groups and cyclic runs, lay-offs, and opening search.

Every valid meld is enumerated once at import (589 of them) and named by its
index, so table melds, candidates, and lay-off transitions are small ints:

* a **group** is 3-4 cards of one rank in distinct suits (the two decks do
  not relax the one-card-per-suit rule);
* a **run** is 3-13 cards of one suit on a contiguous arc of the rank cycle,
  so ``Q-K-A`` and ``K-A-2`` are runs. All 13-card runs are one meld.

Points (what a meld scores for the seat that lays it, and the opening
threshold when one is set): an Ace is 11 in groups and wherever a King
precedes it in a run, and 1 when it starts a shorter run (``A-2-3``); a full
13-card run counts it 11. A lay-off scores the difference it makes.
"""

from __future__ import annotations

from itertools import combinations
from typing import NamedTuple

import numpy as np

from .cards import ACE, NUM_CARDS, NUM_RANKS, NUM_SUITS, RANK_VALUES, card_id

GROUP = 0
RUN = 1
FULL_RUN = NUM_RANKS


class Meld(NamedTuple):
    kind: int
    # group: (rank, suit bitmask, 0); run: (suit, start rank, length)
    a: int
    b: int
    c: int


def _cards(meld: Meld) -> tuple[int, ...]:
    if meld.kind == GROUP:
        return tuple(card_id(suit, meld.a) for suit in range(NUM_SUITS) if meld.b >> suit & 1)
    return tuple(card_id(meld.a, (meld.b + i) % NUM_RANKS) for i in range(meld.c))


def _points(meld: Meld) -> int:
    if meld.kind == GROUP:
        return bin(meld.b).count("1") * RANK_VALUES[meld.a]
    total = 0
    for i in range(meld.c):
        rank = (meld.b + i) % NUM_RANKS
        if rank == ACE:
            total += 1 if i == 0 and meld.c < FULL_RUN else 11
        else:
            total += RANK_VALUES[rank]
    return total


def _run(suit: int, start: int, length: int) -> Meld:
    return Meld(RUN, suit, 0 if length == FULL_RUN else start % NUM_RANKS, length)


def _universe() -> list[Meld]:
    melds = []
    for rank in range(NUM_RANKS):
        for size in (3, 4):
            for suits in combinations(range(NUM_SUITS), size):
                melds.append(Meld(GROUP, rank, sum(1 << s for s in suits), 0))
    for suit in range(NUM_SUITS):
        for length in range(3, FULL_RUN):
            for start in range(NUM_RANKS):
                melds.append(_run(suit, start, length))
        melds.append(_run(suit, 0, FULL_RUN))
    return melds


MELDS: list[Meld] = _universe()
MELD_INDEX: dict[Meld, int] = {meld: index for index, meld in enumerate(MELDS)}
MELD_CARDS: list[tuple[int, ...]] = [_cards(meld) for meld in MELDS]
MELD_POINTS: list[int] = [_points(meld) for meld in MELDS]
MELD_KIND: list[int] = [meld.kind for meld in MELDS]
NUM_MELDS = len(MELDS)
MELD_COUNTS = np.zeros((NUM_MELDS, NUM_CARDS), dtype=np.int64)
for _index, _meld_cards in enumerate(MELD_CARDS):
    MELD_COUNTS[_index, list(_meld_cards)] = 1
MELD_SIZE = MELD_COUNTS.sum(axis=1)
# The melds that contain each card type.
MELDS_WITH = [np.flatnonzero(MELD_COUNTS[:, card]) for card in range(NUM_CARDS)]


def can_meld(counts, card: int) -> bool:
    """Whether the hand (52 copy counts) forms a meld containing ``card`` that leaves a card to discard."""

    if counts.sum() >= 4:
        # Any meld has 3+ cards, so a group (two other suits of the rank) or a
        # run (an arc of 3+ ranks through the card) is enough. Runs wrap.
        suit, rank = divmod(card, NUM_RANKS)
        if sum(1 for other in range(NUM_SUITS) if other != suit and counts[other * NUM_RANKS + rank]) >= 2:
            return True
        base = suit * NUM_RANKS
        below = above = 0
        while below < NUM_RANKS - 1 and counts[base + (rank - below - 1) % NUM_RANKS]:
            below += 1
        while above < NUM_RANKS - 1 - below and counts[base + (rank + above + 1) % NUM_RANKS]:
            above += 1
        return below + above >= 2
    rows = MELDS_WITH[card]
    fits = (MELD_COUNTS[rows] <= counts).all(axis=1) & (MELD_SIZE[rows] < counts.sum())
    return bool(fits.any())


def _extensions(meld: Meld) -> tuple[tuple[int, int], ...]:
    """(card, resulting meld index) for every card that can be laid off on ``meld``."""

    if meld.kind == GROUP:
        if meld.b == 0b1111:
            return ()
        return tuple(
            (card_id(suit, meld.a), MELD_INDEX[Meld(GROUP, meld.a, meld.b | 1 << suit, 0)])
            for suit in range(NUM_SUITS)
            if not meld.b >> suit & 1
        )
    suit, start, length = meld.a, meld.b, meld.c
    if length == FULL_RUN:
        return ()
    low = (card_id(suit, (start - 1) % NUM_RANKS), MELD_INDEX[_run(suit, start - 1, length + 1)])
    high = (card_id(suit, (start + length) % NUM_RANKS), MELD_INDEX[_run(suit, start, length + 1)])
    # A 12-card run's two ends are the same card and make the same full run.
    return (low,) if low == high else (low, high)


EXTENSIONS: list[tuple[tuple[int, int], ...]] = [_extensions(meld) for meld in MELDS]


def meld_index(cards: list[int] | tuple[int, ...]) -> int:
    """Index of the meld made of exactly these card types (KeyError if invalid)."""

    for index in _BY_CARDSET.get(frozenset(cards), ()):
        if len(MELD_CARDS[index]) == len(cards):
            return index
    raise KeyError(f"Not a meld: {cards}")


_BY_CARDSET: dict[frozenset[int], list[int]] = {}
for _index, _cards_of in enumerate(MELD_CARDS):
    _BY_CARDSET.setdefault(frozenset(_cards_of), []).append(_index)


def hand_melds(counts) -> list[int]:
    """Every distinct meld the hand (52 copy counts) can form."""

    melds: list[int] = []
    for rank in range(NUM_RANKS):
        suits = [suit for suit in range(NUM_SUITS) if counts[suit * NUM_RANKS + rank]]
        if len(suits) >= 3:
            for size in range(3, len(suits) + 1):
                for chosen in combinations(suits, size):
                    melds.append(MELD_INDEX[Meld(GROUP, rank, sum(1 << s for s in chosen), 0)])
    for suit in range(NUM_SUITS):
        base = suit * NUM_RANKS
        present = [counts[base + rank] > 0 for rank in range(NUM_RANKS)]
        total = sum(present)
        if total < 3:
            continue
        if total == NUM_RANKS:
            for length in range(3, FULL_RUN):
                for start in range(NUM_RANKS):
                    melds.append(MELD_INDEX[_run(suit, start, length)])
            melds.append(MELD_INDEX[_run(suit, 0, FULL_RUN)])
            continue
        for start in range(NUM_RANKS):
            if not present[start] or present[(start - 1) % NUM_RANKS]:
                continue  # not the first rank of a maximal arc
            arc = 0
            while present[(start + arc) % NUM_RANKS]:
                arc += 1
            for offset in range(arc - 2):
                for length in range(3, arc - offset + 1):
                    melds.append(MELD_INDEX[_run(suit, start + offset, length)])
    return melds


# -- opening search -------------------------------------------------------------


def opening_melds(counts, melds: list[int], need: int) -> list[int]:
    """The melds that can start (or continue) an opening worth ``need`` more points.

    A meld qualifies when it, plus some disjoint melds from the rest of the
    hand, reaches ``need`` while leaving at least one card to discard. The
    search runs over the cards that belong to any meld; a hand that also
    holds a card outside every meld satisfies the leftover rule for free.
    """

    if not melds:
        return []
    involved = sorted({card for index in melds for card in MELD_CARDS[index]})
    if sum(counts[card] * RANK_VALUES[card % NUM_RANKS] for card in involved) < need:
        return []
    position = {card: slot for slot, card in enumerate(involved)}
    spare = sum(int(count) for count in counts) - sum(int(counts[card]) for card in involved)
    shapes = [tuple(position[card] for card in MELD_CARDS[index]) for index in melds]
    by_lowest: list[list[int]] = [[] for _ in involved]
    for local, slots in enumerate(shapes):
        by_lowest[min(slots)].append(local)
    memo: dict[tuple[tuple[int, ...], bool], int] = {}

    def best(state: tuple[int, ...], leave: bool) -> int:
        """Max points of disjoint melds within ``state`` (-inf if a card must remain and cannot)."""

        key = (state, leave)
        cached = memo.get(key)
        if cached is not None:
            return cached
        lowest = next((slot for slot, count in enumerate(state) if count), None)
        if lowest is None:
            result = _IMPOSSIBLE if leave else 0
        else:
            skipped = list(state)
            skipped[lowest] -= 1
            result = best(tuple(skipped), False)
            for local in by_lowest[lowest]:
                remaining = _subtract(state, shapes[local])
                if remaining is not None:
                    result = max(result, MELD_POINTS[melds[local]] + best(remaining, leave))
        memo[key] = result
        return result

    start = tuple(int(counts[card]) for card in involved)
    leave = spare == 0
    chosen = []
    for local, index in enumerate(melds):
        remaining = _subtract(start, shapes[local])
        if remaining is not None and MELD_POINTS[index] + best(remaining, leave) >= need:
            chosen.append(index)
    return chosen


_IMPOSSIBLE = -(10**6)


def _subtract(state: tuple[int, ...], slots: tuple[int, ...]) -> tuple[int, ...] | None:
    remaining = list(state)
    for slot in slots:
        if not remaining[slot]:
            return None
        remaining[slot] -= 1
    return tuple(remaining)
