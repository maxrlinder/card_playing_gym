from __future__ import annotations

import random
from itertools import combinations

import numpy as np
import pytest

from cardgym.games.rummy.cards import NUM_CARDS, parse_card
from cardgym.games.rummy.melds import (
    _BY_CARDSET,
    EXTENSIONS,
    MELD_CARDS,
    MELD_COUNTS,
    MELD_POINTS,
    MELD_SIZE,
    MELDS,
    MELDS_WITH,
    can_meld,
    hand_melds,
    meld_index,
    opening_melds,
)


def cards(*names: str) -> list[int]:
    return [parse_card(name) for name in names]


def counts_of(card_list: list[int]) -> np.ndarray:
    counts = np.zeros(NUM_CARDS, dtype=np.int64)
    for card in card_list:
        counts[card] += 1
    return counts


def test_universe_and_cyclic_runs():
    assert len(MELDS) == 13 * 5 + 4 * (13 * 10 + 1)
    assert MELD_POINTS[meld_index(cards("Ah", "2h", "3h"))] == 6
    assert MELD_POINTS[meld_index(cards("Qh", "Kh", "Ah"))] == 31
    assert MELD_POINTS[meld_index(cards("Kh", "Ah", "2h"))] == 23  # wrapped: Ace after King is 11
    full = meld_index([parse_card(r + "s") for r in "A23456789TJQK"])
    assert MELD_POINTS[full] == 11 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 40
    assert MELD_POINTS[meld_index(cards("Ac", "Ad", "Ah"))] == 33
    with pytest.raises(KeyError):
        meld_index(cards("2h", "4h", "5h"))  # gap
    with pytest.raises(KeyError):
        meld_index(cards("7c", "7d"))  # too short


def test_extensions():
    run = meld_index(cards("3h", "4h", "5h"))
    assert {card for card, _ in EXTENSIONS[run]} == set(cards("2h", "6h"))
    extended = dict(EXTENSIONS[run])[parse_card("2h")]
    assert set(MELD_CARDS[extended]) == set(cards("2h", "3h", "4h", "5h"))
    twelve = meld_index([parse_card(r + "d") for r in "23456789TJQK"])
    assert [card for card, _ in EXTENSIONS[twelve]] == cards("Ad")
    group = meld_index(cards("9c", "9d", "9s"))
    assert [card for card, _ in EXTENSIONS[group]] == cards("9h")


def _brute_melds(counts: np.ndarray) -> set[int]:
    present = [int(card) for card in np.flatnonzero(counts)]
    found = set()
    for size in range(3, min(13, len(present)) + 1):
        for subset in combinations(present, size):
            for index in _BY_CARDSET.get(frozenset(subset), ()):
                found.add(index)
    return found


def _brute_openings(counts: np.ndarray, need: int) -> set[int]:
    melds = sorted(_brute_melds(counts))
    legal = set()
    total = int(counts.sum())

    def search(start: int, remaining: np.ndarray, chosen: list[int], points: int) -> None:
        if chosen and points >= need and remaining.sum() >= 1:
            legal.update(chosen)
        for position in range(start, len(melds)):
            meld = melds[position]
            used = counts_of(list(MELD_CARDS[meld]))
            if np.all(remaining >= used):
                # The same meld may be laid twice from two decks: allow position again.
                search(position, remaining - used, chosen + [meld], points + MELD_POINTS[meld])

    search(0, counts.copy(), [], 0)
    assert total >= 1
    return legal


@pytest.mark.parametrize("seed", range(60))
def test_hand_melds_and_opening_match_brute_force(seed):
    rng = random.Random(seed)
    # Concentrate on a few suits and ranks so hands are meld-rich.
    start = rng.randrange(13)
    pool = [suit * 13 + (start + offset) % 13 for suit in rng.sample(range(4), 3) for offset in range(5)]
    hand = [rng.choice(pool) for _ in range(rng.randrange(8, 15))]
    counts = counts_of(hand)
    counts = np.minimum(counts, 2)
    melds = hand_melds(counts)
    assert len(melds) == len(set(melds))
    assert set(melds) == _brute_melds(counts)
    for need in (10, 30, 45):
        assert set(opening_melds(counts, melds, need)) == _brute_openings(counts, need)


def test_two_small_melds_open_together():
    hand = counts_of(cards("2c", "3c", "4c", "5d", "5h", "5s", "Kd"))  # 9 + 15 = 24 < 30
    assert opening_melds(hand, hand_melds(hand), 30) == []
    hand = counts_of(cards("8c", "9c", "Tc", "5d", "5h", "5s", "Kd"))  # 27 + 15 = 42
    melds = hand_melds(hand)
    assert set(opening_melds(hand, melds, 30)) == set(melds)
    # Without a spare card the opening may not use every card.
    tight = counts_of(cards("8c", "9c", "Tc", "5d", "5h", "5s"))
    assert opening_melds(tight, hand_melds(tight), 30) == []


def test_can_meld_matches_the_meld_table():
    """The group/run shortcut agrees with checking every meld that contains the card."""

    rng = np.random.default_rng(0)
    for _ in range(5000):
        counts = np.zeros(NUM_CARDS, dtype=np.int64)
        for card in rng.integers(0, NUM_CARDS, rng.integers(1, 20)):
            counts[card] = min(2, counts[card] + 1)
        card = int(rng.choice(np.flatnonzero(counts)))
        rows = MELDS_WITH[card]
        exact = ((MELD_COUNTS[rows] <= counts).all(axis=1) & (MELD_SIZE[rows] < counts.sum())).any()
        assert can_meld(counts, card) == bool(exact)
