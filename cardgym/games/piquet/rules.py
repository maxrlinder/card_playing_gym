"""Declarations: point, sequence, and sets (with their tables over suit bytes).

Each category ranks a hand by a key; the better key wins the category and its
holder scores. ``best`` returns the key (None when the hand has nothing to
declare), ``score`` what the winner scores, and ``shown`` the cards the winner
shows (and so reveals to the opponent).

* Point: the suit with the most cards; ties go to the higher card total
  (A 11, K/Q/J/10 10, pips face value). Scores 1 per card.
* Sequence: three or more consecutive cards of a suit; the longer wins, then
  the higher top card. The winner scores every sequence it holds: 3 and 4
  cards score 3 and 4; 5 to 8 cards score 15 to 18.
* Sets: three or four cards of one rank from 10 up; four beat three, then
  the higher rank. The winner scores every set: 14 for four, 3 for three.

Keys compare as tuples; equal point or sequence keys score for neither.
"""

from __future__ import annotations

from .cards import NUM_RANKS, NUM_SUITS, POINT_VALUES, TEN, card_id, mask_of, suit_bits

POINT, SEQUENCE, SETS = 0, 1, 2
CATEGORIES = (POINT, SEQUENCE, SETS)
CATEGORY_NAMES = ("point", "sequence", "sets")
SEQUENCE_SCORES = (0, 0, 0, 3, 4, 15, 16, 17, 18)
SET_SCORES = (0, 0, 0, 3, 14)


def _runs(bits: int) -> tuple[tuple[int, int], ...]:
    """Maximal runs of three or more set bits: (length, top rank)."""

    runs, start = [], None
    for rank in range(NUM_RANKS + 1):
        held = rank < NUM_RANKS and bits >> rank & 1
        if held and start is None:
            start = rank
        elif not held and start is not None:
            if rank - start >= 3:
                runs.append((rank - start, rank - 1))
            start = None
    return tuple(runs)


POINT_LENGTH = tuple(bin(bits).count("1") for bits in range(256))
POINT_VALUE = tuple(sum(POINT_VALUES[r] for r in range(NUM_RANKS) if bits >> r & 1) for bits in range(256))
RUNS = tuple(_runs(bits) for bits in range(256))
RUN_BITS = tuple(
    sum(((1 << length) - 1) << (top - length + 1) for length, top in RUNS[bits]) for bits in range(256)
)


def point_key(hand: int) -> tuple[int, int, int] | None:
    """(length, value, suit) of the best suit; None for an empty hand."""

    best = None
    for suit in range(NUM_SUITS):
        bits = suit_bits(hand, suit)
        if bits:
            key = (POINT_LENGTH[bits], POINT_VALUE[bits], -suit)
            if best is None or key > best:
                best = key
    return None if best is None else (best[0], best[1], -best[2])


def sequence_key(hand: int) -> tuple[int, int] | None:
    best = None
    for suit in range(NUM_SUITS):
        for run in RUNS[suit_bits(hand, suit)]:
            if best is None or run > best:
                best = run
    return best


def _set_ranks(hand: int) -> list[tuple[int, int]]:
    """(size, rank) of every set from the 10 up."""

    sets = []
    for rank in range(TEN, NUM_RANKS):
        size = sum(hand >> card_id(suit, rank) & 1 for suit in range(NUM_SUITS))
        if size >= 3:
            sets.append((size, rank))
    return sets


def sets_key(hand: int) -> tuple[int, int] | None:
    sets = _set_ranks(hand)
    return max(sets) if sets else None


def best(category: int, hand: int) -> tuple[int, ...] | None:
    if category == POINT:
        return point_key(hand)
    if category == SEQUENCE:
        return sequence_key(hand)
    return sets_key(hand)


def compare(category: int, a: tuple[int, ...], b: tuple[int, ...]) -> int:
    """Sign of a - b; a point's suit never breaks a tie."""

    a, b = (a[:2], b[:2]) if category == POINT else (a, b)
    return (a > b) - (a < b)


def score(category: int, hand: int) -> int:
    if category == POINT:
        key = point_key(hand)
        return 0 if key is None else key[0]
    if category == SEQUENCE:
        return sum(SEQUENCE_SCORES[length] for suit in range(NUM_SUITS) for length, _ in RUNS[suit_bits(hand, suit)])
    return sum(SET_SCORES[size] for size, _ in _set_ranks(hand))


def shown(category: int, hand: int) -> int:
    """The cards a winning declaration shows."""

    if category == POINT:
        key = point_key(hand)
        return 0 if key is None else suit_bits(hand, key[2]) << (key[2] * NUM_RANKS)
    if category == SEQUENCE:
        return sum(RUN_BITS[suit_bits(hand, suit)] << (suit * NUM_RANKS) for suit in range(NUM_SUITS))
    return mask_of(
        card_id(suit, rank)
        for _, rank in _set_ranks(hand)
        for suit in range(NUM_SUITS)
        if hand >> card_id(suit, rank) & 1
    )
