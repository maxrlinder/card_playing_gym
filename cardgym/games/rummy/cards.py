"""One or two standard decks without jokers: 52 card types, one copy of each per deck.

A card type is ``suit * 13 + rank`` with ranks ordered A, 2, ..., K (rank 0 is
the Ace). Runs are cyclic, so rank ``r`` is followed by ``(r + 1) % 13`` and
the King by the Ace. With two decks the two physical copies of a type are
interchangeable: everything here counts copies rather than naming them.
"""

from __future__ import annotations

import numpy as np

NUM_SUITS = 4
NUM_RANKS = 13
NUM_CARDS = NUM_SUITS * NUM_RANKS
MAX_DECKS = 2
COPIES = MAX_DECKS  # most copies of a card type in play
NO_CARD = NUM_CARDS  # padding / not-applicable card id

ACE = 0
KING = 12

RANK_NAMES = "A23456789TJQK"
SUIT_NAMES = "cdhs"

# Hand penalty and group value per rank: Ace 11, pips face value, 10-K ten.
RANK_VALUES = (11, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10)
CARD_VALUES = np.array([RANK_VALUES[card % NUM_RANKS] for card in range(NUM_CARDS)], dtype=np.int64)
CARD_RANKS = np.arange(NUM_CARDS) % NUM_RANKS
CARD_SUITS = np.arange(NUM_CARDS) // NUM_RANKS


def card_id(suit: int, rank: int) -> int:
    return suit * NUM_RANKS + rank


def suit_of(card: int) -> int:
    return card // NUM_RANKS


def rank_of(card: int) -> int:
    return card % NUM_RANKS


def card_name(card: int) -> str:
    if card == NO_CARD:
        return "--"
    return RANK_NAMES[rank_of(card)] + SUIT_NAMES[suit_of(card)]


def parse_card(name: str) -> int:
    return card_id(SUIT_NAMES.index(name[1].lower()), RANK_NAMES.index(name[0].upper()))


def hand_penalty(counts: np.ndarray) -> int:
    """Points a hand loses at the end of a round."""

    return int(np.dot(counts.astype(np.int64), CARD_VALUES))


def full_deck(decks: int = 1) -> list[int]:
    return [card for card in range(NUM_CARDS) for _ in range(decks)]
