"""Piquet's 32-card deck: 7 to Ace in four suits, Ace high, every card unique.

A card is ``suit * 8 + rank`` (rank 0 is the 7, rank 7 the Ace), so a hand is
a 32-bit mask and each suit one byte of it.
"""

from __future__ import annotations

import numpy as np

NUM_SUITS = 4
NUM_RANKS = 8
NUM_CARDS = NUM_SUITS * NUM_RANKS
NO_CARD = NUM_CARDS

RANK_NAMES = ("7", "8", "9", "10", "J", "Q", "K", "A")
SUIT_NAMES = ("S", "H", "D", "C")
SUIT_SYMBOLS = ("♠", "♥", "♦", "♣")
TEN, JACK, QUEEN, KING, ACE = 3, 4, 5, 6, 7
# Card values for the point: Ace 11, faces and ten 10, pips their face value.
POINT_VALUES = (7, 8, 9, 10, 10, 10, 10, 11)

FULL_MASK = (1 << NUM_CARDS) - 1


def card_id(suit: int, rank: int) -> int:
    return suit * NUM_RANKS + rank


def suit_of(card: int) -> int:
    return card >> 3


def rank_of(card: int) -> int:
    return card & 7


def card_name(card: int) -> str:
    return RANK_NAMES[rank_of(card)] + SUIT_NAMES[suit_of(card)]


def parse_card(text: str) -> int:
    """``"10H"``, ``"TH"``, ``"AS"``, ``"7c"`` -> card id."""

    text = text.strip().upper()
    rank, suit = text[:-1], text[-1]
    rank = "10" if rank == "T" else rank
    return card_id(SUIT_NAMES.index(suit), RANK_NAMES.index(rank))


def mask_of(cards) -> int:
    mask = 0
    for card in cards:
        mask |= 1 << card
    return mask


def cards_of(mask: int) -> list[int]:
    cards = []
    while mask:
        low = mask & -mask
        cards.append(low.bit_length() - 1)
        mask ^= low
    return cards


def suit_bits(mask: int, suit: int) -> int:
    """The suit's byte of a hand: bit r is rank r."""

    return (mask >> (suit * NUM_RANKS)) & 0xFF


def suit_mask(suit: int) -> int:
    return 0xFF << (suit * NUM_RANKS)


FACE_MASK = mask_of(card_id(s, r) for s in range(NUM_SUITS) for r in (JACK, QUEEN, KING))
CARD_RANKS = np.array([rank_of(card) for card in range(NUM_CARDS)], dtype=np.int64)
CARD_SUITS = np.array([suit_of(card) for card in range(NUM_CARDS)], dtype=np.int64)

# Bit i of a byte, for turning masks into boolean card arrays in one lookup.
_BYTE_BITS = ((np.arange(256)[:, None] >> np.arange(8)[None, :]) & 1).astype(bool)


def mask_array(mask: int) -> np.ndarray:
    """bool [32]: whether each card is in the mask."""

    return _BYTE_BITS[[mask & 0xFF, (mask >> 8) & 0xFF, (mask >> 16) & 0xFF, (mask >> 24) & 0xFF]].reshape(NUM_CARDS)


def format_cards(mask: int) -> str:
    return " ".join(card_name(card) for card in cards_of(mask))
