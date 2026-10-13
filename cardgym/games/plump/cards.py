"""Cards and deck helpers for Plump."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum


class Suit(str, Enum):
    SPADES = "spades"
    HEARTS = "hearts"
    DIAMONDS = "diamonds"
    CLUBS = "clubs"


class Rank(IntEnum):
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14


@dataclass(frozen=True)
class Card:
    """A standard playing card.

    Cards are immutable, hashable, and compare by suit/rank dataclass fields.
    Use ``card_sort_key`` for deterministic game-facing ordering.
    """

    suit: Suit
    rank: Rank

    def __str__(self) -> str:
        return card_str(self)


SUIT_SYMBOLS: dict[Suit, str] = {
    Suit.SPADES: "♠",
    Suit.HEARTS: "♥",
    Suit.DIAMONDS: "♦",
    Suit.CLUBS: "♣",
}

RANK_LABELS: dict[Rank, str] = {
    Rank.TWO: "2",
    Rank.THREE: "3",
    Rank.FOUR: "4",
    Rank.FIVE: "5",
    Rank.SIX: "6",
    Rank.SEVEN: "7",
    Rank.EIGHT: "8",
    Rank.NINE: "9",
    Rank.TEN: "10",
    Rank.JACK: "J",
    Rank.QUEEN: "Q",
    Rank.KING: "K",
    Rank.ACE: "A",
}

SUIT_ORDER: dict[Suit, int] = {
    Suit.SPADES: 0,
    Suit.HEARTS: 1,
    Suit.DIAMONDS: 2,
    Suit.CLUBS: 3,
}


def make_deck() -> list[Card]:
    """Return a standard 52-card deck in deterministic suit/rank order."""

    return [Card(suit, rank) for suit in Suit for rank in Rank]


def card_sort_key(card: Card) -> tuple[int, int]:
    """Sort cards consistently for UI, tests, and ML encoders."""

    return (SUIT_ORDER[card.suit], int(card.rank))


def sort_cards(cards: list[Card] | tuple[Card, ...]) -> list[Card]:
    """Return cards sorted by suit then rank."""

    return sorted(cards, key=card_sort_key)


def card_str(card: Card) -> str:
    """Return a compact human-readable card label, such as ``A♠``."""

    return f"{RANK_LABELS[card.rank]}{SUIT_SYMBOLS[card.suit]}"
