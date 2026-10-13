"""State dataclasses and public API types for the Plump engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union

from .cards import Card, Suit


class Phase(str, Enum):
    NOT_STARTED = "not_started"
    BIDDING = "bidding"
    PLAYING = "playing"
    ROUND_OVER = "round_over"
    GAME_OVER = "game_over"


class EventType(str, Enum):
    ROUND_START = "round_start"
    BID = "bid"
    PLAY = "play"
    TRICK_WIN = "trick_win"
    ROUND_END = "round_end"


class TrumpPolicy(str, Enum):
    REVEAL_NEXT_CARD = "reveal_next_card"
    NONE = "none"


class IllegalActionError(ValueError):
    """Raised when an action violates Plump rules or turn order."""


@dataclass(frozen=True)
class ScoringConfig:
    hit_base_points: int = 10
    zero_bid_success_points: int = 5
    miss_points: int = 0


@dataclass
class GameConfig:
    """Configurable game and variant settings.

    ``deck_order``, ``manual_hands``, and ``manual_trump_suit`` are primarily
    for deterministic tests. They are reused for each reset unless overridden.
    """

    num_players: int = 4
    hand_sizes: list[int] = field(
        default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    )
    forbid_total_bid_equals_hand_size: bool = True
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    trump_policy: TrumpPolicy = TrumpPolicy.NONE
    shuffle: bool = True
    deck_order: Optional[list[Card]] = None
    manual_hands: Optional[dict[int, list[Card]]] = None
    manual_trump_suit: Optional[Suit] = None
    bidding_start_players: Optional[list[int]] = None
    auto_advance_rounds: bool = True


@dataclass(frozen=True)
class Bid:
    player: int
    value: int
    position: int


@dataclass(frozen=True)
class TrickPlay:
    player: int
    card: Card
    position: int


@dataclass
class Trick:
    trick_index: int
    leader: int
    led_suit: Optional[Suit] = None
    plays: list[TrickPlay] = field(default_factory=list)
    winner: Optional[int] = None


@dataclass
class GameEvent:
    type: EventType
    round_index: int
    player: Optional[int] = None
    card: Optional[Card] = None
    bid: Optional[int] = None
    trick_index: Optional[int] = None
    position_in_trick: Optional[int] = None


@dataclass
class RoundState:
    round_index: int
    hand_size: int
    trump_suit: Optional[Suit]
    bidding_start_player: int
    bidding_order: list[int]
    play_start_player: Optional[int] = None
    initial_hands: dict[int, list[Card]] = field(default_factory=dict)
    current_hands: dict[int, list[Card]] = field(default_factory=dict)
    bids: list[Bid] = field(default_factory=list)
    tricks: list[Trick] = field(default_factory=list)
    tricks_won: dict[int, int] = field(default_factory=dict)
    round_scores: dict[int, int] = field(default_factory=dict)
    cumulative_scores_after_round: dict[int, int] = field(default_factory=dict)


@dataclass
class GameState:
    config: GameConfig
    phase: Phase = Phase.NOT_STARTED
    round_index: int = -1
    current_player: Optional[int] = None
    cumulative_scores: dict[int, int] = field(default_factory=dict)
    rounds: list[RoundState] = field(default_factory=list)
    event_log: list[GameEvent] = field(default_factory=list)

    @property
    def current_round(self) -> RoundState:
        if not self.rounds:
            raise RuntimeError("No round has started.")
        return self.rounds[-1]


@dataclass(frozen=True)
class BidAction:
    player: int
    bid: int


@dataclass(frozen=True)
class PlayCardAction:
    player: int
    card: Card


Action = Union[BidAction, PlayCardAction]


@dataclass
class Observation:
    """Player-visible state for GUI, bots, and future ML encoders."""

    player_id: int
    phase: Phase
    round_index: int
    total_rounds: int
    rounds_remaining: int
    hand_size: int
    trump_suit: Optional[Suit]
    current_player: Optional[int]
    bidding_start_player: int
    bidding_order: list[int]
    play_start_player: Optional[int]
    my_hand: list[Card]
    bids: list[Bid]
    tricks_won: dict[int, int]
    current_trick: Optional[Trick]
    completed_tricks: list[Trick]
    played_cards_by_player: dict[int, list[Card]]
    played_cards_total: list[Card]
    voids: dict[int, dict[Suit, bool]]
    legal_bids: list[int]
    legal_cards: list[Card]
    scores: dict[int, int]
    event_log: list[GameEvent]
    hand_size_schedule: list[int]


@dataclass
class StepResult:
    state: GameState
    observation: Optional[Observation]
    rewards: dict[int, int]
    done: bool
    info: dict[str, object] = field(default_factory=dict)
