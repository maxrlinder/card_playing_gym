"""Plump token schema (v6): per-observer event streams and the oracle stream.

An observer's stream is the round from its own seat:

    [GAME] [HAND x N] [TURN] [BID] { [TURN] [BID] } x (P-1)
    { [PLAY x P] [TRICK_WIN] } x N

Every player reference is observer-relative (0 is the observer). The stream
holds exactly what the observer may see: its dealt hand and the public
events. A TURN token ("seat r is about to bid") precedes every bid and is the
position the bid head reads; a play is read off the token before it, whose
NEXT_ACTOR/NEXT_PHASE fields announce who acts next.

Each token is a fixed-width row of categorical fields:

    0 type  1 rel_player  2 rank  3 suit  4 card  5 bid  6 trick
    7 pos_in_trick  8 hand_size  9 num_players  10 next_actor  11 next_phase
    12 bids_made  13 bid_sum  14 hand_size - bid_sum  15 bids_remaining
    16..24 observer's remaining hand (TRICK_WIN rows only)
    25 id of the token's pattern of absent fields

The bidding-state fields (12-15) are populated on GAME, HAND, bid-TURN and
BID rows; a BID row carries the state after that bid. The model embeds the
card as rank + suit only; the exact card id stays in the row for labels.

The critic's oracle stream is the seat-0 stream with the observer's N hand
tokens replaced by all P*N dealt cards, each tagged with its absolute owner,
so its seat ids are absolute and match the critic's value columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .cards import Card, Rank, Suit, sort_cards
from .state import EventType, GameEvent

NUM_CARDS = 52
NUM_SUITS = 4
NUM_RANKS = 13
SUITS: tuple[Suit, ...] = (Suit.SPADES, Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS)
RANKS: tuple[Rank, ...] = tuple(Rank)
IGNORE = -100

# Token types (slot 0). 0 is padding.
TOKEN_PAD = 0
TOKEN_GAME = 1
TOKEN_HAND = 2
TOKEN_BID = 3
TOKEN_PLAY = 4
TOKEN_TRICK_WIN = 5
TOKEN_TURN = 6
NUM_TOKEN_TYPES = 7

# Next-phase ids (slot 11).
NEXT_NONE = 0
NEXT_BID = 1
NEXT_PLAY = 2
NUM_NEXT_PHASES = 3

SLOT_TYPE = 0
SLOT_REL_PLAYER = 1
SLOT_RANK = 2
SLOT_SUIT = 3
SLOT_CARD = 4
SLOT_BID = 5
SLOT_TRICK = 6
SLOT_POS_IN_TRICK = 7
SLOT_HAND_SIZE = 8
SLOT_NUM_PLAYERS = 9
SLOT_NEXT_ACTOR = 10
SLOT_NEXT_PHASE = 11
SLOT_BIDS_MADE = 12
SLOT_BID_SUM = 13
SLOT_BID_PRESSURE = 14
SLOT_BIDS_REMAINING = 15
BASE_WIDTH = 16
REMAINING_HAND_SLOTS = 9
SLOT_REMAINING_HAND = BASE_WIDTH
SLOT_NA_PATTERN = SLOT_REMAINING_HAND + REMAINING_HAND_SLOTS
TOKEN_WIDTH = SLOT_NA_PATTERN + 1

# Every absence mask the schema emits (bit i: base field i holds its NA id).
# Row 0 is reserved for padding tokens and contributes exactly zero.
NA_PATTERN_MASKS = (
    0, 0x00DC, 0x00E0, 0x00FC, 0x0C7E, 0x0CDC,
    0x0CE0, 0xF020, 0xF03C, 0xF0BC, 0xFC20, 0xFCBC,
)
_NA_PATTERN_ID = {mask: index for index, mask in enumerate(NA_PATTERN_MASKS)}
_NA_PATTERN_LOOKUP = np.full(1 << BASE_WIDTH, -1, dtype=np.int8)
for _index, _mask in enumerate(NA_PATTERN_MASKS):
    _NA_PATTERN_LOOKUP[_mask] = _index

_CARD_IDS: dict[Card, int] = {
    Card(suit, rank): suit_index * NUM_RANKS + rank_index
    for suit_index, suit in enumerate(SUITS)
    for rank_index, rank in enumerate(RANKS)
}


def card_id(card: Card) -> int:
    return _CARD_IDS[card]


def card_from_id(index: int) -> Card:
    if not 0 <= index < NUM_CARDS:
        raise ValueError(f"Card id must be in 0..{NUM_CARDS - 1}.")
    return Card(SUITS[index // NUM_RANKS], RANKS[index % NUM_RANKS])


@dataclass(frozen=True)
class Schema:
    """Vocabulary limits; every NA id and table size derives from these two."""

    max_players: int = 5
    max_hand_size: int = 10

    # -- lengths ---------------------------------------------------------
    def seq_len(self, num_players: int, hand_size: int) -> int:
        """GAME + hand + (TURN, BID) per seat + (P plays, TRICK_WIN) per trick."""

        return 1 + hand_size + 2 * num_players + hand_size * (num_players + 1)

    def oracle_seq_len(self, num_players: int, hand_size: int) -> int:
        return self.seq_len(num_players, hand_size) + (num_players - 1) * hand_size

    @staticmethod
    def prefix_len(hand_size: int) -> int:
        """GAME, the hand, and the opening bid's TURN token."""

        return hand_size + 2

    @staticmethod
    def actor_dense_start(hand_size: int) -> int:
        """First stream position after the dealt cards (the opening TURN)."""

        return 1 + hand_size

    @staticmethod
    def oracle_dense_start(num_players: int, hand_size: int) -> int:
        return 1 + num_players * hand_size

    @property
    def max_seq_len(self) -> int:
        return self.seq_len(self.max_players, self.max_hand_size)

    @property
    def oracle_max_seq_len(self) -> int:
        return self.oracle_seq_len(self.max_players, self.max_hand_size)

    @property
    def bid_count(self) -> int:
        return self.max_hand_size + 1

    @property
    def opponents(self) -> int:
        """Seat axis of opponent-only beliefs: relative seats 1..max_players-1."""

        return self.max_players - 1

    # -- NA ids ----------------------------------------------------------
    @property
    def player_na(self) -> int:
        return self.max_players

    @property
    def bid_na(self) -> int:
        return self.max_hand_size + 1

    @property
    def trick_na(self) -> int:
        return self.max_hand_size

    @property
    def bids_made_na(self) -> int:
        return self.max_players + 1

    @property
    def bid_sum_na(self) -> int:
        return self.max_players * self.max_hand_size + 1

    @property
    def bid_pressure_offset(self) -> int:
        return (self.max_players - 1) * self.max_hand_size

    @property
    def base_na_ids(self) -> tuple[int | None, ...]:
        """Sentinel id per base field; None means the field is always present."""

        return (
            None,
            self.player_na,
            NUM_RANKS,
            NUM_SUITS,
            NUM_CARDS,
            self.bid_na,
            self.trick_na,
            self.max_players,
            None,
            None,
            self.player_na,
            NEXT_NONE,
            self.bids_made_na,
            self.bid_sum_na,
            self.bid_sum_na,
            self.bids_made_na,
        )

    @property
    def base_vocab_sizes(self) -> tuple[int, ...]:
        p, n = self.max_players, self.max_hand_size
        return (
            NUM_TOKEN_TYPES, p + 1, NUM_RANKS + 1, NUM_SUITS + 1, NUM_CARDS + 1,
            n + 2, n + 1, p + 1, n + 1, p + 1, p + 1, NUM_NEXT_PHASES,
            p + 2, p * n + 2, p * n + 2, p + 2,
        )


@lru_cache(maxsize=None)
def _template(schema: Schema, num_players: int, hand_size: int) -> tuple[int, ...]:
    token = [0] * TOKEN_WIDTH
    token[SLOT_REL_PLAYER] = schema.player_na
    token[SLOT_RANK] = NUM_RANKS
    token[SLOT_SUIT] = NUM_SUITS
    token[SLOT_CARD] = NUM_CARDS
    token[SLOT_BID] = schema.bid_na
    token[SLOT_TRICK] = schema.trick_na
    token[SLOT_POS_IN_TRICK] = schema.max_players
    token[SLOT_HAND_SIZE] = hand_size
    token[SLOT_NUM_PLAYERS] = num_players
    token[SLOT_NEXT_ACTOR] = schema.player_na
    token[SLOT_NEXT_PHASE] = NEXT_NONE
    token[SLOT_BIDS_MADE] = schema.bids_made_na
    token[SLOT_BID_SUM] = schema.bid_sum_na
    token[SLOT_BID_PRESSURE] = schema.bid_sum_na
    token[SLOT_BIDS_REMAINING] = schema.bids_made_na
    token[SLOT_REMAINING_HAND:SLOT_NA_PATTERN] = [NUM_CARDS] * REMAINING_HAND_SLOTS
    return tuple(token)


def _base(schema: Schema, num_players: int, hand_size: int) -> list[int]:
    return list(_template(schema, num_players, hand_size))


def _relative(player: int, observer: int, num_players: int) -> int:
    return (player - observer) % num_players


def set_na_patterns(tokens: np.ndarray, schema: Schema) -> np.ndarray:
    """Write each row's absent-field pattern id (after all field edits)."""

    flat = tokens.reshape(-1, TOKEN_WIDTH)
    masks = np.zeros(flat.shape[0], dtype=np.uint16)
    for slot, sentinel in enumerate(schema.base_na_ids):
        if sentinel is not None:
            masks |= (flat[:, slot] == sentinel).astype(np.uint16) << slot
    ids = _NA_PATTERN_LOOKUP[masks]
    if np.any(ids < 0):
        invalid = int(masks[np.flatnonzero(ids < 0)[0]])
        raise ValueError(f"Unsupported token NA pattern 0x{invalid:04x}.")
    flat[:, SLOT_NA_PATTERN] = ids
    return tokens


def set_bid_state(token, schema: Schema, num_players: int, hand_size: int, made: int, total: int) -> None:
    if not 0 <= made <= num_players:
        raise ValueError("bids_made is outside the active-player range.")
    if not 0 <= total <= num_players * hand_size:
        raise ValueError("bid_sum is outside the possible round range.")
    token[SLOT_BIDS_MADE] = made
    token[SLOT_BID_SUM] = total
    token[SLOT_BID_PRESSURE] = hand_size - total + schema.bid_pressure_offset
    token[SLOT_BIDS_REMAINING] = num_players - made


def set_remaining_hand(token, cards: list[Card], hand_size: int) -> None:
    """Write the observer's sorted remaining hand into a TRICK_WIN row."""

    if len(cards) > min(hand_size, REMAINING_HAND_SLOTS):
        raise ValueError("Remaining hand exceeds the token's card capacity.")
    values = [card_id(card) for card in sort_cards(cards)]
    token[SLOT_REMAINING_HAND:SLOT_NA_PATTERN] = values + [NUM_CARDS] * (REMAINING_HAND_SLOTS - len(values))


def game_token(schema: Schema, num_players: int, hand_size: int, bidding_position: int) -> list[int]:
    token = _base(schema, num_players, hand_size)
    token[SLOT_TYPE] = TOKEN_GAME
    token[SLOT_POS_IN_TRICK] = bidding_position
    return token


def hand_token(schema: Schema, num_players: int, hand_size: int, card: Card, owner: int = 0) -> list[int]:
    token = _base(schema, num_players, hand_size)
    token[SLOT_TYPE] = TOKEN_HAND
    token[SLOT_REL_PLAYER] = owner
    token[SLOT_RANK] = RANKS.index(card.rank)
    token[SLOT_SUIT] = SUITS.index(card.suit)
    token[SLOT_CARD] = card_id(card)
    return token


def turn_token(
    schema: Schema, num_players: int, hand_size: int, rel_actor: int, made: int, total: int
) -> list[int]:
    """"Seat ``rel_actor`` is about to bid": a near-empty readout position."""

    token = _base(schema, num_players, hand_size)
    token[SLOT_TYPE] = TOKEN_TURN
    token[SLOT_REL_PLAYER] = rel_actor
    token[SLOT_NEXT_ACTOR] = rel_actor
    token[SLOT_NEXT_PHASE] = NEXT_BID
    set_bid_state(token, schema, num_players, hand_size, made, total)
    return token


def event_token(
    schema: Schema,
    event: GameEvent,
    observer: int,
    num_players: int,
    hand_size: int,
    *,
    made: int | None = None,
    total: int | None = None,
    remaining_hand: list[Card] | None = None,
) -> list[int] | None:
    """One public event as seen by ``observer``; None for non-token events."""

    if event.type == EventType.BID:
        token = _base(schema, num_players, hand_size)
        token[SLOT_TYPE] = TOKEN_BID
        token[SLOT_REL_PLAYER] = _relative(event.player, observer, num_players)
        token[SLOT_BID] = event.bid
        if made is None or total is None:
            raise ValueError("BID tokens carry the updated bidding state.")
        set_bid_state(token, schema, num_players, hand_size, made, total)
        return token
    if event.type == EventType.PLAY:
        token = _base(schema, num_players, hand_size)
        token[SLOT_TYPE] = TOKEN_PLAY
        token[SLOT_REL_PLAYER] = _relative(event.player, observer, num_players)
        token[SLOT_RANK] = RANKS.index(event.card.rank)
        token[SLOT_SUIT] = SUITS.index(event.card.suit)
        token[SLOT_CARD] = card_id(event.card)
        token[SLOT_TRICK] = event.trick_index
        token[SLOT_POS_IN_TRICK] = event.position_in_trick
        return token
    if event.type == EventType.TRICK_WIN:
        token = _base(schema, num_players, hand_size)
        token[SLOT_TYPE] = TOKEN_TRICK_WIN
        token[SLOT_REL_PLAYER] = _relative(event.player, observer, num_players)
        token[SLOT_TRICK] = event.trick_index
        if remaining_hand is not None:
            set_remaining_hand(token, remaining_hand, hand_size)
        return token
    return None


def prefix_tokens(
    schema: Schema,
    observer: int,
    num_players: int,
    hand_size: int,
    hand: list[Card],
    bidding_start: int,
) -> np.ndarray:
    """[GAME] [HAND x N] [TURN] for one observer: everything before the first bid."""

    if len(hand) != hand_size:
        raise ValueError("Observer hand does not match the hand size.")
    rows = [game_token(schema, num_players, hand_size, _relative(observer, bidding_start, num_players))]
    rows.extend(hand_token(schema, num_players, hand_size, card) for card in sort_cards(hand))
    for row in rows:
        set_bid_state(row, schema, num_players, hand_size, 0, 0)
    rows.append(
        turn_token(schema, num_players, hand_size, _relative(bidding_start, observer, num_players), 0, 0)
    )
    return set_na_patterns(np.asarray(rows, dtype=np.int64), schema)


def round_events(events: list[GameEvent]) -> list[GameEvent]:
    return [e for e in events if e.type in (EventType.BID, EventType.PLAY, EventType.TRICK_WIN)]


def event_rows(
    schema: Schema,
    events: list[GameEvent],
    observer: int,
    num_players: int,
    hand_size: int,
    *,
    bids_before: list[int],
    hand_before: list[Card],
    turn_before_bids: bool,
) -> list[list[int]]:
    """Token rows for consecutive events, given the state before the first.

    ``turn_before_bids`` inserts each bid's TURN token in front of it (a
    from-scratch stream, whose opening TURN already sits in the prefix). A
    self-play wave instead ends with the next bidder's TURN; see
    :func:`wave_rows`.
    """

    rows: list[list[int]] = []
    bids = list(bids_before)
    hand = list(hand_before)
    for event in events:
        if event.type == EventType.BID:
            if turn_before_bids and bids:
                rows.append(
                    turn_token(
                        schema, num_players, hand_size,
                        _relative(event.player, observer, num_players), len(bids), sum(bids),
                    )
                )
            bids.append(event.bid)
            rows.append(
                event_token(schema, event, observer, num_players, hand_size, made=len(bids), total=sum(bids))
            )
        elif event.type == EventType.PLAY:
            if event.player == observer:
                hand.remove(event.card)
            rows.append(event_token(schema, event, observer, num_players, hand_size))
        elif event.type == EventType.TRICK_WIN:
            rows.append(event_token(schema, event, observer, num_players, hand_size, remaining_hand=hand))
    return rows


def wave_rows(
    schema: Schema,
    events: list[GameEvent],
    num_players: int,
    hand_size: int,
    *,
    bids: list[int],
    hands: dict[int, list[Card]],
    next_actor: int | None,
    next_phase: int,
) -> list[np.ndarray]:
    """Every observer's rows for one self-play step.

    ``events`` happened in this step; ``bids`` and ``hands`` are the state
    after them. The block ends at the next decision's readout: the next
    bidder's TURN token, or the last event row annotated with the next player
    (``next_actor`` None when the round is over). Rows are built once from
    seat 0 and re-relativized per observer; only player fields and the
    TRICK_WIN remaining hands differ between observers.
    """

    bid_state = bids[: len(bids) - sum(1 for e in events if e.type == EventType.BID)]
    rows: list[list[int]] = []
    for event in events:
        if event.type == EventType.BID:
            bid_state.append(event.bid)
            rows.append(
                event_token(schema, event, 0, num_players, hand_size, made=len(bid_state), total=sum(bid_state))
            )
        else:
            row = event_token(schema, event, 0, num_players, hand_size)
            if row is not None:
                rows.append(row)
    if next_actor is not None and next_phase == NEXT_BID:
        rows.append(turn_token(schema, num_players, hand_size, next_actor, len(bids), sum(bids)))
    annotate_next(rows, schema)
    if next_actor is not None:
        rows[-1][SLOT_NEXT_ACTOR] = next_actor
        rows[-1][SLOT_NEXT_PHASE] = next_phase
    base = set_na_patterns(np.asarray(rows, dtype=np.int64), schema)
    # Hands at each TRICK_WIN: the final step can close two tricks (the last
    # decision's trick and the forced final one), so replay this step's plays.
    held = {seat: list(cards) for seat, cards in hands.items()}
    for event in events:
        if event.type == EventType.PLAY:
            held[event.player].append(event.card)
    hands_at_trick_win = []
    for event in events:
        if event.type == EventType.PLAY:
            held[event.player].remove(event.card)
        elif event.type == EventType.TRICK_WIN:
            hands_at_trick_win.append({seat: list(cards) for seat, cards in held.items()})
    trick_wins = np.flatnonzero(base[:, SLOT_TYPE] == TOKEN_TRICK_WIN)
    has_player = base[:, SLOT_REL_PLAYER] != schema.player_na
    has_next = base[:, SLOT_NEXT_ACTOR] != schema.player_na
    blocks = []
    for observer in range(num_players):
        block = base.copy()
        block[has_player, SLOT_REL_PLAYER] = (base[has_player, SLOT_REL_PLAYER] - observer) % num_players
        block[has_next, SLOT_NEXT_ACTOR] = (base[has_next, SLOT_NEXT_ACTOR] - observer) % num_players
        for row, held_then in zip(trick_wins, hands_at_trick_win):
            set_remaining_hand(block[row], held_then[observer], hand_size)
        blocks.append(block)
    return blocks


def annotate_next(rows: list[list[int]], schema: Schema) -> None:
    """Row t announces the actor/phase of the action token at t+1, if any.

    A TURN row already carries its own announcement and is left alone;
    a row followed by a TURN keeps NA (the TURN is the readout).
    """

    for position in range(len(rows) - 1):
        upcoming = rows[position + 1]
        if upcoming[SLOT_TYPE] in (TOKEN_BID, TOKEN_PLAY):
            rows[position][SLOT_NEXT_ACTOR] = upcoming[SLOT_REL_PLAYER]
            rows[position][SLOT_NEXT_PHASE] = NEXT_BID if upcoming[SLOT_TYPE] == TOKEN_BID else NEXT_PLAY
        elif upcoming[SLOT_TYPE] != TOKEN_TURN and rows[position][SLOT_TYPE] != TOKEN_TURN:
            rows[position][SLOT_NEXT_ACTOR] = schema.player_na
            rows[position][SLOT_NEXT_PHASE] = NEXT_NONE


def seat_tokens(
    schema: Schema,
    events: list[GameEvent],
    observer: int,
    num_players: int,
    hand_size: int,
    hand: list[Card],
    bidding_start: int,
    *,
    pending_actor: int | None = None,
    pending_phase: int = NEXT_NONE,
) -> np.ndarray:
    """One observer's stream from scratch (reference builder; used by GUI/eval).

    ``hand`` is the observer's dealt hand. With ``pending_actor`` the stream
    ends at the readout for that actor's ``pending_phase`` decision (a TURN
    token for a bid, or NEXT annotation on the last row for a play).
    """

    prefix = [list(row) for row in prefix_tokens(schema, observer, num_players, hand_size, hand, bidding_start)]
    body = event_rows(
        schema,
        round_events(events),
        observer,
        num_players,
        hand_size,
        bids_before=[],
        hand_before=list(hand),
        turn_before_bids=True,
    )
    # The opening bid's TURN is the last prefix row; annotate from there.
    rows = prefix + body
    annotate_next(rows[len(prefix) - 1 :], schema)
    if pending_actor is not None:
        bids = [e.bid for e in round_events(events) if e.type == EventType.BID]
        rel = _relative(pending_actor, observer, num_players)
        if pending_phase == NEXT_BID:
            if body:
                rows.append(turn_token(schema, num_players, hand_size, rel, len(bids), sum(bids)))
        else:
            rows[-1][SLOT_NEXT_ACTOR] = rel
            rows[-1][SLOT_NEXT_PHASE] = pending_phase
    return set_na_patterns(np.asarray(rows, dtype=np.int64), schema)


def oracle_tokens(
    schema: Schema,
    seat_zero_stream: np.ndarray,
    num_players: int,
    hand_size: int,
    initial_hands: dict[int, list[Card]],
) -> np.ndarray:
    """The perfect-information critic stream for one complete round."""

    if set(initial_hands) != set(range(num_players)):
        raise ValueError("initial_hands must cover every seat.")
    cards = []
    for owner in range(num_players):
        if len(initial_hands[owner]) != hand_size:
            raise ValueError(f"Player {owner} hand does not match the hand size.")
        for card in sort_cards(initial_hands[owner]):
            row = hand_token(schema, num_players, hand_size, card, owner=owner)
            set_bid_state(row, schema, num_players, hand_size, 0, 0)
            cards.append(row)
    oracle = np.concatenate(
        (
            seat_zero_stream[:1],
            np.asarray(cards, dtype=np.int64),
            seat_zero_stream[1 + hand_size :],
        ),
        axis=0,
    )
    # TRICK_WIN rows repeat seat 0's remaining hand; redundant here and it
    # would privilege seat 0, so the oracle derives hands from the cards.
    oracle[:, SLOT_REMAINING_HAND:SLOT_NA_PATTERN] = NUM_CARDS
    set_na_patterns(oracle, schema)
    expected = (schema.oracle_seq_len(num_players, hand_size), TOKEN_WIDTH)
    if oracle.shape != expected:
        raise AssertionError(f"Oracle stream has shape {oracle.shape}, expected {expected}.")
    return oracle
