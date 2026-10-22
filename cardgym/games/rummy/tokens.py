"""Rummy token streams: the seat's own decisions and every opponent turn, in table order.

A seat's stream has two kinds of token, each a row of categorical fields plus
52 per-card codes (one per card type):

* a **decision** token right before each decision the seat makes (draw, meld,
  lay-off, discard): the observation it decides from, and the position its
  policy reads. Its card codes are the explicit state: where the copies of
  each card type are (unseen, the seat's hand, the copies each opponent is
  known to hold, the table split by who put each copy there, the discard
  pile by depth, or not in a one-deck game) and whether a copy can be laid
  off. Its fields add the phase, the seat's own previous action, a card it
  must still use (taken from below the discard top), and opening points laid
  this turn;
* a **turn** token for every completed opponent turn: who played, whether it
  drew from the stock, took the discard top, or dug deeper (the deepest card
  taken and how many cards), what it discarded, how many melds and lay-offs
  it made, and per card type what the turn did with it (taken, melded, laid
  off, discarded). The order of turns is the stream's order; within a turn
  the order is the rules' (draw, melds and lay-offs, discard).

Both kinds carry the public counters: stock and pile sizes, every seat's hand
size, who has opened, players, decks, round. Every stream is observer-relative (seat 0 is the
observer). The critic reads a per-seat oracle stream with the same positions
and layout: hand locations are whole hands, "unseen" is the stock, stock
draws show their card, and one more field shows the next stock card.

So the stream is the round as the seat saw it, one opponent turn at a time;
decision tokens restate the present exactly, and turn tokens carry the order
and attribution of everything opponents did (the "tells").
"""

from __future__ import annotations

import math

import numpy as np

from .cards import COPIES, MAX_DECKS, NO_CARD, NUM_CARDS
from .engine import (
    EVENT_DISCARD,
    EVENT_LAYOFF,
    EVENT_MELD,
    EVENT_STOCK,
    EVENT_TAKE,
    MAX_PLAYERS,
    NUM_PAIRS,
    NUM_STATE_CODES,
    PLAY,
    ROTATE,
    Event,
    RummyEngine,
)
from .melds import MELD_CARDS, MELD_KIND

MELD_SIZES = [len(cards) for cards in MELD_CARDS]

# Token kinds.
KIND_DECISION, KIND_TURN = 0, 1

F_KIND = 0
# Decision tokens (not-applicable values on turn tokens).
F_PHASE = 1  # draw (0), play (1), or a turn token (2)
F_LAST_KIND = 2  # the viewer's previous event kind (0: none yet)
F_LAST_CARD = 3  # its card: drawn, taken, laid off, or discarded
F_LAST_MELD_KIND = 4  # meld it laid or extended: group, run, or none
F_LAST_MELD_SIZE = 5  # that meld's size afterwards (0 = none)
F_PROGRESS = 6  # opening points the viewer has laid this turn while unopened
F_OBLIGATION = 7  # a card the viewer took from below the top and must still use
# Turn tokens (not-applicable values on decision tokens).
F_ACTOR = 8  # the opponent who played the turn, 1..3 (0 = none)
F_DRAW = 9  # none, stock, the discard top, deeper
F_TAKEN = 10  # the deepest card it took (oracle: any card it drew)
F_TAKE_COUNT = 11  # cards it took from the pile
F_DISCARD = 12  # the card it discarded
F_MELDS = 13  # melds it laid
F_LAYOFFS = 14  # lay-offs it made
# Both kinds: the public counters after the token's moment.
F_STOCK = 15
F_PILE = 16
F_HAND = 17  # hand sizes of seats 0..3 (four fields)
F_OPENED = 21  # bitmask of seats past their opening
F_PLAYERS = 22
F_DECKS = 23
F_ROUND = 24  # completed turns // players
F_STOCK_TOP = 25  # oracle only: the next stock card
NUM_FIELDS = 26
CARD_START = NUM_FIELDS
WIDTH = NUM_FIELDS + NUM_CARDS

NO_MELD_KIND = 2
PHASE_TURN = 2
DRAW_NONE, DRAW_STOCK, DRAW_TOP, DRAW_DEEP = range(4)
ABSENT_SEAT = 63  # hand-size value for seats beyond the table
MAX_COUNT = 62  # hand, pile, and take sizes are capped here
MAX_ROUND = 63
MAX_MELDS = 7
MAX_LAYOFFS = 15

FIELD_VOCAB = (
    2,
    3,
    7,
    NUM_CARDS + 1,
    3,
    14,
    30,
    NUM_CARDS + 1,
    MAX_PLAYERS,
    4,
    NUM_CARDS + 1,
    MAX_COUNT + 1,
    NUM_CARDS + 1,
    MAX_MELDS + 1,
    MAX_LAYOFFS + 1,
    COPIES * NUM_CARDS + 1,
    MAX_COUNT + 1,
    *(ABSENT_SEAT + 1,) * MAX_PLAYERS,
    1 << MAX_PLAYERS,
    MAX_PLAYERS + 1,
    MAX_DECKS + 1,
    MAX_ROUND + 1,
    NUM_CARDS + 1,
)
assert len(FIELD_VOCAB) == NUM_FIELDS

# Card codes. Decision tokens: the state code, pair + NUM_PAIRS * layable.
# Turn tokens: EVENT_BASE + what the turn did with the card type, mixed radix
# over (copies taken from the pile 0-2, drawn from the stock 0-1 (oracle
# only), melded copies 0-2, laid-off copies 0-2, discarded 0-1).
EVENT_BASE = NUM_STATE_CODES
EVENT_RADIX = (COPIES + 1, 2, COPIES + 1, COPIES + 1, 2)
EVENT_PLACE = tuple(int(np.prod(EVENT_RADIX[:i])) for i in range(len(EVENT_RADIX)))
NUM_EVENT_CODES = int(np.prod(EVENT_RADIX))
CARD_VOCAB = NUM_STATE_CODES + NUM_EVENT_CODES
TAKEN, STOCKED, MELDED, LAID, DISCARDED = range(5)


def event_codes(counts: np.ndarray) -> np.ndarray:
    """Per-card turn codes from counts [5, 52] (taken, stocked, melded, laid off, discarded)."""

    clipped = np.minimum(counts, np.asarray(EVENT_RADIX)[:, None] - 1)
    return EVENT_BASE + np.asarray(EVENT_PLACE) @ clipped


def max_stream_length(max_turns: int) -> int:
    """A seat's tokens: its decisions and every opponent turn.

    Decisions are a draw and a discard per own turn plus one meld or lay-off
    per tabled card, bounded for the seat that moves most often (two players).
    """

    return 2 * math.ceil(max_turns / 2) + COPIES * NUM_CARDS + max_turns


class ObservationWriter:
    """Writes each seat's decision tokens and the turn tokens of its opponents.

    Call :meth:`record` after every engine event (it returns the turn tokens
    when the event ended a turn) and :meth:`observe` when a seat is about to
    decide. Views 0..P-1 are the seats' public knowledge, view P the oracle
    (absolute seats; rotated per observer when written).
    """

    def __init__(self, engine: RummyEngine) -> None:
        self.engine = engine
        self.players = engine.num_players
        self.last: list[Event | None] = [None] * self.players
        self._new_turn()

    def _new_turn(self) -> None:
        self.turn_counts = np.zeros((len(EVENT_RADIX), NUM_CARDS), dtype=np.int64)
        self.turn_draw = DRAW_NONE
        self.turn_drawn = NO_CARD
        self.turn_taken = 0
        self.turn_melds = 0
        self.turn_layoffs = 0

    def _counters(self, seat: int) -> list[int]:
        engine = self.engine
        players = self.players
        sizes = np.minimum(engine.hands.sum(axis=1), MAX_COUNT).tolist()
        hand = [sizes[(seat + j) % players] if j < players else ABSENT_SEAT for j in range(MAX_PLAYERS)]
        opened = sum(1 << j for j in range(players) if engine.opened[(seat + j) % players])
        return [
            len(engine.stock), min(len(engine.pile), MAX_COUNT), *hand, opened, players, engine.decks,
            min(engine.turns // players, MAX_ROUND),
        ]

    def record(self, event: Event) -> np.ndarray | None:
        """Note ``event``; when it ends a turn, return every seat's turn token [P, 2, WIDTH] int16.

        Row ``[seat, 0]`` is the seat's actor token and ``[seat, 1]`` its oracle
        token; the acting seat's own rows are unused (its stream has no turn
        tokens of its own).
        """

        actor = event.actor
        self.last[actor] = event
        counts = self.turn_counts
        if event.kind == EVENT_STOCK:
            self.turn_draw, self.turn_drawn = DRAW_STOCK, event.card
            counts[STOCKED, event.card] += 1
        elif event.kind == EVENT_TAKE:
            self.turn_draw = DRAW_TOP if len(event.cards) == 1 else DRAW_DEEP
            self.turn_drawn, self.turn_taken = event.card, len(event.cards)
            for card in event.cards:
                counts[TAKEN, card] += 1
        elif event.kind == EVENT_MELD:
            self.turn_melds += 1
            for card in event.cards:
                counts[MELDED, card] += 1
        elif event.kind == EVENT_LAYOFF:
            self.turn_layoffs += 1
            counts[LAID, event.card] += 1
        if event.kind != EVENT_DISCARD:
            return None
        counts[DISCARDED, event.card] += 1
        engine = self.engine
        oracle_codes = event_codes(counts)
        public = counts.copy()
        public[STOCKED] = 0
        actor_codes = event_codes(public)
        taken = NO_CARD if self.turn_draw == DRAW_STOCK else self.turn_drawn
        stock_top = engine.stock[-1] if engine.stock else NO_CARD
        rows = np.zeros((self.players, 2, WIDTH), dtype=np.int64)
        for seat in range(self.players):
            if seat == actor:
                continue
            head = [KIND_TURN, PHASE_TURN, 0, NO_CARD, NO_MELD_KIND, 0, 0, NO_CARD,
                    (actor - seat) % self.players, self.turn_draw]
            turn = [
                min(self.turn_taken, MAX_COUNT), event.card, min(self.turn_melds, MAX_MELDS),
                min(self.turn_layoffs, MAX_LAYOFFS),
            ]
            tail = self._counters(seat)
            rows[seat, 0, :NUM_FIELDS] = head + [taken] + turn + tail + [NO_CARD]
            rows[seat, 1, :NUM_FIELDS] = head + [self.turn_drawn] + turn + tail + [stock_top]
            rows[seat, 0, CARD_START:] = actor_codes
            rows[seat, 1, CARD_START:] = oracle_codes
        self._new_turn()
        return rows.astype(np.int16)

    def observe(self, seat: int) -> tuple[np.ndarray, np.ndarray]:
        """Seat ``seat``'s decision token and its oracle token now; both [WIDTH] int16."""

        engine = self.engine
        players = self.players
        phase = int(engine.phase == PLAY)
        last = self.last[seat]
        if last is None:
            head = [KIND_DECISION, phase, 0, NO_CARD, NO_MELD_KIND, 0]
        elif last.meld >= 0:
            head = [KIND_DECISION, phase, last.kind, last.card, MELD_KIND[last.meld], MELD_SIZES[last.meld]]
        else:
            head = [KIND_DECISION, phase, last.kind, last.card, NO_MELD_KIND, 0]
        head += [min(engine.turn_points, FIELD_VOCAB[F_PROGRESS] - 1), engine.obligation]
        head += [0, DRAW_NONE, NO_CARD, 0, NO_CARD, 0, 0]
        tail = self._counters(seat)
        stock_top = engine.stock[-1] if engine.stock else NO_CARD
        rows = np.empty((2, WIDTH), dtype=np.int64)
        rows[0, :NUM_FIELDS] = head + tail + [NO_CARD]
        rows[1, :NUM_FIELDS] = head + tail + [stock_top]
        pairs = engine.view_pairs(seat)
        pairs[1] = ROTATE[players][seat][pairs[1]]
        rows[:, CARD_START:] = pairs + NUM_PAIRS * engine.layable
        tokens = rows.astype(np.int16)
        return tokens[0], tokens[1]
