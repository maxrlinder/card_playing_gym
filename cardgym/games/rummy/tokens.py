"""Rummy token streams: one token per game event, carrying the viewer's whole ledger.

Every decision (draw, meld, lay-off, discard) is one event, and every event is
one token in every stream, so a game's streams all have ``1 + decisions``
tokens and the token an action produces is the readout of the next decision.

A token is ``NUM_FIELDS`` categorical event fields followed by the viewer's
ledger: 52 codes, one per card type, each naming the locations of both copies
plus two flags (moved by this event, can be laid off on the table). Actor
streams are observer-relative (seat 0 is the observer, locations 1 + j are
seat j's publicly known holdings, everything else is "unseen"). The critic's
oracle stream uses absolute seats, sees every hand and the stock's next card.
"""

from __future__ import annotations

import numpy as np

from .cards import COPIES, NO_CARD, NUM_CARDS
from .engine import (
    EVENT_DEAL,
    EVENT_STOCK,
    MAX_PLAYERS,
    NUM_LEDGER_CODES,
    PLAY,
    Event,
    RummyEngine,
)
from .melds import MELD_CARDS, MELD_KIND

F_TYPE = 0
F_ACTOR = 1  # acting seat; NO_SEAT on the deal
F_CARD = 2  # the event's card when the viewer may see it
F_MELD_KIND = 3  # meld laid or extended: group, run, or none
F_MELD_SIZE = 4  # its size after the event (0 = none)
F_NEXT_ACTOR = 5
F_NEXT_PHASE = 6  # draw, play, or end
F_STOCK = 7
F_HAND = 8  # hand sizes of seats 0..3 (four fields)
F_OPENED = 12  # bitmask of seats past their opening
F_PROGRESS = 13  # opening points laid this turn by a seat not yet open
F_PLAYERS = 14
F_ROUND = 15  # completed turns // players
F_STOCK_TOP = 16  # oracle only: the next stock card
NUM_FIELDS = 17
LEDGER_START = NUM_FIELDS
WIDTH = NUM_FIELDS + NUM_CARDS

NO_SEAT = MAX_PLAYERS
NO_MELD_KIND = 2
PHASE_END = 2
ABSENT_SEAT = 31  # hand-size value for seats beyond the table
MAX_ROUND = 63

FIELD_VOCAB = (
    7,  # type: padding, deal, stock draw, discard take, meld, lay-off, discard
    MAX_PLAYERS + 1,
    NUM_CARDS + 1,
    3,
    14,
    MAX_PLAYERS + 1,
    3,
    104,
    32,
    32,
    32,
    32,
    1 << MAX_PLAYERS,
    30,
    MAX_PLAYERS + 1,
    MAX_ROUND + 1,
    NUM_CARDS + 1,
)
assert len(FIELD_VOCAB) == NUM_FIELDS
LEDGER_VOCAB = NUM_LEDGER_CODES


def max_stream_length(max_turns: int) -> int:
    """Deal + a draw and a discard per turn + at most one meld or lay-off per tabled card."""

    return 1 + 2 * max_turns + COPIES * NUM_CARDS


class StreamWriter:
    """Builds every viewer's token for each engine event (seats 0..P-1, then the oracle)."""

    def __init__(self, engine: RummyEngine) -> None:
        self.engine = engine
        players = engine.num_players
        self.players = players
        seats = np.arange(players)
        # relative[o, s]: seat s as seen by observer o; the oracle keeps absolute seats.
        self.relative = np.vstack([(seats[None, :] - seats[:, None]) % players, seats[None, :]])

    def deal(self) -> np.ndarray:
        rows = self._rows(None)
        rows[:, F_TYPE] = EVENT_DEAL
        return rows

    def event(self, event: Event) -> np.ndarray:
        rows = self._rows(event)
        rows[:, F_TYPE] = event.kind
        rows[:, F_ACTOR] = self.relative[:, event.actor]
        if event.kind == EVENT_STOCK:
            # A stock draw is visible to the drawer and the oracle only.
            rows[:, F_CARD] = NO_CARD
            rows[event.actor, F_CARD] = event.card
            rows[self.players, F_CARD] = event.card
        else:
            rows[:, F_CARD] = event.card
        if event.meld >= 0:
            rows[:, F_MELD_KIND] = MELD_KIND[event.meld]
            rows[:, F_MELD_SIZE] = len(MELD_CARDS[event.meld])
        return rows

    def _rows(self, event: Event | None) -> np.ndarray:
        engine = self.engine
        players = self.players
        rows = np.zeros((players + 1, WIDTH), dtype=np.int64)
        rows[:, F_ACTOR] = NO_SEAT
        rows[:, F_CARD] = NO_CARD
        rows[:, F_MELD_KIND] = NO_MELD_KIND
        if engine.done:
            rows[:, F_NEXT_ACTOR] = NO_SEAT
            rows[:, F_NEXT_PHASE] = PHASE_END
        else:
            rows[:, F_NEXT_ACTOR] = self.relative[:, engine.current]
            rows[:, F_NEXT_PHASE] = int(engine.phase == PLAY)
        rows[:, F_STOCK] = len(engine.stock)
        sizes = engine.hands.sum(axis=1)
        hand = np.full((players + 1, MAX_PLAYERS), ABSENT_SEAT, dtype=np.int64)
        # Column j of a row is the seat the viewer calls j.
        for view in range(players + 1):
            hand[view, self.relative[view]] = sizes
        rows[:, F_HAND : F_HAND + MAX_PLAYERS] = hand
        opened = np.array(engine.opened, dtype=np.int64)
        rows[:, F_OPENED] = (opened[None, :] << self.relative).sum(axis=1)
        rows[:, F_PROGRESS] = engine.turn_points
        rows[:, F_PLAYERS] = players
        rows[:, F_ROUND] = min(engine.turns // players, MAX_ROUND)
        rows[:, F_STOCK_TOP] = NO_CARD
        rows[players, F_STOCK_TOP] = engine.stock[-1] if engine.stock else NO_CARD
        rows[:, LEDGER_START:] = engine.ledgers()
        return rows
