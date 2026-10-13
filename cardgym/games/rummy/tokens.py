"""Rummy token streams: one token per own decision, carrying the viewer's whole ledger.

A seat's stream holds exactly one token per decision that seat makes (draw,
meld, lay-off, discard): the observation it decides from. Opponents' turns
never become tokens of their own in it. A token states the present and
summarizes everything since the seat's previous token:

* the present: decision phase, stock and hand sizes, who has opened, opening
  points laid this turn, and the ledger: 52 codes, one per card type, naming
  the locations of both copies, whether a copy can be laid off on the table,
  and who moved the card type since the previous token (nobody, the viewer,
  or an opponent);
* the summary: the viewer's own previous action, and per opponent whether it
  drew from the stock or took the discard top since then (and which card it
  took), and what it discarded.

Every stream is observer-relative (seat 0 is the observer; ledger location
``1 + j`` holds seat ``j``'s publicly known cards, everything else is
"unseen"). The critic reads a per-seat oracle stream with the same positions
and layout: location ``1 + j`` is seat ``j``'s whole hand, "unseen" is the
stock, stock draws show their card, and one more field shows the next stock
card.

A stream therefore has as many tokens as its seat has decisions, every token
is a readout, and the actor and critic positions of a decision coincide.
"""

from __future__ import annotations

import math

import numpy as np

from .cards import COPIES, NO_CARD, NUM_CARDS
from .engine import (
    EVENT_DISCARD,
    EVENT_STOCK,
    EVENT_TAKE,
    MAX_PLAYERS,
    NUM_LEDGER_CODES,
    NUM_MOVERS,
    NUM_PAIRS,
    PLAY,
    ROTATE,
    Event,
    RummyEngine,
)
from .melds import MELD_CARDS, MELD_KIND

F_PHASE = 0  # the decision: draw (0) or play (1)
F_LAST_KIND = 1  # the viewer's previous event kind (0: none yet)
F_LAST_CARD = 2  # its card: drawn, taken, laid off, or discarded
F_LAST_MELD_KIND = 3  # meld it laid or extended: group, run, or none
F_LAST_MELD_SIZE = 4  # that meld's size afterwards (0 = none)
F_OPP_DRAW = 5  # per opponent 1..3 since the previous token: none, stock, take, absent
F_OPP_TAKE = 8  # per opponent: the card it took (oracle: any card it drew)
F_OPP_DISCARD = 11  # per opponent: the card it discarded
F_STOCK = 14
F_HAND = 15  # hand sizes of seats 0..3 (four fields)
F_OPENED = 19  # bitmask of seats past their opening
F_PROGRESS = 20  # opening points the viewer has laid this turn while unopened
F_PLAYERS = 21
F_ROUND = 22  # completed turns // players
F_STOCK_TOP = 23  # oracle only: the next stock card
NUM_FIELDS = 24
LEDGER_START = NUM_FIELDS
WIDTH = NUM_FIELDS + NUM_CARDS

OPPONENTS = MAX_PLAYERS - 1
NO_MELD_KIND = 2
DRAW_NONE, DRAW_STOCK, DRAW_TAKE, DRAW_ABSENT = range(4)
ABSENT_SEAT = 31  # hand-size value for seats beyond the table
MAX_ROUND = 63

FIELD_VOCAB = (
    2,
    7,
    NUM_CARDS + 1,
    3,
    14,
    *(4,) * OPPONENTS,
    *(NUM_CARDS + 1,) * OPPONENTS,
    *(NUM_CARDS + 1,) * OPPONENTS,
    104,
    *(32,) * MAX_PLAYERS,
    1 << MAX_PLAYERS,
    30,
    MAX_PLAYERS + 1,
    MAX_ROUND + 1,
    NUM_CARDS + 1,
)
assert len(FIELD_VOCAB) == NUM_FIELDS
LEDGER_VOCAB = NUM_LEDGER_CODES


def max_stream_length(max_turns: int) -> int:
    """A seat's decisions: a draw and a discard per own turn, plus one meld or lay-off per tabled card.

    The bound is for the seat that moves most often (two players).
    """

    return 2 * math.ceil(max_turns / 2) + COPIES * NUM_CARDS


class ObservationWriter:
    """Tracks what each viewer has seen change; writes a seat's tokens at its decisions.

    Call :meth:`record` after every engine event and :meth:`observe` when a
    seat is about to decide. Views 0..P-1 are the seats' public knowledge,
    view P the oracle (absolute seats; rotated per observer when written).
    """

    def __init__(self, engine: RummyEngine) -> None:
        self.engine = engine
        players = engine.num_players
        self.players = players
        self.pairs = engine.view_pairs()  # [P + 1, 52] now
        # Absolute seat of the latest event that changed each card's pair, per view.
        self.mover = np.zeros((players + 1, NUM_CARDS), dtype=np.int64)
        # Pairs as each seat saw them at its previous token (its own view, the oracle's).
        self.reference = self.pairs[:players].copy()
        self.oracle_reference = np.repeat(self.pairs[players][None], players, axis=0)
        self.events = 0
        self.token_event = np.zeros(players, dtype=np.int64)  # events seen at the previous token
        self.drawn = [(0, DRAW_NONE, NO_CARD)] * players  # per seat: (event number, kind, card)
        self.discarded = [(0, NO_CARD)] * players
        self.last: list[Event | None] = [None] * players
        seats = np.arange(players)
        # relative[o, s]: seat s as seen by observer o.
        self.relative = (seats[None, :] - seats[:, None]) % players

    def record(self, event: Event) -> None:
        pairs = self.engine.view_pairs()
        self.mover[pairs != self.pairs] = event.actor
        self.pairs = pairs
        self.events += 1
        actor = event.actor
        if event.kind in (EVENT_STOCK, EVENT_TAKE):
            kind = DRAW_STOCK if event.kind == EVENT_STOCK else DRAW_TAKE
            self.drawn[actor] = (self.events, kind, event.card)
        elif event.kind == EVENT_DISCARD:
            self.discarded[actor] = (self.events, event.card)
        self.last[actor] = event

    def observe(self, seat: int) -> tuple[np.ndarray, np.ndarray]:
        """Seat ``seat``'s actor token and oracle token now; both [WIDTH] int16."""

        engine = self.engine
        players = self.players
        rows = np.zeros((2, WIDTH), dtype=np.int64)  # actor, oracle
        rows[:, F_PHASE] = int(engine.phase == PLAY)
        rows[:, F_LAST_CARD] = NO_CARD
        rows[:, F_LAST_MELD_KIND] = NO_MELD_KIND
        last = self.last[seat]
        if last is not None:
            rows[:, F_LAST_KIND] = last.kind
            rows[:, F_LAST_CARD] = last.card
            if last.meld >= 0:
                rows[:, F_LAST_MELD_KIND] = MELD_KIND[last.meld]
                rows[:, F_LAST_MELD_SIZE] = len(MELD_CARDS[last.meld])
        since = self.token_event[seat]
        for offset in range(1, MAX_PLAYERS):
            draw_field, take_field, discard_field = (
                F_OPP_DRAW + offset - 1, F_OPP_TAKE + offset - 1, F_OPP_DISCARD + offset - 1,
            )
            rows[:, take_field] = NO_CARD
            rows[:, discard_field] = NO_CARD
            if offset >= players:
                rows[:, draw_field] = DRAW_ABSENT
                continue
            opponent = (seat + offset) % players
            at, kind, card = self.drawn[opponent]
            if at > since:
                rows[:, draw_field] = kind
                rows[1, take_field] = card
                if kind == DRAW_TAKE:
                    rows[0, take_field] = card
            at, card = self.discarded[opponent]
            if at > since:
                rows[:, discard_field] = card
        rows[:, F_STOCK] = len(engine.stock)
        sizes = engine.hands.sum(axis=1)
        hand = np.full(MAX_PLAYERS, ABSENT_SEAT, dtype=np.int64)
        hand[self.relative[seat]] = sizes
        rows[:, F_HAND : F_HAND + MAX_PLAYERS] = hand
        opened = np.array(engine.opened, dtype=np.int64)
        rows[:, F_OPENED] = int((opened << self.relative[seat]).sum())
        rows[:, F_PROGRESS] = min(engine.turn_points, FIELD_VOCAB[F_PROGRESS] - 1)
        rows[:, F_PLAYERS] = players
        rows[:, F_ROUND] = min(engine.turns // players, MAX_ROUND)
        rows[0, F_STOCK_TOP] = NO_CARD
        rows[1, F_STOCK_TOP] = engine.stock[-1] if engine.stock else NO_CARD
        layable = engine.layable.astype(np.int64)
        own = self.pairs[seat]
        changed = own != self.reference[seat]
        mover = np.where(changed, 1 + (self.mover[seat] - seat) % players, 0)
        rows[0, LEDGER_START:] = own + NUM_PAIRS * (mover + NUM_MOVERS * layable)
        oracle = self.pairs[players]
        changed = oracle != self.oracle_reference[seat]
        mover = np.where(changed, 1 + (self.mover[players] - seat) % players, 0)
        rows[1, LEDGER_START:] = ROTATE[players][seat][oracle] + NUM_PAIRS * (mover + NUM_MOVERS * layable)
        self.reference[seat] = own
        self.oracle_reference[seat] = oracle
        self.token_event[seat] = self.events
        return rows[0].astype(np.int16), rows[1].astype(np.int16)
