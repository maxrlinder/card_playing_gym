"""Piquet token streams: every event of the partie, plus a decision token before each of the seat's own choices.

Both seats see every event (the deal, a carte blanche, each exchange, each
declaration, each card played, the end of a deal) as one token, in order; the
seat about to decide also gets a **decision** token, the position its policy
reads. Selecting a card for discard has no token of its own: the next
decision token shows it. A token is a row of categorical fields plus 32 card
codes, one per card:

* the fields describe the event (who, which card, what was declared, who won
  the trick or the category, points and bonuses) and restate the counters
  after it: role, phase, tricks, both seats' points this deal, talon size,
  the suits the opponent has shown it lacks, and in a Rubicon partie the
  score: both running totals (earlier deals plus this one's points so far,
  exact below 120), the running lead, and the deal number. Decision tokens add the head and its context (cards chosen so
  far and the most allowed; the category and the seat's own best
  combination, and how it compares with Elder's declaration; the suit led);
* the card codes are where each card is, as far as the viewer knows: its own
  hand (and, at a decision, which of those cards are legal), its chosen and
  discarded cards, talon cards it has seen, cards it knows the opponent
  holds (shown by a declaration or carte blanche, or seen in the talon before
  the opponent drew them), cards played in finished tricks and the current
  one by each seat, or unknown.

Every seat reference is relative: "me" and "the opponent". So a token
restates the present exactly and the stream keeps the order of everything
that happened. In a partie each deal starts a fresh context (the stream
splits there, see PiquetSelfPlay.context_starts): earlier deals' cards are
irrelevant, and the score fields carry everything they decided. There is no oracle stream: the actor's own value head is the
PPO baseline. The writer also returns, per token, where each card the viewer
cannot place truly is (the belief head's labels).
"""

from __future__ import annotations

import numpy as np

from .cards import NO_CARD, NUM_CARDS, mask_array, suit_of
from .engine import (
    DECLARE,
    EV_ANNOUNCE,
    EV_CARTE_BLANCHE,
    EV_DEAL,
    EV_DEAL_END,
    EV_EXCHANGE,
    EV_PLAY,
    EV_RESOLVE,
    EV_SELECT,
    EXCHANGE,
    HEADS,
    KNOW_EXACT,
    KNOW_MAYBE,
    KNOW_NONE,
    NUM_WHERE,
    PLAY,
    TALON_SIZE,
    TRICKS,
    W_DISCARD,
    W_HAND,
    W_PLAYED,
    W_SELECTED,
    W_TALON,
    W_TRICK,
    Event,
    PiquetEngine,
)
from .rules import POINT, best, compare

# Token kinds: a decision, or an event.
KIND_DECISION = 0
EVENT_KINDS = {
    EV_DEAL: 1, EV_CARTE_BLANCHE: 2, EV_EXCHANGE: 3, EV_ANNOUNCE: 4, EV_RESOLVE: 5, EV_PLAY: 6, EV_DEAL_END: 7,
}
KIND_DEAL, KIND_CARTE_BLANCHE, KIND_EXCHANGE, KIND_ANNOUNCE, KIND_RESOLVE, KIND_PLAY, KIND_DEAL_END = range(1, 8)

F_KIND = 0
F_HEAD = 1  # decision tokens: 1 + head index (0 = an event)
F_ACTOR = 2  # 1 me, 2 the opponent, 0 none
F_ROLE = 3  # 0 elder, 1 younger (this deal)
F_CARD = 4  # card played; a decision's card led, or the card chosen last for discard
F_COUNT = 5  # cards exchanged; at a discard decision, cards chosen so far
F_LIMIT = 6  # at a discard decision, the most that may be chosen
F_CATEGORY = 7  # 1 + point/sequence/sets
F_STATUS = 8  # declared: 1 nothing, 2 a combination, 3 "good" (conceded)
F_LENGTH = 9  # a declared combination: cards in the point or sequence, size of the set
F_VALUE = 10  # the point's card total
F_TOP = 11  # 1 + top rank of a sequence, or the rank of a set
F_COMPARE = 12  # Younger's decision against a declaration: 1 better, 2 equal
F_WINNER = 13  # trick, category, or the cards: 1 me, 2 the opponent, 3 nobody
F_POINTS = 14  # points the event scored
F_BONUS = 15  # repique, pique, the cards, capot
F_BONUS_SEAT = 16
F_TRICK = 17  # tricks finished this deal
F_TRICK_POS = 18  # 1 leading, 2 following
F_LEAD_SUIT = 19  # 1 + suit led to the current trick
F_MY_TRICKS = 20
F_OPP_TRICKS = 21
F_MY_POINTS = 22  # this deal
F_OPP_POINTS = 23
F_MY_TOTAL = 24  # partie: earlier deals plus this deal's points so far (see total_code)
F_OPP_TOTAL = 25
F_DEAL = 26  # partie: deals finished
F_MODE = 27  # 0 a single deal, 1 a Rubicon partie
F_TALON = 28
F_OPP_VOID = 29  # bitmask of suits the opponent failed to follow
F_PHASE = 30  # exchange, declare, play, done
F_LEAD = 31  # partie: my running total minus the opponent's (see lead_code)
NUM_FIELDS = 32
CARD_START = NUM_FIELDS
WIDTH = NUM_FIELDS + NUM_CARDS

MAX_POINTS = 99  # this deal; a repique deal can pass it (rare)
MAX_DEALS = 6
# Partie totals are exact below EXACT_TOTAL (the rubicon line is 100), then in tens.
EXACT_TOTAL = 120
TOTAL_VOCAB = EXACT_TOTAL + 21  # up to 320
# The running lead is exact within +-EXACT_LEAD, then in tens out to +-200.
EXACT_LEAD = 60
LEAD_STEPS = EXACT_LEAD + 14
LEAD_VOCAB = 2 * LEAD_STEPS + 1
FIELD_VOCAB = (
    8, 1 + len(HEADS), 3, 2, NUM_CARDS + 1, 9, 9, 4, 4, 9, 76, 9, 3, 4, 64, 5, 3,
    TRICKS + 1, 3, 5, TRICKS + 1, TRICKS + 1, MAX_POINTS + 1, MAX_POINTS + 1, TOTAL_VOCAB, TOTAL_VOCAB,
    MAX_DEALS, 2, TALON_SIZE + 1, 16, 4, LEAD_VOCAB,
)
assert len(FIELD_VOCAB) == NUM_FIELDS
# The layout before the partie fields (31 fields; totals in fives below 500,
# no lead): checkpoints trained on it load through PiquetEmbedding, blind to
# the partie fields.
LEGACY_FIELD_VOCAB = FIELD_VOCAB[:24] + (100, 100) + FIELD_VOCAB[26:31]
# Fields only a Rubicon partie fills (0 in a single deal).
PARTIE_FIELDS = (F_MY_TOTAL, F_OPP_TOTAL, F_DEAL, F_MODE, F_LEAD)


def total_code(total: int) -> int:
    if total < EXACT_TOTAL:
        return total
    return min(EXACT_TOTAL + (total - EXACT_TOTAL) // 10, TOTAL_VOCAB - 1)


def lead_code(lead: int) -> int:
    size = abs(lead)
    if size > EXACT_LEAD:
        size = min(EXACT_LEAD - (-(size - EXACT_LEAD) // 10), LEAD_STEPS)
    return LEAD_STEPS + (size if lead >= 0 else -size)

# Card codes: where the viewer knows each card to be.
(
    V_UNKNOWN, V_MINE, V_MINE_LEGAL, V_SELECTED, V_DISCARDED, V_SEEN_TALON,
    V_OPP, V_OPP_MAYBE, V_PLAYED_ME, V_PLAYED_OPP, V_TRICK_ME, V_TRICK_OPP,
) = range(12)
CARD_VOCAB = 12

# Where a card the viewer cannot place truly is, for the belief head: the
# opponent's hand, its discards, or the talon (-1: the viewer can place it).
BELIEF_HAND, BELIEF_DISCARD, BELIEF_TALON = range(3)
BELIEF_CLASSES = 3
UNCERTAIN = np.zeros(CARD_VOCAB, dtype=bool)
UNCERTAIN[[V_UNKNOWN, V_OPP_MAYBE]] = True


def _tables() -> tuple[np.ndarray, np.ndarray]:
    """VIEW[seat, where, knowledge] -> card code; TRUTH[seat, where] -> belief class or -1."""

    view = np.full((2, NUM_WHERE, 3), V_UNKNOWN, dtype=np.int16)
    truth = np.full((2, NUM_WHERE), -1, dtype=np.int8)
    for seat in (0, 1):
        other = 1 - seat
        view[seat, W_HAND + seat] = V_MINE
        view[seat, W_SELECTED + seat] = V_SELECTED
        view[seat, W_DISCARD + seat] = V_DISCARDED
        for where in (W_HAND + other, W_SELECTED + other):
            view[seat, where, KNOW_EXACT] = V_OPP
            view[seat, where, KNOW_MAYBE] = V_OPP_MAYBE
        view[seat, W_DISCARD + other, KNOW_EXACT] = V_OPP_MAYBE
        view[seat, W_DISCARD + other, KNOW_MAYBE] = V_OPP_MAYBE
        view[seat, W_PLAYED + seat] = V_PLAYED_ME
        view[seat, W_PLAYED + other] = V_PLAYED_OPP
        view[seat, W_TRICK + seat] = V_TRICK_ME
        view[seat, W_TRICK + other] = V_TRICK_OPP
        view[seat, W_TALON : W_TALON + TALON_SIZE, KNOW_EXACT] = V_SEEN_TALON
        truth[seat, [W_HAND + other, W_SELECTED + other]] = BELIEF_HAND
        truth[seat, W_DISCARD + other] = BELIEF_DISCARD
        truth[seat, W_TALON : W_TALON + TALON_SIZE] = BELIEF_TALON
    assert KNOW_NONE == 0
    return view, truth


VIEW, TRUTH = _tables()

# Tokens a seat can receive in one deal: the deal, two carte blanches, its
# discard decisions (at most 7 cards and the stop) and both exchanges, three
# categories of two events and one decision, 24 cards played and 12
# decisions, and the end of the deal. Every context is one deal (a partie
# restarts it each deal), so this is also the longest sequence.
DEAL_TOKENS = 1 + 2 + 8 + 2 + 3 * 3 + 24 + 12 + 1


def _relative(seat: int, viewer: int) -> int:
    return 0 if seat < 0 else 1 + (seat != viewer)


def _key_fields(category: int, key: tuple[int, ...] | None) -> tuple[int, int, int]:
    """(length, value, top) of a declared combination."""

    if key is None:
        return 0, 0, 0
    if category == POINT:
        return key[0], key[1], 0
    return key[0], 0, key[1] + 1


class ObservationWriter:
    """Builds both seats' actor and oracle tokens from the engine's state.

    :meth:`event_rows` is called as each event happens (the engine's
    listener), :meth:`decision_row` when a seat is about to decide, and
    :meth:`truth` with every token for the belief labels.
    """

    def __init__(self) -> None:
        self.last_selected = [NO_CARD, NO_CARD]

    def _counters(self, engine: PiquetEngine, seat: int) -> np.ndarray:
        row = np.zeros(WIDTH, dtype=np.int64)
        other = 1 - seat
        row[F_ROLE] = int(seat != engine.elder)
        row[F_CARD] = NO_CARD
        row[F_TRICK] = engine.trick_index
        row[F_MY_TRICKS] = engine.tricks[seat]
        row[F_OPP_TRICKS] = engine.tricks[other]
        row[F_MY_POINTS] = min(engine.points[seat], MAX_POINTS)
        row[F_OPP_POINTS] = min(engine.points[other], MAX_POINTS)
        if engine.rubicon:
            mine = engine.totals[seat] + engine.points[seat]
            theirs = engine.totals[other] + engine.points[other]
            row[F_MY_TOTAL] = total_code(mine)
            row[F_OPP_TOTAL] = total_code(theirs)
            row[F_LEAD] = lead_code(mine - theirs)
            row[F_DEAL] = min(engine.deal_index, MAX_DEALS - 1)
            row[F_MODE] = 1
        row[F_TALON] = len(engine.talon)
        row[F_OPP_VOID] = engine.voids[other]
        row[F_PHASE] = engine.phase
        return row

    @staticmethod
    def _codes(engine: PiquetEngine, seat: int, legal: int = 0) -> np.ndarray:
        view = VIEW[seat, engine.where, engine.info[seat]]
        if legal:
            view[mask_array(legal)] = V_MINE_LEGAL
        return view

    @staticmethod
    def truth(engine: PiquetEngine, seat: int) -> np.ndarray:
        """int8 [32]: each card's belief class from ``seat``'s side (-1 where it holds or saw it)."""

        return TRUTH[seat, engine.where]

    def event_rows(self, engine: PiquetEngine, event: Event) -> np.ndarray | None:
        """Both seats' tokens for ``event`` [2, WIDTH] int16; None if it has none."""

        if event.kind == EV_SELECT:
            self.last_selected[event.actor] = event.card
            return None
        if event.kind == EV_EXCHANGE:
            self.last_selected[event.actor] = NO_CARD
        rows = np.empty((2, WIDTH), dtype=np.int16)
        for seat in (0, 1):
            row = self._counters(engine, seat)
            row[F_KIND] = EVENT_KINDS[event.kind]
            row[F_ACTOR] = _relative(event.actor, seat)
            if event.kind == EV_EXCHANGE:
                row[F_COUNT] = event.count
            elif event.kind in (EV_ANNOUNCE, EV_RESOLVE):
                row[F_CATEGORY] = 1 + event.category
                row[F_STATUS] = event.status
                row[F_LENGTH], row[F_VALUE], row[F_TOP] = _key_fields(event.category, event.key)
            elif event.kind == EV_PLAY:
                row[F_CARD] = event.card
                lead = event.winner < 0
                row[F_TRICK_POS] = 1 if lead else 2
                led = event.card if lead else engine.events[-2].card
                row[F_LEAD_SUIT] = 1 + suit_of(led)
            if event.kind in (EV_RESOLVE, EV_PLAY, EV_DEAL_END) and (event.winner >= 0 or event.kind != EV_PLAY):
                row[F_WINNER] = 3 if event.winner < 0 else _relative(event.winner, seat)
            row[F_POINTS] = min(event.points, FIELD_VOCAB[F_POINTS] - 1)
            row[F_BONUS] = event.bonus
            row[F_BONUS_SEAT] = _relative(event.bonus_seat, seat)
            row[CARD_START:] = self._codes(engine, seat)
            rows[seat] = row
        return rows

    def decision_row(self, engine: PiquetEngine, seat: int) -> np.ndarray:
        """The deciding seat's decision token [WIDTH] int16."""

        row = self._counters(engine, seat)
        row[F_KIND] = KIND_DECISION
        row[F_HEAD] = 1 + HEADS.index(engine.head)
        row[F_ACTOR] = 1
        legal = 0
        if engine.phase == EXCHANGE:
            row[F_COUNT] = engine.selected.bit_count()
            row[F_LIMIT] = engine.max_discards
            row[F_CARD] = self.last_selected[seat]
            legal = engine.legal_mask()
        elif engine.phase == DECLARE:
            category = engine.category
            mine = best(category, engine.hands[seat])
            row[F_CATEGORY] = 1 + category
            row[F_LENGTH], row[F_VALUE], row[F_TOP] = _key_fields(category, mine)
            if seat != engine.elder:
                announced = engine.announced
                row[F_STATUS] = 1 if announced is None else 2
                if announced is not None:
                    row[F_COMPARE] = 1 if compare(category, mine, announced) > 0 else 2
        elif engine.phase == PLAY:
            legal = engine.legal_mask()
            if engine.trick:
                row[F_TRICK_POS] = 2
                row[F_CARD] = engine.trick[0][1]
                row[F_LEAD_SUIT] = 1 + suit_of(engine.trick[0][1])
            else:
                row[F_TRICK_POS] = 1
        row[CARD_START:] = self._codes(engine, seat, legal)
        return row.astype(np.int16)
