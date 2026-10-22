"""The Rummy round: rules, legal moves, scoring, and every viewer's card ledger.

Variant: Scandinavian rummy with 500 Rum's discard pile and scoring. One deck
by default (two as an option), no jokers, cyclic runs, 2-4 players, seven
cards each by default. A turn is

    draw (the stock, or any discard with every card above it) -> any number of melds / lay-offs -> discard

* **Drawing.** Take the stock's top card, or a card in the discard pile
  together with every card above it. The top card alone may be kept; a deeper
  card obliges the seat to meld it or lay it off this turn (the cards above
  it go to the hand), so a deeper card is offered only when that is possible:
  directly, in a meld from the hand, or at the end of a chain of lay-offs
  that extends a table run up to it (2-3-4 on the table, 5 in hand: the 6).
  Until it has been used the seat may only make the melds and lay-offs that
  use it, and the lay-offs that start such a chain.
* **Opening.** None by default: any seat may meld and lay off from its first
  turn. ``opening_points > 0`` restores a threshold: until a seat has opened,
  its melds in one turn must total at least that many points, laid one at a
  time; a meld is legal only if the rest of the hand can still complete the
  threshold (``melds.opening_melds``), and the seat may not discard or lay
  off until it has.
* **Hand floor.** A meld or lay-off never takes the last card: a round is won
  by discarding it.
* **One route per outcome.** An opened seat melds exactly three cards and
  adds any more by lay-offs (4-5-6, then 7), which scores the same as the
  longer meld: every card beyond three is a separate decision to play it or
  hold it. An unopened seat's opening melds keep every size, since the
  threshold search counts them whole.
* **Scoring.** Every card a seat melds or lays off scores its points for that
  seat (``melds.MELD_POINTS``; a lay-off scores what it adds to the meld).
  When the round ends, the cards left in each hand count against it (Ace 11,
  faces 10, pips face value). The round ends when a seat discards its last
  card, or at the end of the turn that drew the last stock card.
  ``max_turns`` is a safety bound with the same scoring, reported as
  ``truncated``.

The ledger is the per-viewer card-location state the model reads: for every
card type, where its copies are, as far as the viewer knows for certain.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from .cards import COPIES, MAX_DECKS, NO_CARD, NUM_CARDS, NUM_RANKS, card_id, full_deck, hand_penalty
from .melds import (
    EXTENSIONS,
    FULL_RUN,
    GROUP,
    MELD_CARDS,
    MELD_KIND,
    MELD_POINTS,
    MELDS,
    RUN,
    can_meld,
    hand_melds,
    opening_melds,
)

MAX_PLAYERS = 4

# Phases.
DRAW, PLAY, DONE = 0, 1, 2

# Action kinds; the model embeds these as candidate kinds.
STOCK, TAKE, MELD_GROUP, MELD_RUN, LAYOFF_GROUP, LAYOFF_RUN, DISCARD = range(7)
NUM_ACTION_KINDS = 7

# Event kinds (token types; 0 is padding and 1 the deal).
EVENT_DEAL, EVENT_STOCK, EVENT_TAKE, EVENT_MELD, EVENT_LAYOFF, EVENT_DISCARD = range(1, 7)

# Ledger locations. Location 1 + j is seat j's hand: for an observer, the
# observer itself (j = 0, fully known) and the copies each opponent is
# publicly known to hold (taken from the discard pile and not since played);
# for the oracle, absolute seat j's whole hand. Location LOC_TABLE + j is the
# table, copies seat j put there (melded or laid off). LOC_PILE + d is the
# discard pile at depth d (0 is the top; the last depth holds everything
# deeper). LOC_ABSENT holds the copies a one-deck game does not have.
# Location 0 is everything else (the observer's "unseen"; the oracle's stock).
LOC_UNSEEN = 0
LOC_TABLE = 1 + MAX_PLAYERS
PILE_DEPTHS = 11
LOC_PILE = 1 + 2 * MAX_PLAYERS
LOC_TOP = LOC_PILE
LOC_ABSENT = LOC_PILE + PILE_DEPTHS
NUM_LOCATIONS = LOC_ABSENT + 1

# The two copies' locations as an unordered pair: 231 codes.
PAIR = np.zeros((NUM_LOCATIONS, NUM_LOCATIONS), dtype=np.int64)
_pairs = [(a, b) for a in range(NUM_LOCATIONS) for b in range(a, NUM_LOCATIONS)]
for _code, (_a, _b) in enumerate(_pairs):
    PAIR[_a, _b] = PAIR[_b, _a] = _code
NUM_PAIRS = len(_pairs)
PAIR_LOCATIONS = np.array(_pairs, dtype=np.int64)  # [231, 2]
# A state code is pair + NUM_PAIRS * layable.
NUM_STATE_CODES = NUM_PAIRS * 2


def _pair_rotations() -> dict[int, np.ndarray]:
    """``ROTATE[players][seat]`` maps an absolute-seat pair code to seat-relative.

    Locations ``1 + j`` and ``LOC_TABLE + j`` (absolute seat ``j``'s hand and
    the copies it tabled) move to ``j - seat`` mod ``players``; every other
    location is unchanged.
    """

    rotations = {}
    for players in range(2, MAX_PLAYERS + 1):
        table = np.zeros((players, NUM_PAIRS), dtype=np.int64)
        for seat in range(players):
            location = np.arange(NUM_LOCATIONS)
            relative = (np.arange(players) - seat) % players
            location[1 : 1 + players] = 1 + relative
            location[LOC_TABLE : LOC_TABLE + players] = LOC_TABLE + relative
            for code, (a, b) in enumerate(_pairs):
                table[seat, code] = PAIR[location[a], location[b]]
        rotations[players] = table
    return rotations


ROTATE = _pair_rotations()


def pair_codes(counts: np.ndarray) -> np.ndarray:
    """Pair codes [..., 52] from copy counts [..., NUM_LOCATIONS, 52] that sum to 2 per card."""

    cumulative = counts.cumsum(axis=-2)
    return PAIR[(cumulative >= 1).argmax(axis=-2), (cumulative >= 2).argmax(axis=-2)]


class Action(NamedTuple):
    kind: int
    card: int = NO_CARD  # TAKE (the deepest card taken) / LAYOFF / DISCARD
    meld: int = -1  # MELD: meld index; LAYOFF: table position; TAKE: cards taken above the card


@dataclass(frozen=True)
class Event:
    kind: int
    actor: int
    card: int  # NO_CARD for melds; the deepest card for a take
    meld: int  # MELD: new meld; LAYOFF: extended meld; else -1
    cards: tuple[int, ...]  # card types that left the actor's hand (or entered it on a draw)


class RummyEngine:
    def __init__(
        self,
        num_players: int,
        hand_size: int,
        seed: int,
        *,
        decks: int = 1,
        opening_points: int = 0,
        max_turns: int = 200,
        first_player: int | None = None,
    ) -> None:
        if not 2 <= num_players <= MAX_PLAYERS:
            raise ValueError("Rummy needs 2-4 players.")
        if not 1 <= decks <= MAX_DECKS:
            raise ValueError("Rummy uses one or two decks.")
        if num_players * hand_size + 2 > NUM_CARDS * decks:
            raise ValueError("Not enough cards for this deal.")
        rng = random.Random(seed)
        deck = full_deck(decks)
        rng.shuffle(deck)
        self.num_players = num_players
        self.hand_size = hand_size
        self.decks = decks
        self.opening_points = opening_points
        self.max_turns = max_turns
        self.hands = np.zeros((num_players, NUM_CARDS), dtype=np.int64)
        for seat in range(num_players):
            for _ in range(hand_size):
                self.hands[seat, deck.pop()] += 1
        self.known = np.zeros((num_players, NUM_CARDS), dtype=np.int64)
        self.pile = [deck.pop()]  # bottom first; the top is the last
        self.pile_counts = np.zeros(NUM_CARDS, dtype=np.int64)
        self.pile_counts[self.pile[0]] += 1
        self.stock = deck  # drawn from the end
        self.table: list[int] = []
        self.table_counts = np.zeros(NUM_CARDS, dtype=np.int64)
        self.placed = np.zeros((num_players, NUM_CARDS), dtype=np.int64)  # tabled copies per seat
        self.points = np.zeros(num_players, dtype=np.int64)  # points melded and laid off
        self.layable = np.zeros(NUM_CARDS, dtype=bool)
        self.opened = [opening_points <= 0] * num_players
        self.current = rng.randrange(num_players) if first_player is None else first_player
        self.phase = DRAW
        self.turns = 0  # completed turns
        self.turn_points = 0  # opening points laid this turn by an unopened seat
        self.obligation = NO_CARD  # a card taken from below the top, still to be used this turn
        self.kept = NO_CARD  # the discard top taken this turn: it may be used or kept, not discarded
        self.winner: int | None = None
        self.truncated = False
        self._stock_out = False
        self._candidates: list[Action] | None = None
        rotation = np.arange(MAX_PLAYERS)
        self._observed = (np.arange(num_players)[:, None] + rotation[None, :num_players]) % num_players

    # -- queries -----------------------------------------------------------

    @property
    def done(self) -> bool:
        return self.phase == DONE

    @property
    def top(self) -> int:
        return self.pile[-1] if self.pile else NO_CARD

    def hand_count(self, seat: int) -> int:
        return int(self.hands[seat].sum())

    def candidates(self) -> list[Action]:
        """Every legal action, each semantically distinct, in a fixed order."""

        if self._candidates is None:
            if self.phase == DRAW:
                self._candidates = self._draw_candidates()
            elif self.phase == PLAY:
                actions = self._play_candidates(self.current, self.hands[self.current])
                card = self.obligation
                if card != NO_CARD:
                    chains = self._chains(self.hands[self.current], card)
                    actions = [
                        action for action in actions
                        if self._uses(action, card) or (action.kind == LAYOFF_RUN and (action.meld, action.card) in chains)
                    ]
                self._candidates = actions
            else:
                self._candidates = []
        return self._candidates

    def _draw_candidates(self) -> list[Action]:
        actions = [Action(STOCK)]
        if not self.pile:
            return actions
        seat = self.current
        hand = self.hands[seat].copy()
        # The taken top cannot go back this turn, so another card must be there to discard.
        if hand.sum() > hand[self.pile[-1]]:
            actions.append(Action(TAKE, self.pile[-1], 0))
        hand[self.pile[-1]] += 1
        for above in range(1, len(self.pile)):
            card = self.pile[-1 - above]
            hand[card] += 1
            if self._usable(seat, hand, card):
                actions.append(Action(TAKE, card, above))
        return actions

    def _usable(self, seat: int, hand: np.ndarray, card: int) -> bool:
        """Whether ``card`` can be melded or laid off this turn from ``hand`` (just drawn)."""

        if self.opened[seat]:
            return bool(self.layable[card]) or can_meld(hand, card) or bool(self._chains(hand, card))
        melds = hand_melds(hand)
        return any(card in MELD_CARDS[meld] for meld in opening_melds(hand, melds, self.opening_points))

    def _chains(self, hand: np.ndarray, card: int) -> set[tuple[int, int]]:
        """(table position, first card) of every lay-off chain that ends with ``card``.

        A chain extends a table run of the card's suit one hand card at a time
        until ``card`` fits its end (2-3-4 on the table, 5 in hand: lay off 5,
        then 6). It must leave a card to discard and the run within 13 cards.
        The run's direct extensions (no hand card between) are in ``layable``.
        """

        suit, rank = divmod(card, NUM_RANKS)
        size = int(hand.sum())
        starts = set()
        for position, index in enumerate(self.table):
            meld = MELDS[index]
            if meld.kind != RUN or meld.a != suit or meld.c == FULL_RUN:
                continue
            for step, end in ((1, meld.b + meld.c - 1), (-1, meld.b)):
                between = (rank - end) * step % NUM_RANKS - 1  # hand cards needed before ``card``
                if between < 1 or meld.c + between + 1 > FULL_RUN or between + 1 >= size:
                    continue
                if all(hand[card_id(suit, (end + step * i) % NUM_RANKS)] for i in range(1, between + 1)):
                    starts.add((position, card_id(suit, (end + step) % NUM_RANKS)))
        return starts

    @staticmethod
    def _uses(action: Action, card: int) -> bool:
        if action.kind in (MELD_GROUP, MELD_RUN):
            return card in MELD_CARDS[action.meld]
        return action.kind in (LAYOFF_GROUP, LAYOFF_RUN) and action.card == card

    def _play_candidates(self, seat: int, hand: np.ndarray) -> list[Action]:
        size = int(hand.sum())
        kept = self.kept
        # Cards that may be discarded: the turn must always be able to end.
        spare = size - (int(hand[kept]) if kept != NO_CARD else 0)

        def leaves_discard(cards) -> bool:
            if kept == NO_CARD:
                return len(cards) < spare
            return spare - sum(card != kept for card in cards) >= 1

        melds = hand_melds(hand)
        actions: list[Action] = []
        if not self.opened[seat]:
            need = self.opening_points - self.turn_points
            for meld in opening_melds(hand, melds, need):
                if leaves_discard(MELD_CARDS[meld]):
                    actions.append(Action(MELD_RUN if MELD_KIND[meld] else MELD_GROUP, NO_CARD, meld))
            if self.turn_points:
                return actions  # an opening in progress must be completed
        else:
            # Opened seats meld three cards at a time and add the rest one lay-off at a
            # time: 4-5-6 then 7 is the only way to table 4-5-6-7, so every extra card
            # is its own play-or-hold decision and no outcome has two routes.
            for meld in melds:
                if len(MELD_CARDS[meld]) == 3 and len(MELD_CARDS[meld]) < size and leaves_discard(MELD_CARDS[meld]):
                    actions.append(Action(MELD_RUN if MELD_KIND[meld] else MELD_GROUP, NO_CARD, meld))
            if size >= 2:
                seen: set[tuple[int, int]] = set()
                for position, meld in enumerate(self.table):
                    for card, _ in EXTENSIONS[meld]:
                        # Identical table melds give identical outcomes.
                        if hand[card] and (card, meld) not in seen and leaves_discard((card,)):
                            seen.add((card, meld))
                            kind = LAYOFF_GROUP if MELD_KIND[meld] == GROUP else LAYOFF_RUN
                            actions.append(Action(kind, card, position))
        for card in np.flatnonzero(hand):
            if card != kept:
                actions.append(Action(DISCARD, int(card)))
        return actions

    # -- transitions -------------------------------------------------------

    def apply(self, action: Action) -> Event:
        if self.phase == DONE:
            raise RuntimeError("The round is over.")
        seat = self.current
        kind = action.kind
        self._candidates = None
        if self.phase == DRAW:
            if kind == STOCK:
                card = self.stock.pop()
                self.hands[seat, card] += 1
                event = Event(EVENT_STOCK, seat, card, -1, (card,))
                self._stock_out = not self.stock
            elif kind == TAKE:
                above = action.meld
                if not 0 <= above < len(self.pile) or self.pile[-1 - above] != action.card:
                    raise ValueError(f"Action {action} does not match the discard pile.")
                taken = tuple(reversed(self.pile[-1 - above :]))  # top first
                del self.pile[-1 - above :]
                for card in taken:
                    self.pile_counts[card] -= 1
                    self.known[seat, card] += 1
                    self.hands[seat, card] += 1
                if above:
                    self.obligation = action.card
                else:
                    self.kept = action.card
                event = Event(EVENT_TAKE, seat, action.card, -1, taken)
            else:
                raise ValueError(f"Action {action} is not a draw.")
            self.phase = PLAY
            return event
        if kind in (MELD_GROUP, MELD_RUN):
            meld = action.meld
            cards = MELD_CARDS[meld]
            for card in cards:
                self._table(seat, card)
            self.table.append(meld)
            self.points[seat] += MELD_POINTS[meld]
            if not self.opened[seat]:
                self.turn_points += MELD_POINTS[meld]
                if self.turn_points >= self.opening_points:
                    self.opened[seat] = True
                    self.turn_points = 0
            if self.obligation in cards:
                self.obligation = NO_CARD
            self._refresh_layable()
            return Event(EVENT_MELD, seat, NO_CARD, meld, cards)
        if kind in (LAYOFF_GROUP, LAYOFF_RUN):
            target = self.table[action.meld]
            extended = dict(EXTENSIONS[target])[action.card]
            self._table(seat, action.card)
            self.table[action.meld] = extended
            self.points[seat] += MELD_POINTS[extended] - MELD_POINTS[target]
            if self.obligation == action.card:
                self.obligation = NO_CARD
            self._refresh_layable()
            return Event(EVENT_LAYOFF, seat, action.card, extended, (action.card,))
        if kind == DISCARD:
            if self.obligation != NO_CARD:
                raise ValueError("A card taken from below the top must be used before discarding.")
            card = action.card
            if card == self.kept:
                raise ValueError("The discard top taken this turn cannot be discarded the same turn.")
            self._remove(seat, card)
            self.pile.append(card)
            self.pile_counts[card] += 1
            self._end_turn(seat)
            return Event(EVENT_DISCARD, seat, card, -1, (card,))
        raise ValueError(f"Action {action} is not a play.")

    def _table(self, seat: int, card: int) -> None:
        self._remove(seat, card)
        self.table_counts[card] += 1
        self.placed[seat, card] += 1

    def _remove(self, seat: int, card: int) -> None:
        if self.hands[seat, card] <= 0:
            raise ValueError("Card not in hand.")
        self.hands[seat, card] -= 1
        # Public knowledge only shrinks: observers cannot tell which copy left.
        if self.known[seat, card]:
            self.known[seat, card] -= 1

    def _refresh_layable(self) -> None:
        self.layable[:] = False
        for meld in self.table:
            for card, _ in EXTENSIONS[meld]:
                self.layable[card] = True

    def _end_turn(self, seat: int) -> None:
        if not self.hands[seat].any():
            self.winner = seat
            self.phase = DONE
            return
        self.turns += 1
        if self._stock_out:
            self.phase = DONE
            return
        if self.turns >= self.max_turns:
            self.truncated = True
            self.phase = DONE
            return
        self.current = (seat + 1) % self.num_players
        self.phase = DRAW
        self.turn_points = 0
        self.kept = NO_CARD

    # -- outcome -----------------------------------------------------------

    def penalties(self) -> np.ndarray:
        return np.array([hand_penalty(self.hands[seat]) for seat in range(self.num_players)], dtype=np.float64)

    def scores(self) -> np.ndarray:
        """Points melded and laid off minus the hand left (final once the round is over)."""

        return self.points.astype(np.float64) - self.penalties()

    # -- ledgers -------------------------------------------------------------

    def location_counts(self, seat: int | None = None) -> np.ndarray:
        """Copies per (view, location, card type): views 0..P-1 are the seats, view P the oracle.

        With ``seat``, only that seat's view and the oracle's: [2, locations, 52].
        """

        players = self.num_players
        seats = np.arange(players) if seat is None else np.array([seat])
        views = len(seats)
        counts = np.zeros((views + 1, NUM_LOCATIONS, NUM_CARDS), dtype=np.int64)
        if self.pile:
            depth = np.minimum(np.arange(len(self.pile))[::-1], PILE_DEPTHS - 1)
            np.add.at(counts[0], (LOC_PILE + depth, np.asarray(self.pile)), 1)
            counts[1:, LOC_PILE : LOC_PILE + PILE_DEPTHS] = counts[0, LOC_PILE : LOC_PILE + PILE_DEPTHS]
        counts[:, LOC_ABSENT] = COPIES - self.decks
        # Observer o, location 1 + j: seat o + j's hand (own) or known copies;
        # LOC_TABLE + j: the copies seat o + j tabled.
        observed = self._observed[seats]
        counts[:views, 1 : 1 + players] = self.known[observed]
        counts[:views, LOC_TABLE : LOC_TABLE + players] = self.placed[observed]
        counts[views, LOC_TABLE : LOC_TABLE + players] = self.placed
        counts[np.arange(views), 1] = self.hands[seats]
        counts[views, 1 : 1 + players] = self.hands
        counts[:, LOC_UNSEEN] = COPIES - counts[:, 1:].sum(axis=1)
        return counts

    def view_pairs(self, seat: int | None = None) -> np.ndarray:
        """Copy-location pair codes [P + 1, 52] (views as in :meth:`location_counts`)."""

        return pair_codes(self.location_counts(seat))
