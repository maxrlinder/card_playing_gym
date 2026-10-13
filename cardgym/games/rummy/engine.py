"""The Rummy round: rules, legal moves, scoring, and every viewer's card ledger.

Variant (``rummy-bot``'s ``rules.txt`` with its documented interpretations):
two decks without jokers, cyclic runs, 2-4 players, a configurable hand size,
and a 30-point opening. A turn is

    draw (stock or the discard top) -> any number of melds / lay-offs -> discard

* **Opening.** Until a seat has opened, its melds this turn must total at
  least ``opening_points``. Melds are laid one at a time; a meld is legal only
  if the rest of the hand can still complete the threshold (``melds.
  opening_melds``), and the seat may not discard or lay off until it has.
  Laying them in sequence within one turn is the same as laying them at once:
  nobody else acts in between.
* **Hand floor.** A meld or lay-off never takes the last card: a round is won
  by discarding it.
* **End.** Discarding the last card wins: the others lose their hand values
  and the winner gains the sum, doubled for going Rummy (opening and going
  out in one turn while the table was empty at its start). Drawing the last
  stock card ends the round at the end of that turn, without a winner if the
  drawer does not go out: everyone loses their hand value. ``max_turns`` is a
  safety bound with the same scoring, reported as ``truncated``.

The ledger is the per-viewer card-location state the model reads: for every
card type, where its two copies are, as far as the viewer knows for certain.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from .cards import COPIES, NO_CARD, NUM_CARDS, full_deck, hand_penalty
from .melds import EXTENSIONS, GROUP, MELD_CARDS, MELD_KIND, MELD_POINTS, hand_melds, opening_melds

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
# for the oracle, absolute seat j's whole hand. Location 0 is everything else
# (the observer's "unseen"; the oracle's stock).
LOC_UNSEEN = 0
LOC_TABLE = 1 + MAX_PLAYERS
LOC_PILE = 2 + MAX_PLAYERS  # discard pile below the top
LOC_TOP = 3 + MAX_PLAYERS  # discard top
NUM_LOCATIONS = 4 + MAX_PLAYERS

# The two copies' locations as an unordered pair: 36 codes.
PAIR = np.zeros((NUM_LOCATIONS, NUM_LOCATIONS), dtype=np.int64)
_pairs = [(a, b) for a in range(NUM_LOCATIONS) for b in range(a, NUM_LOCATIONS)]
for _code, (_a, _b) in enumerate(_pairs):
    PAIR[_a, _b] = PAIR[_b, _a] = _code
NUM_PAIRS = len(_pairs)
PAIR_LOCATIONS = np.array(_pairs, dtype=np.int64)  # [36, 2]
# Who last moved a card type, as the viewer sees it: nobody since the viewer's
# previous token, the viewer itself, or opponent 1..3 (observer-relative).
NUM_MOVERS = 1 + MAX_PLAYERS
# A ledger code is pair + NUM_PAIRS * (mover + NUM_MOVERS * layable).
NUM_LEDGER_CODES = NUM_PAIRS * NUM_MOVERS * 2


def _pair_rotations() -> dict[int, np.ndarray]:
    """``ROTATE[players][seat]`` maps an absolute-seat pair code to seat-relative.

    Location ``1 + j`` (absolute seat ``j``'s hand) becomes ``1 + (j - seat) %
    players``; every other location is unchanged.
    """

    rotations = {}
    for players in range(2, MAX_PLAYERS + 1):
        table = np.zeros((players, NUM_PAIRS), dtype=np.int64)
        for seat in range(players):
            location = np.arange(NUM_LOCATIONS)
            location[1 : 1 + players] = 1 + (np.arange(players) - seat) % players
            for code, (a, b) in enumerate(_pairs):
                table[seat, code] = PAIR[location[a], location[b]]
        rotations[players] = table
    return rotations


ROTATE = _pair_rotations()


class Action(NamedTuple):
    kind: int
    card: int = NO_CARD  # TAKE / LAYOFF / DISCARD
    meld: int = -1  # MELD: meld index; LAYOFF: table position


@dataclass(frozen=True)
class Event:
    kind: int
    actor: int
    card: int  # NO_CARD for melds
    meld: int  # MELD: new meld; LAYOFF: extended meld; else -1
    cards: tuple[int, ...]  # card types that left the actor's hand (or entered it on a draw)


class RummyEngine:
    def __init__(
        self,
        num_players: int,
        hand_size: int,
        seed: int,
        *,
        opening_points: int = 30,
        max_turns: int = 200,
        first_player: int | None = None,
    ) -> None:
        if not 2 <= num_players <= MAX_PLAYERS:
            raise ValueError("Rummy needs 2-4 players.")
        if num_players * hand_size + 2 > NUM_CARDS * COPIES:
            raise ValueError("Not enough cards for this deal.")
        rng = random.Random(seed)
        deck = full_deck()
        rng.shuffle(deck)
        self.num_players = num_players
        self.hand_size = hand_size
        self.opening_points = opening_points
        self.max_turns = max_turns
        self.hands = np.zeros((num_players, NUM_CARDS), dtype=np.int64)
        for seat in range(num_players):
            for _ in range(hand_size):
                self.hands[seat, deck.pop()] += 1
        self.known = np.zeros((num_players, NUM_CARDS), dtype=np.int64)
        self.pile = [deck.pop()]
        self.pile_counts = np.zeros(NUM_CARDS, dtype=np.int64)
        self.pile_counts[self.pile[0]] += 1
        self.stock = deck  # drawn from the end
        self.table: list[int] = []
        self.table_counts = np.zeros(NUM_CARDS, dtype=np.int64)
        self.layable = np.zeros(NUM_CARDS, dtype=bool)
        self.opened = [False] * num_players
        self.current = rng.randrange(num_players) if first_player is None else first_player
        self.phase = DRAW
        self.turns = 0  # completed turns
        self.turn_points = 0  # opening points laid this turn by an unopened seat
        self._turn_start_table_empty = True
        self._turn_start_opened = False
        self.winner: int | None = None
        self.rummy = False
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
                self._candidates = [Action(STOCK)] + ([Action(TAKE, self.top)] if self.pile else [])
            elif self.phase == PLAY:
                self._candidates = self._play_candidates()
            else:
                self._candidates = []
        return self._candidates

    def _play_candidates(self) -> list[Action]:
        seat = self.current
        hand = self.hands[seat]
        size = int(hand.sum())
        melds = hand_melds(hand)
        actions: list[Action] = []
        if not self.opened[seat]:
            need = self.opening_points - self.turn_points
            for meld in opening_melds(hand, melds, need):
                actions.append(Action(MELD_RUN if MELD_KIND[meld] else MELD_GROUP, NO_CARD, meld))
            if self.turn_points:
                return actions  # an opening in progress must be completed
        else:
            for meld in melds:
                if len(MELD_CARDS[meld]) < size:
                    actions.append(Action(MELD_RUN if MELD_KIND[meld] else MELD_GROUP, NO_CARD, meld))
            if size >= 2:
                seen: set[tuple[int, int]] = set()
                for position, meld in enumerate(self.table):
                    for card, _ in EXTENSIONS[meld]:
                        # Identical table melds give identical outcomes.
                        if hand[card] and (card, meld) not in seen:
                            seen.add((card, meld))
                            kind = LAYOFF_GROUP if MELD_KIND[meld] == GROUP else LAYOFF_RUN
                            actions.append(Action(kind, card, position))
        for card in np.flatnonzero(hand):
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
                event = Event(EVENT_STOCK, seat, card, -1, (card,))
                self._stock_out = not self.stock
            elif kind == TAKE:
                card = self.pile.pop()
                self.pile_counts[card] -= 1
                self.known[seat, card] += 1
                event = Event(EVENT_TAKE, seat, card, -1, (card,))
            else:
                raise ValueError(f"Action {action} is not a draw.")
            self.hands[seat, card] += 1
            self.phase = PLAY
            return event
        if kind in (MELD_GROUP, MELD_RUN):
            meld = action.meld
            cards = MELD_CARDS[meld]
            for card in cards:
                self._remove(seat, card)
                self.table_counts[card] += 1
            self.table.append(meld)
            if not self.opened[seat]:
                self.turn_points += MELD_POINTS[meld]
                if self.turn_points >= self.opening_points:
                    self.opened[seat] = True
                    self.turn_points = 0
            self._refresh_layable()
            return Event(EVENT_MELD, seat, NO_CARD, meld, cards)
        if kind in (LAYOFF_GROUP, LAYOFF_RUN):
            target = self.table[action.meld]
            extended = dict(EXTENSIONS[target])[action.card]
            self._remove(seat, action.card)
            self.table_counts[action.card] += 1
            self.table[action.meld] = extended
            self._refresh_layable()
            return Event(EVENT_LAYOFF, seat, action.card, extended, (action.card,))
        if kind == DISCARD:
            card = action.card
            self._remove(seat, card)
            self.pile.append(card)
            self.pile_counts[card] += 1
            self._end_turn(seat)
            return Event(EVENT_DISCARD, seat, card, -1, (card,))
        raise ValueError(f"Action {action} is not a play.")

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
            self.rummy = not self._turn_start_opened and self._turn_start_table_empty
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
        self._turn_start_table_empty = not self.table
        self._turn_start_opened = self.opened[self.current]

    # -- outcome -----------------------------------------------------------

    def penalties(self) -> np.ndarray:
        return np.array([hand_penalty(self.hands[seat]) for seat in range(self.num_players)], dtype=np.float64)

    def scores(self) -> np.ndarray:
        penalties = self.penalties()
        if self.winner is None:
            return -penalties
        factor = 2.0 if self.rummy else 1.0
        scores = -factor * penalties
        scores[self.winner] = factor * (penalties.sum() - penalties[self.winner])
        return scores

    # -- ledgers -------------------------------------------------------------

    def location_counts(self) -> np.ndarray:
        """Copies per (view, location, card type): views 0..P-1 are the seats, view P the oracle."""

        players = self.num_players
        counts = np.zeros((players + 1, NUM_LOCATIONS, NUM_CARDS), dtype=np.int64)
        counts[:, LOC_TABLE] = self.table_counts
        counts[:, LOC_PILE] = self.pile_counts
        if self.pile:
            counts[:, LOC_PILE, self.pile[-1]] -= 1
            counts[:, LOC_TOP, self.pile[-1]] = 1
        # Observer o, location 1 + j: seat o + j's hand (own) or known copies.
        counts[:players, 1 : 1 + players] = self.known[self._observed]
        counts[np.arange(players), 1] = self.hands
        counts[players, 1 : 1 + players] = self.hands
        counts[:, LOC_UNSEEN] = COPIES - counts[:, 1:].sum(axis=1)
        return counts

    def view_pairs(self) -> np.ndarray:
        """Copy-location pair codes [P + 1, 52]: views 0..P-1 are the seats, view P the oracle."""

        cumulative = self.location_counts().cumsum(axis=1)
        return PAIR[(cumulative >= 1).argmax(axis=1), (cumulative >= 2).argmax(axis=1)]
