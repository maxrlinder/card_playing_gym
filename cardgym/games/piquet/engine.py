"""Piquet rules: one deal, or a six-deal Rubicon partie, between two seats.

A deal runs in three phases, each with its own policy head:

1. **Exchange** (``exchange``). Elder (the non-dealer) discards 1 to 5 cards
   and draws as many from the top of the eight-card talon; then Younger
   discards 0 up to every card left and draws the same way. A discard is
   chosen one card at a time (actions 0-31) and closed with ``STOP`` (32),
   so a 1-in-1585 choice becomes a few small ones. Elder may look at the
   cards of its five it left behind (they are the first Younger would draw);
   cards Younger leaves stay face down. A hand without K, Q, or J (carte
   blanche) is shown before its owner exchanges and scores 10.
2. **Declarations** (``declare``), in the order point, sequence, sets. Elder
   declares its best combination of the category or says nothing (action 1
   or 0); Younger then contests (1) or concedes "good" (0) when it holds an
   equal or better one, or, after Elder said nothing, declares or not.
   Saying nothing is how a combination is sunk: it hides the cards and scores
   nothing. The winner scores and shows the winning cards; equal points or
   sequences score for neither.
3. **Tricks** (``play``). Elder leads; follow suit if you can; no trumps.
   Leading a card scores 1, winning a trick the opponent led scores 1, the
   last trick 1 more. Seven or more tricks score 10 ("the cards"), all twelve
   40 instead (capot).

Bonuses: **repique** (+60) for reaching 30 from declarations alone (carte
blanche included, in order) before the opponent scores anything; **pique**
(+30, Elder only, never with repique) for reaching 30 from declarations and
trick play before Younger scores anything. The 10 or 40 for the cards come
after the last trick and do not count toward pique.

A partie is ``deals`` deals with the deal alternating. ``returns`` is the
point difference, or with ``rubicon`` the Rubicon payoff: the winner of the
six deals scores 100 plus the difference, or 100 plus both totals when the
loser stayed under 100 (rubiconed).

Decisions with a single legal action are taken automatically.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .cards import FACE_MASK, NO_CARD, NUM_CARDS, mask_array, mask_of, rank_of, suit_mask, suit_of
from .rules import CATEGORIES, best, compare, score, shown

HAND_SIZE = 12
TALON_SIZE = 8
ELDER_MAX = 5
TRICKS = HAND_SIZE

EXCHANGE, DECLARE, PLAY, DONE = range(4)
HEADS = ("exchange", "declare", "play")
PHASE_HEAD = {EXCHANGE: "exchange", DECLARE: "declare", PLAY: "play"}
STOP = NUM_CARDS  # exchange: done choosing discards
QUIET, SPEAK = 0, 1  # declare: say nothing (or concede "good") / declare (or contest)
ACTIONS = {"exchange": NUM_CARDS + 1, "declare": 2, "play": NUM_CARDS}

# Events. SELECT (one more card chosen for discard) is private and changes
# only what the chooser sees at its next decision.
EV_DEAL, EV_CARTE_BLANCHE, EV_SELECT, EV_EXCHANGE, EV_ANNOUNCE, EV_RESOLVE, EV_PLAY, EV_DEAL_END = range(8)
# What a seat said in a declaration.
SAID_NOTHING, DECLARED, CONCEDED = 1, 2, 3
BONUS_NONE, BONUS_REPIQUE, BONUS_PIQUE, BONUS_CARDS, BONUS_CAPOT = range(5)
CARTE_BLANCHE_POINTS = 10
REPIQUE, PIQUE, CARDS, CAPOT = 60, 30, 10, 40

# Where each card is (absolute seats): hand, chosen for discard, discarded,
# played in a finished trick, on the table in the current trick, or in the
# talon at a depth from its top.
W_HAND, W_SELECTED, W_DISCARD, W_PLAYED, W_TRICK, W_TALON = 0, 2, 4, 6, 8, 10
NUM_WHERE = W_TALON + TALON_SIZE
# What a seat knows about a card it cannot see by its location: nothing, its
# exact location, or "in the opponent's hand or discards" (shown by carte
# blanche before that opponent exchanged).
KNOW_NONE, KNOW_EXACT, KNOW_MAYBE = 0, 1, 2


@dataclass
class Event:
    kind: int
    actor: int = -1
    card: int = NO_CARD
    count: int = 0
    category: int = -1
    status: int = 0
    key: tuple[int, ...] | None = None  # the combination declared (point: length, value, suit)
    winner: int = -1  # trick or category winner; -1 for nobody
    points: int = 0
    bonus: int = BONUS_NONE
    bonus_seat: int = -1
    cards: int = 0  # mask: shown cards; on an exchange, the cards drawn (private)
    discarded: int = 0  # exchange: the discards (private)
    seen: int = 0  # Elder's exchange: the talon cards it left and may look at (private)


@dataclass
class DealResult:
    elder: int
    dealt: list[int] = field(default_factory=lambda: [0, 0])  # hands as dealt
    talon: list[int] = field(default_factory=list)  # as dealt, top first
    held: list[int] = field(default_factory=lambda: [0, 0])  # hands after the exchange
    # Every card played, in order: (seat, card, trick, whether it won the trick).
    plays: list[tuple[int, int, int, bool]] = field(default_factory=list)
    points: list[int] = field(default_factory=lambda: [0, 0])
    tricks: list[int] = field(default_factory=lambda: [0, 0])
    exchanged: list[int] = field(default_factory=lambda: [0, 0])
    category_winners: list[int] = field(default_factory=lambda: [-1, -1, -1])
    trick_winners: list[int] = field(default_factory=list)
    carte_blanche: list[bool] = field(default_factory=lambda: [False, False])
    repique: int = -1
    pique: int = -1
    capot: int = -1


Listener = Callable[["PiquetEngine", Event], None]


class PiquetEngine:
    def __init__(
        self,
        seed: int,
        *,
        deals: int = 1,
        rubicon: bool = False,
        dealer: int | None = None,
        listener: Listener | None = None,
        deck: list[int] | None = None,
    ) -> None:
        if deals < 1:
            raise ValueError("A partie needs at least one deal.")
        self.rng = random.Random(seed)
        self.deals = deals
        self.rubicon = rubicon
        self.listener = listener
        self.events: list[Event] = []
        self.results: list[DealResult] = []
        self.totals = [0, 0]
        self.deal_index = 0
        self.dealer = self.rng.randrange(2) if dealer is None else dealer
        self.done = False
        self._fixed_deck = deck
        self._deal()
        self._advance()

    # -- state ---------------------------------------------------------------

    @property
    def head(self) -> str:
        return PHASE_HEAD[self.phase]

    @property
    def younger(self) -> int:
        return 1 - self.elder

    def _deal(self) -> None:
        if self._fixed_deck is not None and self.deal_index == 0:
            deck = list(self._fixed_deck)
        else:
            deck = list(range(NUM_CARDS))
            self.rng.shuffle(deck)
        self.elder = 1 - self.dealer
        self.hands = [0, 0]
        self.hands[self.elder] = mask_of(deck[:HAND_SIZE])
        self.hands[self.younger] = mask_of(deck[HAND_SIZE : 2 * HAND_SIZE])
        self.talon = deck[2 * HAND_SIZE :]
        self.where = np.empty(NUM_CARDS, dtype=np.int64)
        for seat in (0, 1):
            self.where[mask_array(self.hands[seat])] = W_HAND + seat
        self._place_talon()
        self.info = np.zeros((2, NUM_CARDS), dtype=np.int64)
        self.selected = 0
        self.discards = [0, 0]
        self.exchanged = [False, False]
        self.shown_hand = [0, 0]  # carte blanche
        self.points = [0, 0]
        self.declared = [0, 0]
        self.result = DealResult(elder=self.elder, dealt=list(self.hands), talon=list(self.talon))
        self.category = 0
        self.announced: tuple[int, ...] | None = None
        self.tricks = [0, 0]
        self.trick_index = 0
        self.trick: list[tuple[int, int]] = []
        self.played = [0, 0]
        self.voids = [0, 0]  # suits each seat has shown it lacks
        self.phase = EXCHANGE
        self.current = self.elder
        self._emit(Event(EV_DEAL))
        if not self.hands[self.elder] & FACE_MASK:
            self._carte_blanche(self.elder)
        self._limits(1, min(ELDER_MAX, len(self.talon)))

    def _place_talon(self) -> None:
        for depth, card in enumerate(self.talon):
            self.where[card] = W_TALON + depth

    def _limits(self, low: int, high: int) -> None:
        self.min_discards, self.max_discards = low, high

    def _emit(self, event: Event) -> None:
        self.events.append(event)
        if self.listener is not None:
            self.listener(self, event)

    def _score(self, seat: int, points: int, *, declaration: bool) -> tuple[int, int]:
        """Add points; returns (bonus kind, seat) when this crossed into repique or pique."""

        other = 1 - seat
        self.points[seat] += points
        if declaration:
            self.declared[seat] += points
            if self.result.repique < 0 and self.declared[seat] >= 30 and self.points[other] == 0:
                self.result.repique = seat
                self.points[seat] += REPIQUE
                return BONUS_REPIQUE, seat
        elif (
            seat == self.elder
            and self.result.repique < 0
            and self.result.pique < 0
            and self.points[seat] >= 30
            and self.points[other] == 0
        ):
            self.result.pique = seat
            self.points[seat] += PIQUE
            return BONUS_PIQUE, seat
        return BONUS_NONE, -1

    # -- decisions -------------------------------------------------------------

    def legal_mask(self) -> int:
        """Legal cards as a mask (exchange: selectable cards; play: playable cards)."""

        hand = self.hands[self.current]
        if self.phase == EXCHANGE:
            if bin(self.selected).count("1") >= self.max_discards:
                return 0
            return hand & ~self.selected
        if self.phase == PLAY:
            if self.trick:
                follow = hand & suit_mask(suit_of(self.trick[0][1]))
                if follow:
                    return follow
            return hand
        return 0

    def legal(self) -> np.ndarray:
        if self.done:
            raise RuntimeError("The partie is over.")
        if self.phase == DECLARE:
            return np.ones(2, dtype=bool)
        legal = np.zeros(ACTIONS[self.head], dtype=bool)
        legal[:NUM_CARDS] = mask_array(self.legal_mask())
        if self.phase == EXCHANGE:
            legal[STOP] = bin(self.selected).count("1") >= self.min_discards
        return legal

    def apply(self, action: int) -> None:
        if not self.legal()[action]:
            raise ValueError(f"Illegal {self.head} action {action}.")
        self._apply(action)
        self._advance()

    def _apply(self, action: int) -> None:
        if self.phase == EXCHANGE:
            if action == STOP:
                self._exchange()
            else:
                self.selected |= 1 << action
                self.where[action] = W_SELECTED + self.current
                self._emit(Event(EV_SELECT, self.current, card=action))
        elif self.phase == DECLARE:
            if self.current == self.elder:
                self._announce(action == SPEAK)
            else:
                self._resolve(action == SPEAK)
        else:
            self._play(action)

    def _advance(self) -> None:
        """Take every decision that has a single legal action."""

        while not self.done:
            if self.phase == DECLARE:
                return  # declarations only wait when there is a choice
            legal = self.legal()
            if legal.sum() > 1:
                return
            self._apply(int(np.flatnonzero(legal)[0]))

    # -- exchange --------------------------------------------------------------

    def _carte_blanche(self, seat: int) -> None:
        self.shown_hand[seat] = self.hands[seat]
        self.info[1 - seat][mask_array(self.hands[seat])] = KNOW_EXACT
        self.result.carte_blanche[seat] = True
        bonus, bonus_seat = self._score(seat, CARTE_BLANCHE_POINTS, declaration=True)
        self._emit(
            Event(EV_CARTE_BLANCHE, seat, cards=self.hands[seat], points=CARTE_BLANCHE_POINTS,
                  bonus=bonus, bonus_seat=bonus_seat)
        )

    def _exchange(self) -> None:
        seat = self.current
        count = bin(self.selected).count("1")
        drawn = self.talon[:count]
        self.talon = self.talon[count:]
        drawn_mask = mask_of(drawn)
        discarded = self.selected
        self.hands[seat] = (self.hands[seat] & ~discarded) | drawn_mask
        self.discards[seat] = discarded
        self.where[mask_array(discarded)] = W_DISCARD + seat
        self.where[mask_array(drawn_mask)] = W_HAND + seat
        self._place_talon()
        self.exchanged[seat] = True
        self.result.exchanged[seat] = count
        self.selected = 0
        other = 1 - seat
        # Cards shown by carte blanche may now be in the hand or the discards.
        maybe = mask_array(self.shown_hand[seat]) & (self.info[other] == KNOW_EXACT)
        self.info[other][maybe] = KNOW_MAYBE
        seen = 0
        if seat == self.elder:
            seen = mask_of(self.talon[: ELDER_MAX - count])
            self.info[seat][mask_array(seen)] = KNOW_EXACT
        self._emit(Event(EV_EXCHANGE, seat, count=count, cards=drawn_mask, discarded=discarded, seen=seen))
        if seat == self.elder:
            self.current = self.younger
            if not self.hands[self.younger] & FACE_MASK:
                self._carte_blanche(self.younger)
            self._limits(0, len(self.talon))
        else:
            self.result.held = list(self.hands)
            self.phase = DECLARE
            self.category = 0
            self._elder_declares()

    # -- declarations ----------------------------------------------------------

    def _elder_declares(self) -> None:
        self.current = self.elder
        self.announced = None
        if best(self.category, self.hands[self.elder]) is None:
            self._announce(False)

    def _announce(self, speak: bool) -> None:
        key = best(self.category, self.hands[self.elder]) if speak else None
        self.announced = key
        self._emit(Event(EV_ANNOUNCE, self.elder, category=self.category,
                         status=DECLARED if key is not None else SAID_NOTHING, key=key))
        self.current = self.younger
        mine = best(self.category, self.hands[self.younger])
        if mine is None or (key is not None and compare(self.category, mine, key) < 0):
            self._resolve(False)

    def _resolve(self, speak: bool) -> None:
        category = self.category
        younger, elder = self.younger, self.elder
        key = best(category, self.hands[younger]) if speak else None
        if self.announced is not None:
            if key is None:
                status, winner = CONCEDED, elder
            else:
                order = compare(category, key, self.announced)
                status, winner = DECLARED, (younger if order > 0 else -1)
        else:
            status, winner = (DECLARED, younger) if key is not None else (SAID_NOTHING, -1)
        points = cards = 0
        bonus, bonus_seat = BONUS_NONE, -1
        if winner >= 0:
            points = score(category, self.hands[winner])
            cards = shown(category, self.hands[winner])
            self.info[1 - winner][mask_array(cards)] = KNOW_EXACT
            bonus, bonus_seat = self._score(winner, points, declaration=True)
        self.result.category_winners[category] = winner
        self._emit(Event(EV_RESOLVE, younger, category=category, status=status, key=key, winner=winner,
                         points=points, cards=cards, bonus=bonus, bonus_seat=bonus_seat))
        self.category += 1
        if self.category < len(CATEGORIES):
            self._elder_declares()
        else:
            self.phase = PLAY
            self.current = self.elder

    # -- tricks ----------------------------------------------------------------

    def _play(self, card: int) -> None:
        seat = self.current
        self.hands[seat] &= ~(1 << card)
        self.where[card] = W_TRICK + seat
        if not self.trick:
            self.trick.append((seat, card))
            bonus, bonus_seat = self._score(seat, 1, declaration=False)
            self._emit(Event(EV_PLAY, seat, card=card, points=1, bonus=bonus, bonus_seat=bonus_seat))
            self.current = 1 - seat
            return
        leader, led = self.trick[0]
        follows = suit_of(card) == suit_of(led)
        if not follows:
            self.voids[seat] |= 1 << suit_of(led)
        winner = seat if follows and rank_of(card) > rank_of(led) else leader
        points = int(winner == seat) + int(self.trick_index == TRICKS - 1)
        for player, played in ((leader, led), (seat, card)):
            self.played[player] |= 1 << played
            self.where[played] = W_PLAYED + player
        self.result.plays += [
            (leader, led, self.trick_index, winner == leader),
            (seat, card, self.trick_index, winner == seat),
        ]
        self.trick = []
        self.tricks[winner] += 1
        self.trick_index += 1
        self.result.trick_winners.append(winner)
        bonus, bonus_seat = self._score(winner, points, declaration=False) if points else (BONUS_NONE, -1)
        self._emit(Event(EV_PLAY, seat, card=card, winner=winner, points=points, bonus=bonus, bonus_seat=bonus_seat))
        if self.trick_index == TRICKS:
            self._end_deal()
        else:
            self.current = winner

    def _end_deal(self) -> None:
        result = self.result
        bonus, winner, points = BONUS_NONE, -1, 0
        for seat in (0, 1):
            if self.tricks[seat] == TRICKS:
                bonus, winner, points = BONUS_CAPOT, seat, CAPOT
                result.capot = seat
            elif self.tricks[seat] > TRICKS // 2:
                bonus, winner, points = BONUS_CARDS, seat, CARDS
        if winner >= 0:
            self.points[winner] += points
        result.points = list(self.points)
        result.tricks = list(self.tricks)
        self.phase = DONE
        self._emit(Event(EV_DEAL_END, winner=winner, points=points, bonus=bonus, bonus_seat=winner))
        self.results.append(result)
        for seat in (0, 1):
            self.totals[seat] += self.points[seat]
        self.deal_index += 1
        if self.deal_index >= self.deals:
            self.done = True
            self.current = -1
        else:
            self.dealer = 1 - self.dealer
            self._deal()

    # -- outcome ---------------------------------------------------------------

    def returns(self) -> np.ndarray:
        """Zero-sum reward per seat: the point difference, or the Rubicon payoff."""

        a, b = self.totals
        if not self.rubicon:
            return np.array([a - b, b - a], dtype=np.float64)
        if a == b:
            return np.zeros(2)
        winner = 0 if a > b else 1
        high, low = max(a, b), min(a, b)
        payoff = 100 + (high - low if low >= 100 else high + low)
        returns = np.full(2, -float(payoff))
        returns[winner] = payoff
        return returns

