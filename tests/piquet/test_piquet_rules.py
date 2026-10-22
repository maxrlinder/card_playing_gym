"""Declarations, scoring, and the deal's flow on rigged decks."""

from __future__ import annotations

import random

import numpy as np
import pytest

from cardgym.games.piquet.cards import (
    NUM_CARDS,
    card_id,
    cards_of,
    mask_array,
    mask_of,
    parse_card,
    rank_of,
    suit_mask,
    suit_of,
)
from cardgym.games.piquet.engine import (
    BONUS_PIQUE,
    BONUS_REPIQUE,
    CAPOT,
    CARDS,
    DECLARE,
    ELDER_MAX,
    EV_CARTE_BLANCHE,
    EV_DEAL_END,
    EV_PLAY,
    EV_RESOLVE,
    EXCHANGE,
    PIQUE,
    PLAY,
    QUIET,
    REPIQUE,
    SPEAK,
    STOP,
    W_DISCARD,
    W_HAND,
    W_PLAYED,
    W_SELECTED,
    W_TALON,
    W_TRICK,
    PiquetEngine,
)
from cardgym.games.piquet.heuristic import HeuristicPolicy, RandomPolicy
from cardgym.games.piquet.rules import POINT, SEQUENCE, SETS, best, compare, score, shown


def hand(text: str) -> int:
    return mask_of(parse_card(card) for card in text.split())


def deck(elder: str, younger: str, talon: str) -> list[int]:
    cards = [parse_card(c) for c in f"{elder} {younger} {talon}".split()]
    assert sorted(cards) == list(range(NUM_CARDS)), "a rigged deck must hold every card once"
    return cards


def test_point_sequence_and_sets():
    cards = hand("AH KH QH JH 10H 7S 8S 9S 10S AS AD AC")
    assert best(POINT, cards) == (5, 51, 1)  # hearts beat spades' 5 cards worth 45
    assert score(POINT, cards) == 5
    assert shown(POINT, cards) == hand("AH KH QH JH 10H")
    assert best(SEQUENCE, cards) == (5, 7)  # quint to the Ace
    assert score(SEQUENCE, cards) == 15 + 4  # the quint and 7-8-9-10 of spades
    assert shown(SEQUENCE, cards) == hand("AH KH QH JH 10H 7S 8S 9S 10S")
    assert best(SETS, cards) == (4, 7)
    assert score(SETS, cards) == 14  # four aces; the tens are only two
    assert shown(SETS, cards) == hand("AH AS AD AC")
    # Sixième to huitième; nines never make a set.
    assert score(SEQUENCE, hand("7C 8C 9C 10C JC QC")) == 16
    assert score(SEQUENCE, hand("7C 8C 9C 10C JC QC KC AC")) == 18
    assert best(SETS, hand("9S 9H 9D 9C")) is None
    assert score(SETS, hand("KS KH KD QS QH QD")) == 6


def test_keys_compare_length_then_value_and_points_ignore_the_suit():
    assert compare(POINT, (5, 44, 0), (5, 44, 3)) == 0
    assert compare(POINT, (5, 45, 0), (5, 44, 3)) > 0
    assert compare(POINT, (4, 44, 0), (5, 30, 3)) < 0
    assert compare(SEQUENCE, (4, 7), (4, 6)) > 0
    assert compare(SETS, (3, 7), (4, 3)) < 0  # any quatorze beats any trio


def play_out(engine: PiquetEngine, policy=None) -> None:
    policy = policy or HeuristicPolicy()
    while not engine.done:
        engine.apply(policy.act(engine))


def event_totals(engine: PiquetEngine, deal: int) -> list[int]:
    """Each seat's points this deal, rebuilt from the events alone."""

    totals, index = [0, 0], -1
    for event in engine.events:
        if event.kind == 0:  # a deal starts
            index += 1
        if index != deal:
            continue
        if event.kind == EV_CARTE_BLANCHE:
            totals[event.actor] += event.points
        elif event.kind == EV_RESOLVE and event.winner >= 0:
            totals[event.winner] += event.points
        elif event.kind == EV_PLAY:
            totals[event.actor if event.winner < 0 else event.winner] += event.points
        elif event.kind == EV_DEAL_END and event.winner >= 0:
            totals[event.winner] += event.points
        if event.bonus in (BONUS_REPIQUE, BONUS_PIQUE):
            totals[event.bonus_seat] += REPIQUE if event.bonus == BONUS_REPIQUE else PIQUE
    return totals


@pytest.mark.parametrize("policy", ["heuristic", "random"])
def test_deals_conserve_cards_and_points_add_up(policy):
    for seed in range(60):
        engine = PiquetEngine(seed, deals=2)
        rng = random.Random(seed)
        while not engine.done:
            seat = engine.current
            legal = engine.legal()
            if engine.phase == EXCHANGE:
                held = bin(engine.hands[seat]).count("1")
                assert held == 12
                low, high = (1, ELDER_MAX) if seat == engine.elder else (0, len(engine.talon))
                assert (engine.min_discards, engine.max_discards) == (low, high)
            elif engine.phase == PLAY and engine.trick:
                led = suit_mask(suit_of(engine.trick[0][1]))
                follow = engine.hands[seat] & led
                assert mask_of(np.flatnonzero(legal)) == (follow or engine.hands[seat])
            # Every card is in exactly one place and `where` agrees with the masks.
            where = engine.where
            for player in (0, 1):
                held = engine.hands[player] & ~(engine.selected if player == seat else 0)
                assert np.array_equal(mask_array(held), where == W_HAND + player)
                assert np.array_equal(mask_array(engine.discards[player]), where == W_DISCARD + player)
                assert np.array_equal(mask_array(engine.played[player]), where == W_PLAYED + player)
            assert np.array_equal(mask_array(engine.selected), (where == W_SELECTED) | (where == W_SELECTED + 1))
            assert sorted(np.flatnonzero(where >= W_TALON)) == sorted(engine.talon)
            assert (where == W_TRICK).sum() + (where == W_TRICK + 1).sum() == len(engine.trick)
            action = (HeuristicPolicy() if policy == "heuristic" else RandomPolicy(rng.random())).act(engine)
            engine.apply(action)
        for deal, result in enumerate(engine.results):
            assert sum(result.tricks) == 12
            assert result.points == event_totals(engine, deal)
            assert 1 <= result.exchanged[result.elder] <= ELDER_MAX
            assert 0 <= result.exchanged[1 - result.elder] <= 8 - result.exchanged[result.elder]
        assert engine.totals == [sum(r.points[s] for r in engine.results) for s in (0, 1)]
        assert engine.results[0].elder != engine.results[1].elder  # the deal alternates


# A huitième and point of eight in hearts, a trio of kings.
PIQUE_DECK = deck(
    "7H 8H 9H 10H JH QH KH AH KS KD QS JD",
    "7S 8S 9S 7D 8D 9D 7C 8C 9C 10S JS QD",
    "10D AS AD 10C JC QC KC AC",
)


def test_pique_on_the_opening_lead():
    engine = PiquetEngine(0, dealer=1, deck=PIQUE_DECK)  # seat 0 is elder
    engine.apply(parse_card("JD"))
    engine.apply(STOP)
    assert engine.hands[0] & hand("10D")  # drawn from the top
    assert engine.phase == EXCHANGE and engine.current == 1
    engine.apply(STOP)  # younger keeps its hand
    while engine.phase == DECLARE:
        engine.apply(SPEAK)
    # Point 8, huitième 18, three kings 3: 29, one short of repique.
    assert engine.results == [] and engine.result.category_winners == [0, 0, 0]
    assert engine.points == [29, 0] and engine.result.repique == -1
    engine.apply(parse_card("AH"))  # the lead's point makes 30: pique
    assert engine.result.pique == 0 and engine.points == [30 + PIQUE, 0]
    play_out(engine)


def test_repique_needs_thirty_from_declarations_alone():
    rigged = deck(
        "7H 8H 9H 10H JH QH KH AH AS AD AC KS",
        "7S 8S 9S 10S JS QS 7D 8D 9D 10D JD QD",
        "KD 7C 8C 9C 10C JC QC KC",
    )
    engine = PiquetEngine(0, dealer=1, deck=rigged)
    engine.apply(parse_card("KS"))
    engine.apply(STOP)
    engine.apply(STOP)
    while engine.phase == DECLARE:
        engine.apply(SPEAK)
    # 8 + 18 + 14 = 40 with the opponent at zero: repique, and no pique on top.
    assert engine.result.repique == 0 and engine.points[0] == 40 + REPIQUE
    play_out(engine)
    assert engine.results[0].pique == -1


# Equal points (five cards worth 44) and equal quints to the Jack.
TIE_DECK = deck(
    "7H 8H 9H 10H JH QS KS 7D QD 7C 8C KC",
    "7S 8S 9S 10S JS 8D 9D KD 9C 10C QC QH",
    "AC AS KH AH 10D JD AD JC",
)


def declarations(engine: PiquetEngine, actions: list[int]) -> None:
    engine.apply(parse_card("KC"))
    engine.apply(STOP)
    engine.apply(STOP)
    for action in actions:
        assert engine.phase == DECLARE
        engine.apply(action)


def test_equal_combinations_score_for_neither():
    engine = PiquetEngine(0, dealer=1, deck=TIE_DECK)
    declarations(engine, [SPEAK, SPEAK, SPEAK, SPEAK])  # point and sequence: both declare
    assert engine.phase == PLAY  # no sets on either side
    assert engine.result.category_winners == [-1, -1, -1] and engine.points == [0, 0]


def test_conceding_and_sinking():
    engine = PiquetEngine(0, dealer=1, deck=TIE_DECK)
    declarations(engine, [SPEAK, QUIET, QUIET])  # younger says "good"; elder sinks its quint
    assert engine.current == 1 and engine.category == SEQUENCE
    engine.apply(SPEAK)  # younger declares its quint unopposed
    assert engine.result.category_winners == [0, 1, -1]
    assert engine.points == [5, 15]
    resolve = [e for e in engine.events if e.kind == EV_RESOLVE]
    assert resolve[1].cards == hand("7S 8S 9S 10S JS")  # the winner shows its cards


def test_carte_blanche_is_shown_and_scores_ten():
    rigged = deck(
        "7H 8H 9H 10H AH 7S 8S 9S 10S AS 7D 8D",
        "JH QH KH JS QS KS JD QD KD JC QC KC",
        "9D 10D AD 7C 8C 9C 10C AC",
    )
    engine = PiquetEngine(0, dealer=1, deck=rigged)
    shown_event = next(e for e in engine.events if e.kind == EV_CARTE_BLANCHE)
    assert shown_event.actor == 0 and shown_event.cards == engine.hands[0]
    assert engine.points == [10, 0]
    assert (engine.info[1][mask_array(engine.hands[0])] == 1).all()


def test_elder_sees_the_cards_it_left():
    engine = PiquetEngine(0, dealer=1, deck=TIE_DECK)
    engine.apply(parse_card("KC"))
    engine.apply(STOP)
    left = [parse_card(c) for c in "AS KH AH 10D".split()]
    assert (engine.info[0][left] == 1).all() and engine.info[0][parse_card("JD")] == 0
    assert (engine.info[1][left] == 0).all()


def test_capot_and_the_cards():
    for seed in range(400):
        engine = PiquetEngine(seed)
        play_out(engine)
        result = engine.results[0]
        end = engine.events[-1]
        if result.capot >= 0:
            assert result.tricks[result.capot] == 12 and end.points == CAPOT
        elif max(result.tricks) > 6:
            assert end.winner == int(np.argmax(result.tricks)) and end.points == CARDS
        else:
            assert end.winner == -1 and end.points == 0


def test_rubicon_payoff():
    engine = PiquetEngine(0, deals=6, rubicon=True)
    play_out(engine)
    for totals, expected in (([150, 120], 130), ([150, 80], 330), ([90, 70], 260), ([100, 100], 0)):
        engine.totals = totals
        returns = engine.returns()
        assert returns.sum() == 0 and abs(returns[0]) == expected
        assert (returns[0] > 0) == (totals[0] > totals[1]) or expected == 0


def test_heuristic_beats_random():
    total = 0.0
    for seed in range(100):
        for seat in (0, 1):
            policies = [RandomPolicy(seed), RandomPolicy(seed)]
            policies[seat] = HeuristicPolicy()
            engine = PiquetEngine(seed)
            while not engine.done:
                engine.apply(policies[engine.current].act(engine))
            total += engine.returns()[seat]
    assert total / 200 > 15


def test_cards():
    assert cards_of(hand("AS 7H")) == [parse_card("AS"), parse_card("7H")]
    assert rank_of(parse_card("10D")) == 3 and suit_of(parse_card("10D")) == 2
    assert card_id(3, 7) == parse_card("AC") == NUM_CARDS - 1
