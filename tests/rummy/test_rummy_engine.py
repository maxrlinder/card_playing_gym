from __future__ import annotations

import random

import numpy as np
import pytest

from cardgym.games.rummy.cards import COPIES, NO_CARD, NUM_CARDS, parse_card
from cardgym.games.rummy.engine import (
    DISCARD,
    DONE,
    DRAW,
    LAYOFF_GROUP,
    LAYOFF_RUN,
    LOC_UNSEEN,
    MELD_GROUP,
    MELD_RUN,
    PLAY,
    STOCK,
    TAKE,
    Action,
    RummyEngine,
)
from cardgym.games.rummy.heuristic import HeuristicPolicy, RandomPolicy
from cardgym.games.rummy.melds import MELD_CARDS, MELD_POINTS, meld_index
from cardgym.rl.ppo import relative_rewards


def play_out(engine: RummyEngine, policy) -> int:
    steps = 0
    while not engine.done:
        candidates = engine.candidates()
        assert candidates, "a live round always has a legal action"
        seat = engine.current
        before = int(engine.hands[seat].sum())
        action = candidates[policy.act(engine)]
        if engine.phase == PLAY and action.kind != DISCARD:
            used = len(MELD_CARDS[action.meld]) if action.kind in (MELD_GROUP, MELD_RUN) else 1
            assert used < before, "a meld or lay-off never takes the last card"
        if engine.phase == PLAY and engine.obligation != NO_CARD:
            assert all(a.kind != DISCARD for a in candidates), "a card taken from below the top is used first"
        if engine.phase == PLAY and engine.obligation == NO_CARD:
            discards = {a.card for a in candidates if a.kind == DISCARD}
            assert discards, "the turn can always end"
            assert engine.kept not in discards, "the discard top taken this turn is kept or used"
        if engine.phase == PLAY and not engine.opened[seat]:
            assert action.kind not in (LAYOFF_GROUP, LAYOFF_RUN)
            if engine.turn_points:
                assert all(a.kind in (MELD_GROUP, MELD_RUN) for a in candidates)
        engine.apply(action)
        steps += 1
        counts = engine.location_counts()
        assert counts.min() >= 0
        assert np.all(counts.sum(axis=1) == COPIES)
        oracle_stock = np.bincount(np.array(engine.stock, dtype=np.int64), minlength=NUM_CARDS)
        assert np.array_equal(counts[-1, LOC_UNSEEN], oracle_stock)
    return steps


@pytest.mark.parametrize("players", [2, 3, 4])
@pytest.mark.parametrize("decks", [1, 2])
@pytest.mark.parametrize("policy", ["heuristic", "random"])
def test_rounds_conserve_cards_and_score_melds_minus_hand(players, decks, policy):
    for seed in range(6):
        engine = RummyEngine(players, 7, seed, decks=decks, max_turns=120)
        play_out(engine, HeuristicPolicy() if policy == "heuristic" else RandomPolicy(seed))
        assert engine.phase == DONE
        assert np.array_equal(engine.scores(), engine.points - engine.penalties())
        # Lay-offs score what they add, so the seats' points sum to the table's value.
        assert engine.points.sum() == sum(MELD_POINTS[meld] for meld in engine.table)
        assert engine.placed.sum() == engine.table_counts.sum() == sum(len(MELD_CARDS[m]) for m in engine.table)
        if engine.winner is not None:
            assert engine.penalties()[engine.winner] == 0
        assert abs(relative_rewards(engine.scores()).sum()) < 1e-9


def test_stock_out_ends_the_round_without_a_winner():
    engine = RummyEngine(2, 7, seed=3)
    engine.stock = engine.stock[-2:]  # two cards left
    policy = RandomPolicy(0)
    draws = 0
    while not engine.done:
        if engine.phase == DRAW:
            engine.apply(Action(STOCK))
            draws += 1
        else:
            discards = [a for a in engine.candidates() if a.kind == DISCARD]
            engine.apply(discards[policy.rng.randrange(len(discards))])
    assert draws == 2 and engine.winner is None and not engine.truncated


def test_turn_limit_truncates():
    engine = RummyEngine(3, 7, seed=5, max_turns=4)
    play_out(engine, RandomPolicy(1))
    assert engine.turns <= 4
    assert engine.winner is not None or engine.truncated or not engine.stock


def _rigged(hand: list[str], pile: tuple[str, ...] = ("2d",), opening_points: int = 0) -> RummyEngine:
    """Seat 0 to draw with ``hand``; ``pile`` is bottom first (the last card is the top)."""

    engine = RummyEngine(2, 7, seed=0, decks=2, first_player=0, opening_points=opening_points)
    engine.hands[:] = 0
    for name in hand:
        engine.hands[0, parse_card(name)] += 1
    engine.hands[1, parse_card("9s")] = 7
    engine.pile = [parse_card(name) for name in pile]
    engine.pile_counts[:] = np.bincount(engine.pile, minlength=NUM_CARDS)
    return engine


def cards(*names: str) -> list[int]:
    return [parse_card(name) for name in names]


def test_opening_threshold_option_needs_thirty_points_across_melds():
    engine = _rigged(["8c", "9c", "Tc", "5d", "5h", "5s", "Kd"], opening_points=30)
    engine.apply(Action(TAKE, parse_card("2d"), 0))
    melds = [a for a in engine.candidates() if a.kind in (MELD_GROUP, MELD_RUN)]
    assert {a.meld for a in melds} == {meld_index(cards("8c", "9c", "Tc")), meld_index(cards("5d", "5h", "5s"))}
    engine.apply(next(a for a in melds if a.kind == MELD_GROUP))  # 15 points: not yet open
    assert not engine.opened[0] and engine.turn_points == 15
    assert all(a.kind == MELD_RUN for a in engine.candidates())  # must finish the opening
    engine.apply(engine.candidates()[0])
    assert engine.opened[0] and engine.turn_points == 0
    assert DISCARD in {a.kind for a in engine.candidates()}


def test_melds_and_lay_offs_score_for_the_seat_and_the_hand_against_it():
    engine = _rigged(["8c", "9c", "Tc", "5d", "5h", "5s", "Kd"], pile=("Jc",))
    assert engine.opened[0], "no opening threshold by default"
    engine.apply(Action(TAKE, parse_card("Jc"), 0))
    engine.apply(Action(MELD_RUN, meld=meld_index(cards("8c", "9c", "Tc"))))
    engine.apply(next(a for a in engine.candidates() if a.kind == LAYOFF_RUN and a.card == parse_card("Jc")))
    engine.apply(Action(MELD_GROUP, meld=meld_index(cards("5d", "5h", "5s"))))
    assert [a.kind for a in engine.candidates()] == [DISCARD], "the last card cannot be laid off"
    engine.apply(Action(DISCARD, parse_card("Kd")))
    assert engine.winner == 0 and engine.done
    # 8-9-T (27) + the Jc lay-off (10) + 5-5-5 (15); seat 1 holds seven nines.
    assert engine.scores().tolist() == [52.0, -63.0]
    assert engine.placed[0].sum() == 7


def test_deep_pickup_takes_every_card_above_and_must_use_the_deepest():
    engine = _rigged(["8c", "9c", "5d", "5h", "Kd", "3s", "4s"], pile=("Qh", "7c", "2s", "9d"))
    takes = {(a.card, a.meld) for a in engine.candidates() if a.kind == TAKE}
    # The top is always offered; 2s (2-3-4s) and 7c (7-8-9c) can be used; Qh cannot.
    assert takes == {(parse_card("9d"), 0), (parse_card("2s"), 1), (parse_card("7c"), 2)}
    with pytest.raises(ValueError):
        engine.apply(Action(TAKE, parse_card("Qh"), 2))
    event = engine.apply(Action(TAKE, parse_card("7c"), 2))
    assert event.card == parse_card("7c") and event.cards == tuple(cards("9d", "2s", "7c"))
    assert engine.pile == cards("Qh") and engine.obligation == parse_card("7c")
    assert all(engine.known[0, card] == 1 for card in cards("9d", "2s", "7c"))
    candidates = engine.candidates()
    assert candidates and all(parse_card("7c") in MELD_CARDS[a.meld] for a in candidates)
    with pytest.raises(ValueError):
        engine.apply(Action(DISCARD, parse_card("Kd")))
    engine.apply(Action(MELD_RUN, meld=meld_index(cards("7c", "8c", "9c"))))
    assert engine.obligation == NO_CARD
    assert DISCARD in {a.kind for a in engine.candidates()}


def test_a_deep_card_may_end_a_chain_of_lay_offs():
    # 2-3-4h on the table, 5h in hand: the 6h deep in the pile is usable (lay off 5h, then 6h).
    engine = _rigged(["5h", "Kd", "8c"], pile=("Qs", "6h", "Jc", "9d"))
    run = meld_index(cards("2h", "3h", "4h"))
    engine.table = [run]
    engine.table_counts[cards("2h", "3h", "4h")] += 1
    engine.placed[1, cards("2h", "3h", "4h")] += 1
    engine._refresh_layable()
    takes = {(a.card, a.meld) for a in engine.candidates() if a.kind == TAKE}
    assert (parse_card("6h"), 2) in takes and (parse_card("Qs"), 3) not in takes
    engine.apply(Action(TAKE, parse_card("6h"), 2))
    # Only the chain's first lay-off (5h on the run) is offered; nothing else, no discard.
    assert engine.candidates() == [Action(LAYOFF_RUN, parse_card("5h"), 0)]
    engine.apply(engine.candidates()[0])
    assert engine.obligation == parse_card("6h")
    assert engine.candidates() == [Action(LAYOFF_RUN, parse_card("6h"), 0)]
    engine.apply(engine.candidates()[0])
    assert engine.obligation == NO_CARD and engine.table == [meld_index(cards("2h", "3h", "4h", "5h", "6h"))]
    assert DISCARD in {a.kind for a in engine.candidates()}


def test_a_lay_off_chain_must_leave_a_card_to_discard():
    run = meld_index(cards("2h", "3h", "4h"))
    # Taking 7h brings 6h along: 5h, 6h, 7h would all go on the run and leave nothing to discard.
    engine = _rigged(["5h"], pile=("Qs", "7h", "6h"))
    engine.table = [run]
    engine._refresh_layable()
    assert (parse_card("7h"), 1) not in {(a.card, a.meld) for a in engine.candidates() if a.kind == TAKE}
    engine = _rigged(["5h", "Kd"], pile=("Qs", "7h", "6h"))
    engine.table = [run]
    engine._refresh_layable()
    assert (parse_card("7h"), 1) in {(a.card, a.meld) for a in engine.candidates() if a.kind == TAKE}


def test_opened_seats_meld_three_cards_and_lay_off_the_rest_one_at_a_time():
    engine = _rigged(["4h", "5h", "6h", "7h", "Kd"], pile=("2c",))
    engine.apply(Action(TAKE, parse_card("2c"), 0))
    melds = {a.meld for a in engine.candidates() if a.kind in (MELD_GROUP, MELD_RUN)}
    assert melds == {meld_index(cards("4h", "5h", "6h")), meld_index(cards("5h", "6h", "7h"))}
    engine.apply(Action(MELD_RUN, meld=meld_index(cards("4h", "5h", "6h"))))
    # The 7h is now its own decision: lay it off, or hold it by discarding.
    candidates = engine.candidates()
    assert Action(LAYOFF_RUN, parse_card("7h"), 0) in candidates
    assert DISCARD in {a.kind for a in candidates}
    engine.apply(Action(LAYOFF_RUN, parse_card("7h"), 0))
    assert engine.table == [meld_index(cards("4h", "5h", "6h", "7h"))]
    assert engine.points[0] == MELD_POINTS[meld_index(cards("4h", "5h", "6h", "7h"))]


def test_an_opening_still_offers_melds_of_every_size():
    engine = _rigged(["8c", "9c", "Tc", "Jc", "5d", "5h", "5s"], opening_points=30)
    engine.apply(Action(TAKE, parse_card("2d"), 0))
    sizes = {len(MELD_CARDS[a.meld]) for a in engine.candidates() if a.kind in (MELD_GROUP, MELD_RUN)}
    assert 4 in sizes


def test_the_top_card_may_be_kept_but_not_thrown_back():
    engine = _rigged(["8c", "9c", "5d", "5h", "Kd", "3s", "4s"], pile=("Qh", "7c", "2s", "9d"))
    engine.apply(Action(TAKE, parse_card("9d"), 0))
    assert engine.obligation == NO_CARD and engine.kept == parse_card("9d")
    discards = {a.card for a in engine.candidates() if a.kind == DISCARD}
    assert parse_card("Kd") in discards and parse_card("9d") not in discards
    with pytest.raises(ValueError):
        engine.apply(Action(DISCARD, parse_card("9d")))
    engine.apply(Action(DISCARD, parse_card("Kd")))
    assert parse_card("9d") in np.flatnonzero(engine.hands[0])


def test_the_top_card_is_not_offered_when_nothing_else_could_be_discarded():
    # Two decks: the only card in hand is the other copy of the top.
    engine = _rigged(["9d"], pile=("Qh", "9d"))
    assert {a.kind for a in engine.candidates()} == {STOCK}
    engine = _rigged(["9d", "Kd"], pile=("Qh", "9d"))
    assert TAKE in {a.kind for a in engine.candidates()}


def test_melds_and_lay_offs_leave_a_card_other_than_the_kept_top():
    engine = _rigged(["8c", "8d", "8h"], pile=("Qh", "7c"))
    engine.apply(Action(TAKE, parse_card("7c"), 0))
    # 8c-8d-8h would leave only the kept 7c, which cannot be discarded: not offered.
    candidates = engine.candidates()
    assert all(a.kind == DISCARD for a in candidates)
    assert {a.card for a in candidates} == set(cards("8c", "8d", "8h"))
    engine = _rigged(["8c", "8d", "8h", "Kd"], pile=("Qh", "7c"))
    engine.apply(Action(TAKE, parse_card("7c"), 0))
    engine.apply(Action(MELD_GROUP, meld=meld_index(cards("8c", "8d", "8h"))))
    assert {a.card for a in engine.candidates() if a.kind == DISCARD} == {parse_card("Kd")}


def test_random_policies_cover_every_action_kind():
    seen = set()
    deep = 0
    for seed in range(30):
        rng = random.Random(seed)
        engine = RummyEngine(rng.choice([2, 3, 4]), 7, seed, decks=rng.choice([1, 2]))
        policy = HeuristicPolicy() if seed % 2 else RandomPolicy(seed)
        while not engine.done:
            action = engine.candidates()[policy.act(engine)]
            seen.add(action.kind)
            deep += action.kind == TAKE and action.meld > 0
            engine.apply(action)
    assert seen == {STOCK, TAKE, MELD_GROUP, MELD_RUN, LAYOFF_GROUP, LAYOFF_RUN, DISCARD}
    assert deep > 0
