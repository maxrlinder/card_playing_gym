from __future__ import annotations

import random

import numpy as np
import pytest

from cardgym.games.rummy.cards import COPIES, NUM_CARDS, parse_card
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
from cardgym.games.rummy.melds import MELD_CARDS, meld_index
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
@pytest.mark.parametrize("policy", ["heuristic", "random"])
def test_rounds_conserve_cards_and_score_zero_sum(players, policy):
    for seed in range(6):
        engine = RummyEngine(players, 13, seed, max_turns=120)
        play_out(engine, HeuristicPolicy() if policy == "heuristic" else RandomPolicy(seed))
        assert engine.phase == DONE
        scores = engine.scores()
        penalties = engine.penalties()
        if engine.winner is not None:
            factor = 2 if engine.rummy else 1
            assert scores[engine.winner] == factor * penalties.sum()
            assert penalties[engine.winner] == 0
        else:
            assert np.array_equal(scores, -penalties)
        assert abs(relative_rewards(scores).sum()) < 1e-9


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


def _rigged(hand: list[str], top: str = "2d") -> RummyEngine:
    engine = RummyEngine(2, 7, seed=0, first_player=0)
    engine.hands[:] = 0
    for name in hand:
        engine.hands[0, parse_card(name)] += 1
    engine.hands[1, parse_card("9s")] = 7
    engine.pile = [parse_card(top)]
    engine.pile_counts[:] = 0
    engine.pile_counts[parse_card(top)] = 1
    return engine


def test_opening_needs_thirty_points_across_melds():
    engine = _rigged(["8c", "9c", "Tc", "5d", "5h", "5s", "Kd"], top="2d")
    engine.apply(Action(TAKE, parse_card("2d")))
    melds = [a for a in engine.candidates() if a.kind in (MELD_GROUP, MELD_RUN)]
    assert {a.meld for a in melds} == {meld_index([parse_card(c) for c in ("8c", "9c", "Tc")]),
                                       meld_index([parse_card(c) for c in ("5d", "5h", "5s")])}
    engine.apply(next(a for a in melds if a.kind == MELD_GROUP))  # 15 points: not yet open
    assert not engine.opened[0] and engine.turn_points == 15
    assert all(a.kind == MELD_RUN for a in engine.candidates())  # must finish the opening
    engine.apply(engine.candidates()[0])
    assert engine.opened[0] and engine.turn_points == 0
    kinds = {a.kind for a in engine.candidates()}
    assert DISCARD in kinds


def test_going_rummy_doubles_the_score():
    engine = _rigged(["8c", "9c", "Tc", "5d", "5h", "5s", "Kd"], top="Jc")
    engine.apply(Action(TAKE, parse_card("Jc")))
    while engine.phase == PLAY:
        candidates = engine.candidates()
        # Meld, then lay off, then discard: the Kd is the last card.
        engine.apply(min(candidates, key=lambda a: (a.kind == DISCARD, a.kind in (LAYOFF_GROUP, LAYOFF_RUN))))
    assert engine.winner == 0 and engine.rummy
    assert engine.scores()[0] == 2 * 63 and engine.scores()[1] == -2 * 63


def test_lay_offs_after_opening():
    engine = _rigged(["8c", "9c", "Tc", "Jc", "5d", "5h", "5s"], top="5c")
    engine.apply(Action(TAKE, parse_card("5c")))
    run = meld_index([parse_card(c) for c in ("8c", "9c", "Tc", "Jc")])
    engine.apply(Action(MELD_RUN, meld=run))  # 37 points: open
    group = meld_index([parse_card(c) for c in ("5d", "5h", "5s")])
    engine.apply(Action(MELD_GROUP, meld=group))
    layoffs = [a for a in engine.candidates() if a.kind == LAYOFF_GROUP]
    assert layoffs == [], "the last card cannot be laid off"
    assert [a.kind for a in engine.candidates()] == [DISCARD]
    assert engine.layable[parse_card("5c")] and engine.layable[parse_card("7c")]


def test_random_policies_cover_every_action_kind():
    seen = set()
    for seed in range(30):
        engine = RummyEngine(random.Random(seed).choice([2, 3, 4]), 13, seed)
        policy = HeuristicPolicy() if seed % 2 else RandomPolicy(seed)
        while not engine.done:
            action = engine.candidates()[policy.act(engine)]
            seen.add(action.kind)
            engine.apply(action)
    assert seen == {STOCK, TAKE, MELD_GROUP, MELD_RUN, LAYOFF_GROUP, LAYOFF_RUN, DISCARD}
