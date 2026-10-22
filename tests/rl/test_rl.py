from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from cardgym.core.config import parse_section
from cardgym.rl.batch import RowIndex, bucket_length, build_batch
from cardgym.rl.config import PPOConfig
from cardgym.rl.entropy import ALPHA_MIN, EntropyController
from cardgym.rl.ppo import (
    center_zero_sum,
    gae_advantages,
    masked_probabilities,
    ppo_terms,
    relative_rewards,
    sample_rows,
)
from cardgym.rl.selfplay import SelfPlayCollector
from tests.conftest import tiny_plump_game


def test_ppo_terms_start_at_ratio_one_and_clip():
    logits = torch.tensor([[1.0, 2.0, 0.5], [0.0, 0.0, 0.0]])
    old = torch.softmax(logits.masked_fill(torch.tensor([[False, False, True], [False, False, False]]), -1e9), -1)
    old[0, 2] = 0.0
    terms = ppo_terms(logits, old, torch.tensor([1, 0]), torch.tensor([1.0, -1.0]), clip_ratio=0.2)
    torch.testing.assert_close(terms.ratios, torch.ones(2))
    torch.testing.assert_close(terms.divergences, torch.zeros(2), atol=1e-6, rtol=0)
    torch.testing.assert_close(terms.normalized_entropy[1], torch.tensor(1.0))
    shifted = logits.clone()
    shifted[0, 1] += 5.0  # ratio far above 1 + clip with a positive advantage
    clipped = ppo_terms(shifted, old, torch.tensor([1, 0]), torch.tensor([1.0, -1.0]), clip_ratio=0.2)
    assert clipped.clipped[0] and clipped.losses[0].item() == pytest.approx(-1.2)


def test_masked_sampling_never_picks_illegal_actions():
    rng = np.random.default_rng(0)
    mask = rng.random((200, 9)) < 0.4
    mask[np.arange(200), rng.integers(0, 9, 200)] = True
    probabilities = masked_probabilities(rng.normal(size=(200, 9)), mask)
    np.testing.assert_allclose(probabilities.sum(1), 1.0)
    for uniforms in (np.zeros(200), np.ones(200), rng.random(200)):
        sampled = sample_rows(probabilities, mask, uniforms)
        assert mask[np.arange(200), sampled].all()


def test_zero_sum_projection_and_relative_rewards():
    values = torch.tensor([[1.0, 2.0, 6.0, 9.0], [1.0, -1.0, 4.0, 0.0]])
    centered = center_zero_sum(values, torch.tensor([3, 2]))
    torch.testing.assert_close(centered, torch.tensor([[-2.0, -1.0, 3.0, 0.0], [1.0, -1.0, 0.0, 0.0]]))
    rewards = relative_rewards(np.array([13.0, 5.0, 0.0]))
    assert rewards.sum() == pytest.approx(0.0) and rewards[0] == pytest.approx(10.5)


def test_gae_lambda_one_is_monte_carlo_and_zero_is_one_step():
    values = np.array([[1.0, -1.0], [3.0, -3.0], [2.0, -2.0]])
    returns = np.array([5.0, -5.0])
    np.testing.assert_allclose(gae_advantages(values, returns, 1.0), returns - values)
    one_step = gae_advantages(values, returns, 0.0)
    np.testing.assert_allclose(one_step[0], values[1] - values[0])
    np.testing.assert_allclose(one_step[2], returns - values[2])


def test_entropy_controller_tracks_its_target():
    config = parse_section(PPOConfig, {"entropy_targets": {"bid": 0.5}, "entropy_learning_rate": 0.1}, "ppo")
    controller = EntropyController(("bid",), config, torch.device("cpu"))
    start = controller.alpha("bid")
    controller.update({"bid": 0.2}, iteration=1)
    assert controller.alpha("bid") > start
    controller.reset("bid")
    assert controller.alpha("bid") == pytest.approx(ALPHA_MIN)


def test_entropy_controller_resumes_with_an_added_head():
    config = parse_section(PPOConfig, {"entropy_targets": {"bid": 0.5}, "entropy_learning_rate": 0.1}, "ppo")
    old = EntropyController(("bid", "card"), config, torch.device("cpu"))
    old.update({"bid": 0.2}, iteration=1)
    new = EntropyController(("bid", "card", "gate"), config, torch.device("cpu"))
    new.load_state_dict(old.state_dict())
    assert new.alpha("bid") == pytest.approx(old.alpha("bid"))
    moments = new.optimizer.state[new.log_alpha["bid"]]["exp_avg"]
    assert torch.equal(moments, old.optimizer.state[old.log_alpha["bid"]]["exp_avg"])
    assert new.log_alpha["gate"] not in new.optimizer.state
    new.update({"bid": 0.2, "gate": 0.1}, iteration=2)  # the added head steps from fresh moments


def test_a_gated_head_measures_only_the_choice_within_a_side():
    from cardgym.rl.trainer import _within_side_entropy

    logits = torch.tensor([[1.0, 0.0, 2.0, -1.0, 0.5], [0.3, 0.0, 0.0, 0.0, 0.0]])
    legal = torch.tensor([[True, True, True, True, True], [True, True, True, False, False]])
    side = torch.tensor([[False, False, True, True, True], [False, False, False, True, True]])
    log_probs = torch.where(legal, torch.log_softmax(logits.masked_fill(~legal, float("-inf")), -1), 0.0)
    total, mass = _within_side_entropy(log_probs, legal, side)
    p = torch.softmax(logits[0], -1)
    plays, holds = p[:2], p[2:]
    within = lambda q: -(q / q.sum() * (q / q.sum()).log()).sum()  # noqa: E731
    expected = plays.sum() * within(plays) / np.log(2) + holds.sum() * within(holds) / np.log(3)
    assert total[0].item() == pytest.approx(expected.item(), rel=1e-5)
    assert mass[0].item() == pytest.approx(1.0, rel=1e-5)
    # Chain rule: the full entropy is the split's entropy plus the within-side part.
    split = -(plays.sum() * plays.sum().log() + holds.sum() * holds.sum().log())
    full = -(p * p.log()).sum()
    assert full.item() == pytest.approx(
        (split + plays.sum() * within(plays) + holds.sum() * within(holds)).item(), rel=1e-5
    )
    # One side legal: the plain normalized entropy.
    q = torch.softmax(logits[1, :3], -1)
    assert total[1].item() == pytest.approx((-(q * q.log()).sum() / np.log(3)).item(), rel=1e-5)
    assert mass[1].item() == pytest.approx(1.0)


def test_entropy_target_schedule_interpolates():
    config = parse_section(
        PPOConfig,
        {
            "entropy_targets": {"bid": 0.4},
            "entropy_targets_end": {"bid": 0.2},
            "entropy_target_schedule": "linear",
            "entropy_schedule_start_iteration": 10,
            "entropy_schedule_end_iteration": 20,
        },
        "ppo",
    )
    assert config.entropy_targets_at(5)["bid"] == pytest.approx(0.4)
    assert config.entropy_targets_at(15)["bid"] == pytest.approx(0.3)
    assert config.entropy_targets_at(99)["bid"] == pytest.approx(0.2)


def test_row_index_selects_microbatch_rows():
    group = np.array([0, 1, 0, 0, 1])
    seq = np.array([3, 0, 1, 5, 2])
    index = RowIndex(group, seq)
    assert sorted(index.select(0, 0, 4).tolist()) == [0, 2]
    assert sorted(index.select(1, 0, 10).tolist()) == [1, 4]
    assert bucket_length(17, 16, 121) == 32 and bucket_length(120, 16, 121) == 121


@pytest.fixture(scope="module")
def collected():
    game = tiny_plump_game()
    actor = game.build_actor().eval()
    collector = SelfPlayCollector(
        actor, device=torch.device("cpu"), precision="fp32", kv_dtype="fp32",
        max_rows=17, max_length=game.max_length,  # forces several merged loops
    )
    trajectories = collector.collect(game.new_games(random.Random(3), 1))
    return game, actor, trajectories


def test_batch_weights_give_every_game_equal_mass(collected):
    game, _, trajectories = collected
    batch = build_batch(
        trajectories,
        [game.sequences(t) for t in trajectories],
        max_players=game.max_players,
        actor_labels=game.actor_labels,
        critic_labels=game.critic_labels,
        bucket_width=16,
        critic_bucket_width=32,
        max_length=game.max_length,
        critic_max_length=game.critic_max_length,
    )
    games = len(trajectories)
    dense_actor = sum(g.position_weight.sum() for g in batch.actor_groups)
    dense_critic = sum(g.position_weight.sum() for g in batch.critic_groups)
    assert dense_actor == pytest.approx(1.0) and dense_critic == pytest.approx(1.0)
    per_game = np.zeros(games)
    for rows in batch.decisions.values():
        np.add.at(per_game, rows.game, rows.weight)
    decisions = np.array([len(t.decisions) for t in trajectories])
    players = np.array([t.game.num_players for t in trajectories])
    np.testing.assert_allclose(per_game, decisions / players / games, rtol=1e-6)
    for head, rows in batch.decisions.items():
        assert (rows.probs[np.arange(len(rows)), rows.action] > 0).all(), head
        np.testing.assert_allclose(rows.probs.sum(1), 1.0, rtol=1e-5)


def test_small_buckets_merge_upward_and_microbatches_balance():
    from cardgym.rl.batch import merge_small_buckets, microbatches

    buckets = {32: list(range(40)), 64: [40, 41], 96: [42], 128: [43, 44]}
    merged = merge_small_buckets(buckets, 4)
    # The three long buckets fold into the longest; nothing is ever truncated.
    assert merged == {128: [43, 44, 42, 40, 41], 32: list(range(40))}
    assert merge_small_buckets({32: [0], 64: list(range(1, 9))}, 4) == {64: list(range(1, 9)) + [0]}
    assert merge_small_buckets(buckets, 0) == buckets

    class Group:
        def __init__(self, rows, length):
            self.tokens = np.zeros((rows, length, 1))
            self.length = length

    assert microbatches(Group(20, 400), 7600) == [(0, 10), (10, 20)]
    assert microbatches(Group(5, 10), 1000) == [(0, 5)]
    spans = microbatches(Group(101, 64), 640)
    assert max(b - a for a, b in spans) <= 10 and min(b - a for a, b in spans) >= 9
    # A row slice (one optimizer minibatch) splits only its own rows.
    assert microbatches(Group(20, 400), 7600, rows=(10, 20)) == [(10, 20)]
    assert microbatches(Group(3, 400), 7600, rows=(1, 1)) == []
    halves = [microbatches(Group(101, 64), 640, rows=(a, b)) for a, b in ((0, 50), (50, 101))]
    assert halves[0][0][0] == 0 and halves[0][-1][1] == 50 and halves[1][0][0] == 50 and halves[1][-1][1] == 101
    group = Group(7, 64)
    group.lengths = np.array([10, 30, 5, 40, 20, 25, 15])
    assert microbatches(group, 50, packed=True, rows=(2, 7)) == [(2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]
    assert microbatches(group, 80, packed=True, rows=(0, 3)) == [(0, 3)]


def test_clip_grad_norm_matches_torch():
    from cardgym.rl.trainer import clip_grad_norm

    generator = torch.Generator().manual_seed(0)
    for max_norm in (1e-3, 1e6):
        mine = [torch.randn(5, 7, generator=generator).requires_grad_() for _ in range(3)]
        theirs = [p.detach().clone().requires_grad_() for p in mine]
        for a, b in zip(mine, theirs):
            a.grad = torch.randn(a.shape, generator=generator)
            b.grad = a.grad.clone()
        mine[1].grad = None
        theirs[1].grad = None
        norm = clip_grad_norm(mine, max_norm)
        expected = float(torch.nn.utils.clip_grad_norm_(theirs, max_norm))
        assert norm == pytest.approx(expected, rel=1e-5)
        for a, b in zip(mine, theirs):
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad)
