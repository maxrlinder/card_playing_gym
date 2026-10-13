"""The published rl-v6 release reproduces the original training code's outputs.

``tests/fixtures/plump_rl_v6_golden.npz`` holds token streams and the logits
and values the original plump-bot implementation computed for them from the
same checkpoint the release was converted from. Agreement is to floating-point
summation order (the embedding sums are grouped differently), not bit-for-bit.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from cardgym.core.config import parse_section
from cardgym.games.plump.model import PlumpCritic, PlumpModelConfig
from cardgym.games.plump.policy import PlumpPolicy
from cardgym.rl.checkpoint import load_weights
from tests.conftest import ROOT

RELEASE = ROOT / "checkpoints" / "plump" / "rl-v6"
GOLDEN = ROOT / "tests" / "fixtures" / "plump_rl_v6_golden.npz"


@pytest.fixture(scope="module")
def golden():
    return np.load(GOLDEN)


def test_release_metadata_matches_the_golden_iteration(golden):
    info = json.loads((RELEASE / "info.json").read_text())
    assert info["game"] == "plump"
    assert info["iteration"] == int(golden["iteration"])


def test_actor_reproduces_the_original_logits(golden):
    policy = PlumpPolicy.load(RELEASE)
    tokens = torch.from_numpy(golden["tokens"].astype(np.int64))
    with torch.no_grad():
        hidden = policy.actor.forward_hidden(tokens)
        bid = policy.actor.policy_logits(hidden, "bid")
        card = policy.actor.policy_logits(hidden, "play")
        values = policy.actor.player_values(hidden)
    for row, length in enumerate(golden["lengths"]):
        np.testing.assert_allclose(bid[row, :length].numpy(), golden["bid_logits"][row, :length], atol=2e-4)
        np.testing.assert_allclose(card[row, :length].numpy(), golden["card_logits"][row, :length], atol=2e-4)
        np.testing.assert_allclose(values[row, :length].numpy(), golden["actor_values"][row, :length], atol=2e-4)


def test_critic_reproduces_the_original_values(golden):
    weights = load_weights(RELEASE, "plump")
    critic = PlumpCritic(parse_section(PlumpModelConfig, weights["model"], "model"))
    critic.load_state_dict(weights["critic"])
    critic.eval()
    tokens = torch.from_numpy(golden["oracle"].astype(np.int64))
    with torch.no_grad():
        values = critic.player_values(critic.forward_hidden(tokens))
    for row, length in enumerate(golden["oracle_lengths"]):
        np.testing.assert_allclose(values[row, :length].numpy(), golden["critic_values"][row, :length], atol=5e-4)
