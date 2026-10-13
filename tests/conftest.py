from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch

from cardgym.games.plump.game import PlumpGame, PlumpGameConfig
from cardgym.games.plump.model import PlumpModelConfig
from cardgym.games.plump.training import PlumpLossConfig

ROOT = Path(__file__).resolve().parents[1]


def tiny_plump_model(**overrides) -> PlumpModelConfig:
    fields = dict(d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=64, per_layer_dim=8)
    fields.update(overrides)
    return PlumpModelConfig(**fields)


def tiny_plump_game(**game) -> PlumpGame:
    settings = dict(hand_sizes=(3, 5, 10), player_counts=(3, 5), deals_per_shape=2)
    settings.update(game)
    return PlumpGame(tiny_plump_model(), PlumpGameConfig(**settings), PlumpLossConfig())


def randomize(model: torch.nn.Module, seed: int = 0, scale: float = 0.05) -> torch.nn.Module:
    """Give zero-initialized conditioning tables real signal for equivalence tests."""

    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(scale * torch.randn(parameter.shape, generator=generator))
    return model


@pytest.fixture
def runs_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("CARDGYM_RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path / "runs"


@pytest.fixture(autouse=True)
def _seed():
    random.seed(0)
    torch.manual_seed(0)
