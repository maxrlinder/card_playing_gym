"""Growing a trained trunk: fresh blocks start as identities, training state follows its parameters."""

from __future__ import annotations

import csv

import numpy as np
import pytest
import torch

from cardgym.cli import main
from cardgym.games.plump.game import PlumpGame
from cardgym.games.plump.training import PlumpLossConfig
from cardgym.games.rummy.game import RummyGame, RummyGameConfig
from cardgym.games.rummy.model import RummyModelConfig
from cardgym.games.rummy.tokens import FIELD_VOCAB, LEDGER_VOCAB
from cardgym.games.rummy.training import RummyLossConfig
from cardgym.rl.checkpoint import load_checkpoint
from cardgym.rl.grow import block_positions, grow_trainer_state
from tests.conftest import ROOT, randomize, tiny_plump_game, tiny_plump_model
from tests.nn.test_trunk import streams

SMOKE = ROOT / "configs" / "rummy" / "smoke.toml"


def rummy_game(layers: int) -> RummyGame:
    model = RummyModelConfig(
        d_model=32, n_layers=layers, n_heads=4, n_kv_heads=2, d_ff=64, per_layer_dim=8,
        ledger_width=8, ledger_hidden=16, ledger_layers=1, score_dim=16,
    )
    return RummyGame(model, RummyGameConfig(), RummyLossConfig())


def rummy_tokens(batch: int, length: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    columns = [torch.randint(0, vocab, (batch, length, 1), generator=generator) for vocab in FIELD_VOCAB]
    columns.append(torch.randint(0, LEDGER_VOCAB, (batch, length, 52), generator=generator))
    return torch.cat(columns, dim=-1)


def plump_tokens() -> torch.Tensor:
    batch = streams(4)
    padded = np.zeros((len(batch), max(s.shape[0] for s in batch), batch[0].shape[1]), dtype=np.int64)
    for row, stream in enumerate(batch):
        padded[row, : stream.shape[0]] = stream
    return torch.from_numpy(padded)


def trained_state(old_game, models) -> dict:
    """A trainer state with real signal everywhere (including zero-initialized tables) and Adam moments."""

    state = {}
    for role, model in models.items():
        randomize(model)
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        for parameter in model.parameters():
            parameter.grad = torch.randn_like(parameter)
        optimizer.step()
        state[role] = model.state_dict()
        state[f"{role}_optimizer"] = optimizer.state_dict()
    return state


@pytest.mark.parametrize("kind", ["rummy", "plump"])
def test_grown_models_compute_the_same_function(kind):
    if kind == "rummy":
        old_game, new_game, tokens = rummy_game(4), rummy_game(6), rummy_tokens(3, 9)
    else:
        old_game = tiny_plump_game()
        new_game = PlumpGame(
            tiny_plump_model(n_layers=4), old_game.game_config, PlumpLossConfig()
        )
        tokens = plump_tokens()
    old = {"actor": old_game.build_actor(), "critic": old_game.build_critic()}
    trainer = trained_state(old_game, old)
    old_layers = old_game.model_config.n_layers
    fresh = [2, 3]
    grown = grow_trainer_state(trainer, old_game, new_game, fresh)
    positions = block_positions(old_layers, new_game.model_config.n_layers, fresh)
    for role, build in (("actor", new_game.build_actor), ("critic", new_game.build_critic)):
        model = build()
        model.load_state_dict(grown[role])
        with torch.no_grad():
            torch.testing.assert_close(
                model.forward_hidden(tokens), old[role].eval().forward_hidden(tokens), rtol=1e-5, atol=1e-5
            )
        # Fresh blocks are random, not copies, except for their zero residual outputs.
        for position in fresh:
            block = model.trunk.blocks[position]
            assert block.qkv_proj.weight.abs().sum() > 0
            assert block.out_proj.weight.abs().sum() == 0 and block.mlp[2].weight.abs().sum() == 0
        # Adam moments follow their parameters; fresh parameters have no state.
        names = [name for name, p in model.named_parameters() if p.requires_grad]
        moments = grown[f"{role}_optimizer"]["state"]
        old_names = [name for name, p in old[role].named_parameters() if p.requires_grad]
        old_moments = trainer[f"{role}_optimizer"]["state"]
        source = old_names.index("trunk.blocks.1.mlp.0.weight")
        target = names.index(f"trunk.blocks.{positions[1]}.mlp.0.weight")
        torch.testing.assert_close(moments[target]["exp_avg"], old_moments[source]["exp_avg"])
        assert names.index(f"trunk.blocks.{fresh[0]}.qkv_proj.weight") not in moments
        ple = names.index("trunk.ple_context.weight")
        assert moments[ple]["exp_avg"].shape == model.trunk.ple_context.weight.shape


def test_value_embedding_membership_is_kept():
    old_game, new_game = rummy_game(4), rummy_game(6)
    trainer = trained_state(old_game, {"actor": old_game.build_actor(), "critic": old_game.build_critic()})
    with pytest.raises(ValueError, match="value embedding"):
        grow_trainer_state(trainer, old_game, new_game, [1, 4])  # a lone fresh block shifts parity
    with pytest.raises(ValueError, match="fresh positions"):
        grow_trainer_state(trainer, old_game, new_game, [2])


def test_grow_command_forks_a_run_that_resumes(runs_dir, tmp_path):
    assert main(["rummy", "train", "small", "--config", str(SMOKE), "--iterations", "2"]) == 0
    deeper = tmp_path / "deeper.toml"
    deeper.write_text(SMOKE.read_text().replace("n_layers = 2", "n_layers = 4"))
    assert main(
        ["rummy", "grow", "deep", "--from", "small", "--config", str(deeper), "--fresh", "2,3",
         "--set", "run.iterations=3"]
    ) == 0
    run = runs_dir / "rummy" / "deep"
    payload = load_checkpoint(run / "checkpoints" / "iter_000002.pt")
    assert payload["iteration"] == 2 and payload["config"]["model"]["n_layers"] == 4
    assert payload["trainer"]["optimizer_steps"] > 0
    assert main(["rummy", "train", "deep", "--config", str(deeper), "--set", "run.iterations=3"]) == 0
    with open(run / "metrics.csv", newline="") as handle:
        history = list(csv.DictReader(handle))
    assert [int(row["iteration"]) for row in history] == [3]
    # Only n_layers may change.
    wider = tmp_path / "wider.toml"
    wider.write_text(deeper.read_text().replace("d_ff = 64", "d_ff = 128"))
    with pytest.raises(ValueError, match="only in n_layers"):
        main(["rummy", "grow", "wide", "--from", "small", "--config", str(wider), "--fresh", "2,3"])

