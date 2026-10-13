"""End-to-end: train, resume, reconfigure, export, fork, evaluate, play."""

from __future__ import annotations

import csv
import json

import pytest

from cardgym.cli import main
from cardgym.core.config import apply_overrides, load_toml
from cardgym.games.plump.evaluation import evaluate
from cardgym.games.plump.gui.app import GuiController
from cardgym.games.plump.heuristic import HeuristicPolicy
from cardgym.games.plump.policy import PlumpPolicy
from cardgym.rl.setup import load_config, normalized_raw
from tests.conftest import ROOT

SMOKE = str(ROOT / "configs" / "plump" / "smoke.toml")


def rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def test_presets_resolve():
    for name in ("smoke", "rl-v6"):
        config = load_config(ROOT / "configs" / "plump" / f"{name}.toml")
        normalized = normalized_raw(config)
        assert set(normalized) == {"run", "rollout", "ppo", "model", "game", "losses"}
    production = load_config(ROOT / "configs" / "plump" / "rl-v6.toml")
    release = json.loads((ROOT / "checkpoints" / "plump" / "rl-v6" / "info.json").read_text())
    assert release["game"] == production.run.game
    with pytest.raises(ValueError, match="Unknown keys"):
        from cardgym.rl.setup import resolve_config

        resolve_config(apply_overrides(load_toml(SMOKE), []) | {"ppo": {"learning_rat": 1.0}})


def test_train_resume_reconfigure_export_and_fork(runs_dir, tmp_path, capsys):
    assert main(["plump", "train", "smoke", "--config", SMOKE, "--iterations", "2"]) == 0
    run = runs_dir / "plump" / "smoke"
    history = rows(run / "metrics.csv")
    assert [int(r["iteration"]) for r in history] == [1, 2]
    assert abs(float(history[-1]["behavior_kl"])) < 1e-6  # cached rollout == replay
    assert float(history[-1]["games"]) == 12

    with pytest.raises(ValueError, match="differs from the recorded config"):
        main(["plump", "train", "smoke", "--config", SMOKE, "--iterations", "3"])
    assert main(
        ["plump", "train", "smoke", "--config", SMOKE, "--iterations", "3",
         "--reconfigure", "--reason", "one more", "--reset-entropy", "play"]
    ) == 0
    assert [int(r["iteration"]) for r in rows(run / "metrics.csv")] == [1, 2, 3]
    assert (run / "checkpoints" / "resume_000002_reconfigured.pt").is_file()
    assert (run / "config-history" / "iter_000002.toml").is_file()

    # Rewinding to an interval checkpoint drops the newer metric rows.
    assert main(
        ["plump", "train", "smoke", "--config", SMOKE, "--iterations", "3", "--resume-checkpoint", "1",
         "--prepare-only"]
    ) == 0
    assert [int(r["iteration"]) for r in rows(run / "metrics.csv")] == [1]

    release = tmp_path / "release"
    assert main(["plump", "export", "smoke", "--out", str(release)]) == 0
    assert {p.name for p in release.iterdir()} == {"actor.pt", "critic.pt", "info.json"}
    assert main(["plump", "train", "fork", "--config", SMOKE, "--init-from", str(release), "--reset-iteration",
                 "--iterations", "1"]) == 0
    assert [int(r["iteration"]) for r in rows(runs_dir / "plump" / "fork" / "metrics.csv")] == [1]
    with pytest.raises(ValueError, match="does not match"):
        main(["plump", "train", "wider", "--config", SMOKE, "--init-from", str(release), "--set", "model.d_model=48"])

    assert main(["plump", "dashboard", "smoke"]) == 0
    assert (run / "dashboard.png").is_file()
    assert main(["games"]) == 0
    assert "plump" in capsys.readouterr().out


def test_evaluation_plays_every_hand_from_every_position(runs_dir, tmp_path):
    assert main(["plump", "train", "tiny", "--config", SMOKE, "--iterations", "1"]) == 0
    model = PlumpPolicy.load(runs_dir / "plump" / "tiny" / "checkpoints" / "iter_000001.pt")
    report = evaluate(model, HeuristicPolicy(), player_counts=(3,), hand_sizes=(3, 4), deals=2, seed=5)
    assert report["rounds"] == 2 * 2 * 3 * 3
    low, high = report["relative_reward_ci95"]
    assert low <= report["relative_reward"] <= high
    again = evaluate(model, HeuristicPolicy(), player_counts=(3,), hand_sizes=(3, 4), deals=2, seed=5)
    assert again["relative_reward"] == report["relative_reward"]


def test_gui_round_against_random_bots():
    controller = GuiController(None)
    view = controller.new_game(opponents=2, hand_size=3, human_bid_position=1, seed=3)
    assert view["human_turn"] and view["phase"] == "bidding"
    view = controller.bid(view["legal_bids"][0])
    while not view["done"]:
        if view["human_turn"]:
            card = next(card for card in view["my_hand"] if card["legal"])
            view = controller.play(card["suit"], card["rank"])
        else:
            view = controller.advance_bot()
    assert view["game_over"] and len(view["completed_rounds"]) == 1
