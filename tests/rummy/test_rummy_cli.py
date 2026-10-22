"""End-to-end: train the smoke preset, export a release, evaluate it."""

from __future__ import annotations

import csv
import json

from cardgym.cli import main
from cardgym.rl.setup import load_config, normalized_raw
from tests.conftest import ROOT

SMOKE = str(ROOT / "configs" / "rummy" / "smoke.toml")


def test_presets_resolve():
    for name in ("smoke", "base", "deep"):
        config = load_config(ROOT / "configs" / "rummy" / f"{name}.toml")
        assert set(normalized_raw(config)) == {"run", "rollout", "ppo", "model", "game", "losses"}
        assert config.run.game == "rummy"


def test_train_export_evaluate(runs_dir, tmp_path, capsys):
    assert main(["rummy", "train", "smoke", "--config", SMOKE, "--iterations", "2"]) == 0
    run = runs_dir / "rummy" / "smoke"
    with open(run / "metrics.csv", newline="") as handle:
        history = list(csv.DictReader(handle))
    assert [int(row["iteration"]) for row in history] == [1, 2]
    # Cached rollout == replay: exact while every decision is fresh (iteration 1);
    # later updates also train on decisions carried over from older weights.
    assert abs(float(history[0]["behavior_kl"])) < 1e-6
    assert float(history[-1]["policy_lag"]) >= 0
    assert float(history[-1]["hands_accuracy"]) > 0
    # The hold gate: an entropy in bits, a probability, and a sampled rate.
    assert 0 < float(history[-1]["entropy_hold"]) <= 1
    assert 0 < float(history[-1]["hold_probability"]) < 1
    assert float(history[-1]["entropy_target_hold"]) == 0.5
    assert 0 <= float(history[-1]["hold_rate"]) <= 1

    release = tmp_path / "release"
    assert main(["rummy", "export", "smoke", "--out", str(release)]) == 0
    report = tmp_path / "report.json"
    assert main(
        ["rummy", "evaluate", str(release), "--opponent", "random", "--players", "2,3",
         "--hand-size", "7", "--deals", "2", "--output", str(report)]
    ) == 0
    result = json.loads(report.read_text())
    assert set(result["by_players"]) == {"2", "3"}
    assert result["by_players"]["3"]["model_alone"]["games"] == 2 * 2 * 3  # deals x decks x lone seats
    assert "relative reward" in capsys.readouterr().out


def test_asynchronous_rollout(runs_dir, tmp_path):
    """The rollout process collects the next batch during the update: one more version of lag."""

    config = tmp_path / "async.toml"
    config.write_text(
        (ROOT / "configs" / "rummy" / "smoke.toml").read_text().replace(
            'mode = "continuous"\n', 'mode = "continuous"\nasynchronous = true\n'
        )
    )
    assert main(["rummy", "train", "async", "--config", str(config), "--iterations", "3"]) == 0
    with open(runs_dir / "rummy" / "async" / "metrics.csv", newline="") as handle:
        history = list(csv.DictReader(handle))
    assert [int(row["iteration"]) for row in history] == [1, 2, 3]
    assert float(history[0]["policy_lag"]) == 0  # primed with the initial weights
    assert all(float(row["policy_lag"]) >= 1 for row in history[1:])
    assert all(float(row["rollout_sec"]) > 0 for row in history)


def test_rollout_workers_share_each_lane(runs_dir, tmp_path):
    """Two rollout processes, each with its share of every lane, deliver one update's games together."""

    config = tmp_path / "workers.toml"
    config.write_text(
        (ROOT / "configs" / "rummy" / "smoke.toml").read_text().replace(
            'mode = "continuous"\n', 'mode = "continuous"\nasynchronous = true\nworkers = 2\n'
        )
    )
    assert main(["rummy", "train", "workers", "--config", str(config), "--iterations", "3"]) == 0
    with open(runs_dir / "rummy" / "workers" / "metrics.csv", newline="") as handle:
        history = list(csv.DictReader(handle))
    assert [int(row["iteration"]) for row in history] == [1, 2, 3]
    assert all(int(float(row["games"])) == 8 for row in history)  # 4 lanes x 2 games, split 1 + 1
    assert all(float(row["policy_lag"]) >= 1 for row in history[1:])


def test_split_lanes_divides_quota_and_games_in_flight():
    from cardgym.rl.continuous import split_lanes
    from cardgym.rl.setup import load_config as load

    lanes = load(SMOKE).game.lanes(1.5)
    shares = [lane for worker in range(3) for lane in split_lanes(lanes, worker, 3)]
    for lane in lanes:
        mine = [share for share in shares if share.name == lane.name]
        assert sum(share.quota for share in mine) == lane.quota
        assert all(share.in_flight >= 1 for share in mine)
