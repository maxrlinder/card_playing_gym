"""End-to-end: train the smoke preset without a critic, export a release, evaluate it."""

from __future__ import annotations

import csv
import json

import pytest

from cardgym.cli import main
from cardgym.rl.checkpoint import load_weights
from cardgym.rl.setup import load_config, normalized_raw
from tests.conftest import ROOT

SMOKE = str(ROOT / "configs" / "piquet" / "smoke.toml")


def test_presets_resolve():
    for name in ("smoke", "base", "partie"):
        config = load_config(ROOT / "configs" / "piquet" / f"{name}.toml")
        assert set(normalized_raw(config)) == {"run", "rollout", "ppo", "model", "game", "losses"}
        assert config.run.game == "piquet"
        assert config.game.build_critic() is None


def test_train_export_evaluate(runs_dir, tmp_path, capsys):
    # One optimizer step per epoch, so iteration 1's replay must equal the rollout exactly.
    assert main(["piquet", "train", "smoke", "--config", SMOKE, "--iterations", "2", "--set", "ppo.minibatches=1"]) == 0
    run = runs_dir / "piquet" / "smoke"
    with open(run / "metrics.csv", newline="") as handle:
        history = list(csv.DictReader(handle))
    assert [int(row["iteration"]) for row in history] == [1, 2]
    assert abs(float(history[0]["behavior_kl"])) < 1e-6
    last = history[-1]
    for key in ("belief_accuracy", "tricks_accuracy", "declarations_accuracy", "bonus_accuracy", "trick_accuracy"):
        assert 0 < float(last[key]) <= 1, key
    assert float(last["points_rmse"]) > 0
    assert float(last["value_rmse"]) > 0 and float(last["actor_value_rmse"]) > 0
    assert not last.get("critic_grad_norm")  # no critic was trained
    # auxiliary_grad_refresh = 2: the second optimizer step reuses the first one's scale.
    assert float(last.get("lagged_auxiliary_steps") or 0) == 1
    assert float(last["auxiliary_grad_scale"]) > 0

    release = tmp_path / "release"
    assert main(["piquet", "export", "smoke", "--out", str(release)]) == 0
    weights = load_weights(release, "piquet")
    assert weights["critic"] is None and weights["entropy"]
    report = tmp_path / "report.json"
    evaluate = ["piquet", "evaluate", str(release), "--opponent", "random", "--deals", "3"]
    assert main([*evaluate, "--output", str(report)]) == 0
    result = json.loads(report.read_text())
    assert result["games"] == 6  # each deal both ways
    assert "points per deal" in capsys.readouterr().out
    assert main(["piquet", "evaluate", str(release), "--mode", "rubicon", "--deals", "1"]) == 0

    # Resuming restores the critic-less trainer.
    assert main([
        "piquet", "train", "smoke", "--config", SMOKE, "--iterations", "3", "--set", "ppo.minibatches=1",
        "--reconfigure", "--reason", "resume test",
    ]) == 0
    with open(run / "metrics.csv", newline="") as handle:
        assert [int(row["iteration"]) for row in csv.DictReader(handle)] == [1, 2, 3]


@pytest.mark.parametrize("workers", [1, 2])
def test_asynchronous_rollout(runs_dir, tmp_path, workers):
    config = tmp_path / "async.toml"
    config.write_text(
        (ROOT / "configs" / "piquet" / "smoke.toml").read_text().replace(
            'mode = "continuous"\n', f'mode = "continuous"\nasynchronous = true\nworkers = {workers}\n'
        )
    )
    assert main(["piquet", "train", "async", "--config", str(config), "--iterations", "3"]) == 0
    with open(runs_dir / "piquet" / "async" / "metrics.csv", newline="") as handle:
        history = list(csv.DictReader(handle))
    assert [int(row["iteration"]) for row in history] == [1, 2, 3]
    assert all(float(row["policy_lag"]) >= 1 for row in history[1:])


def test_simulate(capsys):
    assert main(["piquet", "simulate", "--games", "20", "--opponent", "random"]) == 0
    assert "points per deal" in capsys.readouterr().out


def test_rollout_recorded_values_match_the_value_pass():
    """With the sampling weights unchanged, recorded values give the value pass's baseline exactly."""

    import random

    from cardgym.rl.batch import build_batch
    from cardgym.rl.setup import build_components

    components = build_components(load_config(SMOKE))
    game, trainer, collector = components.config.game, components.trainer, components.collector
    assert collector.record_values
    trajectories = collector.collect(game.lanes(1.0), random.Random(0), version=0)
    batch = build_batch(
        trajectories, [game.sequences(t) for t in trajectories],
        max_players=game.max_players, actor_labels=game.actor_labels, critic_labels=game.critic_labels,
        bucket_width=16, critic_bucket_width=16, max_length=game.max_length, critic_max_length=game.max_length,
    )
    assert all(rows.values is not None for rows in batch.decisions.values())
    pending: list = []
    trainer._actor_baseline(batch, pending)
    passed = trainer._baseline(batch, pending)
    from_pass = {head: rows.advantages.copy() for head, rows in batch.decisions.items()}
    recorded = trainer._baseline(batch, [])
    for head, rows in batch.decisions.items():
        assert abs(rows.advantages - from_pass[head]).max() < 1e-4, head
    assert abs(passed["value_rmse"] - recorded["value_rmse"]) < 1e-3


def test_policy_warmup_trains_only_the_auxiliary_losses_first(runs_dir, tmp_path):
    config = tmp_path / "warmup.toml"
    config.write_text(
        (ROOT / "configs" / "piquet" / "smoke.toml").read_text().replace(
            "auxiliary_grad_refresh = 2\n", "auxiliary_grad_refresh = 1\npolicy_warmup_steps = 2\n"
        )
    )
    assert main(["piquet", "train", "warmup", "--config", str(config), "--iterations", "2"]) == 0
    with open(runs_dir / "piquet" / "warmup" / "metrics.csv", newline="") as handle:
        first, second = list(csv.DictReader(handle))
    # Two minibatches per update: the first update is all warm-up, the second all policy.
    assert float(first["policy_warmup_steps"]) == 2 and float(first["policy_grad_norm"]) == 0
    assert not float(second.get("policy_warmup_steps") or 0) and float(second["policy_grad_norm"]) > 0
