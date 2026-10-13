"""End-to-end: train the smoke preset, export a release, evaluate it."""

from __future__ import annotations

import csv
import json

from cardgym.cli import main
from cardgym.rl.setup import load_config, normalized_raw
from tests.conftest import ROOT

SMOKE = str(ROOT / "configs" / "rummy" / "smoke.toml")


def test_presets_resolve():
    for name in ("smoke", "base"):
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

    release = tmp_path / "release"
    assert main(["rummy", "export", "smoke", "--out", str(release)]) == 0
    report = tmp_path / "report.json"
    assert main(
        ["rummy", "evaluate", str(release), "--opponent", "random", "--players", "2,3",
         "--hand-size", "7", "--deals", "2", "--output", str(report)]
    ) == 0
    result = json.loads(report.read_text())
    assert set(result["by_players"]) == {"2", "3"}
    assert result["by_players"]["3"]["model_alone"]["games"] == 2 * 3
    assert "relative reward" in capsys.readouterr().out
