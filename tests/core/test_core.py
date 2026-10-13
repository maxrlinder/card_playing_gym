from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal

import pytest

from cardgym.core.config import (
    apply_overrides,
    config_diff,
    dump_toml,
    parse_section,
    section_dict,
)
from cardgym.core.metrics import MetricsLog
from cardgym.core.runs import RunDirectory


@dataclass(frozen=True)
class Section:
    rate: float = 1.0
    count: int = 2
    mode: Literal["a", "b"] = "a"
    sizes: tuple[int, ...] = (3, 4)
    targets: dict[str, float] = field(default_factory=dict)
    flag: bool = False

    def validate(self) -> None:
        if self.count < 0:
            raise ValueError("count must be non-negative")


def test_parse_section_coerces_and_defaults():
    section = parse_section(Section, {"rate": 2, "sizes": [5], "targets": {"x": 1}}, "s")
    assert section == Section(rate=2.0, sizes=(5,), targets={"x": 1.0})
    assert isinstance(section.rate, float)


@pytest.mark.parametrize(
    "data, message",
    [
        ({"unknown": 1}, "Unknown keys"),
        ({"count": 1.5}, "integer"),
        ({"mode": "c"}, "not one of"),
        ({"flag": 1}, "true/false"),
        ({"count": -1}, "non-negative"),
    ],
)
def test_parse_section_is_strict(data, message):
    with pytest.raises(ValueError, match=message):
        parse_section(Section, data, "s")


def test_overrides_dump_and_diff_round_trip():
    import tomllib

    raw = {"run": {"game": "plump", "iterations": 3}, "ppo": {"targets": {"bid": 0.5}}}
    changed = apply_overrides(raw, ["run.iterations=5", "ppo.targets={ bid = 0.25 }"])
    assert raw["run"]["iterations"] == 3, "overrides must not mutate the input"
    assert tomllib.loads(dump_toml(changed)) == changed
    assert config_diff(raw, changed) == [
        "ppo.targets: recorded={'bid': 0.5} requested={'bid': 0.25}",
        "run.iterations: recorded=3 requested=5",
    ]
    with pytest.raises(ValueError, match="Unknown configuration key"):
        apply_overrides(raw, ["run.iteratons=5"])
    assert section_dict(Section(sizes=(1, 2)))["sizes"] == [1, 2]


def test_metrics_widen_header_and_truncate(tmp_path):
    log = MetricsLog(tmp_path / "metrics.csv")
    log.append({"iteration": 1, "loss": 0.5})
    log.append({"iteration": 2, "loss": 0.25, "extra": 1.0})
    rows = log.read()
    assert [row["extra"] for row in rows] == ["", "1"]
    assert log.last_value("loss") == 0.25
    assert log.truncate_after(1) == 1
    assert [row["iteration"] for row in MetricsLog(tmp_path / "metrics.csv").read()] == ["1"]


def test_run_directory_manifest_and_lock(runs_dir):
    run = RunDirectory("plump", "demo")
    run.create({"run": {"game": "plump"}}, ["cardgym"])
    assert run.exists and run.recorded_config() == {"run": {"game": "plump"}}
    checkpoint = run.interval_checkpoint(3)
    checkpoint.write_bytes(b"weights")
    run.record_latest(checkpoint, 3)
    assert run.resolve_checkpoint("latest") == checkpoint.resolve()
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="mismatch"):
        run.resolve_checkpoint("latest")
    with run.acquire_lock():
        with pytest.raises(RuntimeError, match="already owned"):
            with RunDirectory("plump", "demo").acquire_lock():
                pass
    assert not run.lock.exists()
    archive = run.record_reconfiguration(
        {"run": {"game": "plump", "seed": 1}},
        iteration=3,
        reason="test",
        changes=["run.seed"],
        resume_checkpoint=checkpoint,
    )
    assert archive.read_text().startswith("[run]")
    assert json.loads(run.metadata.read_text())["config_migrations"][0]["reason"] == "test"
    with pytest.raises(ValueError):
        RunDirectory("plump", "../escape")
