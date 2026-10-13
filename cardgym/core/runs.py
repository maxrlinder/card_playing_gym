"""Run directories: ownership, recorded config, checkpoints, and metadata.

Layout::

    runs/<game>/<name>/
      config.toml            resolved configuration; resume must match it
      config-history/        configs replaced by --reconfigure
      metadata.json          command, git revision, versions, status
      metrics.csv            one row per update
      train.log              human-readable progress
      checkpoints/
        iter_000000.pt       full training state, every interval
        latest.json          size + sha256 manifest of the resume point

``CARDGYM_RUNS_DIR`` relocates the ``runs`` root.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch

from .config import dump_toml, load_toml
from .io import atomic_write_json, atomic_write_text, file_sha256, timestamp

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def runs_root() -> Path:
    configured = os.environ.get("CARDGYM_RUNS_DIR")
    return Path(configured).expanduser() if configured else PROJECT_ROOT / "runs"


class RunDirectory:
    def __init__(self, game: str, name: str, *, root: Path | None = None) -> None:
        if not RUN_NAME.fullmatch(name):
            raise ValueError("Run names may contain letters, digits, '.', '_', and '-'.")
        self.game = game
        self.name = name
        self.path = ((root or runs_root()) / game / name).resolve()
        self.checkpoints = self.path / "checkpoints"
        self.config = self.path / "config.toml"
        self.config_history = self.path / "config-history"
        self.metadata = self.path / "metadata.json"
        self.metrics = self.path / "metrics.csv"
        self.train_log = self.path / "train.log"
        self.dashboard = self.path / "dashboard.png"
        self.evaluations = self.path / "evaluations"
        self.lock = self.path / "run.lock"

    @property
    def exists(self) -> bool:
        return self.config.is_file()

    def create(self, raw_config: dict[str, Any], command: list[str]) -> None:
        self.checkpoints.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.config, dump_toml(raw_config))
        atomic_write_json(
            self.metadata,
            {
                "game": self.game,
                "run": self.name,
                "created_at": timestamp(),
                "command": command,
                "git": _git_metadata(),
                "python": sys.version,
                "torch": torch.__version__,
                "status": "created",
            },
        )

    def recorded_config(self) -> dict[str, Any]:
        return load_toml(self.config)

    def update_metadata(self, **updates: Any) -> None:
        data: dict[str, Any] = {}
        if self.metadata.exists():
            data = json.loads(self.metadata.read_text())
        data.update(updates)
        data["updated_at"] = timestamp()
        atomic_write_json(self.metadata, data)

    def record_reconfiguration(
        self,
        raw_config: dict[str, Any],
        *,
        iteration: int,
        reason: str,
        changes: list[str],
        resume_checkpoint: Path,
    ) -> Path:
        """Adopt a new config, archiving the old one with an audit entry."""

        self.config_history.mkdir(parents=True, exist_ok=True)
        archive = self.config_history / f"iter_{iteration:06d}.toml"
        suffix = 1
        while archive.exists():
            archive = self.config_history / f"iter_{iteration:06d}_{suffix}.toml"
            suffix += 1
        atomic_write_text(archive, self.config.read_text())
        atomic_write_text(self.config, dump_toml(raw_config))
        metadata = json.loads(self.metadata.read_text()) if self.metadata.exists() else {}
        migrations = list(metadata.get("config_migrations", []))
        migrations.append(
            {
                "at": timestamp(),
                "iteration": iteration,
                "reason": reason,
                "changes": changes,
                "previous_config": str(archive.relative_to(self.path)),
                "resume_checkpoint": resume_checkpoint.name,
                "resume_sha256": file_sha256(resume_checkpoint),
            }
        )
        metadata["config_migrations"] = migrations
        metadata["updated_at"] = timestamp()
        atomic_write_json(self.metadata, metadata)
        return archive

    def interval_checkpoint(self, iteration: int) -> Path:
        return self.checkpoints / f"iter_{iteration:06d}.pt"

    def record_latest(self, checkpoint: Path, iteration: int) -> None:
        atomic_write_json(
            self.checkpoints / "latest.json",
            {
                "iteration": iteration,
                "path": checkpoint.name,
                "size": checkpoint.stat().st_size,
                "sha256": file_sha256(checkpoint),
                "written_at": timestamp(),
            },
        )

    def resolve_checkpoint(self, selector: str | int = "latest") -> Path:
        """``latest`` (checksum-verified), an iteration number, or a path."""

        if isinstance(selector, int) or str(selector).isdigit():
            path = self.interval_checkpoint(int(selector))
        elif selector == "latest":
            manifest_path = self.checkpoints / "latest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(manifest_path)
            manifest = json.loads(manifest_path.read_text())
            path = self.checkpoints / manifest["path"]
            if path.stat().st_size != int(manifest["size"]):
                raise RuntimeError(f"Checkpoint size mismatch: {path}")
            if file_sha256(path) != manifest["sha256"]:
                raise RuntimeError(f"Checkpoint checksum mismatch: {path}")
        else:
            path = Path(str(selector)).expanduser()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path.resolve()

    def log(self, message: str) -> None:
        line = f"[{timestamp()}] {message}"
        print(line, flush=True)
        self.path.mkdir(parents=True, exist_ok=True)
        with self.train_log.open("a") as handle:
            handle.write(line + "\n")

    @contextmanager
    def acquire_lock(self) -> Iterator[None]:
        """Exclusive ownership; a lock whose process is gone is reclaimed."""

        self.path.mkdir(parents=True, exist_ok=True)
        if self.lock.exists():
            try:
                pid = int(self.lock.read_text().strip())
            except ValueError:
                pid = -1
            if pid > 0 and _process_exists(pid):
                raise RuntimeError(f"Run {self.name!r} is already owned by process {pid}.")
            self.lock.unlink()
        descriptor = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        try:
            os.write(descriptor, f"{os.getpid()}\n".encode())
            os.close(descriptor)
            yield
        finally:
            try:
                self.lock.unlink()
            except FileNotFoundError:
                pass


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _git_metadata() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()

    try:
        return {
            "commit": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "dirty": bool(git("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "branch": None, "dirty": None}
