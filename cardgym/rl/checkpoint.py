"""Training checkpoints (full resumable state) and release models (weights).

A training checkpoint is one ``.pt`` file holding models, optimizers, entropy
temperatures, iteration counters, RNG streams, and the resolved config.

A release is a directory meant to be published:

    actor.pt     actor weights + the [model] section that builds it
    critic.pt    critic weights + entropy temperatures, to resume training
    info.json    provenance (source run, iteration, checksums)

Each file stays well under GitHub's 50 MB soft limit for plump-sized models.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cardgym.core.io import atomic_torch_save, atomic_write_json, file_sha256, timestamp

CHECKPOINT_FORMAT = "cardgym.checkpoint"
RELEASE_FORMAT = "cardgym.release"
VERSION = 1


def save_checkpoint(
    path: Path,
    *,
    game: str,
    iteration: int,
    raw_config: dict[str, Any],
    trainer,
    rng: random.Random,
) -> None:
    payload = {
        "format": CHECKPOINT_FORMAT,
        "version": VERSION,
        "game": game,
        "iteration": iteration,
        "config": raw_config,
        "trainer": trainer.state_dict(include_optimizers=True),
        "rng": {
            "trainer": rng.getstate(),
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        },
    }
    atomic_torch_save(payload, Path(path))


def load_checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a cardgym training checkpoint.")
    if payload.get("version") != VERSION:
        raise ValueError(f"Unsupported checkpoint version {payload.get('version')}.")
    return payload


def restore_rng(payload: dict[str, Any], rng: random.Random) -> None:
    state = payload.get("rng", {})
    if "trainer" in state:
        rng.setstate(state["trainer"])
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])


def export_release(
    directory: Path,
    *,
    game: str,
    iteration: int,
    model_config: dict[str, Any],
    actor_state: dict[str, torch.Tensor],
    critic_state: dict[str, torch.Tensor],
    entropy_state: dict[str, Any] | None,
    source: dict[str, Any],
) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    common = {"format": RELEASE_FORMAT, "version": VERSION, "game": game, "iteration": iteration}
    actor_path = directory / "actor.pt"
    critic_path = directory / "critic.pt"
    atomic_torch_save(
        {**common, "kind": "actor", "model": model_config, "state_dict": _cpu(actor_state)},
        actor_path,
    )
    atomic_torch_save(
        {
            **common,
            "kind": "critic",
            "model": model_config,
            "state_dict": _cpu(critic_state),
            "entropy": entropy_state,
        },
        critic_path,
    )
    atomic_write_json(
        directory / "info.json",
        {
            "game": game,
            "iteration": iteration,
            "exported_at": timestamp(),
            "source": source,
            "files": {
                path.name: {"sha256": file_sha256(path), "bytes": path.stat().st_size}
                for path in (actor_path, critic_path)
            },
        },
    )
    return directory


def load_release_file(path: Path, kind: str) -> dict[str, Any]:
    path = Path(path)
    if path.is_dir():
        path = path / f"{kind}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != RELEASE_FORMAT or payload.get("kind") != kind:
        raise ValueError(f"{path} is not a cardgym release {kind}.")
    return payload


def load_weights(path: Path, game: str) -> dict[str, Any]:
    """Actor/critic weights (+ entropy, model config, iteration) from any source.

    Accepts a release directory, a release ``actor.pt``, or a training
    checkpoint. Critic weights are None when only an actor file is given.
    """

    path = Path(path)
    if path.is_dir() or _format(path) == RELEASE_FORMAT:
        directory = path if path.is_dir() else path.parent
        actor = load_release_file(directory, "actor")
        critic_file = directory / "critic.pt"
        critic = load_release_file(critic_file, "critic") if critic_file.is_file() else None
        if actor["game"] != game:
            raise ValueError(f"{path} holds a {actor['game']} model, not {game}.")
        return {
            "iteration": int(actor["iteration"]),
            "model": actor["model"],
            "actor": actor["state_dict"],
            "critic": None if critic is None else critic["state_dict"],
            "entropy": None if critic is None else critic.get("entropy"),
        }
    payload = load_checkpoint(path)
    if payload["game"] != game:
        raise ValueError(f"{path} holds a {payload['game']} checkpoint, not {game}.")
    return {
        "iteration": int(payload["iteration"]),
        "model": payload["config"]["model"],
        "actor": payload["trainer"]["actor"],
        "critic": payload["trainer"]["critic"],
        "entropy": payload["trainer"].get("entropy"),
    }


def release_info(path: Path) -> dict[str, Any] | None:
    info = Path(path) / "info.json"
    return json.loads(info.read_text()) if info.is_file() else None


def _format(path: Path) -> str | None:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except Exception:  # noqa: BLE001 - anything unreadable is simply not a release
        return None
    return payload.get("format") if isinstance(payload, dict) else None


def _cpu(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state.items()}
