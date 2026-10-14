"""Resolve a run configuration and assemble the training components."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from cardgym.core.config import apply_overrides, load_toml, parse_section, section_dict
from cardgym.core.devices import resolve_device
from cardgym.games import game_class
from cardgym.games.base import Game
from cardgym.nn.metal_trunk import metal_trunk_requested, unsupported_reason

from .config import PPOConfig, RolloutConfig, RunConfig
from .continuous import ContinuousCollector
from .selfplay import SelfPlayCollector
from .trainer import PPOTrainer

SECTIONS = ("run", "rollout", "ppo", "model", "game", "losses")


@dataclass(frozen=True)
class ResolvedConfig:
    raw: dict[str, Any]
    run: RunConfig
    rollout: RolloutConfig
    ppo: PPOConfig
    game: Game


def resolve_config(raw: dict[str, Any]) -> ResolvedConfig:
    unknown = sorted(set(raw) - set(SECTIONS))
    if unknown:
        raise ValueError(f"Unknown configuration sections: {unknown}")
    run = parse_section(RunConfig, raw.get("run"), "run")
    cls = game_class(run.game)
    game = cls(
        parse_section(cls.ModelConfig, raw.get("model"), "model"),
        parse_section(cls.GameConfig, raw.get("game"), "game"),
        parse_section(cls.LossConfig, raw.get("losses"), "losses"),
    )
    ppo = parse_section(PPOConfig, raw.get("ppo"), "ppo")
    unknown_heads = sorted(set(ppo.entropy_targets) - set(cls.policy_heads))
    if unknown_heads:
        raise ValueError(f"[ppo] entropy_targets names unknown heads {unknown_heads}.")
    return ResolvedConfig(
        raw=raw,
        run=run,
        rollout=parse_section(RolloutConfig, raw.get("rollout"), "rollout"),
        ppo=ppo,
        game=game,
    )


def load_config(path: str | Path, overrides: list[str] | None = None) -> ResolvedConfig:
    return resolve_config(apply_overrides(load_toml(path), overrides or []))


def normalized_raw(config: ResolvedConfig) -> dict[str, Any]:
    """Every section with every field spelled out, defaults included."""

    return {
        "run": section_dict(config.run),
        "rollout": section_dict(config.rollout),
        "ppo": section_dict(config.ppo),
        "model": section_dict(config.game.model_config),
        "game": section_dict(config.game.game_config),
        "losses": section_dict(config.game.loss_config),
    }


@dataclass
class Components:
    config: ResolvedConfig
    device: torch.device
    trainer: PPOTrainer
    collector: SelfPlayCollector | ContinuousCollector


def build_components(config: ResolvedConfig, *, device: str | None = None) -> Components:
    game = config.game
    target = resolve_device(device if device is not None else config.run.device)
    actor = game.build_actor()
    critic = game.build_critic()
    trainer = PPOTrainer(
        actor,
        critic,
        config.ppo,
        game.hooks(target, config.run.precision),
        device=target,
        precision=config.run.precision,
        policy_heads=game.policy_heads,
        max_players=game.max_players,
    )
    if (
        metal_trunk_requested()
        and target.type == "mps"
        and config.run.precision == "bf16"
    ):
        for model, length in ((trainer.actor, game.max_length), (trainer.critic, game.critic_max_length)):
            if unsupported_reason(model.trunk_config, length) is None:
                model.enable_metal_trunk(length)
    if config.rollout.mode == "continuous":
        collector = ContinuousCollector(
            trainer.actor,
            device=target,
            precision=config.run.precision,
            kv_dtype=config.rollout.kv_dtype,
            max_length=game.max_length,
            cohorts=config.rollout.cohorts,
        )
    else:
        collector = SelfPlayCollector(
            trainer.actor,
            device=target,
            precision=config.run.precision,
            kv_dtype=config.rollout.kv_dtype,
            max_rows=config.rollout.max_rows,
            max_length=game.max_length,
            cohorts=config.rollout.cohorts,
        )
    return Components(config=config, device=target, trainer=trainer, collector=collector)


def metal_status(components: Components) -> str:
    if not metal_trunk_requested():
        return "off"
    enabled = [
        name
        for name, model in (("actor", components.trainer.actor), ("critic", components.trainer.critic))
        if model.trunk.metal is not None
    ]
    return ",".join(enabled) if enabled else "requested but unsupported here"
