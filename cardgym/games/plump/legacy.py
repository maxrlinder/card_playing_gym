"""Import checkpoints written by the original plump-bot trainer (schema v6).

Only the configuration the rl-v6 line trained is supported: TRICK_WIN and
bid-TURN tokens on, embedding RMSNorm on, factored card embeddings, and no
absolute position embeddings. Weights are renamed into this package's layout:

* the core-field and bidding-state tables merge into one field table, minus
  the single dummy exact-card row the factored model never read; the same
  row is dropped from the per-layer and value-embedding tables;
* the trunk moves under ``trunk.``; the actor's scalar value head (unused by
  PPO) is dropped; the critic keeps only its trainable readouts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .model import PlumpModelConfig
from .tokens import SLOT_CARD, Schema

REQUIRED = {
    "schema_version": 6,
    "trick_win_token": True,
    "turn_token": "bid",
    "embedding_rms_norm": True,
    "card_embedding_mode": "factored",
    "absolute_position_embedding": False,
    "dropout": 0.0,
}


def model_config(old: dict[str, Any]) -> PlumpModelConfig:
    unsupported = {key: old.get(key) for key, value in REQUIRED.items() if old.get(key) != value}
    if unsupported:
        raise ValueError(f"Unsupported legacy model configuration: {unsupported}")
    return PlumpModelConfig(
        max_players=int(old["max_players"]),
        max_hand_size=int(old["max_hand_size"]),
        d_model=int(old["d_model"]),
        n_layers=int(old["n_layers"]),
        n_heads=int(old["n_heads"]),
        n_kv_heads=int(old["n_kv_heads"] or old["n_heads"]),
        d_ff=int(old["d_ff"]),
        per_layer_dim=int(old["per_layer_embedding_dim"]),
        value_embeddings=old["value_embedding_mode"],
    )


def _card_row(schema: Schema) -> int:
    return sum(schema.base_vocab_sizes[:SLOT_CARD])


def _embedding(old: dict[str, torch.Tensor], schema: Schema, prefix: str = "") -> dict[str, torch.Tensor]:
    card = _card_row(schema)

    def without_card(name: str) -> torch.Tensor:
        weight = old[prefix + name]
        return torch.cat((weight[:card], weight[card + 1 :]))

    core = old[prefix + "slot_embedding.weight"]
    state = {
        "embedding.fields.weight": torch.cat(
            (core[:card], core[card + 1 :], old[prefix + "bid_pressure_embedding.weight"])
        ),
        "embedding.na_pattern.weight": old[prefix + "na_pattern_embedding.weight"],
        "trunk.input_norm.weight": old[prefix + "embedding_norm.weight"],
        "trunk.final_norm.weight": old[prefix + "final_norm.weight"],
        "trunk.final_norm.bias": old[prefix + "final_norm.bias"],
    }
    if prefix + "per_layer_slot_embedding.weight" in old:
        state["embedding.ple_fields.weight"] = without_card("per_layer_slot_embedding.weight")
        state["embedding.ple_na_pattern.weight"] = old[prefix + "per_layer_na_pattern_embedding.weight"]
        state["trunk.ple_context.weight"] = old[prefix + "per_layer_context_projection.weight"]
    if prefix + "value_slot_embedding.weight" in old:
        state["embedding.value_fields.weight"] = without_card("value_slot_embedding.weight")
        state["embedding.value_na_pattern.weight"] = old[prefix + "value_na_pattern_embedding.weight"]
    for key, value in old.items():
        if key.startswith(prefix + "blocks."):
            state["trunk." + key[len(prefix) :]] = value
    return state


def _copy(old: dict[str, torch.Tensor], mapping: dict[str, str]) -> dict[str, torch.Tensor]:
    state = {}
    for source, target in mapping.items():
        for key, value in old.items():
            if key.startswith(source + "."):
                state[target + key[len(source) :]] = value
    return state


def convert_actor(old: dict[str, torch.Tensor], config: PlumpModelConfig) -> dict[str, torch.Tensor]:
    state = _embedding(old, config.schema())
    state.update(
        _copy(
            old,
            {
                "bid_head": "bid_head",
                "card_rank_output_embedding": "card_rank",
                "card_suit_output_embedding": "card_suit",
                "player_value_head": "value_head",
                "suit_presence_head": "suit_head",
                "rank_boundary_head": "rank_boundary_head",
                "trick_count_head": "trick_head",
                "next_winner_head": "next_winner_head",
                "bid_hit_head": "bid_hit_head",
            },
        )
    )
    state["card_rank_bias"] = old["card_head.rank_bias"]
    state["card_suit_bias"] = old["card_head.suit_bias"]
    return state


def convert_critic(old: dict[str, torch.Tensor], config: PlumpModelConfig) -> dict[str, torch.Tensor]:
    state = _embedding(old, config.schema(), prefix="backbone.")
    state.update(
        _copy(
            old,
            {
                "player_value_head": "value_head",
                "backbone.trick_count_head": "trick_head",
                "backbone.next_winner_head": "next_winner_head",
                "card_order_head": "card_order_head",
            },
        )
    )
    return state


def load_legacy_checkpoint(path: Path) -> dict[str, Any]:
    """Model config, converted actor/critic weights, entropy state, iteration."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 6 or "critic_state_dict" not in payload:
        raise ValueError(f"{path} is not a plump-bot schema-v6 PPO checkpoint.")
    config = model_config(payload["model_config"])
    actors = payload.get("actor_state_dicts") or [payload["model_state_dict"]]
    log_alpha = payload.get("entropy_log_alpha") or {}
    return {
        "iteration": int(payload["iteration"]),
        "model_config": config,
        "actor": convert_actor(actors[0], config),
        "critic": convert_critic(payload["critic_state_dict"], config),
        "entropy": {
            "log_alpha": {head: value.detach().clone() for head, value in log_alpha.items()},
            "optimizer": None,
        },
        "training_config": payload.get("resolved_config"),
    }
