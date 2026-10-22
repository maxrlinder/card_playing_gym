"""Deepen a trained run: insert fresh blocks into its trunk and keep training.

``cardgym <game> grow NEW --from RUN --config NEW.toml --fresh 4,5,10,11``
forks RUN's checkpoint into a new run whose [model] differs only in
``n_layers``. The trained blocks keep their order and sit at the positions
not named by ``--fresh``. A fresh block is the trunk's ordinary random
initialization, except that its two residual output projections (attention
``out_proj`` and the MLP's down projection) start at zero. Its per-layer
embedding (PLE) and value-embedding slots also start at zero. So each fresh
block adds nothing at first, and the grown model computes exactly what the
trained one did. Gradients still reach it at once: an output projection's
gradient is its input times the residual gradient, which is not zero.

Packed per-layer tensors (the trunk's PLE context projection, a game's
packed PLE and value-embedding lookups) are re-sliced along their packed
axis. A tensor whose size differs between the two models is recognized as
one by that axis matching the PLE or value width. Value embeddings feed only
the blocks ``TrunkConfig.value_layers`` names (``alternating``: every other
block, counted from the last). Every trained block must keep its membership,
which for ``alternating`` means fresh blocks come in adjacent pairs.

The fork is a full training checkpoint: Adam moments follow their
parameters, fresh parameters start with empty Adam state, and the optimizer
step count, entropy temperatures, RNG streams, and iteration carry over. So
no learning-rate warmup is needed and nothing makes a cold-start jump.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from cardgym.core.config import section_dict
from cardgym.core.io import atomic_torch_save
from cardgym.core.runs import RunDirectory
from cardgym.games.base import Game
from cardgym.nn.trunk import TrunkConfig

from .checkpoint import load_checkpoint
from .setup import load_config, resolve_config

_BLOCK = re.compile(r"^(.*\btrunk\.blocks\.)(\d+)(\..+)$")
# Residual output projections of a fresh block: zero keeps the block an identity.
_RESIDUAL_OUTPUTS = ("out_proj.weight", "mlp.2.weight", "mlp.2.bias")


def add_grow_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("run", help="the new run's name; runs/<game>/<run>")
    parser.add_argument("--from", dest="source", required=True, help="the trained run to fork")
    parser.add_argument("--checkpoint", default="latest", help="source checkpoint: 'latest', an iteration, or a path")
    parser.add_argument("--config", type=Path, required=True, help="the new run's config ([model] n_layers deeper)")
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="SECTION.KEY=VALUE",
        help="override one config value (TOML syntax)",
    )
    parser.add_argument(
        "--fresh", required=True,
        help="comma-separated block positions (0-based, in the new stack) that start fresh",
    )


def block_positions(old_layers: int, new_layers: int, fresh: Sequence[int]) -> list[int]:
    """New position of each trained block: the positions ``fresh`` leaves, in order."""

    chosen = sorted(set(fresh))
    if len(chosen) != len(fresh):
        raise ValueError("--fresh names a position twice.")
    if new_layers <= old_layers:
        raise ValueError(f"The new trunk ({new_layers} layers) must be deeper than {old_layers}.")
    if len(chosen) != new_layers - old_layers:
        raise ValueError(
            f"Growing {old_layers} -> {new_layers} layers needs {new_layers - old_layers} fresh positions, "
            f"got {len(chosen)}."
        )
    if any(not 0 <= position < new_layers for position in chosen):
        raise ValueError(f"Fresh positions must lie in 0..{new_layers - 1}.")
    return [position for position in range(new_layers) if position not in chosen]


def check_value_layers(old: TrunkConfig, new: TrunkConfig, positions: Sequence[int]) -> None:
    moved = [
        layer for layer in range(old.n_layers)
        if (layer in old.value_layers) != (positions[layer] in new.value_layers)
    ]
    if moved:
        raise ValueError(
            f"Trained blocks {moved} would gain or lose their value embedding at their new positions "
            f"(value_embeddings = {old.value_embeddings!r}: the new stack feeds blocks {list(new.value_layers)}). "
            "Insert fresh blocks in adjacent pairs."
        )


def _rename(name: str, positions: Sequence[int]) -> str:
    match = _BLOCK.match(name)
    if match is None:
        return name
    return f"{match.group(1)}{positions[int(match.group(2))]}{match.group(3)}"


def _slots(old: TrunkConfig, new: TrunkConfig, positions: Sequence[int], kind: str) -> tuple[int, dict[int, int]]:
    """(slot width, new slot -> old slot) along a packed per-layer axis."""

    if kind == "ple":
        return old.per_layer_dim, {positions[layer]: layer for layer in range(old.n_layers)}
    new_slot = {layer: slot for slot, layer in enumerate(new.value_layers)}
    return old.kv_width, {
        new_slot[positions[layer]]: slot for slot, layer in enumerate(old.value_layers)
    }


def _packed_kind(old: TrunkConfig, new: TrunkConfig, old_size: int, new_size: int, name: str) -> str:
    """"ple" or "value": by the axis sizes, or by the name when both widths agree."""

    kinds = []
    if old.per_layer_dim and (old_size, new_size) == (old.ple_width, new.ple_width):
        kinds.append("ple")
    if old.value_layers and (old_size, new_size) == (old.value_width, new.value_width):
        kinds.append("value")
    if len(kinds) == 2:
        kinds = [kind for kind in kinds if kind in name]
    if len(kinds) != 1:
        raise ValueError(
            f"Cannot tell how to re-slice {name!r} along an axis of {old_size} -> {new_size} "
            f"(PLE {old.ple_width} -> {new.ple_width}, values {old.value_width} -> {new.value_width})."
        )
    return kinds[0]


def repack(
    tensor: torch.Tensor,
    shape: Sequence[int],
    old: TrunkConfig,
    new: TrunkConfig,
    positions: Sequence[int],
    name: str,
) -> torch.Tensor:
    """Re-slice a packed per-layer tensor to the new stack; fresh slots are zero."""

    axes = [axis for axis in range(tensor.dim()) if tensor.shape[axis] != shape[axis]]
    if tensor.dim() != len(shape) or len(axes) != 1:
        raise ValueError(f"Expected one packed axis between {tuple(tensor.shape)} and {tuple(shape)}.")
    axis = axes[0]
    width, sources = _slots(old, new, positions, _packed_kind(old, new, tensor.shape[axis], shape[axis], name))
    out = torch.zeros(tuple(shape), dtype=tensor.dtype)
    for target, source in sources.items():
        out.narrow(axis, target * width, width).copy_(tensor.narrow(axis, source * width, width))
    return out


def grow_state(
    state: dict[str, torch.Tensor],
    fresh_state: dict[str, torch.Tensor],
    old: TrunkConfig,
    new: TrunkConfig,
    positions: Sequence[int],
) -> dict[str, torch.Tensor]:
    """The trained ``state`` laid out for the deeper model whose initial state is ``fresh_state``."""

    grown = {name: value.detach().cpu().clone() for name, value in fresh_state.items()}
    trained = set(positions)
    for name in grown:
        match = _BLOCK.match(name)
        if match is not None and int(match.group(2)) not in trained and match.group(3)[1:] in _RESIDUAL_OUTPUTS:
            grown[name].zero_()
    for name, value in state.items():
        target = _rename(name, positions)
        if target not in grown:
            raise KeyError(f"The deeper model has no parameter for {name!r} (as {target!r}).")
        value = value.detach().cpu()
        if tuple(value.shape) == tuple(grown[target].shape):
            grown[target] = value.clone().to(grown[target].dtype)
        else:
            grown[target] = repack(value, grown[target].shape, old, new, positions, target).to(grown[target].dtype)
    return grown


def grow_optimizer(
    optimizer_state: dict[str, Any],
    old_names: Sequence[str],
    new_parameters: Sequence[tuple[str, torch.Tensor]],
    old: TrunkConfig,
    new: TrunkConfig,
    positions: Sequence[int],
) -> dict[str, Any]:
    """An Adam ``state_dict`` re-indexed for the deeper model.

    Moments follow their parameters (packed tensors are re-sliced, fresh
    slots zero); fresh parameters get no entry, so Adam initializes them on
    their first step.
    """

    if len(optimizer_state["param_groups"]) != 1:
        raise ValueError("Expected one Adam parameter group.")
    index = {_rename(name, positions): slot for slot, name in enumerate(old_names)}
    states: dict[int, dict[str, Any]] = {}
    for slot, (name, parameter) in enumerate(new_parameters):
        source = index.get(name)
        if source is None or source not in optimizer_state["state"]:
            continue
        entry = {}
        for key, value in optimizer_state["state"][source].items():
            if torch.is_tensor(value) and value.dim() > 0 and tuple(value.shape) != tuple(parameter.shape):
                value = repack(value.cpu(), parameter.shape, old, new, positions, name)
            entry[key] = value.clone() if torch.is_tensor(value) else value
        states[slot] = entry
    group = dict(optimizer_state["param_groups"][0])
    group["params"] = list(range(len(new_parameters)))
    return {"state": states, "param_groups": [group]}


def _trainable(model: torch.nn.Module) -> list[tuple[str, torch.Tensor]]:
    return [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]


def grow_trainer_state(
    trainer: dict[str, Any], old_game: Game, new_game: Game, fresh: Sequence[int], *, seed: int = 0
) -> dict[str, Any]:
    """A training checkpoint's ``trainer`` state for ``new_game``'s deeper models."""

    old_trunk = old_game.model_config.trunk()
    new_trunk = new_game.model_config.trunk()
    positions = block_positions(old_trunk.n_layers, new_trunk.n_layers, fresh)
    check_value_layers(old_trunk, new_trunk, positions)
    grown = dict(trainer)
    torch.manual_seed(seed)
    for role, build in (("actor", "build_actor"), ("critic", "build_critic")):
        if trainer.get(role) is None:
            continue  # a game without a critic
        old_model = getattr(old_game, build)()
        new_model = getattr(new_game, build)()
        grown[role] = grow_state(trainer[role], new_model.state_dict(), old_trunk, new_trunk, positions)
        key = f"{role}_optimizer"
        if key in trainer:
            grown[key] = grow_optimizer(
                trainer[key],
                [name for name, _ in _trainable(old_model)],
                _trainable(new_model),
                old_trunk,
                new_trunk,
                positions,
            )
    return grown


def grow_command(game_name: str, args: argparse.Namespace) -> int:
    source = RunDirectory(game_name, args.source)
    checkpoint = source.resolve_checkpoint(args.checkpoint)
    payload = load_checkpoint(checkpoint)
    old = resolve_config(payload["config"])
    new = load_config(args.config, args.overrides)
    if new.run.game != game_name:
        raise ValueError(f"{args.config} configures {new.run.game!r}, not {game_name!r}.")
    old_model, new_model = section_dict(old.game.model_config), section_dict(new.game.model_config)
    changed = sorted(key for key in old_model.keys() | new_model.keys() if old_model.get(key) != new_model.get(key))
    if changed != ["n_layers"]:
        raise ValueError(f"[model] may differ from {source.name}'s only in n_layers; it differs in {changed}.")
    fresh = [int(position) for position in str(args.fresh).split(",") if position.strip()]
    run = RunDirectory(game_name, args.run)
    if run.exists:
        raise ValueError(f"Run {run.name!r} already exists.")
    iteration = int(payload["iteration"])
    with run.acquire_lock():
        trainer = grow_trainer_state(payload["trainer"], old.game, new.game, fresh, seed=new.run.seed)
        run.create(new.raw, [Path(sys.argv[0]).name, *sys.argv[1:]])
        target = run.interval_checkpoint(iteration)
        atomic_torch_save({**payload, "config": new.raw, "trainer": trainer}, target)
        run.record_latest(target, iteration)
        positions = block_positions(old.game.model_config.n_layers, new.game.model_config.n_layers, fresh)
        run.update_metadata(
            status="prepared",
            prepared_iteration=iteration,
            parent=str(checkpoint),
            parent_iteration=iteration,
            grown={"from_layers": old_model["n_layers"], "to_layers": new_model["n_layers"],
                   "fresh": sorted(fresh), "trained_positions": positions},
        )
        run.log(
            f"Grown from {source.name}/{checkpoint.name} at iteration {iteration}: "
            f"{old_model['n_layers']} -> {new_model['n_layers']} layers, trained blocks at {positions}, "
            f"fresh blocks at {sorted(fresh)} (identity at start)."
        )
    print(run.path)
    return 0
