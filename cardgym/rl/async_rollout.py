"""Rollout in its own process, overlapped with the update.

An update is GPU work; a collection is mostly host work (game rules, token
building, sampling, kernel dispatch). In one process they run back to back.
Here a rollout process collects the next update's games while the trainer
updates on the previous batch, so an iteration costs about max(collect,
update) rather than their sum.

The cost is one more version of policy lag: the batch trained at iteration
i + 1 is collected with the weights from before update i. Every decision
records its version and the probabilities it was sampled from, and PPO's
ratio is taken against those, so the update stays correct; ``policy_lag``
reports the staleness.

The rollout process builds the game, actor, and continuous collector from
the run's config. It also derives each finished game's training sequences
and the rollout summary (host work kept off the trainer's critical path)
and returns them with slim trajectories. Weights travel through shared-memory
CPU tensors: the trainer writes them, then sends the request; the rollout
process loads them before collecting, and the trainer writes again only after
it has received that batch.
"""

from __future__ import annotations

import queue
import random
import traceback
from dataclasses import dataclass
from typing import Any

import torch

from .batch import GameSequences
from .selfplay import CollectionStats, Trajectory


@dataclass
class SlimGame:
    """What the trainer needs of a finished game once its sequences are built."""

    num_players: int


@dataclass
class RolloutBatch:
    trajectories: list[Trajectory]  # games replaced by SlimGame, streams dropped
    sequences: list[GameSequences]
    summary: dict[str, float]
    stats: CollectionStats


def _serve(raw_config: dict[str, Any], seed: int, metal: bool, shared: dict, requests, results) -> None:
    try:
        from cardgym.core.devices import resolve_device
        from cardgym.nn.metal_trunk import unsupported_reason

        from .continuous import ContinuousCollector
        from .setup import resolve_config

        config = resolve_config(raw_config)
        game = config.game
        device = resolve_device(config.run.device)
        torch.manual_seed(seed)
        actor = game.build_actor().to(device).eval()
        if (
            metal
            and device.type == "mps"
            and config.run.precision == "bf16"
            and unsupported_reason(actor.trunk_config) is None
        ):
            actor.enable_metal_trunk(game.max_length)
        collector = ContinuousCollector(
            actor,
            device=device,
            precision=config.run.precision,
            kv_dtype=config.rollout.kv_dtype,
            max_length=game.max_length,
        )
        rng = random.Random(seed)
        lanes = game.lanes(config.rollout.concurrency)
        while True:
            version = requests.get()
            if version is None:
                return
            actor.load_state_dict(shared)
            trajectories = collector.collect(lanes, rng, version=version)
            batch = RolloutBatch(
                trajectories=[
                    Trajectory(
                        game=SlimGame(t.game.num_players), streams=[], decisions=t.decisions, returns=t.returns
                    )
                    for t in trajectories
                ],
                sequences=[game.sequences(t) for t in trajectories],
                summary=game.rollout_summary(trajectories),
                stats=collector.stats,
            )
            results.put(("ok", batch))
    except Exception:  # surfaced in the trainer, which owns the run log
        results.put(("error", traceback.format_exc()))


class RolloutProcess:
    """A rollout process serving one request at a time."""

    def __init__(self, raw_config: dict[str, Any], actor: torch.nn.Module, *, seed: int, metal: bool) -> None:
        context = torch.multiprocessing.get_context("spawn")
        self.shared = {
            name: value.detach().to("cpu").clone().share_memory_() for name, value in actor.state_dict().items()
        }
        self.requests = context.Queue()
        self.results = context.Queue()
        self.process = context.Process(
            target=_serve,
            args=(raw_config, seed, metal, self.shared, self.requests, self.results),
            daemon=True,
            name="cardgym-rollout",
        )
        self.process.start()
        self.pending = False

    def request(self, actor: torch.nn.Module, version: int) -> None:
        """Collect a batch with ``actor``'s current weights, tagged ``version``."""

        if self.pending:
            raise RuntimeError("The rollout process is already collecting.")
        with torch.no_grad():
            for name, value in actor.state_dict().items():
                self.shared[name].copy_(value)
        self.requests.put(version)
        self.pending = True

    def receive(self) -> RolloutBatch:
        while True:
            try:
                status, payload = self.results.get(timeout=5.0)
                break
            except queue.Empty:
                if not self.process.is_alive():
                    raise RuntimeError(f"The rollout process died (exit code {self.process.exitcode}).")
        self.pending = False
        if status == "error":
            raise RuntimeError(f"The rollout process failed:\n{payload}")
        return payload

    def stop(self) -> None:
        if self.process.is_alive():
            self.requests.put(None)
            self.process.join(timeout=30)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)
