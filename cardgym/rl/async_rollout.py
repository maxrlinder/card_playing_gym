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

Collection is host work (rules, tokens, candidates; ~50 us per decision for
Rummy), so ``[rollout] workers`` runs several such processes, each with its
share of every lane (``split_lanes``) and its own seed, all reading the same
shared weights. Their batches are concatenated in worker order.
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


def _serve(
    raw_config: dict[str, Any], seed: int, metal: bool, shared: dict, requests, results, worker: int, workers: int
) -> None:
    try:
        from cardgym.core.devices import resolve_device
        from cardgym.nn.metal_trunk import unsupported_reason

        from .continuous import ContinuousCollector, split_lanes
        from .setup import records_values, resolve_config

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
            cohorts=config.rollout.cohorts,
            record_values=records_values(config),
        )
        rng = random.Random(seed)
        lanes = split_lanes(game.lanes(config.rollout.concurrency), worker, workers)
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


def merge_summaries(summaries: list[dict[str, float]], games: list[int]) -> dict[str, float]:
    """Game-weighted means of the workers' rollout summaries; ``longest*`` keys take the maximum."""

    total = max(sum(games), 1)
    merged: dict[str, float] = {}
    for key in summaries[0]:
        values = [summary[key] for summary in summaries]
        if key.startswith("longest"):
            merged[key] = max(values)
        else:
            merged[key] = sum(value * count for value, count in zip(values, games)) / total
    return merged


def merge_stats(stats: list[CollectionStats]) -> CollectionStats:
    """Counts and host times add up; the collection took as long as its slowest worker."""

    merged = CollectionStats()
    for name in ("games", "decisions", "waves", "forward_rows", "padded_rows", "compactions",
                 "forward_sec", "wait_sec", "sample_sec", "step_sec", "in_flight"):
        setattr(merged, name, sum(getattr(entry, name) for entry in stats))
    merged.peak_rows = max(entry.peak_rows for entry in stats)
    merged.collect_sec = max(entry.collect_sec for entry in stats)
    decisions = max(merged.decisions, 1)
    merged.policy_lag = sum(entry.policy_lag * entry.decisions for entry in stats) / decisions
    return merged


class RolloutProcess:
    """Rollout worker processes serving one request at a time."""

    def __init__(
        self, raw_config: dict[str, Any], actor: torch.nn.Module, *, seed: int, metal: bool, workers: int = 1
    ) -> None:
        context = torch.multiprocessing.get_context("spawn")
        self.shared = {
            name: value.detach().to("cpu").clone().share_memory_() for name, value in actor.state_dict().items()
        }
        self.requests = [context.Queue() for _ in range(workers)]
        self.results = [context.Queue() for _ in range(workers)]
        self.processes = [
            context.Process(
                target=_serve,
                args=(raw_config, seed + 7_919 * worker, metal, self.shared, self.requests[worker],
                      self.results[worker], worker, workers),
                daemon=True,
                name=f"cardgym-rollout-{worker}",
            )
            for worker in range(workers)
        ]
        for process in self.processes:
            process.start()
        self.pending = False

    def request(self, actor: torch.nn.Module, version: int) -> None:
        """Collect a batch with ``actor``'s current weights, tagged ``version``."""

        if self.pending:
            raise RuntimeError("The rollout processes are already collecting.")
        with torch.no_grad():
            for name, value in actor.state_dict().items():
                self.shared[name].copy_(value)
        for requests in self.requests:
            requests.put(version)
        self.pending = True

    def receive(self) -> RolloutBatch:
        batches = [self._receive(worker) for worker in range(len(self.processes))]
        self.pending = False
        if len(batches) == 1:
            return batches[0]
        return RolloutBatch(
            trajectories=[t for batch in batches for t in batch.trajectories],
            sequences=[s for batch in batches for s in batch.sequences],
            summary=merge_summaries([b.summary for b in batches], [len(b.trajectories) for b in batches]),
            stats=merge_stats([b.stats for b in batches]),
        )

    def _receive(self, worker: int) -> RolloutBatch:
        while True:
            try:
                status, payload = self.results[worker].get(timeout=5.0)
                break
            except queue.Empty:
                process = self.processes[worker]
                if not process.is_alive():
                    raise RuntimeError(f"Rollout process {worker} died (exit code {process.exitcode}).")
        if status == "error":
            raise RuntimeError(f"Rollout process {worker} failed:\n{payload}")
        return payload

    def stop(self) -> None:
        for process, requests in zip(self.processes, self.requests):
            if process.is_alive():
                requests.put(None)
        for process in self.processes:
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
