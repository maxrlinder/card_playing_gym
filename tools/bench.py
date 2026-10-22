"""Measure a run's PPO update and rollout on its own weights and recorded batches.

    CARDGYM_METAL_TRUNK=1 .venv/bin/python tools/bench.py record GAME RUN --out FILE [--batches 3]
    CARDGYM_METAL_TRUNK=1 .venv/bin/python tools/bench.py update GAME RUN --batches FILE [--repeat 2] [--set k=v]
    CARDGYM_METAL_TRUNK=1 .venv/bin/python tools/bench.py rollout GAME RUN [--collections 6]

``record`` collects batches in-process with the run's latest weights and
config (after warm-up collections, so the continuous pool is in its steady
state) and pickles the PPO batches. ``update`` replays them through
``PPOTrainer.update`` and splits the time into the critic baseline (or, for a
game without a critic, the actor's value pass), the actor steps, and the
critic epochs (each phase ends with a device synchronize).
``rollout`` times synchronous collections. ``--set`` overrides config values
(not [model]) for the measurement.
"""

from __future__ import annotations

import argparse
import pickle
import random
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cardgym.core.config import apply_overrides  # noqa: E402
from cardgym.core.devices import autorelease_pool  # noqa: E402
from cardgym.core.runs import RunDirectory  # noqa: E402
from cardgym.rl.batch import build_batch  # noqa: E402
from cardgym.rl.checkpoint import load_checkpoint  # noqa: E402
from cardgym.rl.setup import build_components, metal_status, resolve_config  # noqa: E402


def _components(game: str, run_name: str, overrides: list[str]):
    run = RunDirectory(game, run_name)
    payload = load_checkpoint(run.resolve_checkpoint("latest"))
    config = resolve_config(apply_overrides(payload["config"], overrides))
    components = build_components(config)
    components.trainer.load_state_dict(payload["trainer"])
    print(
        f"{run_name} iteration {payload['iteration']} on {components.device}, metal {metal_status(components)}, "
        f"{config.game.model_config.n_layers} layers, "
        f"{sum(lane.quota or 0 for lane in config.game.lanes(config.rollout.concurrency))} games per update"
    )
    return components, int(payload["iteration"])


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def record(args: argparse.Namespace) -> int:
    components, iteration = _components(args.game, args.run, args.overrides)
    config, game, collector = components.config, components.config.game, components.collector
    rng = random.Random(args.seed)
    batches = []
    for index in range(args.warmup + args.batches):
        started = time.perf_counter()
        trajectories = collector.collect(game.lanes(config.rollout.concurrency), rng, version=iteration)
        elapsed = time.perf_counter() - started
        sequences = [game.sequences(t) for t in trajectories]
        summary = game.rollout_summary(trajectories)
        stats = collector.stats
        print(
            f"collection {index}: {elapsed:.2f}s, {stats.decisions} decisions, {stats.waves} waves, "
            f"tokens/stream {summary['tokens_per_stream']:.1f}, longest {summary['longest_stream']:.0f}"
        )
        if index < args.warmup:
            continue
        batches.append(
            build_batch(
                trajectories, sequences,
                max_players=game.max_players,
                actor_labels=game.actor_labels,
                critic_labels=game.critic_labels,
                bucket_width=config.ppo.sequence_bucket_width,
                critic_bucket_width=config.ppo.critic_sequence_bucket_width,
                max_length=game.max_length,
                critic_max_length=game.critic_max_length,
                min_group_sequences=config.ppo.min_group_sequences,
            )
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as handle:
        pickle.dump({"iteration": iteration, "trajectories": None, "batches": batches}, handle)
    print(f"wrote {len(batches)} batches to {args.out}")
    return 0


class _Phases:
    """Wrap the trainer's phases with synchronized timers."""

    def __init__(self, trainer) -> None:
        self.trainer = trainer
        self.times: dict[str, list[float]] = {}
        for name, label in (("_baseline", "baseline"), ("_actor_baseline", "value"),
                            ("_actor_backward", "actor"), ("_critic_epoch", "critic")):
            self._wrap(name, label)

    def _wrap(self, name: str, label: str) -> None:
        method = getattr(self.trainer, name)
        device = self.trainer.device

        def timed(*args, **kwargs):
            _sync(device)
            started = time.perf_counter()
            result = method(*args, **kwargs)
            _sync(device)
            self.times.setdefault(label, []).append(time.perf_counter() - started)
            return result

        setattr(self.trainer, name, timed)


def update(args: argparse.Namespace) -> int:
    components, iteration = _components(args.game, args.run, args.overrides)
    trainer = components.trainer
    with args.batches.open("rb") as handle:
        recorded = pickle.load(handle)
    batches = recorded["batches"]
    phases = _Phases(trainer)
    totals = []
    for repeat in range(args.repeat + 1):
        for index, batch in enumerate(batches):
            phases.times.clear()
            with autorelease_pool():
                _sync(trainer.device)
                started = time.perf_counter()
                stats = trainer.update(batch, iteration + 1)
                _sync(trainer.device)
                elapsed = time.perf_counter() - started
            actor = sum(phases.times.get("actor", []))
            other = elapsed - sum(sum(v) for v in phases.times.values())
            line = (
                f"pass {repeat} batch {index}: {elapsed:.3f}s = baseline (host) {sum(phases.times['baseline']):.3f} "
                f"+ value pass {sum(phases.times.get('value', [])):.3f} "
                f"+ actor {actor:.3f} ({len(phases.times['actor'])} steps) "
                f"+ critic {sum(phases.times.get('critic', [])):.3f} "
                f"+ steps/other {other:.3f} | positions {stats['positions']:.0f} rows {stats['policy_rows']:.0f} "
                f"peak {stats['peak_device_bytes'] / 2**30:.2f} GiB"
            )
            print(line, flush=True)
            if repeat > 0:
                totals.append(elapsed)
    if totals:
        print(f"update: mean {statistics.mean(totals):.3f}s, min {min(totals):.3f}s over {len(totals)}")
    return 0


def rollout(args: argparse.Namespace) -> int:
    components, iteration = _components(args.game, args.run, args.overrides)
    config, game, collector = components.config, components.config.game, components.collector
    rng = random.Random(args.seed)
    times = []
    for index in range(args.collections):
        with autorelease_pool():
            started = time.perf_counter()
            trajectories = collector.collect(game.lanes(config.rollout.concurrency), rng, version=iteration)
            sequences = [game.sequences(t) for t in trajectories]
            elapsed = time.perf_counter() - started
        stats = collector.stats
        print(
            f"collection {index}: {elapsed:.2f}s ({stats.collect_sec:.2f} collect, "
            f"{elapsed - stats.collect_sec:.2f} sequences) | {stats.decisions} decisions, {stats.waves} waves, "
            f"{stats.decisions / stats.collect_sec:.0f}/s | launch {stats.forward_sec:.2f} wait {stats.wait_sec:.2f} "
            f"sample {stats.sample_sec:.2f} step {stats.step_sec:.2f} | in flight {stats.in_flight}",
            flush=True,
        )
        if index >= 2:
            times.append(elapsed)
        del sequences
    if times:
        print(f"rollout: mean {statistics.mean(times):.2f}s over {len(times)} steady-state collections")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, func in (("record", record), ("update", update), ("rollout", rollout)):
        command = commands.add_parser(name)
        command.add_argument("game")
        command.add_argument("run")
        command.add_argument("--set", dest="overrides", action="append", default=[], metavar="SECTION.KEY=VALUE")
        command.add_argument("--seed", type=int, default=0)
        command.set_defaults(func=func)
        if name == "record":
            command.add_argument("--out", type=Path, required=True)
            command.add_argument("--batches", type=int, default=3)
            command.add_argument("--warmup", type=int, default=2)
        elif name == "update":
            command.add_argument("--batches", type=Path, required=True)
            command.add_argument("--repeat", type=int, default=2)
        else:
            command.add_argument("--collections", type=int, default=6)
    args = parser.parse_args()
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
