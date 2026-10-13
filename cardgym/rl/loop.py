"""The training loop behind ``cardgym <game> train``.

Each iteration: deal the schedule, decode it in self-play, derive training
streams, run one PPO update, append a metrics row, and checkpoint on the
interval. Re-running the same command resumes; a changed config is refused
unless ``--reconfigure`` records why (the old config is archived and a
config-compatible resume checkpoint is written first).
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from cardgym.core.config import config_diff, parse_section, section_dict
from cardgym.core.dashboard import DashboardRefresher, Panel
from cardgym.core.devices import autorelease_pool
from cardgym.core.metrics import MetricsLog
from cardgym.core.runs import RunDirectory

from .batch import build_batch
from .checkpoint import load_checkpoint, load_weights, restore_rng, save_checkpoint
from .setup import ResolvedConfig, build_components, load_config, metal_status, resolve_config

# Exit status asking tools/train_supervised.sh to restart from the checkpoint.
RECYCLE_EXIT_CODE = 75


def add_train_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("run", help="run name; runs/<game>/<run>")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="override one config value (TOML syntax)",
    )
    parser.add_argument("--iterations", type=int, help="shortcut for --set run.iterations=N")
    parser.add_argument("--device", help="shortcut for --set run.device=...")
    parser.add_argument(
        "--init-from",
        type=Path,
        help="new run only: start from a release directory or checkpoint's weights",
    )
    parser.add_argument(
        "--reset-iteration",
        action="store_true",
        help="with --init-from: count iterations from zero",
    )
    parser.add_argument("--reconfigure", action="store_true", help="adopt a changed config")
    parser.add_argument("--reason", default="", help="why the config changed (audit trail)")
    parser.add_argument(
        "--resume-checkpoint",
        default="latest",
        help="existing run: 'latest', an iteration number, or a path",
    )
    parser.add_argument(
        "--reset-entropy",
        action="append",
        default=[],
        metavar="HEAD",
        help="with --reconfigure: return a head's entropy temperature to its floor",
    )
    parser.add_argument(
        "--recycle-after",
        type=int,
        default=0,
        help=f"exit with status {RECYCLE_EXIT_CODE} at the first checkpoint after N iterations",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="create/fork/reconfigure the run and write its checkpoint, but do not train",
    )


def default_panels(heads: tuple[str, ...]) -> list[Panel]:
    return [
        Panel(
            "Policy entropy (normalized)",
            tuple(f"entropy_{h}" for h in heads) + tuple(f"entropy_target_{h}" for h in heads),
        ),
        Panel("Entropy temperatures", tuple(f"entropy_alpha_{h}" for h in heads)),
        Panel("PPO", ("clip_fraction", "behavior_kl")),
        Panel("Losses", ("loss_policy", "loss_critic_value", "actor_value_loss")),
        Panel(
            "Value error (reward units)",
            ("value_rmse", "value_zero_rmse", "critic_dense_rmse", "actor_value_rmse"),
        ),
        Panel("Value correlation", ("value_correlation", "critic_dense_correlation")),
        Panel(
            "Gradient norms",
            ("actor_grad_norm", "policy_grad_norm", "auxiliary_grad_norm", "critic_grad_norm"),
        ),
        Panel("Seconds per iteration", ("iteration_sec", "collect_sec", "build_sec", "update_sec")),
        Panel("Learning rate", ("learning_rate",), smooth=False),
    ]


def train(game_name: str, args: argparse.Namespace) -> int:
    overrides = list(args.overrides)
    if args.iterations is not None:
        overrides.append(f"run.iterations={args.iterations}")
    if args.device is not None:
        overrides.append(f'run.device="{args.device}"')
    requested = load_config(args.config, overrides)
    if requested.run.game != game_name:
        raise ValueError(f"{args.config} configures {requested.run.game!r}, not {game_name!r}.")
    run = RunDirectory(game_name, args.run)
    existed = run.exists
    if existed and args.init_from is not None:
        raise ValueError("--init-from requires a new run name.")
    if not existed and args.reconfigure:
        raise ValueError("--reconfigure requires an existing run.")
    if args.reset_entropy and not args.reconfigure:
        raise ValueError("--reset-entropy requires --reconfigure.")
    if args.recycle_after < 0:
        raise ValueError("--recycle-after must be >= 0.")

    with run.acquire_lock():
        changes: list[str] = []
        if existed:
            recorded = run.recorded_config()
            changes = config_diff(recorded, requested.raw)
            if changes and not args.reconfigure:
                raise ValueError(
                    "Run configuration differs from the recorded config "
                    "(pass --reconfigure --reason ... to adopt it):\n  " + "\n  ".join(changes)
                )
            config = requested if changes else resolve_config(recorded)
        else:
            run.create(requested.raw, [Path(sys.argv[0]).name, *sys.argv[1:]])
            config = requested
        return _train(run, config, args, existed=existed, changes=changes)


def _train(
    run: RunDirectory,
    config: ResolvedConfig,
    args: argparse.Namespace,
    *,
    existed: bool,
    changes: list[str],
) -> int:
    seed = config.run.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    components = build_components(config, device="cpu" if args.prepare_only else None)
    trainer = components.trainer
    game = config.game
    rng = random.Random(seed)
    metrics = MetricsLog(run.metrics)

    def save(path: Path, iteration: int) -> None:
        save_checkpoint(
            path,
            game=game.name,
            iteration=iteration,
            raw_config=config.raw,
            trainer=trainer,
            rng=rng,
        )

    if existed:
        checkpoint = run.resolve_checkpoint(args.resume_checkpoint)
        payload = load_checkpoint(checkpoint)
        trainer.load_state_dict(payload["trainer"])
        restore_rng(payload, rng)
        iteration = int(payload["iteration"])
        for head in dict.fromkeys(args.reset_entropy):
            before = trainer.entropy.alpha(head)
            trainer.entropy.reset(head)
            changes.append(f"entropy alpha ({head}): {before:.6g} -> {trainer.entropy.alpha(head):.6g} (reset)")
        dropped = metrics.truncate_after(iteration)
        run.record_latest(checkpoint, iteration)
        run.log(f"Resumed {run.name} from {checkpoint.name} at iteration {iteration}.")
        if dropped:
            run.log(f"Discarded {dropped} metric rows newer than the resumed checkpoint.")
        if changes:
            resume = run.checkpoints / f"resume_{iteration:06d}_reconfigured.pt"
            suffix = 1
            while resume.exists():
                resume = run.checkpoints / f"resume_{iteration:06d}_reconfigured_{suffix}.pt"
                suffix += 1
            save(resume, iteration)
            run.record_latest(resume, iteration)
            archive = run.record_reconfiguration(
                config.raw,
                iteration=iteration,
                reason=args.reason,
                changes=changes,
                resume_checkpoint=resume,
            )
            run.log(f"Reconfigured at iteration {iteration}; previous config archived as {archive.name}.")
    elif args.init_from is not None:
        source = args.init_from.expanduser().resolve()
        weights = load_weights(source, game.name)
        source_model = parse_section(type(game.model_config), weights["model"], "model")
        if source_model != game.model_config:
            differences = config_diff(
                {"model": section_dict(source_model)}, {"model": section_dict(game.model_config)}
            )
            raise ValueError(
                f"[model] does not match {source}:\n  " + "\n  ".join(differences)
            )
        trainer.actor.load_state_dict(weights["actor"])
        if weights["critic"] is not None:
            trainer.critic.load_state_dict(weights["critic"])
        else:
            run.log("Source has no critic weights; the critic starts fresh.")
        if weights["entropy"]:
            trainer.entropy.load_state_dict(weights["entropy"], load_optimizer=False)
        iteration = 0 if args.reset_iteration else weights["iteration"]
        initial = run.interval_checkpoint(iteration)
        save(initial, iteration)
        run.record_latest(initial, iteration)
        run.update_metadata(parent=str(source), parent_iteration=weights["iteration"])
        run.log(f"Initialized {run.name} from {source} at iteration {iteration}.")
    else:
        iteration = 0
        initial = run.interval_checkpoint(0)
        save(initial, 0)
        run.record_latest(initial, 0)
        run.log(f"Created {run.name} on {components.device}; saved {initial.name}.")

    if args.prepare_only:
        run.update_metadata(status="prepared", prepared_iteration=iteration)
        return 0

    target = config.run.iterations
    run.update_metadata(status="running", device=str(components.device), target_iterations=target)
    run.log(
        f"Training {game.name} on {components.device} ({config.run.precision}, "
        f"metal trunk {metal_status(components)}): "
        f"actor {_count(trainer.actor)} params, critic {_count(trainer.critic)} params."
    )
    dashboard = DashboardRefresher(
        run.metrics,
        run.dashboard,
        default_panels(game.policy_heads) + game.dashboard_panels(),
        f"{game.name} / {run.name}",
        config.run.dashboard_every,
    )
    elapsed = metrics.last_value("elapsed_sec")
    first = iteration + 1
    for iteration in range(first, target + 1):
        with autorelease_pool():
            row = _iteration(components, rng, iteration)
            elapsed += row["iteration_sec"]
            row["elapsed_sec"] = elapsed
            metrics.append(row)
            dashboard.maybe_refresh(iteration)
            due = config.run.checkpoint_every > 0 and iteration % config.run.checkpoint_every == 0
            if due or iteration == target:
                checkpoint = run.interval_checkpoint(iteration)
                save(checkpoint, iteration)
                run.record_latest(checkpoint, iteration)
            run.log(_log_line(row, game))
            if (
                args.recycle_after
                and due
                and iteration != target
                and iteration - first + 1 >= args.recycle_after
            ):
                dashboard.finish()
                run.log(f"Recycling the process at checkpoint {iteration}; a supervisor resumes it.")
                return RECYCLE_EXIT_CODE
    dashboard.maybe_refresh(target, force=True)
    dashboard.finish()
    run.update_metadata(status="complete", completed_iteration=target)
    return 0


def _iteration(components, rng: random.Random, iteration: int) -> dict[str, float]:
    config = components.config
    game = config.game
    trainer = components.trainer
    collector = components.collector
    started = time.perf_counter()
    trainer.actor.eval()
    trajectories = collector.collect(game.new_games(rng, iteration))
    collector.release()
    collected = time.perf_counter()
    batch = build_batch(
        trajectories,
        [game.sequences(trajectory) for trajectory in trajectories],
        max_players=game.max_players,
        actor_labels=game.actor_labels,
        critic_labels=game.critic_labels,
        bucket_width=config.ppo.sequence_bucket_width,
        critic_bucket_width=config.ppo.critic_sequence_bucket_width,
        max_length=game.max_length,
        critic_max_length=game.critic_max_length,
        min_group_sequences=config.ppo.min_group_sequences,
    )
    built = time.perf_counter()
    stats = trainer.update(batch, iteration)
    finished = time.perf_counter()
    collection = collector.stats
    row: dict[str, float] = {
        "iteration": iteration,
        "iteration_sec": finished - started,
        "collect_sec": collected - started,
        "build_sec": built - collected,
        "games": collection.games,
        "decisions": collection.decisions,
        "waves": collection.waves,
        "forward_rows": collection.forward_rows,
        "padding_fraction": collection.padded_rows / max(collection.forward_rows, 1),
        "compactions": collection.compactions,
        # Host time blocked on the device: the rest of collect_sec is host work.
        "collect_wait_sec": collection.wait_sec,
    }
    row.update(game.rollout_summary(trajectories))
    row.update(stats)
    return row


def _log_line(row: dict[str, float], game) -> str:
    heads = " ".join(
        f"{head} {row.get(f'entropy_{head}', 0.0):.3f}" for head in game.policy_heads
    )
    summary = game.log_summary(row)
    return (
        f"iter {int(row['iteration']):6d} | {row['iteration_sec']:6.1f}s "
        f"(collect {row['collect_sec']:.1f} update {row['update_sec']:.1f}) "
        f"| games {int(row['games'])} decisions {int(row['decisions'])} "
        f"| entropy {heads} | clip {row.get('clip_fraction', 0.0):.3f} "
        f"| V rmse {row.get('value_rmse', 0.0):.2f}"
        + (f" | {summary}" if summary else "")
        + (" | SKIPPED non-finite step" if row.get("skipped_actor_steps") or row.get("skipped_critic_steps") else "")
    )


def _count(model: torch.nn.Module) -> str:
    return f"{sum(p.numel() for p in model.parameters()):,}"
