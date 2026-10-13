"""Plump registration: config sections, self-play schedule, CLI commands."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from cardgym.core.dashboard import Panel
from cardgym.games.base import Game
from cardgym.rl.batch import GameSequences
from cardgym.rl.selfplay import Trajectory

from .model import PlumpActor, PlumpCritic, PlumpModelConfig
from .selfplay import PlumpSelfPlay
from .training import ACTOR_LABELS, CRITIC_LABELS, PlumpHooks, PlumpLossConfig, training_sequences


@dataclass(frozen=True)
class PlumpGameConfig:
    """The [game] section: which rounds one update plays."""

    hand_sizes: tuple[int, ...] = (3, 4, 5, 6, 7, 8, 9, 10)
    player_counts: tuple[int, ...] = (3, 4, 5)
    # Independent self-play rounds per (players, hand size) shape; every
    # seat of every round is a learning seat.
    deals_per_shape: int = 116

    def validate(self) -> None:
        if not self.hand_sizes or not self.player_counts or self.deals_per_shape < 1:
            raise ValueError("[game] needs hand sizes, player counts, and deals_per_shape >= 1.")
        if any(not 3 <= size <= 10 for size in self.hand_sizes):
            raise ValueError("[game] hand sizes must be 3..10.")
        if any(not 3 <= players <= 5 for players in self.player_counts):
            raise ValueError("[game] player counts must be 3..5.")


class PlumpGame(Game):
    """Plump: bid exactly how many tricks you will take (3-5 players, no trump)."""

    name = "plump"
    ModelConfig = PlumpModelConfig
    GameConfig = PlumpGameConfig
    LossConfig = PlumpLossConfig
    policy_heads = ("bid", "play")
    actor_labels = ACTOR_LABELS
    critic_labels = CRITIC_LABELS

    def __init__(self, model: PlumpModelConfig, game: PlumpGameConfig, losses: PlumpLossConfig) -> None:
        super().__init__(model, game, losses)
        if max(game.player_counts) > model.max_players or max(game.hand_sizes) > model.max_hand_size:
            raise ValueError("[game] shapes exceed the model's max_players/max_hand_size.")
        self.schema = model.schema()

    @property
    def max_players(self) -> int:
        return self.model_config.max_players

    @property
    def max_length(self) -> int:
        return self.schema.max_seq_len

    @property
    def critic_max_length(self) -> int:
        return self.schema.oracle_max_seq_len

    def build_actor(self) -> PlumpActor:
        return PlumpActor(self.model_config)

    def build_critic(self) -> PlumpCritic:
        return PlumpCritic(self.model_config)

    def new_games(self, rng: random.Random, iteration: int) -> list[PlumpSelfPlay]:
        games = []
        for players in self.game_config.player_counts:
            for hand_size in self.game_config.hand_sizes:
                for _ in range(self.game_config.deals_per_shape):
                    games.append(
                        PlumpSelfPlay(
                            self.schema,
                            players,
                            hand_size,
                            bidding_start=rng.randrange(players),
                            seed=rng.getrandbits(48),
                        )
                    )
        return games

    def sequences(self, trajectory: Trajectory) -> GameSequences:
        return training_sequences(self.schema, trajectory)

    def hooks(self, device: torch.device, precision: str) -> PlumpHooks:
        return PlumpHooks(self.loss_config, self.schema, device, precision)

    def rollout_summary(self, trajectories: list[Trajectory]) -> dict[str, float]:
        hits = bids = 0
        points = []
        overbid = []
        for trajectory in trajectories:
            round_state = trajectory.game.env.state.current_round
            for bid in round_state.bids:
                bids += 1
                hits += int(round_state.tricks_won[bid.player] == bid.value)
            points.extend(round_state.round_scores.values())
            overbid.append(sum(bid.value for bid in round_state.bids) - round_state.hand_size)
        return {
            "bid_hit_rate": hits / max(bids, 1),
            "points_per_seat": float(np.mean(points)) if points else 0.0,
            "bid_sum_minus_tricks": float(np.mean(overbid)) if overbid else 0.0,
        }

    def log_summary(self, summary: dict[str, float]) -> str:
        return f"bid hit {summary.get('bid_hit_rate', 0.0):.3f}"

    def dashboard_panels(self) -> list[Panel]:
        stages = (0, 4, 8)
        return [
            Panel("Bid hit rate (self-play)", ("bid_hit_rate",)),
            Panel("Points per seat (self-play)", ("points_per_seat", "bid_sum_minus_tricks")),
            Panel(
                "Actor auxiliary losses",
                ("loss_suit", "loss_rank_boundary", "loss_trick", "loss_bid_hit", "loss_next_winner"),
            ),
            Panel(
                "Belief accuracy, 10 cards (before plays 1/5/9)",
                tuple(f"suit_accuracy_10c_{s}" for s in stages)
                + tuple(f"trick_accuracy_10c_{s}" for s in stages)
                + tuple(f"rank_boundary_balanced_accuracy_10c_{s}" for s in stages),
            ),
            Panel(
                "Oracle critic auxiliaries",
                (
                    "oracle_trick_accuracy",
                    "oracle_next_winner_accuracy",
                    "oracle_card_order_top1",
                    "next_winner_accuracy",
                ),
            ),
        ]

    @classmethod
    def add_commands(cls, subparsers: argparse._SubParsersAction) -> None:
        evaluate = subparsers.add_parser(
            "evaluate", help="play a model against the heuristic (or another model) on a fixed deal bank"
        )
        evaluate.add_argument("model", type=Path, help="release directory, actor.pt, or checkpoint")
        evaluate.add_argument("--opponent", default="heuristic", help="'heuristic' or another model path")
        evaluate.add_argument("--deals", type=int, default=32, help="deals per (players, hand size) shape")
        evaluate.add_argument("--seed", type=int, default=1234)
        evaluate.add_argument("--mode", choices=("argmax", "sample"), default="argmax")
        evaluate.add_argument("--device", default="cpu")
        evaluate.add_argument("--batch-size", type=int, default=256)
        evaluate.add_argument("--output", type=Path, help="write the JSON report here")
        evaluate.set_defaults(func=_evaluate)

        play = subparsers.add_parser("play", help="play against a model in the browser")
        play.add_argument("model", type=Path, nargs="?", default=Path("checkpoints/plump/rl-v6"))
        play.add_argument("--device", default="cpu")
        play.add_argument("--host", default="127.0.0.1")
        play.add_argument("--port", type=int, default=8765)
        play.set_defaults(func=_play)

        legacy = subparsers.add_parser(
            "import-legacy", help="convert a plump-bot (schema v6) checkpoint into a release"
        )
        legacy.add_argument("checkpoint", type=Path)
        legacy.add_argument("--out", type=Path, required=True)
        legacy.set_defaults(func=_import_legacy)


def _evaluate(args: argparse.Namespace) -> int:
    from .evaluation import evaluate_command

    return evaluate_command(args)


def _play(args: argparse.Namespace) -> int:
    from .gui.app import serve

    return serve(args.model, device=args.device, host=args.host, port=args.port)


def _import_legacy(args: argparse.Namespace) -> int:
    from cardgym.core.config import section_dict
    from cardgym.core.io import file_sha256
    from cardgym.rl.checkpoint import export_release

    from .legacy import load_legacy_checkpoint

    converted = load_legacy_checkpoint(args.checkpoint)
    actor = PlumpActor(converted["model_config"])
    actor.load_state_dict(converted["actor"])
    critic = PlumpCritic(converted["model_config"])
    critic.load_state_dict(converted["critic"])
    directory = export_release(
        args.out,
        game="plump",
        iteration=converted["iteration"],
        model_config=section_dict(converted["model_config"]),
        actor_state=actor.state_dict(),
        critic_state=critic.state_dict(),
        entropy_state=converted["entropy"],
        source={
            "legacy_checkpoint": args.checkpoint.name,
            "legacy_sha256": file_sha256(args.checkpoint),
            "legacy_training_config": converted["training_config"],
        },
    )
    print(directory)
    return 0
