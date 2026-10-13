"""Rummy registration: config sections, self-play schedule, CLI commands."""

from __future__ import annotations

import argparse
import functools
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from cardgym.core.dashboard import Panel
from cardgym.games.base import Game
from cardgym.rl.batch import GameSequences
from cardgym.rl.continuous import Lane
from cardgym.rl.selfplay import Trajectory

from .engine import EVENT_MELD, EVENT_TAKE, MAX_PLAYERS
from .model import RummyActor, RummyCritic, RummyModelConfig
from .selfplay import RummySelfPlay
from .tokens import max_stream_length
from .training import ACTOR_LABELS, CRITIC_LABELS, RummyHooks, RummyLossConfig, training_sequences


@dataclass(frozen=True)
class RummyGameConfig:
    """The [game] section: which rounds one update plays."""

    player_counts: tuple[int, ...] = (2, 3, 4)
    hand_sizes: tuple[int, ...] = (13,)
    games_per_shape: int = 96
    opening_points: int = 30
    # Safety bound: discard take/discard cycles need not consume the stock.
    max_turns: int = 200

    def validate(self) -> None:
        if not self.player_counts or not self.hand_sizes or self.games_per_shape < 1:
            raise ValueError("[game] needs player counts, hand sizes, and games_per_shape >= 1.")
        if any(not 2 <= players <= MAX_PLAYERS for players in self.player_counts):
            raise ValueError("[game] player counts must be 2..4.")
        if any(not 3 <= size <= 20 for size in self.hand_sizes):
            raise ValueError("[game] hand sizes must be 3..20.")
        if self.opening_points < 0 or self.max_turns < 1:
            raise ValueError("[game] opening_points must be >= 0 and max_turns >= 1.")


class RummyGame(Game):
    """Rummy: two decks, no jokers, cyclic runs, 30-point opening (2-4 players)."""

    name = "rummy"
    ModelConfig = RummyModelConfig
    GameConfig = RummyGameConfig
    LossConfig = RummyLossConfig
    policy_heads = ("draw", "play")
    actor_labels = ACTOR_LABELS
    critic_labels = CRITIC_LABELS

    @property
    def max_players(self) -> int:
        return MAX_PLAYERS

    @property
    def max_length(self) -> int:
        return max_stream_length(self.game_config.max_turns)

    @property
    def critic_max_length(self) -> int:
        return self.max_length

    def build_actor(self) -> RummyActor:
        return RummyActor(self.model_config)

    def build_critic(self) -> RummyCritic:
        return RummyCritic(self.model_config)

    def new_games(self, rng: random.Random, iteration: int) -> list[RummySelfPlay]:
        config = self.game_config
        return [
            self.deal(players, hand_size, rng)
            for players in config.player_counts
            for hand_size in config.hand_sizes
            for _ in range(config.games_per_shape)
        ]

    def deal(self, players: int, hand_size: int, rng: random.Random) -> RummySelfPlay:
        config = self.game_config
        return RummySelfPlay(
            players,
            hand_size,
            seed=rng.getrandbits(48),
            opening_points=config.opening_points,
            max_turns=config.max_turns,
        )

    def lanes(self, concurrency: float) -> list[Lane]:
        config = self.game_config
        quota = config.games_per_shape
        return [
            Lane(
                name=f"{players}p{hand_size}",
                players=players,
                in_flight=max(1, math.ceil(quota * concurrency)),
                quota=quota,
                deal=functools.partial(self.deal, players, hand_size),
            )
            for players in config.player_counts
            for hand_size in config.hand_sizes
        ]

    def sequences(self, trajectory: Trajectory) -> GameSequences:
        return training_sequences(trajectory)

    def hooks(self, device: torch.device, precision: str) -> RummyHooks:
        return RummyHooks(self.loss_config, device, precision)

    def rollout_summary(self, trajectories: list[Trajectory]) -> dict[str, float]:
        games = [trajectory.game for trajectory in trajectories]
        engines = [game.engine for game in games]
        draws = sum(decision.head == "draw" for t in trajectories for decision in t.decisions)
        takes = sum(event.kind == EVENT_TAKE for game in games for event in game.events)
        melds = [sum(event.kind == EVENT_MELD for event in game.events) for game in games]
        tokens = [len(stream) for t in trajectories for stream in t.streams if len(stream)]
        return {
            "win_rate": float(np.mean([engine.winner is not None for engine in engines])),
            "rummy_rate": float(np.mean([engine.rummy for engine in engines])),
            "truncated_rate": float(np.mean([engine.truncated for engine in engines])),
            "turns_per_game": float(np.mean([engine.turns for engine in engines])),
            "events_per_game": float(np.mean([len(game.events) for game in games])),
            "tokens_per_stream": float(np.mean(tokens)) if tokens else 0.0,
            "longest_stream": float(max(tokens, default=0)),
            "take_rate": takes / max(draws, 1),
            "melds_per_game": float(np.mean(melds)),
            "mean_hand_penalty": float(np.mean([engine.penalties().mean() for engine in engines])),
        }

    def log_summary(self, summary: dict[str, float]) -> str:
        return (
            f"win {summary.get('win_rate', 0.0):.2f} turns {summary.get('turns_per_game', 0.0):.0f} "
            f"take {summary.get('take_rate', 0.0):.2f}"
        )

    def dashboard_panels(self) -> list[Panel]:
        return [
            Panel("Round outcome (self-play)", ("win_rate", "rummy_rate", "truncated_rate", "take_rate")),
            Panel("Round length (self-play)", ("turns_per_game", "melds_per_game", "tokens_per_stream")),
            Panel("Mean hand penalty at the end", ("mean_hand_penalty",)),
            Panel("Actor auxiliary losses", ("loss_hands", "loss_winner", "loss_penalty")),
            Panel(
                "Beliefs and outcome prediction",
                ("hands_accuracy", "hands_held_recall", "winner_accuracy", "oracle_winner_accuracy"),
            ),
        ]

    @classmethod
    def add_commands(cls, subparsers: argparse._SubParsersAction) -> None:
        evaluate = subparsers.add_parser("evaluate", help="play a model against a rule-based opponent")
        evaluate.add_argument("model", type=Path, help="release directory, actor.pt, or checkpoint")
        evaluate.add_argument("--opponent", choices=("heuristic", "random"), default="heuristic")
        evaluate.add_argument("--players", default="2,3,4", help="comma-separated player counts")
        evaluate.add_argument("--hand-size", type=int, default=13)
        evaluate.add_argument("--deals", type=int, default=16, help="deals per player count")
        evaluate.add_argument("--seed", type=int, default=1234)
        evaluate.add_argument("--device", default="cpu")
        evaluate.add_argument("--output", type=Path, help="write the JSON report here")
        evaluate.set_defaults(func=_evaluate)

        simulate = subparsers.add_parser("simulate", help="rule-based self-play statistics (no model)")
        simulate.add_argument("--policy", choices=("heuristic", "random"), default="heuristic")
        simulate.add_argument("--players", default="2,3,4")
        simulate.add_argument("--hand-size", type=int, default=13)
        simulate.add_argument("--games", type=int, default=100)
        simulate.add_argument("--max-turns", type=int, default=200)
        simulate.set_defaults(func=_simulate)


def _evaluate(args: argparse.Namespace) -> int:
    from .evaluation import evaluate_command

    return evaluate_command(args)


def _simulate(args: argparse.Namespace) -> int:
    from .engine import RummyEngine
    from .heuristic import HeuristicPolicy, RandomPolicy

    for players in (int(p) for p in args.players.split(",")):
        events, turns, wins, scores = [], [], 0, []
        started = time.perf_counter()
        for game in range(args.games):
            engine = RummyEngine(players, args.hand_size, seed=game, max_turns=args.max_turns)
            policy = HeuristicPolicy() if args.policy == "heuristic" else RandomPolicy(game)
            count = 0
            while not engine.done:
                engine.apply(engine.candidates()[policy.act(engine)])
                count += 1
            events.append(count + 1)
            turns.append(engine.turns)
            wins += engine.winner is not None
            scores.append(engine.scores())
        elapsed = time.perf_counter() - started
        print(
            f"{players}p {args.policy}: events/game {np.mean(events):.0f} (max {max(events)}), "
            f"turns {np.mean(turns):.0f}, rounds with a winner {wins / args.games:.2f}, "
            f"mean |score| {np.mean(np.abs(np.concatenate(scores))):.1f}, "
            f"{1e6 * elapsed / sum(events):.0f} us/event"
        )
    return 0
