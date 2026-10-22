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

from .cards import MAX_DECKS, NUM_CARDS
from .engine import DISCARD, EVENT_MELD, EVENT_TAKE, MAX_PLAYERS, STOCK, TAKE
from .model import RummyActor, RummyCritic, RummyModelConfig
from .selfplay import RummySelfPlay
from .tokens import F_HAND, F_PLAYERS, max_stream_length
from .training import ACTOR_LABELS, CRITIC_LABELS, RummyHooks, RummyLossConfig, training_sequences


@dataclass(frozen=True)
class RummyGameConfig:
    """The [game] section: which rounds one update plays.

    Every (players, decks, hand size) combination is a shape with its own
    rollout lane, so an update mixes them all.
    """

    player_counts: tuple[int, ...] = (2, 3, 4)
    decks: tuple[int, ...] = (1, 2)
    hand_sizes: tuple[int, ...] = (7,)
    games_per_shape: int = 72
    opening_points: int = 0
    # Safety bound: discard take/discard cycles need not consume the stock.
    max_turns: int = 200

    def validate(self) -> None:
        if not self.player_counts or not self.decks or not self.hand_sizes or self.games_per_shape < 1:
            raise ValueError("[game] needs player counts, decks, hand sizes, and games_per_shape >= 1.")
        if any(not 2 <= players <= MAX_PLAYERS for players in self.player_counts):
            raise ValueError("[game] player counts must be 2..4.")
        if any(not 1 <= decks <= MAX_DECKS for decks in self.decks):
            raise ValueError("[game] decks must be 1 or 2.")
        if any(not 3 <= size <= 20 for size in self.hand_sizes):
            raise ValueError("[game] hand sizes must be 3..20.")
        for players, decks, size in self.shapes():
            if players * size + 2 > NUM_CARDS * decks:
                raise ValueError(f"[game] {players} players x {size} cards do not fit in {decks} deck(s).")
        if self.opening_points < 0 or self.max_turns < 1:
            raise ValueError("[game] opening_points must be >= 0 and max_turns >= 1.")

    def shapes(self) -> list[tuple[int, int, int]]:
        return [
            (players, decks, size)
            for players in self.player_counts
            for decks in self.decks
            for size in self.hand_sizes
        ]


_PLAY_KINDS = np.array([kind not in (STOCK, TAKE, DISCARD) for kind in range(DISCARD + 1)])


def _hold_decisions(trajectories: list[Trajectory]) -> dict[str, float]:
    """How often a seat that could table a card ended its turn instead (sampled actions).

    The endgame is an opponent down to two cards or fewer: holding then risks
    the cards' penalty when that opponent goes out.
    """

    holds = [0, 0]  # [all, endgame]
    counts = [0, 0]
    for trajectory in trajectories:
        for decision in trajectory.decisions:
            if decision.head != "play":
                continue
            kind = decision.features["kind"].astype(np.int64)
            legal = decision.probs > 0
            if not (_PLAY_KINDS[kind] & legal).any() or not ((kind == DISCARD) & legal).any():
                continue
            token = trajectory.streams[decision.seat][decision.position]
            players = int(token[F_PLAYERS])
            endgame = int(min(token[F_HAND + 1 : F_HAND + players])) <= 2
            held = int(kind[decision.action] == DISCARD)
            holds[0] += held
            counts[0] += 1
            holds[1] += held * endgame
            counts[1] += endgame
    return {
        "hold_rate": holds[0] / max(counts[0], 1),
        "hold_rate_endgame": holds[1] / max(counts[1], 1),
    }


class RummyGame(Game):
    """Scandinavian rummy: 1-2 decks, no jokers, cyclic runs, deep discard pickups, 500 Rum scoring."""

    name = "rummy"
    ModelConfig = RummyModelConfig
    GameConfig = RummyGameConfig
    LossConfig = RummyLossConfig
    policy_heads = ("draw", "play")
    entropy_gates = tuple(RummyHooks.gates)
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
            self.deal(players, decks, hand_size, rng)
            for players, decks, hand_size in config.shapes()
            for _ in range(config.games_per_shape)
        ]

    def deal(self, players: int, decks: int, hand_size: int, rng: random.Random) -> RummySelfPlay:
        config = self.game_config
        return RummySelfPlay(
            players,
            hand_size,
            seed=rng.getrandbits(48),
            decks=decks,
            opening_points=config.opening_points,
            max_turns=config.max_turns,
        )

    def lanes(self, concurrency: float) -> list[Lane]:
        config = self.game_config
        quota = config.games_per_shape
        return [
            Lane(
                name=f"{players}p{decks}d{hand_size}",
                players=players,
                in_flight=max(1, math.ceil(quota * concurrency)),
                quota=quota,
                deal=functools.partial(self.deal, players, decks, hand_size),
            )
            for players, decks, hand_size in config.shapes()
        ]

    def sequences(self, trajectory: Trajectory) -> GameSequences:
        return training_sequences(trajectory)

    def hooks(self, device: torch.device, precision: str) -> RummyHooks:
        return RummyHooks(self.loss_config, device, precision)

    def rollout_summary(self, trajectories: list[Trajectory]) -> dict[str, float]:
        games = [trajectory.game for trajectory in trajectories]
        engines = [game.engine for game in games]
        draws = sum(decision.head == "draw" for t in trajectories for decision in t.decisions)
        takes = [event for game in games for event in game.events if event.kind == EVENT_TAKE]
        deep = [len(event.cards) for event in takes if len(event.cards) > 1]
        melds = [sum(event.kind == EVENT_MELD for event in game.events) for game in games]
        tokens = [len(stream) for t in trajectories for stream in t.streams if len(stream)]
        holds = _hold_decisions(trajectories)
        return {
            **holds,
            "win_rate": float(np.mean([engine.winner is not None for engine in engines])),
            "deep_take_rate": len(deep) / max(draws, 1),
            "cards_per_deep_take": float(np.mean(deep)) if deep else 0.0,
            "truncated_rate": float(np.mean([engine.truncated for engine in engines])),
            "turns_per_game": float(np.mean([engine.turns for engine in engines])),
            "events_per_game": float(np.mean([len(game.events) for game in games])),
            "tokens_per_stream": float(np.mean(tokens)) if tokens else 0.0,
            "longest_stream": float(max(tokens, default=0)),
            "take_rate": len(takes) / max(draws, 1),
            "melds_per_game": float(np.mean(melds)),
            "mean_hand_penalty": float(np.mean([engine.penalties().mean() for engine in engines])),
            "mean_melded_points": float(np.mean([engine.points.mean() for engine in engines])),
        }

    def log_summary(self, summary: dict[str, float]) -> str:
        return (
            f"win {summary.get('win_rate', 0.0):.2f} turns {summary.get('turns_per_game', 0.0):.0f} "
            f"take {summary.get('take_rate', 0.0):.2f} hold {summary.get('hold_rate', 0.0):.3f}"
        )

    def dashboard_panels(self) -> list[Panel]:
        return [
            Panel("Round outcome (self-play)", ("win_rate", "truncated_rate", "take_rate", "deep_take_rate")),
            Panel(
                "Round length (self-play)",
                ("turns_per_game", "melds_per_game", "tokens_per_stream"),
                top=80.0,
            ),
            Panel("Points per seat (melded, left in hand)", ("mean_melded_points", "mean_hand_penalty")),
            Panel(
                "Hold or play (when a card could be tabled)",
                ("hold_probability", "hold_rate", "hold_rate_endgame", "hold_eligible_fraction"),
            ),
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
        evaluate.add_argument("--decks", default="1,2", help="comma-separated deck counts")
        evaluate.add_argument("--hand-size", type=int, default=7)
        evaluate.add_argument("--deals", type=int, default=16, help="deals per player and deck count")
        evaluate.add_argument("--seed", type=int, default=1234)
        evaluate.add_argument("--device", default="cpu")
        evaluate.add_argument("--output", type=Path, help="write the JSON report here")
        evaluate.set_defaults(func=_evaluate)

        simulate = subparsers.add_parser("simulate", help="rule-based self-play statistics (no model)")
        simulate.add_argument("--policy", choices=("heuristic", "random"), default="heuristic")
        simulate.add_argument("--players", default="2,3,4")
        simulate.add_argument("--decks", type=int, default=1)
        simulate.add_argument("--hand-size", type=int, default=7)
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
            engine = RummyEngine(players, args.hand_size, seed=game, decks=args.decks, max_turns=args.max_turns)
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
            f"{players}p {args.decks}d {args.policy}: events/game {np.mean(events):.0f} (max {max(events)}), "
            f"turns {np.mean(turns):.0f}, rounds with a winner {wins / args.games:.2f}, "
            f"mean |score| {np.mean(np.abs(np.concatenate(scores))):.1f}, "
            f"{1e6 * elapsed / sum(events):.0f} us/event"
        )
    return 0
