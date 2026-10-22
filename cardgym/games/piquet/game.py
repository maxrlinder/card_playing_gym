"""Piquet registration: config sections, self-play lanes, rollout statistics, CLI commands."""

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

from .engine import HEADS, QUIET
from .model import PiquetActor, PiquetModelConfig
from .selfplay import MODES, PiquetSelfPlay
from .tokens import DEAL_TOKENS
from .training import LABELS, PiquetHooks, PiquetLossConfig, training_sequences


@dataclass(frozen=True)
class PiquetGameConfig:
    """The [game] section: what one update plays.

    ``modes`` are rollout lanes: "deal" is one deal scored by the point
    difference; "rubicon" a six-deal partie scored by the Rubicon payoff,
    each deal a fresh context with the score in its tokens (its returns are
    several times larger, so mixing both in one run lets the parties
    dominate the normalized advantages).
    """

    modes: tuple[str, ...] = ("deal",)
    games_per_mode: int = 1024

    def validate(self) -> None:
        if not self.modes or any(mode not in MODES for mode in self.modes):
            raise ValueError(f"[game] modes must be a non-empty subset of {MODES}.")
        if self.games_per_mode < 1:
            raise ValueError("[game] games_per_mode must be positive.")


class PiquetGame(Game):
    """Piquet: two players, 32 cards, exchange, declarations, and tricks (Rubicon scoring optional)."""

    name = "piquet"
    ModelConfig = PiquetModelConfig
    GameConfig = PiquetGameConfig
    LossConfig = PiquetLossConfig
    policy_heads = HEADS
    actor_labels = LABELS

    @property
    def max_players(self) -> int:
        return 2

    @property
    def max_length(self) -> int:
        # A partie restarts the context every deal.
        return DEAL_TOKENS

    def build_actor(self) -> PiquetActor:
        return PiquetActor(self.model_config)

    def new_games(self, rng: random.Random, iteration: int) -> list[PiquetSelfPlay]:
        return [
            PiquetSelfPlay(rng.getrandbits(48), mode=mode)
            for mode in self.game_config.modes
            for _ in range(self.game_config.games_per_mode)
        ]

    @staticmethod
    def deal(mode: str, rng: random.Random) -> PiquetSelfPlay:
        return PiquetSelfPlay(rng.getrandbits(48), mode=mode)

    def lanes(self, concurrency: float) -> list[Lane]:
        quota = self.game_config.games_per_mode
        return [
            Lane(
                name=mode,
                players=2,
                in_flight=max(1, math.ceil(quota * concurrency)),
                quota=quota,
                deal=functools.partial(self.deal, mode),
            )
            for mode in self.game_config.modes
        ]

    def sequences(self, trajectory: Trajectory) -> GameSequences:
        return training_sequences(trajectory)

    def hooks(self, device: torch.device, precision: str) -> PiquetHooks:
        return PiquetHooks(self.loss_config, device, precision)

    def rollout_summary(self, trajectories: list[Trajectory]) -> dict[str, float]:
        results = [result for t in trajectories for result in t.game.engine.results]
        if not results:
            return {}
        deals = len(results)
        declares = [d for t in trajectories for d in t.decisions if d.head == "declare"]
        tokens = [len(stream) for t in trajectories for stream in t.streams]
        elder = [r.points[r.elder] - r.points[1 - r.elder] for r in results]
        parties = [t.game.engine for t in trajectories if t.game.engine.rubicon]
        partie = {}
        if parties:
            # The loser's final total under 100: rubiconed (the payoff adds both totals).
            partie = {
                "rubicon_rate": float(np.mean([min(e.totals) < 100 and e.totals[0] != e.totals[1] for e in parties])),
                "partie_payoff": float(np.mean([abs(e.returns()[0]) for e in parties])),
                "partie_margin": float(np.mean([abs(e.totals[0] - e.totals[1]) for e in parties])),
            }
        return partie | {
            "elder_margin": float(np.mean(elder)),
            "points_per_deal": float(np.mean([sum(r.points) for r in results])) / 2,
            "elder_exchanged": float(np.mean([r.exchanged[r.elder] for r in results])),
            "younger_exchanged": float(np.mean([r.exchanged[1 - r.elder] for r in results])),
            "elder_tricks": float(np.mean([r.tricks[r.elder] for r in results])),
            "repique_rate": sum(r.repique >= 0 for r in results) / deals,
            "pique_rate": sum(r.pique >= 0 for r in results) / deals,
            "capot_rate": sum(r.capot >= 0 for r in results) / deals,
            "split_rate": sum(r.tricks[0] == 6 for r in results) / deals,
            "elder_point_rate": sum(r.category_winners[0] == r.elder for r in results) / deals,
            "elder_sequence_rate": sum(r.category_winners[1] == r.elder for r in results) / deals,
            "elder_sets_rate": sum(r.category_winners[2] == r.elder for r in results) / deals,
            "quiet_rate": sum(d.action == QUIET for d in declares) / max(len(declares), 1),
            "tokens_per_stream": float(np.mean(tokens)),
            "longest_stream": float(max(tokens)),
        }

    def log_summary(self, summary: dict[str, float]) -> str:
        line = (
            f"elder {summary.get('elder_margin', 0.0):+.1f} pts {summary.get('points_per_deal', 0.0):.1f} "
            f"swap {summary.get('elder_exchanged', 0.0):.1f}/{summary.get('younger_exchanged', 0.0):.1f} "
            f"pique {summary.get('pique_rate', 0.0):.3f} capot {summary.get('capot_rate', 0.0):.3f}"
        )
        if "partie_payoff" in summary:
            line += f" | payoff {summary['partie_payoff']:.0f} rubicon {summary['rubicon_rate']:.3f}"
        return line

    def dashboard_panels(self) -> list[Panel]:
        return [
            Panel("Deal (self-play)", ("elder_margin", "points_per_deal")),
            Panel("Partie (self-play)", ("partie_payoff", "partie_margin")),
            Panel("Rubicon rate (loser under 100)", ("rubicon_rate",)),
            Panel("Exchange (cards)", ("elder_exchanged", "younger_exchanged")),
            Panel(
                "Bonuses and declarations",
                ("repique_rate", "pique_rate", "capot_rate", "split_rate", "quiet_rate"),
            ),
            Panel("Elder wins the category", ("elder_point_rate", "elder_sequence_rate", "elder_sets_rate")),
            Panel(
                "Prediction losses",
                tuple(f"loss_{name}" for name in LABELS),
            ),
            Panel(
                "Prediction accuracy (deal outcome)",
                ("tricks_accuracy", "declarations_accuracy", "bonus_accuracy", "trick_accuracy", "fate_accuracy"),
            ),
            Panel(
                "Prediction accuracy (opponent)",
                ("belief_accuracy", "opp_suits_accuracy", "opp_next_accuracy"),
            ),
        ]

    @classmethod
    def add_commands(cls, subparsers: argparse._SubParsersAction) -> None:
        evaluate = subparsers.add_parser("evaluate", help="duplicate deals against a rule-based opponent or a model")
        evaluate.add_argument("model", type=Path, help="release directory, actor.pt, or checkpoint")
        evaluate.add_argument("--opponent", choices=("heuristic", "random"), default="heuristic")
        evaluate.add_argument(
            "--against", type=Path, default=None,
            help="play this model instead of --opponent (same trunk shape); an older checkpoint plays score-blind",
        )
        evaluate.add_argument("--mode", choices=MODES, default="deal")
        evaluate.add_argument("--deals", type=int, default=256, help="duplicate pairs (each played both ways)")
        evaluate.add_argument("--seed", type=int, default=1234)
        evaluate.add_argument("--device", default="cpu")
        evaluate.add_argument("--output", type=Path, help="write the JSON report here")
        evaluate.set_defaults(func=_evaluate)

        simulate = subparsers.add_parser("simulate", help="rule-based self-play statistics (no model)")
        simulate.add_argument("--policy", choices=("heuristic", "random"), default="heuristic")
        simulate.add_argument("--opponent", choices=("heuristic", "random"), default=None)
        simulate.add_argument("--games", type=int, default=1000)
        simulate.set_defaults(func=_simulate)


def _evaluate(args: argparse.Namespace) -> int:
    from .evaluation import evaluate_command

    return evaluate_command(args)


def _simulate(args: argparse.Namespace) -> int:
    from .engine import PiquetEngine
    from .heuristic import HeuristicPolicy, RandomPolicy

    def policy(name: str, seed: int):
        return HeuristicPolicy() if name == "heuristic" else RandomPolicy(seed)

    opponent = args.opponent or args.policy
    margins, first, started, decisions = [], [], time.perf_counter(), 0
    for game in range(args.games):
        for swap in (0, 1):
            seats = [policy(args.policy, game), policy(opponent, game + 1)]
            if swap:
                seats.reverse()
            engine = PiquetEngine(game)
            while not engine.done:
                engine.apply(seats[engine.current].act(engine))
                decisions += 1
            first.append(engine.returns()[swap])
            result = engine.results[0]
            margins.append(result.points[result.elder] - result.points[1 - result.elder])
    elapsed = time.perf_counter() - started
    print(
        f"{args.policy} vs {opponent}: {np.mean(first):+.2f} +- {np.std(first) / np.sqrt(len(first)):.2f} "
        f"points per deal; elder margin {np.mean(margins):+.2f}; {1e6 * elapsed / decisions:.0f} us/decision"
    )
    return 0
