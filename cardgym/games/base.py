"""The contract between a game and the shared training machinery.

A game supplies rules, tokenization, models, and auxiliary objectives; the
gym supplies self-play decoding, PPO, run management, and the CLI. To add a
game, implement :class:`Game` and register it in ``cardgym.games.GAMES``.
"""

from __future__ import annotations

import abc
import argparse
import random
from typing import Any, ClassVar

import torch

from cardgym.core.dashboard import Panel
from cardgym.nn.model import SequenceModel
from cardgym.rl.batch import GameSequences, LabelSpec
from cardgym.rl.continuous import Lane
from cardgym.rl.selfplay import SelfPlayGame, Trajectory
from cardgym.rl.trainer import TrainingHooks


class PolicyModel(SequenceModel):
    """An actor: named policy heads plus a per-seat value probe."""

    policy_heads: tuple[str, ...] = ()

    def policy_logits(
        self,
        hidden: torch.Tensor,
        head: str,
        features: dict[str, torch.Tensor] | None = None,
        tokens: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Logits [N, actions] from readout hidden states [N, d].

        ``features`` are the decisions' padded candidate arrays (pointer
        heads); ``tokens`` [N, W] are the readout tokens themselves, so a head
        can read raw fields (such as a card ledger) at the decision point;
        ``context`` [N, ...] is the readouts' slice of :meth:`pop_context`.
        """

        raise NotImplementedError

    def pop_context(self) -> torch.Tensor | None:
        """Per-position features [B, T, ...] of the latest forward, handed over once.

        A model whose policy heads reuse something its embedding computed
        (Rummy's per-card features) keeps it from the forward and returns it
        here; callers index it like the hidden states. Returning it clears it,
        so no autograd graph outlives the step that built it.
        """

        return None

    def player_values(self, hidden: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class CriticModel(SequenceModel):
    def player_values(self, hidden: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Game(abc.ABC):
    """A registered game. Instances hold its resolved config sections."""

    name: ClassVar[str]
    ModelConfig: ClassVar[type]
    GameConfig: ClassVar[type]
    LossConfig: ClassVar[type]
    policy_heads: ClassVar[tuple[str, ...]]

    def __init__(self, model: Any, game: Any, losses: Any) -> None:
        self.model_config = model
        self.game_config = game
        self.loss_config = losses

    # -- shapes ----------------------------------------------------------
    @property
    @abc.abstractmethod
    def max_players(self) -> int: ...

    @property
    @abc.abstractmethod
    def max_length(self) -> int:
        """Longest actor stream."""

    @property
    @abc.abstractmethod
    def critic_max_length(self) -> int:
        """Longest critic stream."""

    # -- models ----------------------------------------------------------
    @abc.abstractmethod
    def build_actor(self) -> PolicyModel: ...

    @abc.abstractmethod
    def build_critic(self) -> CriticModel: ...

    # -- self-play -------------------------------------------------------
    @abc.abstractmethod
    def new_games(self, rng: random.Random, iteration: int) -> list[SelfPlayGame]:
        """One update's schedule of fresh self-play games."""

    def lanes(self, concurrency: float) -> list[Lane]:
        """Continuous rollout: the game kinds to keep running and each one's per-update quota."""

        raise NotImplementedError(f"{self.name} does not support [rollout] mode = \"continuous\".")

    @abc.abstractmethod
    def sequences(self, trajectory: Trajectory) -> GameSequences:
        """Training streams and labels for one finished game."""

    actor_labels: ClassVar[dict[str, LabelSpec]] = {}
    critic_labels: ClassVar[dict[str, LabelSpec]] = {}

    @abc.abstractmethod
    def hooks(self, device: torch.device, precision: str) -> TrainingHooks: ...

    def rollout_summary(self, trajectories: list[Trajectory]) -> dict[str, float]:
        return {}

    def dashboard_panels(self) -> list[Panel]:
        return []

    def log_summary(self, summary: dict[str, float]) -> str:
        """A few rollout numbers for the per-iteration log line."""

        return ""

    @classmethod
    def add_commands(cls, subparsers: argparse._SubParsersAction) -> None:
        """Register game-specific CLI commands (``cardgym <game> ...``)."""
