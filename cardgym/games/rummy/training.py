"""Rummy training streams, labels, and auxiliary objectives.

Each seat has an actor stream and an oracle (critic) stream with the same
positions (its decision tokens and its opponents' turn tokens), so a
decision's actor position is also its critic position. Auxiliaries:

* ``hands`` (actor, decision tokens): copies (0/1/2) of every card type in
  every opponent's hand, supervised only where the viewer cannot already know
  the answer (some copy of the type is unseen). This is the belief state the
  trunk must build from pickups, passes, and discards;
* ``winner`` (actor and critic, every position): who goes out, or nobody;
* ``penalty`` (actor and critic, every position): each seat's final hand
  value / 100.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from cardgym.core.devices import autocast_context
from cardgym.rl.batch import GameSequences, LabelSpec, SequenceGroup, TrainingSequence
from cardgym.rl.selfplay import Trajectory
from cardgym.rl.trainer import DenseRows, Totals

from .cards import NUM_CARDS
from .engine import DISCARD, LOC_UNSEEN, MAX_PLAYERS, NUM_PAIRS, PAIR_LOCATIONS
from .model import COUNT_CLASSES, NO_WINNER, OPPONENTS
from .selfplay import RummySelfPlay
from .tokens import CARD_START, F_KIND, F_PHASE, KIND_DECISION

IGNORE = -1
PENALTY_SCALE = 100.0

# Unseen copies per ledger pair code.
UNSEEN_COPIES = (PAIR_LOCATIONS == LOC_UNSEEN).sum(axis=1)


@dataclass(frozen=True)
class RummyLossConfig:
    """The [losses] section: auxiliary loss coefficients (0 disables a head)."""

    hands: float = 0.2
    winner: float = 0.1
    penalty: float = 0.1

    def validate(self) -> None:
        if min(self.hands, self.winner, self.penalty) < 0:
            raise ValueError("[losses] coefficients must be non-negative.")


ACTOR_LABELS = {
    "hands": LabelSpec(True, IGNORE, "int8"),
    "winner": LabelSpec(False, IGNORE, "int64"),
    "penalty": LabelSpec(False, 0.0, "float32"),
}
CRITIC_LABELS = {
    "winner": LabelSpec(False, IGNORE, "int64"),
    "penalty": LabelSpec(False, 0.0, "float32"),
}


def training_sequences(trajectory: Trajectory) -> GameSequences:
    game = trajectory.game
    assert isinstance(game, RummySelfPlay)
    engine = game.engine
    players = game.num_players
    penalties = engine.penalties() / PENALTY_SCALE
    actor: dict[int, TrainingSequence] = {}
    critic: dict[int, TrainingSequence] = {}
    for seat in range(players):
        stream = trajectory.streams[seat]
        length = stream.shape[0]
        if length == 0:
            continue  # the round ended before this seat's first decision
        oracle = np.stack(game.oracle[seat])
        hands = np.stack(game.hands[seat])  # [L, P, 52] true hands at each token
        if oracle.shape[0] != length or hands.shape[0] != length:
            raise AssertionError("Actor and oracle streams must have the same positions.")
        decision = stream[:, F_KIND] == KIND_DECISION
        # Decision tokens' card codes are state codes (pair + NUM_PAIRS * layable).
        unseen = (UNSEEN_COPIES[stream[:, CARD_START:] % NUM_PAIRS] > 0) & decision[:, None]  # [L, 52]
        beliefs = np.full((length, OPPONENTS, NUM_CARDS), IGNORE, dtype=np.int8)
        for offset in range(1, players):
            opponent = hands[:, (seat + offset) % players]
            beliefs[:, offset - 1] = np.where(unseen, opponent, IGNORE)
        relative_penalty = np.zeros(MAX_PLAYERS, dtype=np.float32)
        relative_penalty[:players] = np.roll(penalties, -seat)
        winner = np.int64(NO_WINNER if engine.winner is None else (engine.winner - seat) % players)
        actor[seat] = TrainingSequence(
            stream, 0, {"hands": beliefs, "winner": winner, "penalty": relative_penalty}
        )
        critic[seat] = TrainingSequence(oracle, 0, {"winner": winner, "penalty": relative_penalty})
    critic_positions = []
    for decision in trajectory.decisions:
        phase = 0 if decision.head == "draw" else 1
        token = critic[decision.seat].tokens[decision.position]
        if token[F_KIND] != KIND_DECISION or token[F_PHASE] != phase:
            raise AssertionError("Critic address is not tied to the acting seat's decision.")
        critic_positions.append(decision.position)
    return GameSequences(actor=actor, critic=critic, critic_positions=critic_positions)


def _labels(group: SequenceGroup, name: str, dense: DenseRows, rows: int, upload, per_position: bool):
    array = group.labels[name][dense.start : dense.start + rows]
    return upload(array[dense.seq, dense.position] if per_position else array[dense.seq])


def _players(group: SequenceGroup, dense: DenseRows, rows: int, upload) -> torch.Tensor:
    return upload(group.player_counts[dense.start : dense.start + rows][dense.seq])


def _winner_terms(logits: torch.Tensor, targets: torch.Tensor, players: torch.Tensor):
    classes = torch.arange(logits.shape[-1], device=logits.device)[None, :]
    allowed = (classes < players[:, None]) | (classes == NO_WINNER)
    masked = logits.float().masked_fill(~allowed, -1e9)
    loss = F.cross_entropy(masked, targets.clamp_min(0), reduction="none")
    return loss, masked.argmax(-1) == targets


def _penalty_terms(prediction: torch.Tensor, targets: torch.Tensor, players: torch.Tensor):
    active = (torch.arange(prediction.shape[-1], device=prediction.device)[None, :] < players[:, None]).float()
    error = (prediction.float() - targets) * active
    return 0.5 * error.square().sum(-1) / players.float()


class RummyHooks:
    # Hold or play: at a play decision that could table a card (meld or lay-off)
    # and could also end the turn, the binary split between ending it now (any
    # discard: holding everything still playable) and tabling one more card.
    # With three-card melds and single-card lay-offs (see the engine), playing
    # some cards and holding the rest is a sequence of these splits. Its
    # normalized entropy has its own target ([ppo] entropy_targets.hold).
    gates = {"hold": "play"}

    @staticmethod
    def gate_side(gate: str, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """The candidates that hold: the discards."""

        if gate != "hold":
            raise ValueError(f"Unknown rummy gate {gate!r}.")
        return features["kind"] == DISCARD

    def __init__(self, losses: RummyLossConfig, device: torch.device, precision: str) -> None:
        self.losses = losses
        self.device = device
        self.precision = precision
        # Host arrays -> device; the trainer installs its non-blocking uploader.
        self.upload = lambda array: torch.from_numpy(np.ascontiguousarray(array)).to(device)

    def actor_auxiliary(self, model, hidden, dense: DenseRows, group: SequenceGroup, totals: Totals):
        losses = self.losses
        if max(losses.hands, losses.winner, losses.penalty) <= 0:
            return None
        device = self.device
        rows = dense.rows
        flat = hidden.reshape(-1, hidden.shape[-1])
        weight = dense.weight
        total = torch.zeros((), device=device)
        players = _players(group, dense, rows, self.upload)
        hands = None
        if losses.hands > 0:
            # Only decision tokens carry labels; a row without any adds nothing.
            labels = group.labels["hands"][dense.start : dense.start + rows][dense.seq, dense.position]
            keep = np.flatnonzero((labels != IGNORE).any(axis=(1, 2)))
            at = self.upload(keep)
        with autocast_context(device, self.precision):
            if losses.hands > 0:
                hands = model.hands_head(flat[dense.flat[at]])
            winner = model.winner_head(flat)[dense.flat] if losses.winner > 0 else None
            penalty = model.penalty_head(flat)[dense.flat] if losses.penalty > 0 else None
        if hands is not None:
            targets = self.upload(labels[keep]).long()
            logits = hands.float().view(keep.size, OPPONENTS, NUM_CARDS, COUNT_CLASSES)
            labeled = targets != IGNORE
            nll = F.cross_entropy(logits.permute(0, 3, 1, 2), targets.clamp_min(0), reduction="none")
            per_row = (nll * labeled).sum(dim=(1, 2)) / labeled.sum(dim=(1, 2)).clamp_min(1)
            loss = (per_row * weight[at]).sum()
            total = total + losses.hands * loss
            totals["hands_loss"].append(loss.detach())
            correct = (logits.argmax(-1) == targets) & labeled
            held = labeled & (targets > 0)
            totals["hands_correct"].append(correct.sum().detach())
            totals["hands_total"].append(labeled.sum().detach())
            totals["hands_held_correct"].append((correct & held).sum().detach())
            totals["hands_held_total"].append(held.sum().detach())
        if winner is not None:
            targets = _labels(group, "winner", dense, rows, self.upload, False)
            per_row, correct = _winner_terms(winner, targets, players)
            loss = (per_row * weight).sum()
            total = total + losses.winner * loss
            totals["winner_loss"].append(loss.detach())
            totals["winner_correct"].append((correct.float() * weight).sum().detach())
            totals["winner_weight"].append(weight.sum().detach())
        if penalty is not None:
            targets = _labels(group, "penalty", dense, rows, self.upload, False)
            loss = (_penalty_terms(penalty, targets, players) * weight).sum()
            total = total + losses.penalty * loss
            totals["penalty_loss"].append(loss.detach())
        return total

    def critic_auxiliary(self, critic, hidden, dense: DenseRows, group: SequenceGroup, totals: Totals, record: bool):
        losses = self.losses
        if max(losses.winner, losses.penalty) <= 0:
            return None
        device = self.device
        rows = dense.rows
        flat = hidden.reshape(-1, hidden.shape[-1])
        weight = dense.weight
        players = _players(group, dense, rows, self.upload)
        total = torch.zeros((), device=device)
        with autocast_context(device, self.precision):
            winner = critic.winner_head(flat)[dense.flat] if losses.winner > 0 else None
            penalty = critic.penalty_head(flat)[dense.flat] if losses.penalty > 0 else None
        if winner is not None:
            targets = _labels(group, "winner", dense, rows, self.upload, False)
            per_row, correct = _winner_terms(winner, targets, players)
            loss = (per_row * weight).sum()
            total = total + losses.winner * loss
            totals["oracle_winner_loss"].append(loss.detach())
            if record:
                totals["oracle_winner_correct"].append((correct.float() * weight).sum().detach())
                totals["oracle_winner_weight"].append(weight.sum().detach())
        if penalty is not None:
            targets = _labels(group, "penalty", dense, rows, self.upload, False)
            loss = (_penalty_terms(penalty, targets, players) * weight).sum()
            total = total + losses.penalty * loss
            totals["oracle_penalty_loss"].append(loss.detach())
        return total

    def summarize(self, reduced: dict[str, float]) -> dict[str, float]:
        stats: dict[str, float] = {}
        for name in ("hands", "winner", "penalty", "oracle_winner", "oracle_penalty"):
            if f"{name}_loss" in reduced:
                stats[f"loss_{name}"] = reduced[f"{name}_loss"]
        for key, (numerator, denominator) in {
            "hands_accuracy": ("hands_correct", "hands_total"),
            "hands_held_recall": ("hands_held_correct", "hands_held_total"),
            "winner_accuracy": ("winner_correct", "winner_weight"),
            "oracle_winner_accuracy": ("oracle_winner_correct", "oracle_winner_weight"),
        }.items():
            if reduced.get(denominator, 0.0) > 0:
                stats[key] = reduced[numerator] / reduced[denominator]
        return stats
