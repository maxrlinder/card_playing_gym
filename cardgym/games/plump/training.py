"""Plump training streams and auxiliary objectives.

Every auxiliary is a dense per-position loss over the stream's genuine
positions from the opening bid onward, weighted so each stream carries a
fixed total mass regardless of its length (see ``cardgym.rl.batch``). The
loss coefficients live in the [losses] config section; ``trick`` and
``next_winner`` weight the actor's and the critic's copies alike.
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

from .labels import BELIEF_STAGES, belief_stages, oracle_labels, replay_labels
from .selfplay import PlumpSelfPlay
from .tokens import (
    IGNORE,
    NEXT_BID,
    NEXT_PLAY,
    NUM_CARDS,
    NUM_SUITS,
    SLOT_NEXT_ACTOR,
    SLOT_NEXT_PHASE,
    Schema,
    oracle_tokens,
)


@dataclass(frozen=True)
class PlumpLossConfig:
    """The [losses] section: auxiliary loss coefficients (0 disables a head)."""

    suit: float = 0.1
    rank_boundary: float = 0.2
    trick: float = 0.2
    next_winner: float = 0.2
    bid_hit: float = 0.2
    card_order: float = 0.05  # critic only

    def validate(self) -> None:
        if min(self.suit, self.rank_boundary, self.trick, self.next_winner, self.bid_hit, self.card_order) < 0:
            raise ValueError("[losses] coefficients must be non-negative.")


ACTOR_LABELS = {
    "suit": LabelSpec(True, IGNORE, "int8"),
    "rank_boundary": LabelSpec(True, IGNORE, "int8"),
    "trick_targets": LabelSpec(False, IGNORE, "int64"),
    "trick_masks": LabelSpec(True, False, "bool"),
    "next_winner": LabelSpec(True, IGNORE, "int64"),
    "bid_hit": LabelSpec(False, IGNORE, "int64"),
    "belief_stages": LabelSpec(False, -1, "int64"),
}
CRITIC_LABELS = {
    "trick_targets": LabelSpec(False, IGNORE, "int64"),
    "trick_masks": LabelSpec(True, False, "bool"),
    "next_winner": LabelSpec(True, IGNORE, "int64"),
    "card_order": LabelSpec(False, NUM_CARDS, "int8"),
    "cards_played": LabelSpec(True, 0, "int8"),
}


def training_sequences(schema: Schema, trajectory: Trajectory) -> GameSequences:
    game = trajectory.game
    assert isinstance(game, PlumpSelfPlay)
    players, hand_size = game.num_players, game.hand_size
    events = game.env.state.event_log
    actor: dict[int, TrainingSequence] = {}
    for seat in range(players):
        stream = trajectory.streams[seat]
        if stream.shape[0] != schema.seq_len(players, hand_size):
            raise AssertionError("A finished stream must cover the whole round.")
        labels = replay_labels(schema, game.initial_hands, events, seat, players, hand_size)
        plays = [d.position for d in trajectory.decisions if d.seat == seat and d.head == "play"]
        labels["belief_stages"] = belief_stages(plays, hand_size)
        actor[seat] = TrainingSequence(stream, schema.actor_dense_start(hand_size), labels)
    oracle = oracle_tokens(schema, trajectory.streams[0], players, hand_size, game.initial_hands)
    shift = (players - 1) * hand_size
    critic_positions = []
    for decision in trajectory.decisions:
        position = decision.position + shift
        phase = NEXT_BID if decision.head == "bid" else NEXT_PLAY
        if oracle[position, SLOT_NEXT_ACTOR] != decision.seat or oracle[position, SLOT_NEXT_PHASE] != phase:
            raise AssertionError("Critic address is not tied to the acting seat and phase.")
        critic_positions.append(position)
    critic = TrainingSequence(
        oracle,
        schema.oracle_dense_start(players, hand_size),
        oracle_labels(schema, actor[0].labels, oracle, events, players, hand_size),
    )
    return GameSequences(actor=actor, critic=critic, critic_positions=critic_positions)


def _chunk(group: SequenceGroup, name: str, dense: DenseRows, rows: int, device, per_position: bool = True):
    array = group.labels[name][dense.start : dense.start + rows]
    selected = array[dense.seq, dense.position] if per_position else array[dense.seq]
    return torch.from_numpy(np.ascontiguousarray(selected)).to(device)


def _masked_bce(logits: torch.Tensor, targets: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    labeled = targets != IGNORE
    bce = F.binary_cross_entropy_with_logits(logits, targets.clamp_min(0).to(logits.dtype), reduction="none")
    return (bce * labeled).sum(dim=dims) / labeled.sum(dim=dims).clamp_min(1)


def _feasible_ce(logits: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor):
    """Cross-entropy over feasible classes per seat; returns per-row loss and masked logits."""

    masked = logits.masked_fill(~masks, -1e9)
    gathered = torch.log_softmax(masked, dim=-1).gather(2, targets.clamp_min(0)[:, :, None]).squeeze(-1)
    labeled = (targets != IGNORE) & masks.any(dim=-1)
    per_row = (-gathered * labeled).sum(dim=1) / labeled.sum(dim=1).clamp_min(1)
    return per_row, masked, labeled


def _seat_ce(logits: torch.Tensor, targets: torch.Tensor, players: torch.Tensor):
    active = torch.arange(logits.shape[-1], device=logits.device)[None, :] < players[:, None]
    masked = logits.masked_fill(~active, -1e9)
    gathered = torch.log_softmax(masked, dim=-1).gather(1, targets.clamp_min(0)[:, None]).squeeze(-1)
    labeled = targets != IGNORE
    return -gathered * labeled, masked, labeled


def plackett_luce(
    logits: torch.Tensor,
    order: torch.Tensor,
    played: torch.Tensor,
    active: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Listwise NLL of each seat's remaining play order, and next-card accuracy.

    ``logits`` [rows, 52] score exact cards; hands are disjoint, so scores are
    normalized within each seat's remaining hand:
    ``prod_j exp(s[c_j]) / sum_{k >= j} exp(s[c_k])``. The NLL averages the
    non-trivial choices and then the seats, so hand size and table width do
    not rescale it.
    """

    logits = logits.float()
    order = order.long()
    played = played.long()
    if logits.ndim == 2:
        logits = logits[:, None, :].expand(-1, order.shape[1], -1)
    valid = order < NUM_CARDS
    scores = logits.gather(-1, order.clamp(0, NUM_CARDS - 1)).masked_fill(~valid, -1e9)
    suffix = torch.logcumsumexp(scores.flip(-1), dim=-1).flip(-1)
    slot = torch.arange(order.shape[-1], device=logits.device).view(1, 1, -1)
    cards = valid.sum(dim=-1)
    choices = (cards - played - 1).clamp_min(0)
    choice = active[..., None] & (slot >= played[..., None]) & (slot < (cards - 1)[..., None])
    nll = torch.where(choice, suffix - scores, torch.zeros_like(scores))
    per_seat = nll.sum(dim=-1) / choices.clamp_min(1)
    labeled = active & (choices > 0)
    per_state = (per_seat * labeled).sum(dim=-1) / labeled.sum(dim=-1).clamp_min(1)
    remaining = valid & (slot >= played[..., None])
    predicted = scores.masked_fill(~remaining, -1e9).argmax(dim=-1)
    return per_state, (predicted == played) & labeled, labeled


class PlumpHooks:
    def __init__(self, losses: PlumpLossConfig, schema: Schema, device: torch.device, precision: str) -> None:
        self.losses = losses
        self.schema = schema
        self.device = device
        self.precision = precision

    # -- actor ----------------------------------------------------------- #

    def actor_auxiliary(self, model, hidden, dense: DenseRows, group: SequenceGroup, totals: Totals):
        losses = self.losses
        heads = [
            name
            for name, coef in (
                ("suit", losses.suit),
                ("rank_boundary", losses.rank_boundary),
                ("trick", losses.trick),
                ("next_winner", losses.next_winner),
            )
            if coef > 0
        ]
        if not heads and losses.bid_hit <= 0:
            return None
        schema, device = self.schema, self.device
        rows = hidden.shape[0]
        flat = hidden.reshape(-1, hidden.shape[-1])
        with autocast_context(device, self.precision):
            # Project the whole rectangle, then drop padding: faster on MPS
            # than gathering wide hidden rows first.
            logits = {name: value[dense.flat] for name, value in model.auxiliary_logits(flat, heads).items()}
            bid_hit = model.bid_hit_head(flat[dense.flat]) if losses.bid_hit > 0 else None
        stages = self._stage_rows(group, dense, rows)
        weight = dense.weight
        total = torch.zeros((), device=device)
        count = len(dense)
        if losses.suit > 0:
            suit_logits = logits["suit"].float().view(count, schema.opponents, NUM_SUITS)
            targets = _chunk(group, "suit", dense, rows, device)
            loss = (_masked_bce(suit_logits, targets, (1, 2)) * weight).sum()
            total = total + losses.suit * loss
            totals["suit_loss"].append(loss.detach())
            self._stage_binary(totals, "suit", suit_logits, targets, stages)
        if losses.rank_boundary > 0:
            boundary = logits["rank_boundary"].float().view(count, schema.opponents + 1, NUM_SUITS, 2)
            targets = _chunk(group, "rank_boundary", dense, rows, device)
            loss = (_masked_bce(boundary, targets, (1, 2, 3)) * weight).sum()
            total = total + losses.rank_boundary * loss
            totals["rank_boundary_loss"].append(loss.detach())
            self._stage_balanced(totals, boundary, targets, stages)
        if losses.trick > 0:
            trick = logits["trick"].float().view(count, schema.max_players, schema.bid_count)
            targets = _chunk(group, "trick_targets", dense, rows, device, per_position=False)
            masks = _chunk(group, "trick_masks", dense, rows, device)
            per_row, masked, labeled = _feasible_ce(trick, targets, masks)
            loss = (per_row * weight).sum()
            total = total + losses.trick * loss
            totals["trick_loss"].append(loss.detach())
            self._stage_classes(totals, "trick", masked, targets, labeled, stages)
        if bid_hit is not None:
            targets = _chunk(group, "bid_hit", dense, rows, device, per_position=False)
            loss = (_masked_bce(bid_hit.float(), targets, (1,)) * weight).sum()
            total = total + losses.bid_hit * loss
            totals["bid_hit_loss"].append(loss.detach())
        if losses.next_winner > 0:
            targets = _chunk(group, "next_winner", dense, rows, device)
            players = torch.from_numpy(group.player_counts[dense.start : dense.start + rows][dense.seq]).to(device)
            per_row, masked, labeled = _seat_ce(logits["next_winner"].float(), targets, players)
            loss = (per_row * weight).sum()
            total = total + losses.next_winner * loss
            totals["next_winner_loss"].append(loss.detach())
            totals["next_winner_correct"].append(((masked.argmax(-1) == targets) & labeled).sum().detach())
            totals["next_winner_total"].append(labeled.sum().detach())
        return total

    def _stage_rows(self, group: SequenceGroup, dense: DenseRows, rows: int) -> list[torch.Tensor]:
        """Dense-row index of each 10-card belief stage readout, per stage."""

        lookup = np.full((rows, group.length), -1, dtype=np.int64)
        lookup[dense.seq, dense.position] = np.arange(len(dense))
        stages = group.labels["belief_stages"][dense.start : dense.start + rows]
        selected = []
        for column in range(len(BELIEF_STAGES)):
            seq = np.flatnonzero(stages[:, column] >= 0)
            index = lookup[seq, stages[seq, column]]
            if np.any(index < 0):
                raise AssertionError("A belief stage readout has no dense row.")
            selected.append(torch.from_numpy(index).to(self.device))
        return selected

    @staticmethod
    def _stage_binary(totals: Totals, head: str, logits, targets, stages) -> None:
        for stage, index in zip(BELIEF_STAGES, stages):
            if not index.numel():
                continue
            selected, labels = logits[index], targets[index]
            labeled = labels != IGNORE
            correct = ((selected >= 0) == (labels == 1)) & labeled
            totals[f"{head}_stage_{stage}_correct"].append(correct.sum().detach())
            totals[f"{head}_stage_{stage}_total"].append(labeled.sum().detach())

    @staticmethod
    def _stage_balanced(totals: Totals, logits, targets, stages) -> None:
        for stage, index in zip(BELIEF_STAGES, stages):
            if not index.numel():
                continue
            predicted = logits[index] >= 0
            labels = targets[index]
            positive = labels == 1
            negative = labels == 0
            prefix = f"rank_boundary_stage_{stage}"
            totals[f"{prefix}_tp"].append((predicted & positive).sum().detach())
            totals[f"{prefix}_positive"].append(positive.sum().detach())
            totals[f"{prefix}_tn"].append((~predicted & negative).sum().detach())
            totals[f"{prefix}_negative"].append(negative.sum().detach())

    @staticmethod
    def _stage_classes(totals: Totals, head: str, masked, targets, labeled, stages) -> None:
        for stage, index in zip(BELIEF_STAGES, stages):
            if not index.numel():
                continue
            correct = (masked[index].argmax(-1) == targets[index]) & labeled[index]
            totals[f"{head}_stage_{stage}_correct"].append(correct.sum().detach())
            totals[f"{head}_stage_{stage}_total"].append(labeled[index].sum().detach())

    # -- critic ---------------------------------------------------------- #

    def critic_auxiliary(self, critic, hidden, dense: DenseRows, group: SequenceGroup, totals: Totals, record: bool):
        losses = self.losses
        coefficients = (
            ("trick", losses.trick),
            ("card_order", losses.card_order),
            ("next_winner", losses.next_winner),
        )
        heads = [name for name, coef in coefficients if coef > 0]
        if not heads:
            return None
        schema, device = self.schema, self.device
        rows = hidden.shape[0]
        with autocast_context(device, self.precision):
            flat = hidden.reshape(-1, hidden.shape[-1])
            logits = {name: value[dense.flat] for name, value in critic.auxiliary_logits(flat, heads).items()}
        weight = dense.weight
        total = torch.zeros((), device=device)
        trick_targets = _chunk(group, "trick_targets", dense, rows, device, per_position=False)
        if losses.trick > 0:
            trick = logits["trick"].float().view(len(dense), schema.max_players, schema.bid_count)
            masks = _chunk(group, "trick_masks", dense, rows, device)
            per_row, masked, labeled = _feasible_ce(trick, trick_targets, masks)
            loss = (per_row * weight).sum()
            total = total + losses.trick * loss
            totals["oracle_trick_loss"].append(loss.detach())
            if record:
                totals["oracle_trick_correct"].append(((masked.argmax(-1) == trick_targets) & labeled).sum().detach())
                totals["oracle_trick_total"].append(labeled.sum().detach())
        if losses.next_winner > 0:
            targets = _chunk(group, "next_winner", dense, rows, device)
            players = torch.from_numpy(group.player_counts[dense.start : dense.start + rows][dense.seq]).to(device)
            per_row, masked, labeled = _seat_ce(logits["next_winner"].float(), targets, players)
            loss = (per_row * weight).sum()
            total = total + losses.next_winner * loss
            totals["oracle_next_winner_loss"].append(loss.detach())
            if record:
                totals["oracle_next_winner_correct"].append(((masked.argmax(-1) == targets) & labeled).sum().detach())
                totals["oracle_next_winner_total"].append(labeled.sum().detach())
        if losses.card_order > 0:
            order = _chunk(group, "card_order", dense, rows, device, per_position=False)
            played = _chunk(group, "cards_played", dense, rows, device)
            active_seats = (trick_targets != IGNORE).sum(dim=1)
            active = torch.arange(schema.max_players, device=device)[None, :] < active_seats[:, None]
            per_state, correct, labeled = plackett_luce(logits["card_order"], order, played, active)
            loss = (per_state * weight).sum()
            total = total + losses.card_order * loss
            totals["oracle_card_order_loss"].append(loss.detach())
            if record:
                totals["oracle_card_order_correct"].append(correct.sum().detach())
                totals["oracle_card_order_total"].append(labeled.sum().detach())
        return total

    # -- metrics --------------------------------------------------------- #

    def summarize(self, reduced: dict[str, float]) -> dict[str, float]:
        stats: dict[str, float] = {}

        def ratio(numerator: str, denominator: str) -> float | None:
            total = reduced.get(denominator, 0.0)
            return reduced.get(numerator, 0.0) / total if total > 0 else None

        for name in ("suit", "rank_boundary", "trick", "bid_hit", "next_winner"):
            if f"{name}_loss" in reduced:
                stats[f"loss_{name}"] = reduced[f"{name}_loss"]
        for name in ("trick", "next_winner", "card_order"):
            if f"oracle_{name}_loss" in reduced:
                stats[f"loss_oracle_{name}"] = reduced[f"oracle_{name}_loss"]
        for key, (numerator, denominator) in {
            "next_winner_accuracy": ("next_winner_correct", "next_winner_total"),
            "oracle_trick_accuracy": ("oracle_trick_correct", "oracle_trick_total"),
            "oracle_next_winner_accuracy": ("oracle_next_winner_correct", "oracle_next_winner_total"),
            "oracle_card_order_top1": ("oracle_card_order_correct", "oracle_card_order_total"),
        }.items():
            value = ratio(numerator, denominator)
            if value is not None:
                stats[key] = value
        for stage in BELIEF_STAGES:
            for head in ("suit", "trick"):
                value = ratio(f"{head}_stage_{stage}_correct", f"{head}_stage_{stage}_total")
                if value is not None:
                    stats[f"{head}_accuracy_10c_{stage}"] = value
            recalls = [
                reduced[f"rank_boundary_stage_{stage}_{hit}"] / reduced[f"rank_boundary_stage_{stage}_{kind}"]
                for hit, kind in (("tp", "positive"), ("tn", "negative"))
                if reduced.get(f"rank_boundary_stage_{stage}_{kind}", 0.0) > 0
            ]
            if recalls:
                stats[f"rank_boundary_balanced_accuracy_10c_{stage}"] = sum(recalls) / len(recalls)
        return stats
