"""Piquet training streams, labels, and prediction-head objectives.

Each seat contributes its own stream (a partie's split into one sequence
per deal, the contexts the policy saw); there is no critic stream (the PPO
baseline is the actor's value head at each decision). Every label is per
position, each labelled with the outcome of its own deal:

* ``belief`` [L, 32]: for each card the viewer cannot place (unknown, or
  shown by carte blanche before its holder exchanged), whether it is in the
  opponent's hand, its discards, or the talon. The card counting the trunk
  must do;
* ``tricks`` [L]: tricks the seat takes this deal (0-12);
* ``declarations`` [L, 3]: who wins the point, the sequence, and the sets
  (me, the opponent, nobody), only until the category is settled;
* ``points`` [L, 2]: both seats' points this deal / 50 (regression);
* ``bonus`` [L, 3]: repique, pique, capot: none, me, or the opponent;
* ``trick`` [L]: who wins the trick being played (or about to be led), in
  the play phase;
* ``opp_suits`` [L, 4]: the opponent's suit lengths after the exchange (0-8),
  the shape its guards and long suit come from;
* ``opp_next`` [L]: the next card the opponent plays, in the play phase
  (opponent modelling);
* ``fate`` [L, 32]: for every card not yet played, the trick it will fall in
  and whether it wins it, or that it is never played.
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

from .cards import NUM_CARDS, NUM_SUITS, suit_bits
from .engine import TRICKS
from .model import (
    BELIEF_CLASSES,
    BONUSES,
    CATEGORIES,
    FATE_CLASSES,
    NEVER_PLAYED,
    OUTCOME_CLASSES,
    PLAYERS,
    SUIT_LENGTHS,
)
from .selfplay import PiquetSelfPlay
from .tokens import CARD_START, F_KIND, KIND_DECISION, UNCERTAIN

IGNORE = -1
POINTS_SCALE = 50.0
NOBODY = 2
# Prediction rows are padded to a multiple of this: MPS compiles a graph per
# tensor shape, and with every microbatch a different size the heads' graphs
# were recompiled each time (1.1 s an update at 16k-position microbatches).
ROW_BUCKET = 8192


@dataclass(frozen=True)
class PiquetLossConfig:
    """The [losses] section: prediction-head coefficients (0 disables a head)."""

    belief: float = 0.2
    tricks: float = 0.1
    declarations: float = 0.1
    points: float = 0.1
    bonus: float = 0.05
    trick: float = 0.1
    opp_suits: float = 0.1
    opp_next: float = 0.1
    fate: float = 0.1

    def validate(self) -> None:
        if min(getattr(self, name) for name in LABELS) < 0:
            raise ValueError("[losses] coefficients must be non-negative.")


LABELS = {
    "belief": LabelSpec(True, IGNORE, "int8"),
    "tricks": LabelSpec(True, IGNORE, "int8"),
    "declarations": LabelSpec(True, IGNORE, "int8"),
    "points": LabelSpec(True, 0.0, "float32"),
    "bonus": LabelSpec(True, IGNORE, "int8"),
    "trick": LabelSpec(True, IGNORE, "int8"),
    "opp_suits": LabelSpec(True, IGNORE, "int8"),
    "opp_next": LabelSpec(True, IGNORE, "int8"),
    "fate": LabelSpec(True, IGNORE, "int8"),
}
# (classes per entry, entries per position) of each classification head.
CLASSIFIERS = {
    "belief": (BELIEF_CLASSES, NUM_CARDS),
    "tricks": (TRICKS + 1, 1),
    "declarations": (OUTCOME_CLASSES, CATEGORIES),
    "bonus": (OUTCOME_CLASSES, BONUSES),
    "trick": (PLAYERS, 1),
    "opp_suits": (SUIT_LENGTHS, NUM_SUITS),
    "opp_next": (NUM_CARDS, 1),
    "fate": (FATE_CLASSES, NUM_CARDS),
}


def _winner(seat: int, viewer: int) -> int:
    """Declarations: 0 me, 1 the opponent, 2 nobody."""

    return NOBODY if seat < 0 else int(seat != viewer)


def _bonus(seat: int, viewer: int) -> int:
    """Bonuses: 0 none, 1 me, 2 the opponent."""

    return 0 if seat < 0 else 1 + (seat != viewer)


def seat_labels(game: PiquetSelfPlay, seat: int, stream: np.ndarray) -> dict[str, np.ndarray]:
    length = stream.shape[0]
    meta = np.asarray(game.meta[seat], dtype=np.int64).reshape(-1, 4)
    truth = np.stack(game.truth[seat])
    if meta.shape[0] != length or truth.shape[0] != length:
        raise AssertionError("Every token needs its label metadata.")
    deal, settled, finished, played = meta.T
    results = game.engine.results
    tricks = np.array([result.tricks[seat] for result in results])
    points = np.array([[result.points[seat], result.points[1 - seat]] for result in results]) / POINTS_SCALE
    winners = np.array([[_winner(w, seat) for w in result.category_winners] for result in results])
    bonuses = np.array([[_bonus(s, seat) for s in (result.repique, result.pique, result.capot)] for result in results])
    trick_winners = np.array([[_winner(w, seat) for w in result.trick_winners] for result in results])
    in_play = (settled == CATEGORIES) & (finished < TRICKS)
    trick = np.full(length, IGNORE, dtype=np.int8)
    trick[in_play] = trick_winners[deal[in_play], finished[in_play]]
    open_category = settled[:, None] <= np.arange(CATEGORIES)[None, :]
    other = 1 - seat
    opp_suits = np.array([[bin(suit_bits(r.held[other], s)).count("1") for s in range(NUM_SUITS)] for r in results])
    # Per deal: the order in which each card is played (-1: never), its fate,
    # and the opponent's cards in play order with their order index.
    order = np.full((len(results), NUM_CARDS), -1)
    fates = np.full((len(results), NUM_CARDS), NEVER_PLAYED)
    opp_next = np.full((len(results), 2 * TRICKS + 1), IGNORE)
    for index, result in enumerate(results):
        upcoming = IGNORE
        for step in range(len(result.plays) - 1, -1, -1):
            player, card, trick_number, won = result.plays[step]
            order[index, card] = step
            fates[index, card] = 2 * trick_number + int(won)
            if player == other:
                upcoming = card
            opp_next[index, step] = upcoming
    pending = order[deal] >= played[:, None]
    fate = np.where(pending | (order[deal] < 0), fates[deal], IGNORE)
    next_card = np.where(settled == CATEGORIES, opp_next[deal, np.minimum(played, 2 * TRICKS)], IGNORE)
    return {
        "belief": np.where(UNCERTAIN[stream[:, CARD_START:]], truth, IGNORE).astype(np.int8),
        "tricks": tricks[deal].astype(np.int8),
        "declarations": np.where(open_category, winners[deal], IGNORE).astype(np.int8),
        "points": points[deal].astype(np.float32),
        "bonus": bonuses[deal].astype(np.int8),
        "trick": trick,
        "opp_suits": opp_suits[deal].astype(np.int8),
        "opp_next": next_card.astype(np.int8),
        "fate": fate.astype(np.int8),
    }


def training_sequences(trajectory: Trajectory) -> GameSequences:
    game = trajectory.game
    assert isinstance(game, PiquetSelfPlay)
    actor = {}
    for seat in (0, 1):
        stream = trajectory.streams[seat].astype(np.int64)
        labels = seat_labels(game, seat, stream)
        deals = np.asarray([meta[0] for meta in game.meta[seat]])
        bounds = [0, *np.flatnonzero(np.diff(deals)) + 1, len(stream)]
        segments = [
            TrainingSequence(stream[a:b], 0, {name: value[a:b] for name, value in labels.items()}, start=a)
            for a, b in zip(bounds[:-1], bounds[1:])
        ]
        actor[seat] = segments if len(segments) > 1 else segments[0]
    for decision in trajectory.decisions:
        if trajectory.streams[decision.seat][decision.position, F_KIND] != KIND_DECISION:
            raise AssertionError("A decision must read its seat's decision token.")
    return GameSequences(actor=actor, critic=None, critic_positions=[])


def _classification(logits: torch.Tensor, targets: torch.Tensor, classes: int):
    """Per-row mean cross entropy over labelled entries, and per-entry correctness."""

    logits = logits.float().view(*targets.shape, classes)
    labeled = targets != IGNORE
    nll = F.cross_entropy(logits.reshape(-1, classes), targets.clamp_min(0).reshape(-1), reduction="none")
    nll = nll.view(targets.shape) * labeled
    per_row = nll.sum(-1) / labeled.sum(-1).clamp_min(1)
    correct = (logits.argmax(-1) == targets) & labeled
    return per_row, labeled, correct


class PiquetHooks:
    def __init__(self, losses: PiquetLossConfig, device: torch.device, precision: str) -> None:
        self.losses = losses
        self.device = device
        self.precision = precision
        # Host arrays -> device; the trainer installs its non-blocking uploader.
        self.upload = lambda array: torch.from_numpy(np.ascontiguousarray(array)).to(device)

    def _labels(self, group: SequenceGroup, name: str, dense: DenseRows, rows: int, padded: int) -> torch.Tensor:
        array = group.labels[name][dense.start : dense.start + rows][dense.seq, dense.position]
        if padded > array.shape[0]:
            fill = np.full((padded - array.shape[0], *array.shape[1:]), LABELS[name].fill, dtype=array.dtype)
            array = np.concatenate([array, fill])
        return self.upload(array)

    def actor_auxiliary(self, model, hidden, dense: DenseRows, group: SequenceGroup, totals: Totals):
        coefficients = {name: getattr(self.losses, name) for name in LABELS}
        if max(coefficients.values()) <= 0:
            return None
        count = len(dense)
        padded = -(-count // ROW_BUCKET) * ROW_BUCKET
        # Padding rows read position 0 with weight zero and ignored labels.
        at = torch.cat([dense.flat, dense.flat.new_zeros(padded - count)])
        weight = torch.cat([dense.weight, dense.weight.new_zeros(padded - count)])
        with autocast_context(self.device, self.precision):
            outputs = model.auxiliary(hidden.reshape(-1, hidden.shape[-1]).index_select(0, at))
        total = torch.zeros((), device=self.device)
        for name, coefficient in coefficients.items():
            if coefficient <= 0:
                continue
            prediction = outputs[name]
            targets = self._labels(group, name, dense, dense.rows, padded)
            if name == "points":
                per_row = 0.5 * (prediction.float() - targets).square().mean(-1)
                totals["points_squared_error"].append((weight * 2 * per_row).sum().detach())
                totals["points_weight"].append(weight.sum().detach())
            else:
                classes, entries = CLASSIFIERS[name]
                targets = targets.long().view(-1, entries)
                per_row, labeled, correct = _classification(prediction, targets, classes)
                totals[f"{name}_correct"].append(correct.sum().detach())
                totals[f"{name}_total"].append(labeled.sum().detach())
            loss = (per_row * weight).sum()
            total = total + coefficient * loss
            totals[f"{name}_loss"].append(loss.detach())
        return total

    def critic_auxiliary(self, critic, hidden, dense, group, totals, record):
        return None

    def summarize(self, reduced: dict[str, float]) -> dict[str, float]:
        stats: dict[str, float] = {}
        for name in LABELS:
            if f"{name}_loss" in reduced:
                stats[f"loss_{name}"] = reduced[f"{name}_loss"]
            if reduced.get(f"{name}_total", 0.0) > 0:
                stats[f"{name}_accuracy"] = reduced[f"{name}_correct"] / reduced[f"{name}_total"]
        if reduced.get("points_weight", 0.0) > 0:
            stats["points_rmse"] = POINTS_SCALE * (reduced["points_squared_error"] / reduced["points_weight"]) ** 0.5
        return stats
