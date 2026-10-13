"""Training batches: bucketed sequence groups plus per-head decision rows.

Each trajectory contributes one actor stream per learning seat and one
canonical critic stream (the game decides what the critic may see; plump's is
a perfect-information "oracle" sequence). Streams are tail-padded to a bucket
length and stacked; causal attention makes padding after a readout exact.

Weighting (G games, P seats, every seat learning):

* every decision row of seat ``i`` in game ``g`` has weight ``1 / (G P)``, so
  the policy objective is ``(1/G) sum_g (1/P) sum_i sum_t L[g, i, t]`` and a
  longer game contributes more decisions;
* dense (per-position) losses spread each stream's mass evenly over its
  positions from ``dense_start`` on: an actor stream carries ``1 / (G P)``,
  a critic stream ``1 / G``. Prefix and padding positions weigh zero.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .selfplay import Trajectory


@dataclass(frozen=True)
class LabelSpec:
    """How to pad and stack one game-specific label array."""

    per_position: bool  # leading axis is the stream's position axis
    fill: float | int | bool
    dtype: str


@dataclass
class TrainingSequence:
    tokens: np.ndarray  # [L, W]
    dense_start: int
    labels: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class GameSequences:
    """What a game derives from one finished trajectory for training."""

    actor: dict[int, TrainingSequence]  # absolute seat -> its own stream
    critic: TrainingSequence
    # Critic-stream position of each trajectory decision's pre-action state.
    critic_positions: list[int]


@dataclass
class SequenceGroup:
    tokens: np.ndarray  # [B, L, W]
    position_weight: np.ndarray  # float32 [B, L]
    player_counts: np.ndarray  # int64 [B]
    # Terminal returns per seat: observer-relative for actor rows (entry 0 is
    # the observer), absolute for critic rows.
    returns: np.ndarray  # float32 [B, max_players]
    labels: dict[str, np.ndarray]

    @property
    def length(self) -> int:
        return self.tokens.shape[1]


@dataclass
class DecisionRows:
    """Every decision of one policy head, addressed into the groups."""

    group: np.ndarray  # actor group index
    seq: np.ndarray  # row within the group
    position: np.ndarray
    action: np.ndarray
    probs: np.ndarray  # float32 [N, actions], zero outside the legal set
    weight: np.ndarray  # float32
    returns: np.ndarray  # acting seat's terminal return
    seat: np.ndarray  # absolute acting seat
    game: np.ndarray
    order: np.ndarray  # decision index within its game
    critic_group: np.ndarray
    critic_seq: np.ndarray
    critic_position: np.ndarray
    features: dict[str, np.ndarray] | None
    advantages: np.ndarray  # float32, filled by the trainer

    def __len__(self) -> int:
        return int(self.group.shape[0])


@dataclass
class PPOBatch:
    actor_groups: list[SequenceGroup]
    critic_groups: list[SequenceGroup]
    decisions: dict[str, DecisionRows]
    games: int


def bucket_length(length: int, width: int, capacity: int) -> int:
    if width <= 0:
        return length
    return min(((length + width - 1) // width) * width, max(capacity, length))


def build_batch(
    trajectories: list[Trajectory],
    sequences: list[GameSequences],
    *,
    max_players: int,
    actor_labels: dict[str, LabelSpec],
    critic_labels: dict[str, LabelSpec],
    bucket_width: int,
    critic_bucket_width: int,
    max_length: int,
    critic_max_length: int,
) -> PPOBatch:
    games = len(trajectories)
    if games == 0:
        raise ValueError("A training batch needs at least one game.")
    game_weight = 1.0 / games

    actor_entries: list[tuple[TrainingSequence, float, int, np.ndarray]] = []
    actor_slot: dict[tuple[int, int], int] = {}
    critic_entries: list[tuple[TrainingSequence, float, int, np.ndarray]] = []
    for game_index, (trajectory, derived) in enumerate(zip(trajectories, sequences)):
        players = trajectory.game.num_players
        seat_weight = game_weight / len(derived.actor)
        for seat, sequence in derived.actor.items():
            relative = np.zeros(max_players, dtype=np.float32)
            relative[:players] = np.roll(trajectory.returns, -seat)
            actor_slot[(game_index, seat)] = len(actor_entries)
            actor_entries.append((sequence, seat_weight, players, relative))
        absolute = np.zeros(max_players, dtype=np.float32)
        absolute[:players] = trajectory.returns
        critic_entries.append((derived.critic, game_weight, players, absolute))

    actor_groups, actor_address = _stack(
        actor_entries, actor_labels, bucket_width, max_length
    )
    critic_groups, critic_address = _stack(
        critic_entries, critic_labels, critic_bucket_width, critic_max_length
    )

    columns: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    features: dict[str, list] = defaultdict(list)
    for game_index, (trajectory, derived) in enumerate(zip(trajectories, sequences)):
        if len(derived.critic_positions) != len(trajectory.decisions):
            raise AssertionError("Every decision needs a critic address.")
        critic_group, critic_seq = critic_address[game_index]
        for order, (record, critic_position) in enumerate(
            zip(trajectory.decisions, derived.critic_positions)
        ):
            slot = actor_slot.get((game_index, record.seat))
            if slot is None:
                continue
            group, seq = actor_address[slot]
            weight = actor_entries[slot][1]
            column = columns[record.head]
            column["group"].append(group)
            column["seq"].append(seq)
            column["position"].append(record.position)
            column["action"].append(record.action)
            column["probs"].append(record.probs)
            column["weight"].append(weight)
            column["returns"].append(float(trajectory.returns[record.seat]))
            column["seat"].append(record.seat)
            column["game"].append(game_index)
            column["order"].append(order)
            column["critic_group"].append(critic_group)
            column["critic_seq"].append(critic_seq)
            column["critic_position"].append(critic_position)
            features[record.head].append(record.features)

    decisions = {
        head: _decision_rows(column, features[head]) for head, column in sorted(columns.items())
    }
    return PPOBatch(
        actor_groups=actor_groups,
        critic_groups=critic_groups,
        decisions=decisions,
        games=games,
    )


def _stack(
    entries: list[tuple[TrainingSequence, float, int, np.ndarray]],
    specs: dict[str, LabelSpec],
    width: int,
    capacity: int,
) -> tuple[list[SequenceGroup], list[tuple[int, int]]]:
    buckets: dict[int, list[int]] = defaultdict(list)
    for index, (sequence, _, _, _) in enumerate(entries):
        buckets[bucket_length(sequence.tokens.shape[0], width, capacity)].append(index)
    groups: list[SequenceGroup] = []
    address: list[tuple[int, int]] = [(-1, -1)] * len(entries)
    for length, members in sorted(buckets.items()):
        count = len(members)
        token_width = entries[members[0]][0].tokens.shape[1]
        tokens = np.zeros((count, length, token_width), dtype=np.int64)
        position_weight = np.zeros((count, length), dtype=np.float32)
        player_counts = np.zeros(count, dtype=np.int64)
        returns = np.zeros((count, entries[members[0]][3].shape[0]), dtype=np.float32)
        labels: dict[str, np.ndarray] = {}
        for name, spec in specs.items():
            sample = entries[members[0]][0].labels[name]
            shape = (count, length, *sample.shape[1:]) if spec.per_position else (count, *sample.shape)
            labels[name] = np.full(shape, spec.fill, dtype=spec.dtype)
        for row, index in enumerate(members):
            sequence, weight, players, seat_returns = entries[index]
            size = sequence.tokens.shape[0]
            tokens[row, :size] = sequence.tokens
            dense = size - sequence.dense_start
            if dense <= 0:
                raise AssertionError("A training stream has no dense-supervision positions.")
            position_weight[row, sequence.dense_start : size] = weight / dense
            player_counts[row] = players
            returns[row] = seat_returns
            for name, spec in specs.items():
                value = sequence.labels[name]
                if spec.per_position:
                    labels[name][row, :size] = value
                else:
                    labels[name][row] = value
            address[index] = (len(groups), row)
        groups.append(
            SequenceGroup(
                tokens=tokens,
                position_weight=position_weight,
                player_counts=player_counts,
                returns=returns,
                labels=labels,
            )
        )
    return groups, address


def _decision_rows(column: dict[str, list], features: list) -> DecisionRows:
    width = max(len(probs) for probs in column["probs"])
    probs = np.zeros((len(column["probs"]), width), dtype=np.float32)
    for row, value in enumerate(column["probs"]):
        probs[row, : len(value)] = value
    stacked = None
    if features and features[0] is not None:
        count = max(next(iter(entry.values())).shape[0] for entry in features)
        stacked = {}
        for key, first in features[0].items():
            array = np.zeros((len(features), count, *first.shape[1:]), dtype=first.dtype)
            for row, entry in enumerate(features):
                array[row, : entry[key].shape[0]] = entry[key]
            stacked[key] = array
    as_int = lambda name: np.asarray(column[name], dtype=np.int64)  # noqa: E731
    return DecisionRows(
        group=as_int("group"),
        seq=as_int("seq"),
        position=as_int("position"),
        action=as_int("action"),
        probs=probs,
        weight=np.asarray(column["weight"], dtype=np.float32),
        returns=np.asarray(column["returns"], dtype=np.float32),
        seat=as_int("seat"),
        game=as_int("game"),
        order=as_int("order"),
        critic_group=as_int("critic_group"),
        critic_seq=as_int("critic_seq"),
        critic_position=as_int("critic_position"),
        features=stacked,
        advantages=np.zeros(len(column["group"]), dtype=np.float32),
    )


class RowIndex:
    """Find the rows of one (group, [start, stop)) microbatch in O(log N)."""

    def __init__(self, group: np.ndarray, seq: np.ndarray) -> None:
        self.stride = int(seq.max()) + 1 if seq.size else 1
        key = group * self.stride + seq
        self.order = np.argsort(key, kind="stable")
        self.sorted_key = key[self.order]

    def select(self, group: int, start: int, stop: int) -> np.ndarray:
        stop = min(stop, self.stride)
        start = min(start, self.stride)
        low = np.searchsorted(self.sorted_key, group * self.stride + start, side="left")
        high = np.searchsorted(self.sorted_key, group * self.stride + stop, side="left")
        return self.order[low:high]


def microbatches(group: SequenceGroup, positions: int) -> list[tuple[int, int]]:
    rows = max(positions // max(group.length, 1), 1)
    count = group.tokens.shape[0]
    return [(start, min(start + rows, count)) for start in range(0, count, rows)]
