"""Game-independent PPO update with a separate (privileged) critic.

One update, given a :class:`PPOBatch`:

1. Critic baseline. The critic's first epoch runs first; its forward passes
   (the critic before any step of this update) are the baseline: each
   decision's is the acting seat's zero-sum-centered value at that
   decision's pre-action state. Advantages are Monte Carlo
   ``R - V(s)`` or GAE over the game's decision states, then normalized.
   A game without a critic (``build_critic`` returns None) gets the same
   baseline from a no-grad pass of the actor's value head over the actor
   streams, or (``actor_baseline = "rollout"``) from the values the
   collector recorded with each sampled decision, which costs no pass at
   all; that head is then trained only by the actor's value loss
   (``actor_value_coef``), and step 3 is skipped.
2. Actor. Per epoch, ``minibatches`` optimizer steps, each accumulating
   every microbatch of its equal row slice of every group: the clipped PPO loss per policy head (minus each head's entropy
   temperature times its normalized entropy) plus the game's auxiliary
   losses and the optional per-seat value probe. With an auxiliary gradient
   budget the two halves are differentiated separately and the auxiliary
   gradient is rescaled to a fixed fraction of the policy gradient's norm
   before the single global clip (with ``auxiliary_grad_refresh`` > 1, only
   every N-th step measures; the others reuse its scale in one backward).
3. Critic. ``critic_epochs`` full passes of dense per-seat value regression
   plus the game's critic auxiliaries, each one optimizer step (the first
   ran in step 1).

Reporting scalars stay on device until the update ends: converting inside the
microbatch loop would insert a host synchronization between every forward and
backward.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import torch

from cardgym.core.devices import (
    Uploader,
    autocast_context,
    device_allocated_bytes,
    release_device_memory,
)
from cardgym.nn.compile import compiled
from cardgym.nn.packing import Packing

from .batch import DecisionRows, PPOBatch, RowIndex, SequenceGroup, microbatches
from .config import PPOConfig
from .entropy import EntropyController
from .ppo import (
    center_zero_sum,
    gae_advantages,
    ppo_terms,
    weighted_mean_std,
    weighted_regression_stats,
)

Totals = dict[str, list[torch.Tensor]]


@dataclass
class Microbatch:
    """Group rows [start, stop) on the device, padded [B, L, W] or packed [N, W].

    Every per-position model output is addressed through :meth:`flat`: a
    (sequence, position) pair's row in ``output.reshape(-1, ...)``.
    """

    start: int
    stop: int
    tokens: torch.Tensor
    length: int  # padded L
    packing: Packing | None = None

    @property
    def bounds(self) -> torch.Tensor | None:
        return None if self.packing is None else self.packing.bounds

    @property
    def longest(self) -> int:
        return self.length if self.packing is None else self.packing.longest

    def flat(self, sequence: np.ndarray, position: np.ndarray) -> np.ndarray:
        if self.packing is None:
            return sequence * self.length + position
        return self.packing.flat(sequence, position)

    def per_row(self, values: np.ndarray) -> np.ndarray:
        """A per-sequence array repeated for each flat row."""

        if self.packing is None:
            return np.repeat(values, self.length, axis=0)
        return self.packing.per_row(values)


@dataclass
class _CriticProgress:
    """What the critic epochs of one update accumulate for its statistics."""

    totals: Totals = field(default_factory=lambda: defaultdict(list))
    epoch_losses: list[list[torch.Tensor]] = field(default_factory=list)
    grad_norm: float = 0.0
    skipped: int = 0


@dataclass
class DenseRows:
    """The genuine (weighted) positions of one microbatch."""

    seq: np.ndarray  # row within the microbatch
    position: np.ndarray
    weight: torch.Tensor  # float32 [N]
    flat: torch.Tensor  # each position's row in the flattened per-position outputs
    start: int  # first group row of the microbatch
    rows: int  # sequences in the microbatch

    def __len__(self) -> int:
        return int(self.seq.shape[0])


class TrainingHooks(Protocol):
    """Game-specific auxiliary objectives; return None when inactive."""

    def actor_auxiliary(
        self, model, hidden: torch.Tensor, dense: DenseRows, group: SequenceGroup, totals: Totals
    ) -> torch.Tensor | None: ...

    def critic_auxiliary(
        self,
        critic,
        hidden: torch.Tensor,
        dense: DenseRows,
        group: SequenceGroup,
        totals: Totals,
        record: bool,
    ) -> torch.Tensor | None: ...

    def summarize(self, reduced: dict[str, float]) -> dict[str, float]: ...

    # Optional: ``gates`` maps a gate name to its policy head, and
    # ``gate_side(gate, features)`` marks one side of that head's candidates
    # [N, A] bool; the gate's binary entropy then gets its own entropy target.


def _squared_norm(tensors: Sequence[torch.Tensor | None] | None) -> torch.Tensor | None:
    """Sum of squares of every entry, on the device.

    Squares and sums rather than ``vector_norm``: on MPS a norm of the
    actor's 8.7M gradient entries measured 11 ms, as one tensor or per tensor;
    this takes 1.5 ms.
    """

    present = [] if tensors is None else [t for t in tensors if t is not None]
    if not present:
        return None
    return torch.stack([square.sum() for square in torch._foreach_mul(present, present)]).sum()


def _norms(*groups: Sequence[torch.Tensor | None] | None) -> list[float]:
    """The global norm of each group of tensors, with one host read."""

    squares = [_squared_norm(group) for group in groups]
    present = [square for square in squares if square is not None]
    values = iter(torch.stack(present).sqrt().tolist() if present else [])
    return [0.0 if square is None else next(values) for square in squares]


def clip_grad_norm(parameters: Sequence[torch.Tensor], max_norm: float) -> float:
    """``torch.nn.utils.clip_grad_norm_`` with the fast norm above; returns the norm before clipping."""

    grads = [p.grad for p in parameters if p.grad is not None]
    (norm,) = _norms(grads)
    scale = max_norm / (norm + 1e-6)
    if grads and math.isfinite(norm) and scale < 1.0:
        torch._foreach_mul_(grads, scale)
    return norm


def _reduce(totals: Totals) -> dict[str, float]:
    """Sum each key's device scalars in float64 on the host, with one transfer.

    A host read per scalar is a device synchronization each; an update
    collects hundreds of them.
    """

    keys = [key for key, values in totals.items() if values]
    if not keys:
        return {}
    flat = [value.reshape(()).float() for key in keys for value in totals[key]]
    host = torch.stack(flat).cpu().double().numpy()
    reduced, offset = {}, 0
    for key in keys:
        count = len(totals[key])
        reduced[key] = math.fsum(host[offset : offset + count])
        offset += count
    return reduced


def _accumulate(
    stash: list[torch.Tensor | None] | None, gradients: Sequence[torch.Tensor | None]
) -> list[torch.Tensor | None]:
    if stash is None:
        return [None if g is None else g.detach() for g in gradients]
    destinations, additions = [], []
    for index, gradient in enumerate(gradients):
        if gradient is None:
            continue
        if stash[index] is None:
            stash[index] = gradient.detach()
        else:
            destinations.append(stash[index])
            additions.append(gradient)
    if destinations:
        torch._foreach_add_(destinations, additions)
    return stash


def configure_adam(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """Select the Adam kernel for ``device`` without resetting saved moments.

    Fused Adam (MPS, CUDA) needs its step counters on the device; ordinary
    Adam wants them on the CPU. Checkpoints can come from either.
    """

    fused = device.type in ("mps", "cuda")
    counter_device = device if fused else torch.device("cpu")
    for group in optimizer.param_groups:
        group["fused"] = fused
        group["foreach"] = False if fused else None
    for state in optimizer.state.values():
        if "step" in state and torch.is_tensor(state["step"]):
            state["step"] = state["step"].to(counter_device)


def standardize_grads(parameters: Sequence[torch.Tensor]) -> None:
    """Give every gradient its parameter's (contiguous) layout.

    Fused Adam requires parameter, gradient, and moments to share a layout,
    but some MPS backward kernels (Conv1d weights) return permuted strides.
    """

    for parameter in parameters:
        grad = parameter.grad
        if grad is not None and grad.stride() != parameter.stride():
            parameter.grad = grad.contiguous()


class LossScale:
    """Dynamic loss scaling for fp16 autocast (bf16 shares fp32's range and needs none).

    fp16 activations' gradients flush to zero below ~6e-8, so every loss is
    multiplied by ``scale`` before its backward and every gradient divided
    by it right after, before the norms, the auxiliary budget, the clip, and
    the step: everything downstream sees true gradients. A step whose
    gradient is not finite is skipped (as any non-finite step is) and halves
    the scale; ``interval`` finite steps in a row double it.
    """

    def __init__(self, scale: float = 2.0**12, interval: int = 500) -> None:
        self.scale = scale
        self.interval = interval
        self.good_steps = 0

    def update(self, finite: bool) -> None:
        if not finite:
            self.scale = max(self.scale / 2.0, 1.0)
            self.good_steps = 0
            return
        self.good_steps += 1
        if self.good_steps >= self.interval:
            self.scale = min(self.scale * 2.0, 2.0**24)
            self.good_steps = 0

    def state_dict(self) -> dict:
        return {"scale": self.scale, "good_steps": self.good_steps}

    def load_state_dict(self, state: dict) -> None:
        self.scale = float(state["scale"])
        self.good_steps = int(state["good_steps"])


# Candidate-list widths at which a head's decisions are split (see _width_buckets).
WIDTH_BUCKETS = (8, 16)
MIN_BUCKET_ROWS = 256


def _width_buckets(rows: DecisionRows, selected: np.ndarray) -> list[np.ndarray]:
    """``selected`` split by candidate-list width, so each part is scored at its own width.

    Lists are padded to the longest in the microbatch; Rummy's play lists
    average 10 legal candidates against a longest of ~37, and scoring at
    widths 8 / 16 / longest does 42% of the padded work. The policy loss is a
    weighted sum over rows, so splitting them is exact. Parts smaller than
    MIN_BUCKET_ROWS join the next wider one (small MPS autograd batches have
    misbehaved before).
    """

    if not selected.size:
        return []
    legal = rows.probs[selected] > 0
    width = legal.shape[1] - np.argmax(legal[:, ::-1], axis=1)
    parts, pending = [], np.zeros(0, dtype=selected.dtype)
    low = 0
    for high in (*WIDTH_BUCKETS, int(width.max())):
        if high <= low:
            continue
        pending = np.concatenate([pending, selected[(width > low) & (width <= high)]])
        if pending.size >= MIN_BUCKET_ROWS:
            parts.append(pending)
            pending = np.zeros(0, dtype=selected.dtype)
        low = high
    if pending.size:
        if parts:
            parts[-1] = np.concatenate([parts[-1], pending])
        else:
            parts.append(pending)
    return parts


def _within_side_entropy(log_probs, legal, side) -> tuple[torch.Tensor, torch.Tensor]:
    """A gated head's entropy once the side is chosen, per row: (sum, eligible mass).

    The chain rule splits a row's entropy into the gate's binary entropy plus
    the expected entropy within the chosen side. This is the second part,
    with each side normalized by its own size like a head:
    ``sum = sum_s P(s) * H(a | s) / log|s|`` over sides with two or more legal
    actions, and ``mass = sum_s P(s)`` over the same sides. The weights P(s)
    are constants, so this term does not reward moving probability toward
    the side with more choices; the split is measured and floored by the
    gate's own term. A row with one side legal is the plain normalized
    entropy (P = 1).
    """

    floor = torch.full_like(log_probs, -1e4)  # finite: an empty side must not make NaN gradients
    total = torch.zeros(log_probs.shape[0], device=log_probs.device)
    mass = torch.zeros_like(total)
    for members in (legal & side, legal & ~side):
        count = members.sum(dim=-1)
        log_side = torch.where(members, log_probs, floor).logsumexp(dim=-1)
        conditional = torch.where(members, log_probs - log_side[:, None], torch.zeros_like(log_probs))
        entropy = -(conditional.exp() * members.float() * conditional).sum(dim=-1)
        choice = count >= 2
        normalized = torch.where(choice, entropy / count.clamp_min(2).float().log(), torch.zeros_like(entropy))
        probability = log_side.exp().detach() * choice.float()
        total = total + probability * normalized
        mass = mass + probability
    return total, mass


class PPOTrainer:
    def __init__(
        self,
        actor,
        critic,
        config: PPOConfig,
        hooks: TrainingHooks,
        *,
        device: torch.device,
        precision: str,
        policy_heads: tuple[str, ...],
        max_players: int,
    ) -> None:
        config.validate()
        self.config = config
        self.hooks = hooks
        self.device = device
        self.precision = precision
        self.policy_heads = policy_heads
        self.max_players = max_players
        if critic is not None and config.actor_baseline == "rollout":
            raise ValueError('[ppo] actor_baseline = "rollout" is for games without a critic.')
        if critic is None and config.actor_value_coef <= 0:
            raise ValueError(
                "Without a critic the actor's value head is the baseline: [ppo] actor_value_coef must be positive."
            )
        self.actor = actor.to(device)
        self.critic = None if critic is None else critic.to(device)
        # Full causal forwards, compiled on CUDA (cardgym.nn.compile).
        self._actor_hidden = compiled(self.actor.forward_hidden, device)
        self._critic_hidden = None if self.critic is None else compiled(self.critic.forward_hidden, device)
        self.actor_parameters = [p for p in self.actor.parameters() if p.requires_grad]
        self.critic_parameters = (
            [] if self.critic is None else [p for p in self.critic.parameters() if p.requires_grad]
        )
        fused = device.type in ("mps", "cuda")
        self.actor_optimizer = torch.optim.Adam(
            self.actor_parameters, lr=config.learning_rate, betas=config.adam_betas, fused=fused
        )
        self.critic_optimizer = None if self.critic is None else torch.optim.Adam(
            self.critic_parameters,
            lr=config.critic_learning_rate,
            betas=config.adam_betas,
            fused=fused,
        )
        # A gate is active only when the config gives it a target.
        self.gates: dict[str, str] = {
            gate: head for gate, head in getattr(hooks, "gates", {}).items() if gate in config.entropy_targets
        }
        if len(set(self.gates.values())) != len(self.gates):
            raise ValueError("At most one entropy gate per policy head.")
        self.entropy = EntropyController(tuple(policy_heads) + tuple(self.gates), config, device)
        self.loss_scale = LossScale() if precision == "fp16" else None
        # Every host array an update reads goes up without draining the queue.
        self.upload = Uploader(device)
        if device.type == "mps":
            self.cache_ceiling = int(config.device_cache_gb * 2**30 or 0.6 * torch.mps.recommended_max_memory())
        elif device.type == "cuda":
            total = torch.cuda.get_device_properties(device).total_memory
            self.cache_ceiling = int(config.device_cache_gb * 2**30 or 0.6 * total)
        else:
            self.cache_ceiling = 0
        if hasattr(hooks, "upload"):
            hooks.upload = self.upload
        self.optimizer_steps = 0
        self._policy_grads: list[torch.Tensor | None] | None = None
        # The auxiliary budget's last measured scale and norms (lagged steps reuse them).
        self._auxiliary_scale: float | None = None
        self._auxiliary_norms = (0.0, 0.0)
        self._auxiliary_grads: list[torch.Tensor | None] | None = None

    # ------------------------------------------------------------------ #

    def update(self, batch: PPOBatch, iteration: int) -> dict[str, float]:
        started = time.perf_counter()
        stats: dict[str, float] = {"skipped_actor_steps": 0.0}
        totals: Totals = defaultdict(list)
        # The critic's first epoch comes first: its forward passes are also
        # the advantage baseline (the critic does not depend on the actor).
        critic = _CriticProgress()
        pending: list = []
        recorded = self.critic is None and self.config.actor_baseline == "rollout" and all(
            rows.values is not None for rows in batch.decisions.values()
        )
        if recorded:
            pass  # _baseline reads the values recorded at sampling
        elif self.critic is None:
            self._actor_baseline(batch, pending)
        else:
            self._critic_epoch(batch, critic, record=True, baseline=pending)
        stats.update(self._baseline(batch, pending))
        del pending
        stats["policy_rows"] = float(sum(len(rows) for rows in batch.decisions.values()))
        stats["positions"] = float(sum(g.tokens.shape[0] * g.length for g in batch.actor_groups))
        targets = self.entropy.targets(iteration)
        for head, target in targets.items():
            stats[f"entropy_target_{head}"] = target

        indexes = {head: RowIndex(rows.group, rows.seq) for head, rows in batch.decisions.items()}
        for _ in range(self.config.epochs):
            self.actor.train()
            epoch_totals: Totals = defaultdict(list)
            for minibatch in range(self.config.minibatches):
                self.actor_optimizer.zero_grad(set_to_none=True)
                warming = self.optimizer_steps < self.config.policy_warmup_steps
                # Warm-up steps backpropagate the auxiliary losses alone, unscaled.
                lagged = 1.0 if warming else self._lagged_auxiliary_scale()
                self._actor_backward(batch, indexes, epoch_totals, minibatch, lagged, train_policy=not warming)
                self._unscale(self.actor_parameters, self._policy_grads, self._auxiliary_grads)
                if warming:
                    policy_norm, auxiliary_norm, scale = 0.0, float("nan"), 1.0
                    stats["policy_warmup_steps"] = stats.get("policy_warmup_steps", 0.0) + 1.0
                elif lagged is None:
                    policy_norm, auxiliary_norm, scale = self._apply_auxiliary_budget()
                else:
                    (policy_norm, auxiliary_norm), scale = self._auxiliary_norms, lagged
                    stats["lagged_auxiliary_steps"] = stats.get("lagged_auxiliary_steps", 0.0) + 1.0
                grad_norm = clip_grad_norm(self.actor_parameters, self.config.max_grad_norm)
                self._set_learning_rate(iteration)
                if math.isfinite(grad_norm):
                    standardize_grads(self.actor_parameters)
                    self.actor_optimizer.step()
                    self.optimizer_steps += 1
                else:
                    # A non-finite gradient must never reach the weights or Adam's moments.
                    self.actor_optimizer.zero_grad(set_to_none=True)
                    stats["skipped_actor_steps"] += 1.0
                if self.loss_scale is not None:
                    self.loss_scale.update(math.isfinite(grad_norm))
            # The entropy temperatures see the whole epoch's entropy, once.
            reduced = _reduce(epoch_totals)
            observed = {
                head: reduced.get(f"entropy_{head}", 0.0) / reduced[f"entropy_weight_{head}"]
                for head in self.entropy.heads
                if reduced.get(f"entropy_weight_{head}", 0.0) > 0
            }
            self.entropy.update(observed, iteration)
            for key, values in epoch_totals.items():
                totals[key].extend(values)
            stats["actor_grad_norm"] = grad_norm
            stats["policy_grad_norm"] = policy_norm
            stats["auxiliary_grad_norm"] = auxiliary_norm
            stats["auxiliary_grad_scale"] = scale
        stats.update(self._summarize_actor(totals))
        for head in self.entropy.heads:
            stats[f"entropy_alpha_{head}"] = self.entropy.alpha(head)
        stats["learning_rate"] = float(self.actor_optimizer.param_groups[0]["lr"])
        if self.loss_scale is not None:
            stats["loss_scale"] = self.loss_scale.scale
        stats["peak_device_bytes"] = float(device_allocated_bytes(self.device))
        if self.critic is not None:
            for _ in range(1, self.config.critic_epochs):
                self._critic_epoch(batch, critic, record=False)
            stats.update(self._critic_stats(critic))
        # _critic_stats (or, without a critic, _summarize_actor) ended on a host
        # read of the losses: every upload is done.
        self.upload.release()
        # The allocator's cached blocks are the next update's activations:
        # returning them every update and allocating again measured 0.27 s of
        # a 2.3 s update. They go back only past a ceiling.
        if device_allocated_bytes(self.device) > self.cache_ceiling:
            release_device_memory(self.device)
        stats["update_sec"] = time.perf_counter() - started
        return stats

    def _set_learning_rate(self, iteration: int) -> None:
        warmup = self.config.lr_warmup_updates
        scale = 1.0 if warmup <= 0 else min(1.0, (self.optimizer_steps + 1) / warmup)
        scale *= self.config.lr_scale(iteration)
        for group in self.actor_optimizer.param_groups:
            group["lr"] = self.config.learning_rate * scale

    # -- advantages ------------------------------------------------------ #

    def _baseline(self, batch: PPOBatch, pending: list) -> dict[str, float]:
        """Advantages and value statistics from the critic's pre-step values.

        ``pending`` holds (group index, group, microbatch, centered values on
        the device) from the first critic epoch's forward passes: the critic
        before any step of this update, exactly what a separate inference
        pass would compute. Converting here, after the epoch, costs one host
        synchronization instead of one per microbatch.
        """

        index = {
            head: RowIndex(rows.critic_group, rows.critic_seq)
            for head, rows in batch.decisions.items()
        }
        predictions = {
            head: np.zeros(len(rows), dtype=np.float64) for head, rows in batch.decisions.items()
        }
        state_values: dict[str, np.ndarray] = {}
        gae = self.config.advantage_estimator == "gae"
        dense_prediction, dense_target, dense_weight = [], [], []
        hosted = torch.cat([values for *_, values in pending]).cpu().numpy() if pending else None
        offset = 0
        for group_index, group, micro, values in pending:
            values_np = hosted[offset : offset + values.shape[0]]
            offset += values.shape[0]
            start, stop = micro.start, micro.stop
            for head, rows in batch.decisions.items():
                selected = index[head].select(group_index, start, stop)
                if not selected.size:
                    continue
                seq = rows.critic_seq[selected] - start
                pos = rows.critic_position[selected]
                # Absolute seat i sits at column (i - shift) mod players.
                players = group.player_counts[start:stop][seq][:, None]
                seats = np.arange(self.max_players)[None, :]
                columns = (seats - rows.critic_shift[selected][:, None]) % players
                seat_values = np.take_along_axis(values_np[micro.flat(seq, pos)], columns, axis=1) * (seats < players)
                predictions[head][selected] = seat_values[np.arange(selected.size), rows.seat[selected]]
                if gae:
                    state_values.setdefault(head, np.zeros((len(rows), self.max_players)))
                    state_values[head][selected] = seat_values
            seq_dense, pos_dense = np.nonzero(group.position_weight[start:stop] > 0)
            players = group.player_counts[start:stop][seq_dense]
            active = np.arange(self.max_players)[None, :] < players[:, None]
            weight = group.position_weight[start:stop][seq_dense, pos_dense].astype(np.float64)
            dense_prediction.append(values_np[micro.flat(seq_dense, pos_dense)][active])
            dense_target.append(group.returns[start:stop][seq_dense][active])
            dense_weight.append(np.repeat(weight / players, players))

        if not pending:
            self._recorded_values(batch, predictions, state_values if gae else None)
        stats: dict[str, float] = {}
        if gae:
            self._gae(batch, state_values)
        else:
            for head, rows in batch.decisions.items():
                rows.advantages[:] = rows.returns - predictions[head]
        all_prediction = np.concatenate([predictions[h] for h in batch.decisions])
        all_target = np.concatenate([rows.returns for rows in batch.decisions.values()]).astype(np.float64)
        all_weight = np.concatenate([rows.weight for rows in batch.decisions.values()]).astype(np.float64)
        acting = weighted_regression_stats(all_prediction, all_target, all_weight)
        stats.update({f"value_{key}": value for key, value in acting.items()})
        if dense_prediction:
            dense = weighted_regression_stats(
                np.concatenate(dense_prediction).astype(np.float64),
                np.concatenate(dense_target).astype(np.float64),
                np.concatenate(dense_weight),
            )
            stats["critic_dense_rmse"] = dense["rmse"]
            stats["critic_dense_correlation"] = dense["correlation"]
        advantages = np.concatenate([rows.advantages for rows in batch.decisions.values()]).astype(np.float64)
        mean, std = weighted_mean_std(advantages, all_weight)
        stats["advantage_mean"] = mean
        stats["advantage_std"] = std
        if self.config.advantage_normalize:
            for rows in batch.decisions.values():
                rows.advantages[:] = (rows.advantages - mean) / std
        return stats

    def _recorded_values(
        self,
        batch: PPOBatch,
        predictions: dict[str, np.ndarray],
        state_values: dict[str, np.ndarray] | None,
    ) -> None:
        """The baseline from values recorded at sampling: centered, acting seat first."""

        for head, rows in batch.decisions.items():
            if rows.values is None:
                raise AssertionError("Every decision needs a recorded value or a baseline pass.")
            players = np.array(
                [batch.actor_groups[g].player_counts[s] for g, s in zip(rows.group, rows.seq)], dtype=np.int64
            )
            seats = np.arange(self.max_players)[None, :]
            active = seats < players[:, None]
            raw = rows.values.astype(np.float64) * active
            centered = (raw - raw.sum(-1, keepdims=True) / players[:, None]) * active
            predictions[head][:] = centered[:, 0]
            if state_values is not None:
                # Absolute seat i is column (i - seat) mod players of the acting seat's frame.
                columns = (seats - rows.seat[:, None]) % players[:, None]
                state_values[head] = np.take_along_axis(centered, columns, axis=1) * active

    def _gae(self, batch: PPOBatch, state_values: dict[str, np.ndarray]) -> None:
        """GAE over each game's decision states (all heads, game order)."""

        entries: dict[int, list[tuple[int, str, int]]] = defaultdict(list)
        for head, rows in batch.decisions.items():
            for row in range(len(rows)):
                entries[int(rows.game[row])].append((int(rows.order[row]), head, row))
        for game, items in entries.items():
            items.sort()
            values = np.stack([state_values[head][row] for _, head, row in items])
            returns = np.zeros(self.max_players)
            returns[: batch.game_returns[game].shape[0]] = batch.game_returns[game]
            advantages = gae_advantages(values, returns, self.config.gae_lambda)
            for state, (_, head, row) in enumerate(items):
                rows = batch.decisions[head]
                rows.advantages[row] = advantages[state, int(rows.seat[row])]

    # -- actor ----------------------------------------------------------- #

    def _microbatches(self, model, group: SequenceGroup, part: int = 0, parts: int = 1):
        """The group's microbatches on the device: packed rows where the model can run them.

        With ``parts`` > 1, only the ``part``-th of that many equal row slices.
        """

        packed = model.packs(self.device)
        count = group.tokens.shape[0]
        rows = (count * part // parts, count * (part + 1) // parts)
        for start, stop in microbatches(group, self.config.microbatch_positions, packed=packed, rows=rows):
            tokens = group.tokens[start:stop]
            if not packed:
                yield Microbatch(start, stop, self.upload(tokens), group.length)
                continue
            packing = Packing(group.lengths[start:stop], upload=self.upload)
            yield Microbatch(start, stop, self.upload(packing.pack(tokens)), group.length, packing)

    def _dense_rows(self, group: SequenceGroup, micro: Microbatch) -> DenseRows:
        seq, pos = np.nonzero(group.position_weight[micro.start : micro.stop] > 0)
        weight = group.position_weight[micro.start : micro.stop][seq, pos]
        return DenseRows(
            seq=seq,
            position=pos,
            weight=self.upload(weight.astype(np.float32)),
            flat=self.upload(micro.flat(seq, pos)),
            start=micro.start,
            rows=micro.stop - micro.start,
        )

    def _lagged_auxiliary_scale(self) -> float | None:
        """The scale to reuse this step, or None when the budget is off or this step measures."""

        refresh = self.config.auxiliary_grad_refresh
        if self.config.auxiliary_grad_fraction <= 0 or refresh <= 1 or self._auxiliary_scale is None:
            return None
        return None if self.optimizer_steps % refresh == 0 else self._auxiliary_scale

    def _actor_backward(
        self,
        batch: PPOBatch,
        indexes: dict[str, RowIndex],
        totals: Totals,
        minibatch: int = 0,
        lagged: float | None = None,
        train_policy: bool = True,
    ) -> None:
        from cardgym.nn.metal_trunk import DualTrunkPass

        self._policy_grads = None
        self._auxiliary_grads = None
        alpha = {head: self.entropy.alpha(head) for head in self.entropy.heads}
        budget = self.config.auxiliary_grad_fraction > 0 and lagged is None
        for group_index, group in enumerate(batch.actor_groups):
            for micro in self._microbatches(self.actor, group, minibatch, self.config.minibatches):
                dual = None
                with autocast_context(self.device, self.precision):
                    if budget and self.actor.trunk.uses_metal(self.device, micro.longest):
                        dual = DualTrunkPass(self.actor, micro.tokens, micro.bounds)
                        hidden, context = dual.hidden, dual.context
                    else:
                        hidden = self._actor_hidden(micro.tokens, micro.bounds)
                        context = self.actor.pop_context()
                dense = self._dense_rows(group, micro)
                auxiliary = torch.zeros((), device=self.device)
                hooked = self.hooks.actor_auxiliary(self.actor, hidden, dense, group, totals)
                if hooked is not None:
                    auxiliary = auxiliary + hooked
                if self.config.actor_value_coef > 0:
                    auxiliary = auxiliary + self.config.actor_value_coef * self._value_loss(
                        self.actor, hidden, dense, group, totals, "actor_value"
                    )
                policy = torch.zeros((), device=self.device)
                for head, rows in batch.decisions.items():
                    selected = indexes[head].select(group_index, micro.start, micro.stop)
                    for part in _width_buckets(rows, selected):
                        policy = policy + self._policy_loss(
                            hidden, micro, context, rows, part, head, alpha, totals
                        )
                self._backward(auxiliary, policy if train_policy else policy.detach(), dual, lagged)

    def _policy_loss(
        self,
        hidden: torch.Tensor,
        micro: Microbatch,
        context: torch.Tensor | None,
        rows: DecisionRows,
        selected: np.ndarray,
        head: str,
        alpha: dict[str, float],
        totals: Totals,
    ) -> torch.Tensor:
        at = self.upload(micro.flat(rows.seq[selected] - micro.start, rows.position[selected]))
        tokens = micro.tokens.reshape(-1, micro.tokens.shape[-1])
        # Candidate lists are padded to the batch's longest; trim them to this
        # microbatch's longest (rows.probs is zero past each row's actions).
        width = int(np.flatnonzero(rows.probs[selected].any(axis=0)).max()) + 1
        features = (
            None
            if rows.features is None
            else {
                key: self.upload(value[selected][:, :width])
                for key, value in rows.features.items()
            }
        )
        with autocast_context(self.device, self.precision):
            logits = self.actor.policy_logits(
                hidden.reshape(-1, hidden.shape[-1])[at],
                head,
                features,
                tokens[at],
                None if context is None else context.reshape(-1, *context.shape[-2:])[at],
            )
        old_probs = self.upload(rows.probs[selected][:, :width])
        logits = logits.float()[:, :width]
        terms = ppo_terms(
            logits,
            old_probs,
            self.upload(rows.action[selected]),
            self.upload(rows.advantages[selected]),
            clip_ratio=self.config.clip_ratio,
        )
        weight = self.upload(rows.weight[selected])
        loss = (weight * terms.losses).sum()
        gate = next((gate for gate, gate_head in self.gates.items() if gate_head == head), None)
        side = None if gate is None else self.hooks.gate_side(gate, features)[:, :width]
        if side is None:
            eligible = weight * terms.entropy_eligible.float()
            entropy_term = (eligible * terms.normalized_entropy).sum()
        else:
            # The gate owns the choice of side; the head's entropy is the rest.
            within, mass = _within_side_entropy(terms.log_probs, old_probs > 0, side)
            eligible = weight * mass
            entropy_term = (weight * within).sum()
        totals["policy_loss"].append(loss.detach())
        totals[f"policy_loss_{head}"].append(loss.detach())
        totals["row_weight"].append(weight.sum().detach())
        totals["entropy_raw"].append((weight * terms.entropy).sum().detach())
        totals[f"entropy_{head}"].append(entropy_term.detach())
        totals[f"entropy_weight_{head}"].append(eligible.sum().detach())
        totals["clip"].append((weight * terms.clipped.float()).sum().detach())
        # Cached reduced-precision rollout vs full-sequence replay: should
        # stay numerical noise, or the first epoch is silently off-policy.
        totals["behavior_kl"].append((weight * terms.divergences).sum().detach())
        if alpha[head]:
            loss = loss - alpha[head] * entropy_term
        if side is not None:
            gate_term = self._gate_entropy(terms.log_probs, old_probs > 0, side, weight, gate, totals)
            if alpha[gate]:
                loss = loss - alpha[gate] * gate_term
        return loss

    @staticmethod
    def _gate_entropy(log_probs, legal, side, weight, gate: str, totals: Totals) -> torch.Tensor:
        """Weighted sum of the binary entropy (in bits) of P(side) over rows where both sides are legal."""

        # A large finite floor, not -inf: rows with an empty side must not turn the gradient into NaN.
        floor = torch.full_like(log_probs, -1e4)
        inside, outside = legal & side, legal & ~side
        log_in = torch.where(inside, log_probs, floor).logsumexp(dim=-1)
        log_out = torch.where(outside, log_probs, floor).logsumexp(dim=-1)
        eligible = weight * (inside.any(dim=-1) & outside.any(dim=-1)).float()
        p_in, p_out = log_in.exp(), log_out.exp()
        bits = -(p_in * log_in + p_out * log_out) / math.log(2.0)
        term = (eligible * bits).sum()
        totals[f"entropy_{gate}"].append(term.detach())
        totals[f"entropy_weight_{gate}"].append(eligible.sum().detach())
        totals[f"{gate}_probability"].append((eligible * p_in).sum().detach())
        return term

    def _value_loss(
        self,
        model,
        hidden: torch.Tensor,
        dense: DenseRows,
        group: SequenceGroup,
        totals: Totals,
        prefix: str,
        values: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Dense per-seat zero-sum value regression, unit mass per stream."""

        if values is None:
            with autocast_context(self.device, self.precision):
                values = model.player_values(hidden.reshape(-1, hidden.shape[-1]))
        prediction = values[dense.flat].float()
        players = self.upload(group.player_counts[dense.start :][dense.seq])
        targets = self.upload(group.returns[dense.start :][dense.seq])
        active = (torch.arange(prediction.shape[1], device=self.device)[None, :] < players[:, None]).float()
        prediction = center_zero_sum(prediction, players)
        raw = prediction - targets
        error = raw / self.config.value_reward_scale
        loss = (0.5 * dense.weight * (error.square() * active).sum(-1) / players.float()).sum()
        per_seat = dense.weight / players.float()
        totals[f"{prefix}_loss"].append(loss.detach())
        totals[f"{prefix}_squared_error"].append((per_seat * (raw.square() * active).sum(-1)).sum().detach())
        totals[f"{prefix}_target_squared"].append((per_seat * (targets.square() * active).sum(-1)).sum().detach())
        totals[f"{prefix}_weight"].append(dense.weight.sum().detach())
        return loss

    def _scaled(self, loss: torch.Tensor) -> torch.Tensor:
        return loss if self.loss_scale is None else loss * self.loss_scale.scale

    def _unscale(self, parameters: Sequence[torch.Tensor], *stashes: list[torch.Tensor | None] | None) -> None:
        """Divide the loss scale out of ``.grad`` and the stashed gradients."""

        if self.loss_scale is None:
            return
        gradients = [p.grad for p in parameters if p.grad is not None]
        for stash in stashes:
            gradients += [] if stash is None else [g for g in stash if g is not None]
        if gradients:
            torch._foreach_mul_(gradients, 1.0 / self.loss_scale.scale)

    def _backward(self, auxiliary: torch.Tensor, policy: torch.Tensor, dual, lagged: float | None = None) -> None:
        auxiliary, policy = self._scaled(auxiliary), self._scaled(policy)
        if lagged is not None:
            total = policy + lagged * auxiliary
            if total.requires_grad:
                total.backward()
            return
        if dual is not None:
            auxiliary_grads, policy_grads = dual.gradients([auxiliary, policy], self.actor_parameters)
            if auxiliary_grads is not None:
                self._auxiliary_grads = _accumulate(self._auxiliary_grads, auxiliary_grads)
            if policy_grads is not None:
                self._policy_grads = _accumulate(self._policy_grads, policy_grads)
            return
        if self.config.auxiliary_grad_fraction <= 0:
            total = auxiliary + policy
            if total.requires_grad:
                total.backward()
            return
        if auxiliary.requires_grad:
            gradients = torch.autograd.grad(
                auxiliary,
                self.actor_parameters,
                retain_graph=policy.requires_grad,
                allow_unused=True,
            )
            self._auxiliary_grads = _accumulate(self._auxiliary_grads, gradients)
        if policy.requires_grad:
            gradients = torch.autograd.grad(policy, self.actor_parameters, allow_unused=True)
            self._policy_grads = _accumulate(self._policy_grads, gradients)

    def _apply_auxiliary_budget(self) -> tuple[float, float, float]:
        """Rescale the auxiliary gradient to its share and fold it into ``.grad``.

        Returns the policy and auxiliary gradient norms (before the rebalance
        and the clip) and the scale applied. Only their ratio matters to Adam;
        the absolute size stays the business of the single clip.
        """

        auxiliary = self._auxiliary_grads
        policy = self._policy_grads
        self._auxiliary_grads = self._policy_grads = None
        if self.config.auxiliary_grad_fraction <= 0:
            return 0.0, 0.0, 0.0
        policy_norm, auxiliary_norm = _norms(policy, auxiliary)
        if auxiliary_norm <= 0:
            scale = 0.0
        elif policy_norm <= 0:
            scale = 1.0
        else:
            scale = self.config.auxiliary_grad_fraction * policy_norm / auxiliary_norm
        both = [
            index for index in range(len(self.actor_parameters))
            if policy is not None and policy[index] is not None
            and auxiliary is not None and auxiliary[index] is not None
        ]
        if both:
            # One fused kernel for every parameter both losses reach.
            torch._foreach_add_([policy[i] for i in both], [auxiliary[i] for i in both], alpha=scale)
        for index, parameter in enumerate(self.actor_parameters):
            p = None if policy is None else policy[index]
            a = None if auxiliary is None else auxiliary[index]
            if p is not None:
                parameter.grad = p
            elif a is not None:
                parameter.grad = a.mul_(scale) if scale != 1.0 else a
            else:
                parameter.grad = None
        if policy_norm > 0 and auxiliary_norm > 0 and math.isfinite(scale):
            self._auxiliary_scale = scale
            self._auxiliary_norms = (policy_norm, auxiliary_norm)
        return policy_norm, auxiliary_norm, scale

    def _summarize_actor(self, totals: Totals) -> dict[str, float]:
        reduced = _reduce(totals)
        epochs = self.config.epochs
        row_weight = max(reduced.get("row_weight", 0.0), 1e-12)
        stats = {
            "loss_policy": reduced.get("policy_loss", 0.0) / epochs,
            "entropy": reduced.get("entropy_raw", 0.0) / row_weight,
            "clip_fraction": reduced.get("clip", 0.0) / row_weight,
            "behavior_kl": reduced.get("behavior_kl", 0.0) / row_weight,
        }
        for head in self.entropy.heads:
            weight = reduced.get(f"entropy_weight_{head}", 0.0)
            stats[f"entropy_{head}"] = reduced.get(f"entropy_{head}", 0.0) / weight if weight else 0.0
        for gate in self.gates:
            weight = reduced.get(f"entropy_weight_{gate}", 0.0)
            stats[f"{gate}_probability"] = reduced.get(f"{gate}_probability", 0.0) / weight if weight else 0.0
            # Share of all policy decisions (both heads) where the gate applies.
            stats[f"{gate}_eligible_fraction"] = weight / row_weight
        value_weight = reduced.get("actor_value_weight", 0.0)
        if value_weight > 0:
            stats["actor_value_loss"] = reduced.get("actor_value_loss", 0.0) / epochs
            stats["actor_value_rmse"] = math.sqrt(
                max(reduced.get("actor_value_squared_error", 0.0), 0.0) / value_weight
            )
        for key, value in self.hooks.summarize(reduced).items():
            stats[key] = value / epochs if key.startswith("loss_") else value
        return stats

    @torch.no_grad()
    def _actor_baseline(self, batch: PPOBatch, pending: list) -> None:
        """The actor's zero-sum-centered values on its own streams, before any step of this update."""

        for group_index, group in enumerate(batch.actor_groups):
            for micro in self._microbatches(self.actor, group):
                with autocast_context(self.device, self.precision):
                    hidden = self._actor_hidden(micro.tokens, micro.bounds)
                    self.actor.pop_context()
                    values = self.actor.player_values(hidden.reshape(-1, hidden.shape[-1]))
                counts = self.upload(micro.per_row(group.player_counts[micro.start : micro.stop]))
                pending.append((group_index, group, micro, center_zero_sum(values.float(), counts)))

    # -- critic ---------------------------------------------------------- #

    def _critic_epoch(
        self, batch: PPOBatch, progress: "_CriticProgress", *, record: bool, baseline: list | None = None
    ) -> None:
        """One full pass of dense value regression, one optimizer step.

        With ``baseline``, each microbatch's zero-sum-centered values (detached,
        on the device) are appended to it for :meth:`_baseline`.
        """

        self.critic.train()
        self.critic_optimizer.zero_grad(set_to_none=True)
        pending: list[torch.Tensor] = []
        for group_index, group in enumerate(batch.critic_groups):
            for micro in self._microbatches(self.critic, group):
                with autocast_context(self.device, self.precision):
                    hidden = self._critic_hidden(micro.tokens, micro.bounds)
                dense = self._dense_rows(group, micro)
                values = None
                if baseline is not None:
                    with autocast_context(self.device, self.precision):
                        values = self.critic.player_values(hidden.reshape(-1, hidden.shape[-1]))
                    counts = self.upload(micro.per_row(group.player_counts[micro.start : micro.stop]))
                    baseline.append(
                        (group_index, group, micro, center_zero_sum(values.detach().float(), counts))
                    )
                loss = self._value_loss(
                    self.critic, hidden, dense, group, progress.totals, "critic_value", values=values
                )
                hooked = self.hooks.critic_auxiliary(
                    self.critic, hidden, dense, group, progress.totals, record
                )
                if hooked is not None:
                    loss = loss + hooked
                self._scaled(loss).backward()
                pending.append(loss.detach())
        self._unscale(self.critic_parameters)
        progress.grad_norm = clip_grad_norm(self.critic_parameters, self.config.max_grad_norm)
        if math.isfinite(progress.grad_norm):
            standardize_grads(self.critic_parameters)
            self.critic_optimizer.step()
        else:
            self.critic_optimizer.zero_grad(set_to_none=True)
            progress.skipped += 1
        progress.epoch_losses.append(pending)

    def _critic_stats(self, progress: "_CriticProgress") -> dict[str, float]:
        epoch_losses = [_reduce({"loss": pending}).get("loss", 0.0) for pending in progress.epoch_losses]
        reduced = _reduce(progress.totals)
        stats = {
            "critic_grad_norm": progress.grad_norm,
            "skipped_critic_steps": float(progress.skipped),
            "loss_critic_value": reduced.get("critic_value_loss", 0.0) / self.config.critic_epochs,
            "critic_loss_first_epoch": epoch_losses[0],
            "critic_loss_last_epoch": epoch_losses[-1],
            "critic_loss_reduction": (
                (epoch_losses[0] - epoch_losses[-1]) / epoch_losses[0] if epoch_losses[0] > 0 else 0.0
            ),
        }
        for key, value in self.hooks.summarize(reduced).items():
            stats[key] = value / self.config.critic_epochs if key.startswith("loss_") else value
        return stats

    # -- state ----------------------------------------------------------- #

    def state_dict(self, *, include_optimizers: bool = True) -> dict:
        state = {
            "actor": self.actor.state_dict(),
            "critic": None if self.critic is None else self.critic.state_dict(),
            "entropy": self.entropy.state_dict(),
            "optimizer_steps": self.optimizer_steps,
        }
        if self.loss_scale is not None:
            state["loss_scale"] = self.loss_scale.state_dict()
        if include_optimizers:
            state["actor_optimizer"] = self.actor_optimizer.state_dict()
            if self.critic_optimizer is not None:
                state["critic_optimizer"] = self.critic_optimizer.state_dict()
        return state

    def load_state_dict(self, state: dict, *, load_optimizers: bool = True) -> None:
        self.actor.load_state_dict(state["actor"])
        if self.critic is not None:
            self.critic.load_state_dict(state["critic"])
        self.entropy.load_state_dict(state.get("entropy", {}), load_optimizer=load_optimizers)
        if load_optimizers and "actor_optimizer" in state:
            self.actor_optimizer.load_state_dict(state["actor_optimizer"])
            if self.critic_optimizer is not None:
                self.critic_optimizer.load_state_dict(state["critic_optimizer"])
            self.optimizer_steps = int(state.get("optimizer_steps", 0))
        if self.loss_scale is not None and "loss_scale" in state:
            self.loss_scale.load_state_dict(state["loss_scale"])
        configure_adam(self.actor_optimizer, self.device)
        if self.critic_optimizer is not None:
            configure_adam(self.critic_optimizer, self.device)
