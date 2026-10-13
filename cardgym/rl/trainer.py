"""Game-independent PPO update with a separate (privileged) critic.

One update, given a :class:`PPOBatch`:

1. Critic baseline. The frozen critic evaluates every critic stream once;
   each decision's baseline is the acting seat's zero-sum-centered value at
   that decision's pre-action state. Advantages are Monte Carlo
   ``R - V(s)`` or GAE over the game's decision states, then normalized.
2. Actor. Every microbatch of every group accumulates into one optimizer
   step: the clipped PPO loss per policy head (minus each head's entropy
   temperature times its normalized entropy) plus the game's auxiliary
   losses and the optional per-seat value probe. With an auxiliary gradient
   budget the two halves are differentiated separately and the auxiliary
   gradient is rescaled to a fixed fraction of the policy gradient's norm
   before the single global clip.
3. Critic. ``critic_epochs`` full passes of dense per-seat value regression
   plus the game's critic auxiliaries, each one optimizer step.

Reporting scalars stay on device until the update ends: converting inside the
microbatch loop would insert a host synchronization between every forward and
backward.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

from cardgym.core.devices import (
    autocast_context,
    device_allocated_bytes,
    release_device_memory,
)

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
class DenseRows:
    """The genuine (weighted) positions of one microbatch."""

    seq: np.ndarray  # row within the microbatch
    position: np.ndarray
    weight: torch.Tensor  # float32 [N]
    flat: torch.Tensor  # seq * length + position, for dense projections
    start: int  # first group row of the microbatch

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


def _norm(gradients: Sequence[torch.Tensor | None] | None) -> float:
    present = [] if gradients is None else [g for g in gradients if g is not None]
    if not present:
        return 0.0
    return float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in present])))


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

    Fused MPS Adam needs its step counters on the device; ordinary Adam
    wants them on the CPU. Checkpoints can come from either.
    """

    fused = device.type == "mps"
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
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.actor_parameters = [p for p in self.actor.parameters() if p.requires_grad]
        self.critic_parameters = [p for p in self.critic.parameters() if p.requires_grad]
        fused = device.type == "mps"
        self.actor_optimizer = torch.optim.Adam(
            self.actor_parameters, lr=config.learning_rate, betas=config.adam_betas, fused=fused
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic_parameters,
            lr=config.critic_learning_rate,
            betas=config.adam_betas,
            fused=fused,
        )
        self.entropy = EntropyController(policy_heads, config, device)
        self.optimizer_steps = 0
        self._policy_grads: list[torch.Tensor | None] | None = None
        self._auxiliary_grads: list[torch.Tensor | None] | None = None

    # ------------------------------------------------------------------ #

    def update(self, batch: PPOBatch, iteration: int) -> dict[str, float]:
        started = time.perf_counter()
        stats: dict[str, float] = {"skipped_actor_steps": 0.0}
        totals: Totals = defaultdict(list)
        stats.update(self._prepare_advantages(batch))
        stats["policy_rows"] = float(sum(len(rows) for rows in batch.decisions.values()))
        stats["positions"] = float(sum(g.tokens.shape[0] * g.length for g in batch.actor_groups))
        targets = self.entropy.targets(iteration)
        for head, target in targets.items():
            stats[f"entropy_target_{head}"] = target

        indexes = {head: RowIndex(rows.group, rows.seq) for head, rows in batch.decisions.items()}
        for _ in range(self.config.epochs):
            self.actor.train()
            self.actor_optimizer.zero_grad(set_to_none=True)
            epoch_totals: Totals = defaultdict(list)
            self._actor_backward(batch, indexes, epoch_totals)
            reduced = {key: math.fsum(float(v) for v in values) for key, values in epoch_totals.items()}
            policy_norm, auxiliary_norm, scale = self._apply_auxiliary_budget()
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(self.actor_parameters, self.config.max_grad_norm)
            )
            self._set_learning_rate(iteration)
            if math.isfinite(grad_norm):
                standardize_grads(self.actor_parameters)
                self.actor_optimizer.step()
                self.optimizer_steps += 1
            else:
                # A non-finite gradient must never reach the weights or Adam's moments.
                self.actor_optimizer.zero_grad(set_to_none=True)
                stats["skipped_actor_steps"] += 1.0
            observed = {
                head: reduced.get(f"entropy_{head}", 0.0) / reduced[f"entropy_weight_{head}"]
                for head in self.policy_heads
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
        for head in self.policy_heads:
            stats[f"entropy_alpha_{head}"] = self.entropy.alpha(head)
        stats["learning_rate"] = float(self.actor_optimizer.param_groups[0]["lr"])
        stats["peak_device_bytes"] = float(device_allocated_bytes(self.device))
        stats.update(self._train_critic(batch))
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

    @torch.inference_mode()
    def _prepare_advantages(self, batch: PPOBatch) -> dict[str, float]:
        self.critic.eval()
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
        for group_index, group in enumerate(batch.critic_groups):
            for start, stop in microbatches(group, self.config.microbatch_positions):
                tokens = torch.from_numpy(group.tokens[start:stop]).to(self.device)
                with autocast_context(self.device, self.precision):
                    values = self.critic.player_values(self.critic.forward_hidden(tokens))
                counts = torch.from_numpy(group.player_counts[start:stop]).to(self.device)
                flat_counts = counts[:, None].expand(-1, values.shape[1]).reshape(-1)
                values = center_zero_sum(
                    values.float().reshape(-1, values.shape[-1]), flat_counts
                ).view_as(values)
                values_np = values.cpu().numpy()
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
                    seat_values = np.take_along_axis(values_np[seq, pos], columns, axis=1) * (seats < players)
                    predictions[head][selected] = seat_values[np.arange(selected.size), rows.seat[selected]]
                    if gae:
                        state_values.setdefault(head, np.zeros((len(rows), self.max_players)))
                        state_values[head][selected] = seat_values
                seq_dense, pos_dense = np.nonzero(group.position_weight[start:stop] > 0)
                players = group.player_counts[start:stop][seq_dense]
                active = np.arange(self.max_players)[None, :] < players[:, None]
                weight = group.position_weight[start:stop][seq_dense, pos_dense].astype(np.float64)
                dense_prediction.append(values_np[seq_dense, pos_dense][active])
                dense_target.append(group.returns[start:stop][seq_dense][active])
                dense_weight.append(np.repeat(weight / players, players))

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

    def _dense_rows(self, group: SequenceGroup, start: int, stop: int) -> DenseRows:
        seq, pos = np.nonzero(group.position_weight[start:stop] > 0)
        weight = group.position_weight[start:stop][seq, pos]
        return DenseRows(
            seq=seq,
            position=pos,
            weight=torch.from_numpy(weight.astype(np.float32)).to(self.device),
            flat=torch.from_numpy(seq * group.length + pos).to(self.device),
            start=start,
        )

    def _actor_backward(
        self, batch: PPOBatch, indexes: dict[str, RowIndex], totals: Totals
    ) -> None:
        from cardgym.nn.metal_trunk import DualTrunkPass

        self._policy_grads = None
        self._auxiliary_grads = None
        alpha = {head: self.entropy.alpha(head) for head in self.policy_heads}
        budget = self.config.auxiliary_grad_fraction > 0
        for group_index, group in enumerate(batch.actor_groups):
            for start, stop in microbatches(group, self.config.microbatch_positions):
                tokens = torch.from_numpy(group.tokens[start:stop]).to(self.device)
                dual = None
                with autocast_context(self.device, self.precision):
                    if budget and self.actor.trunk.uses_metal(self.device, tokens.shape[1]):
                        dual = DualTrunkPass(self.actor, tokens)
                        hidden = dual.hidden
                    else:
                        hidden = self.actor.forward_hidden(tokens)
                context = self.actor.pop_context()
                dense = self._dense_rows(group, start, stop)
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
                    selected = indexes[head].select(group_index, start, stop)
                    if selected.size:
                        policy = policy + self._policy_loss(
                            hidden, tokens, context, rows, selected, start, head, alpha[head], totals
                        )
                self._backward(auxiliary, policy, dual)

    def _policy_loss(
        self,
        hidden: torch.Tensor,
        tokens: torch.Tensor,
        context: torch.Tensor | None,
        rows: DecisionRows,
        selected: np.ndarray,
        start: int,
        head: str,
        alpha: float,
        totals: Totals,
    ) -> torch.Tensor:
        seq = torch.from_numpy(rows.seq[selected] - start).to(self.device)
        pos = torch.from_numpy(rows.position[selected]).to(self.device)
        features = (
            None
            if rows.features is None
            else {key: torch.from_numpy(value[selected]).to(self.device) for key, value in rows.features.items()}
        )
        with autocast_context(self.device, self.precision):
            logits = self.actor.policy_logits(
                hidden[seq, pos], head, features, tokens[seq, pos], None if context is None else context[seq, pos]
            )
        old_probs = torch.from_numpy(rows.probs[selected]).to(self.device)
        logits = logits.float()[:, : old_probs.shape[1]]
        terms = ppo_terms(
            logits,
            old_probs,
            torch.from_numpy(rows.action[selected]).to(self.device),
            torch.from_numpy(rows.advantages[selected]).to(self.device),
            clip_ratio=self.config.clip_ratio,
        )
        weight = torch.from_numpy(rows.weight[selected]).to(self.device)
        loss = (weight * terms.losses).sum()
        eligible = weight * terms.entropy_eligible.float()
        entropy_term = (eligible * terms.normalized_entropy).sum()
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
        return loss - alpha * entropy_term if alpha else loss

    def _value_loss(
        self,
        model,
        hidden: torch.Tensor,
        dense: DenseRows,
        group: SequenceGroup,
        totals: Totals,
        prefix: str,
    ) -> torch.Tensor:
        """Dense per-seat zero-sum value regression, unit mass per stream."""

        with autocast_context(self.device, self.precision):
            values = model.player_values(hidden.reshape(-1, hidden.shape[-1]))
        prediction = values[dense.flat].float()
        players = torch.from_numpy(group.player_counts[dense.start :][dense.seq]).to(self.device)
        targets = torch.from_numpy(group.returns[dense.start :][dense.seq]).to(self.device)
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

    def _backward(self, auxiliary: torch.Tensor, policy: torch.Tensor, dual) -> None:
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
        policy_norm = _norm(policy)
        auxiliary_norm = _norm(auxiliary)
        if auxiliary_norm <= 0:
            scale = 0.0
        elif policy_norm <= 0:
            scale = 1.0
        else:
            scale = self.config.auxiliary_grad_fraction * policy_norm / auxiliary_norm
        for index, parameter in enumerate(self.actor_parameters):
            p = None if policy is None else policy[index]
            a = None if auxiliary is None else auxiliary[index]
            if a is not None and scale != 1.0:
                a.mul_(scale)
            parameter.grad = a if p is None else (p if a is None else p.add_(a))
        return policy_norm, auxiliary_norm, scale

    def _summarize_actor(self, totals: Totals) -> dict[str, float]:
        reduced = {key: math.fsum(float(v) for v in values) for key, values in totals.items()}
        epochs = self.config.epochs
        row_weight = max(reduced.get("row_weight", 0.0), 1e-12)
        stats = {
            "loss_policy": reduced.get("policy_loss", 0.0) / epochs,
            "entropy": reduced.get("entropy_raw", 0.0) / row_weight,
            "clip_fraction": reduced.get("clip", 0.0) / row_weight,
            "behavior_kl": reduced.get("behavior_kl", 0.0) / row_weight,
        }
        for head in self.policy_heads:
            weight = reduced.get(f"entropy_weight_{head}", 0.0)
            stats[f"entropy_{head}"] = reduced.get(f"entropy_{head}", 0.0) / weight if weight else 0.0
        value_weight = reduced.get("actor_value_weight", 0.0)
        if value_weight > 0:
            stats["actor_value_loss"] = reduced.get("actor_value_loss", 0.0) / epochs
            stats["actor_value_rmse"] = math.sqrt(
                max(reduced.get("actor_value_squared_error", 0.0), 0.0) / value_weight
            )
        for key, value in self.hooks.summarize(reduced).items():
            stats[key] = value / epochs if key.startswith("loss_") else value
        return stats

    # -- critic ---------------------------------------------------------- #

    def _train_critic(self, batch: PPOBatch) -> dict[str, float]:
        epoch_losses: list[float] = []
        totals: Totals = defaultdict(list)
        grad_norm = 0.0
        skipped = 0
        for epoch in range(self.config.critic_epochs):
            self.critic.train()
            self.critic_optimizer.zero_grad(set_to_none=True)
            pending: list[torch.Tensor] = []
            for group in batch.critic_groups:
                for start, stop in microbatches(group, self.config.microbatch_positions):
                    tokens = torch.from_numpy(group.tokens[start:stop]).to(self.device)
                    with autocast_context(self.device, self.precision):
                        hidden = self.critic.forward_hidden(tokens)
                    dense = self._dense_rows(group, start, stop)
                    loss = self._value_loss(self.critic, hidden, dense, group, totals, "critic_value")
                    hooked = self.hooks.critic_auxiliary(
                        self.critic, hidden, dense, group, totals, epoch == 0
                    )
                    if hooked is not None:
                        loss = loss + hooked
                    loss.backward()
                    pending.append(loss.detach())
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(self.critic_parameters, self.config.max_grad_norm)
            )
            if math.isfinite(grad_norm):
                standardize_grads(self.critic_parameters)
                self.critic_optimizer.step()
            else:
                self.critic_optimizer.zero_grad(set_to_none=True)
                skipped += 1
            epoch_losses.append(math.fsum(float(value) for value in pending))
        reduced = {key: math.fsum(float(v) for v in values) for key, values in totals.items()}
        stats = {
            "critic_grad_norm": grad_norm,
            "skipped_critic_steps": float(skipped),
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
            "critic": self.critic.state_dict(),
            "entropy": self.entropy.state_dict(),
            "optimizer_steps": self.optimizer_steps,
        }
        if include_optimizers:
            state["actor_optimizer"] = self.actor_optimizer.state_dict()
            state["critic_optimizer"] = self.critic_optimizer.state_dict()
        return state

    def load_state_dict(self, state: dict, *, load_optimizers: bool = True) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.entropy.load_state_dict(state.get("entropy", {}), load_optimizer=load_optimizers)
        if load_optimizers and "actor_optimizer" in state:
            self.actor_optimizer.load_state_dict(state["actor_optimizer"])
            self.critic_optimizer.load_state_dict(state["critic_optimizer"])
            self.optimizer_steps = int(state.get("optimizer_steps", 0))
        configure_adam(self.actor_optimizer, self.device)
        configure_adam(self.critic_optimizer, self.device)
