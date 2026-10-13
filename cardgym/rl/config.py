"""Game-independent configuration sections: [run], [rollout], [ppo]."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

Precision = Literal["fp32", "fp16", "bf16"]


@dataclass(frozen=True)
class RunConfig:
    game: str = ""
    iterations: int = 1000
    checkpoint_every: int = 100
    # Refresh dashboard.png every N iterations (0 = only at the end).
    dashboard_every: int = 0
    device: str = "auto"
    # Autocast compute precision; master weights and Adam state stay fp32.
    precision: Precision = "fp32"
    seed: int = 0

    def validate(self) -> None:
        if not self.game:
            raise ValueError("[run] game must name a registered game.")
        if self.iterations < 0 or self.checkpoint_every < 0 or self.dashboard_every < 0:
            raise ValueError("[run] iteration counts must be non-negative.")


@dataclass(frozen=True)
class RolloutConfig:
    # Storage dtype of the KV cache during self-play decoding.
    kv_dtype: Precision = "fp32"
    # Cache rows (one per observer stream) one merged decode loop may hold;
    # a larger schedule is cut into several loops.
    max_rows: int = 12288
    # Game groups decoded alternately: the device runs one group's wave while
    # the host steps another's games (2 overlaps them; 1 is lockstep).
    cohorts: int = 1
    # "batch": deal one update's games together and decode until the last one
    # ends. "continuous": keep games in flight, replace each as it ends, and
    # carry the unfinished ones into the next update (the game must define
    # lanes; max_rows and cohorts do not apply).
    mode: Literal["batch", "continuous"] = "batch"
    # continuous: games in flight per lane, as a multiple of its per-update quota.
    concurrency: float = 1.0

    def validate(self) -> None:
        if self.max_rows < 1:
            raise ValueError("[rollout] max_rows must be positive.")
        if self.cohorts < 1:
            raise ValueError("[rollout] cohorts must be at least 1.")
        if self.mode not in ("batch", "continuous"):
            raise ValueError(f"Unknown [rollout] mode {self.mode!r}.")
        if self.concurrency <= 0:
            raise ValueError("[rollout] concurrency must be positive.")


@dataclass(frozen=True)
class PPOConfig:
    # --- actor optimizer ---
    learning_rate: float = 3e-5
    adam_betas: tuple[float, float] = (0.9, 0.999)
    # Linear ramp over the first N optimizer steps. Adam's first steps are
    # sign steps of size ~lr per parameter whatever the gradient, so a cold
    # start without a ramp jumps far past any reasonable policy change.
    lr_warmup_updates: int = 100
    lr_schedule: Literal["constant", "linear"] = "constant"
    lr_decay_start_iteration: int = 0
    lr_decay_end_iteration: int = 0
    lr_decay_final_scale: float = 1.0
    epochs: int = 1
    max_grad_norm: float = 1.0

    # --- policy objective ---
    clip_ratio: float = 0.1
    advantage_estimator: Literal["monte_carlo", "gae"] = "monte_carlo"
    gae_lambda: float = 0.95
    advantage_normalize: bool = True
    # Rescale the auxiliary-loss gradient to this fraction of the policy
    # gradient's norm before the shared clip (0 = sum the losses as-is).
    # Adam is invariant to a uniform rescale, so only the ratio matters.
    auxiliary_grad_fraction: float = 0.0

    # --- values ---
    critic_learning_rate: float = 3e-4
    critic_epochs: int = 2
    # Value errors are divided by this before squaring.
    value_reward_scale: float = 5.0
    # Weight of the actor's own per-seat value probe. It never feeds
    # advantages; it measures what observer information can predict.
    actor_value_coef: float = 0.0

    # --- entropy ---
    # adaptive: one temperature per policy head, raised below its target
    # normalized entropy and lowered above it.
    entropy_mode: Literal["off", "fixed", "adaptive"] = "adaptive"
    entropy_coef: float = 0.01
    entropy_learning_rate: float = 1e-3
    entropy_targets: dict[str, float] = field(default_factory=dict)
    entropy_target_schedule: Literal["constant", "linear", "cosine"] = "constant"
    entropy_schedule_start_iteration: int = 0
    entropy_schedule_end_iteration: int = 0
    entropy_targets_end: dict[str, float] = field(default_factory=dict)

    # --- execution ---
    # Positions per physical microbatch; the update still accumulates the
    # whole batch into one optimizer step. This is the update's memory lever.
    microbatch_positions: int = 16384
    # Tail-pad sequences to multiples of this to merge neighbouring lengths
    # into one batch (0 = exact lengths). Causality makes this exact.
    sequence_bucket_width: int = 16
    critic_sequence_bucket_width: int = 32
    # Merge length buckets until each holds this many sequences (0 = off):
    # long streams otherwise leave groups of one or two sequences.
    min_group_sequences: int = 0

    def validate(self) -> None:
        if self.learning_rate <= 0 or self.critic_learning_rate <= 0:
            raise ValueError("[ppo] learning rates must be positive.")
        if self.epochs < 1 or self.critic_epochs < 1:
            raise ValueError("[ppo] epochs must be >= 1.")
        if not 0.0 < self.clip_ratio < 1.0:
            raise ValueError("[ppo] clip_ratio must be in (0, 1).")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("[ppo] gae_lambda must be in [0, 1].")
        if self.auxiliary_grad_fraction < 0:
            raise ValueError("[ppo] auxiliary_grad_fraction must be non-negative.")
        if self.lr_schedule == "linear" and (
            self.lr_decay_end_iteration <= self.lr_decay_start_iteration
        ):
            raise ValueError("[ppo] a linear lr_schedule needs end > start iteration.")
        if self.entropy_target_schedule != "constant":
            if self.entropy_schedule_end_iteration <= self.entropy_schedule_start_iteration:
                raise ValueError("[ppo] an entropy schedule needs end > start iteration.")
            if set(self.entropy_targets_end) != set(self.entropy_targets):
                raise ValueError("[ppo] entropy_targets_end must name the same heads.")
        if self.microbatch_positions < 1:
            raise ValueError("[ppo] microbatch_positions must be positive.")

    def lr_scale(self, iteration: int) -> float:
        if self.lr_schedule == "constant":
            return 1.0
        span = self.lr_decay_end_iteration - self.lr_decay_start_iteration
        progress = min(max((iteration - self.lr_decay_start_iteration) / span, 0.0), 1.0)
        return 1.0 + progress * (self.lr_decay_final_scale - 1.0)

    def entropy_targets_at(self, iteration: int) -> dict[str, float]:
        if self.entropy_target_schedule == "constant":
            return dict(self.entropy_targets)
        span = self.entropy_schedule_end_iteration - self.entropy_schedule_start_iteration
        progress = min(max((iteration - self.entropy_schedule_start_iteration) / span, 0.0), 1.0)
        fraction = (
            progress
            if self.entropy_target_schedule == "linear"
            else 0.5 - 0.5 * math.cos(math.pi * progress)
        )
        return {
            head: start + fraction * (self.entropy_targets_end[head] - start)
            for head, start in self.entropy_targets.items()
        }
