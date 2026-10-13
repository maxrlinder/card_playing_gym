"""PPO and zero-sum value math, independent of any game.

Every probability-sensitive quantity (masked log-softmax, likelihood ratios,
KL, entropy, advantages) is computed in fp32 regardless of autocast.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class PPOTerms:
    losses: torch.Tensor           # [N] clipped surrogate, to be minimized
    divergences: torch.Tensor      # [N] KL(old || new) over legal actions
    entropy: torch.Tensor          # [N] nats
    normalized_entropy: torch.Tensor  # [N] entropy / log(legal count)
    entropy_eligible: torch.Tensor  # [N] more than one legal action
    ratios: torch.Tensor           # [N]
    clipped: torch.Tensor          # [N] |ratio - 1| > clip


def ppo_terms(
    logits: torch.Tensor,
    old_probs: torch.Tensor,
    actions: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip_ratio: float,
) -> PPOTerms:
    """Masked PPO terms. Legal actions are exactly those with ``old_probs > 0``.

    ``-min(r A, clip(r, 1-e, 1+e) A)`` with ``r = pi(a) / pi_old(a)``. The
    entropy is normalized by ``log |legal|`` so one temperature means the
    same thing at every decision width; forced moves are ineligible.
    """

    logits = logits.float()
    old_probs = old_probs.float()
    advantages = advantages.float().detach()
    legal = old_probs > 0
    log_probs = torch.log_softmax(logits.masked_fill(~legal, float("-inf")), dim=-1)
    safe_log_probs = torch.where(legal, log_probs, torch.zeros_like(log_probs))
    old_log_probs = torch.log(old_probs.clamp_min(1e-12))
    action_log_probs = log_probs.gather(1, actions[:, None]).squeeze(1)
    old_action_log_probs = old_log_probs.gather(1, actions[:, None]).squeeze(1)
    ratios = torch.exp(action_log_probs - old_action_log_probs)
    clipped_ratios = ratios.clamp(1.0 - clip_ratio, 1.0 + clip_ratio)
    losses = -torch.minimum(ratios * advantages, clipped_ratios * advantages)
    divergences = (old_probs * (old_log_probs - safe_log_probs) * legal.float()).sum(dim=-1)
    probabilities = safe_log_probs.exp() * legal.float()
    entropy = -(probabilities * safe_log_probs).sum(dim=-1)
    legal_count = legal.sum(dim=-1)
    eligible = legal_count > 1
    normalized = torch.where(
        eligible, entropy / legal_count.clamp_min(2).float().log(), torch.zeros_like(entropy)
    )
    return PPOTerms(
        losses=losses,
        divergences=divergences,
        entropy=entropy,
        normalized_entropy=normalized,
        entropy_eligible=eligible,
        ratios=ratios,
        clipped=(ratios - 1.0).abs() > clip_ratio,
    )


def masked_probabilities(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Softmax over legal entries only; illegal entries get exactly zero."""

    constrained = np.where(mask, logits, -np.inf)
    stable = constrained - constrained.max(axis=-1, keepdims=True)
    weights = np.exp(stable)
    weights[~mask] = 0.0
    return weights / weights.sum(axis=-1, keepdims=True)


def sample_rows(probabilities: np.ndarray, mask: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
    """Inverse-CDF sample one legal index per row; robust to rounding at the tail."""

    width = probabilities.shape[1]
    sampled = (np.cumsum(probabilities, axis=1) < uniforms[:, None]).sum(axis=1)
    last_legal = width - 1 - np.argmax(mask[:, ::-1], axis=1)
    sampled = np.minimum(sampled, last_legal)
    bad = ~mask[np.arange(len(sampled)), sampled]
    sampled[bad] = last_legal[bad]
    return sampled


def center_zero_sum(values: torch.Tensor, num_players: torch.Tensor) -> torch.Tensor:
    """Project [rows, max_players] values onto the active seats' zero-sum subspace.

    Relative terminal rewards sum to zero in every realized game, so their
    conditional expectations must too. Subtracting the active mean is the
    orthogonal projection under equal per-seat MSE; padded seats return zero.
    """

    active = torch.arange(values.shape[1], device=values.device)[None, :] < num_players[:, None]
    active_float = active.to(values.dtype)
    mean = (values * active_float).sum(dim=-1, keepdim=True) / num_players[:, None].to(values.dtype)
    return (values - mean) * active_float


def relative_rewards(scores: np.ndarray) -> np.ndarray:
    """``score_i - mean(score_j, j != i)``: a zero-sum transform of raw scores."""

    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape[0] < 2:
        raise ValueError("Relative rewards require at least two players.")
    opponents = scores.shape[0] - 1
    return scores - (scores.sum() - scores) / opponents


def gae_advantages(
    state_values: np.ndarray,
    terminal_returns: np.ndarray,
    gae_lambda: float,
) -> np.ndarray:
    """Undiscounted GAE over one game's decision states, for every seat.

    ``state_values[s, i]`` is the frozen critic value of seat ``i`` at the
    game's ``s``-th decision state (any actor's), in order. A transition runs
    from one decision state to the next, so opponents' actions and automatic
    bookkeeping never open a gap. Rewards are zero until the terminal
    transition, whose successor value is the realized return. ``lambda = 1``
    telescopes to the Monte Carlo ``R - V(s_t)``; ``lambda = 0`` is one-step
    TD. The caller reads the acting seat's column.
    """

    advantages = np.zeros_like(state_values)
    next_value = np.asarray(terminal_returns, dtype=state_values.dtype)
    running = np.zeros_like(next_value)
    for index in range(state_values.shape[0] - 1, -1, -1):
        running = (next_value - state_values[index]) + gae_lambda * running
        advantages[index] = running
        next_value = state_values[index]
    return advantages


def weighted_mean_std(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    total = float(weights.sum())
    if total <= 0:
        return 0.0, 1.0
    mean = float(np.dot(weights, values) / total)
    variance = float(np.dot(weights, (values - mean) ** 2) / total)
    return mean, math.sqrt(max(variance, 1e-12))


def weighted_regression_stats(
    prediction: np.ndarray, target: np.ndarray, weight: np.ndarray
) -> dict[str, float]:
    """RMSE, zero-predictor RMSE, correlation, and prediction spread."""

    total = float(weight.sum())
    if total <= 0:
        return {"rmse": 0.0, "zero_rmse": 0.0, "correlation": 0.0, "prediction_std": 0.0}
    error = prediction - target
    p_mean = float(np.dot(weight, prediction) / total)
    t_mean = float(np.dot(weight, target) / total)
    p_var = float(np.dot(weight, (prediction - p_mean) ** 2) / total)
    t_var = float(np.dot(weight, (target - t_mean) ** 2) / total)
    covariance = float(np.dot(weight, (prediction - p_mean) * (target - t_mean)) / total)
    denominator = math.sqrt(max(p_var * t_var, 0.0))
    return {
        "rmse": math.sqrt(float(np.dot(weight, error * error)) / total),
        "zero_rmse": math.sqrt(float(np.dot(weight, target * target)) / total),
        "correlation": covariance / denominator if denominator else 0.0,
        "prediction_std": math.sqrt(max(p_var, 0.0)),
    }
