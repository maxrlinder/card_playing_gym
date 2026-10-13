"""Per-head entropy temperatures for the PPO actor.

The objective adds ``alpha_h * H_norm`` for each policy head ``h``, where
``H_norm = H / log|legal|`` is normalized so one temperature means the same
at every decision width. In adaptive mode each ``log alpha_h`` follows the
dual-ascent loss ``log_alpha * (observed - target)``: it rises while the
observed entropy sits below target and falls above it. Targets are floors in
practice: an alpha at its lower clamp has no effect.
"""

from __future__ import annotations

import math

import torch

from .config import PPOConfig

ALPHA_MIN = 1e-6
ALPHA_MAX = 10.0


class EntropyController:
    def __init__(self, heads: tuple[str, ...], config: PPOConfig, device: torch.device) -> None:
        self.heads = heads
        self.config = config
        self.device = device
        initial = math.log(max(config.entropy_coef, 1e-8))
        self.log_alpha = {
            head: torch.nn.Parameter(torch.tensor(initial, device=device)) for head in heads
        }
        self.optimizer = (
            torch.optim.Adam(list(self.log_alpha.values()), lr=config.entropy_learning_rate)
            if config.entropy_mode == "adaptive"
            else None
        )

    def alpha(self, head: str) -> float:
        if self.config.entropy_mode == "off":
            return 0.0
        if self.config.entropy_mode == "fixed":
            return self.config.entropy_coef
        return float(self.log_alpha[head].detach().exp().clamp(max=ALPHA_MAX))

    def targets(self, iteration: int) -> dict[str, float]:
        return self.config.entropy_targets_at(iteration)

    def update(self, observed: dict[str, float], iteration: int) -> None:
        """One controller step from this update's normalized entropies."""

        if self.optimizer is None:
            return
        targets = self.targets(iteration)
        self.optimizer.zero_grad(set_to_none=True)
        loss = torch.zeros((), device=self.device)
        for head, target in targets.items():
            if head not in self.log_alpha or head not in observed:
                continue
            loss = loss + self.log_alpha[head] * (observed[head] - target)
        if not loss.requires_grad:
            return
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            for parameter in self.log_alpha.values():
                parameter.clamp_(math.log(ALPHA_MIN), math.log(ALPHA_MAX))

    def reset(self, head: str) -> None:
        """Return one temperature to its floor and forget its Adam moments.

        Momentum built while the temperature climbed would otherwise keep
        pushing it up after the reset.
        """

        if head not in self.log_alpha:
            raise ValueError(f"No entropy temperature for head {head!r}.")
        parameter = self.log_alpha[head]
        with torch.no_grad():
            parameter.fill_(math.log(ALPHA_MIN))
        if self.optimizer is not None:
            self.optimizer.state.pop(parameter, None)

    def state_dict(self) -> dict:
        return {
            "log_alpha": {head: value.detach().cpu() for head, value in self.log_alpha.items()},
            "optimizer": None if self.optimizer is None else self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict, *, load_optimizer: bool = True) -> None:
        with torch.no_grad():
            for head, value in state.get("log_alpha", {}).items():
                if head in self.log_alpha:
                    self.log_alpha[head].copy_(value.to(self.device))
        if load_optimizer and self.optimizer is not None and state.get("optimizer"):
            self.optimizer.load_state_dict(state["optimizer"])
