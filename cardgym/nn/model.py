"""Base class for a game model: token embedder in front of the shared trunk."""

from __future__ import annotations

import torch
from torch import nn

from .kv_cache import KVCache
from .trunk import Trunk, TrunkConfig, init_weights


def value_mlp(d_model: int, outputs: int) -> nn.Sequential:
    """Hidden -> GELU MLP -> per-seat outputs (actor value probe, critic value)."""

    head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, outputs))
    head.apply(init_weights)
    return head


class SequenceModel(nn.Module):
    """Subclasses implement :meth:`embed` and add their own readout heads.

    ``embed(tokens [B, T, W])`` returns the raw embedding sum [B, T, d] plus
    the packed per-layer and value-embedding lookups (or None when the trunk
    does not use them). Everything after that is shared.
    """

    def __init__(self, trunk_config: TrunkConfig) -> None:
        super().__init__()
        self.trunk = Trunk(trunk_config)

    @property
    def trunk_config(self) -> TrunkConfig:
        return self.trunk.config

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def embed(
        self, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        raise NotImplementedError

    def prepare(self, tokens: torch.Tensor):
        """Embed and condition: the trunk's (x, ple, values) inputs."""

        return self.trunk.condition(*self.embed(tokens))

    def forward_hidden(self, tokens: torch.Tensor) -> torch.Tensor:
        """Full causal forward; final-norm hidden states [B, T, d]."""

        return self.trunk(*self.prepare(tokens))

    def forward_prefill(
        self, tokens: torch.Tensor, cache: KVCache, rows: torch.Tensor | None
    ) -> torch.Tensor:
        """Encode fresh prefixes [B, T, W] into cache rows; last hidden [B, d]."""

        return self.trunk.forward_prefill(*self.prepare(tokens), cache, rows)

    def forward_ragged(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        cache: KVCache,
        readout_offsets: torch.Tensor,
        span: int | None = None,
    ) -> torch.Tensor:
        """Append a padded run per cache row 0..B-1; hidden at each readout [B, d]."""

        return self.trunk.forward_ragged(
            *self.prepare(tokens), cache, positions, readout_offsets, span
        )

    def new_cache(
        self,
        capacity: int,
        max_len: int,
        dtype: torch.dtype = torch.float32,
    ) -> KVCache:
        config = self.trunk_config
        return KVCache(
            n_layers=config.n_layers,
            kv_heads=config.n_kv_heads,
            head_dim=config.head_dim,
            capacity=capacity,
            max_len=max_len,
            device=self.device,
            dtype=dtype,
        )

    def enable_metal_trunk(self, max_length: int) -> None:
        """Route bf16 MPS trunk forwards (and backwards) through fused Metal kernels."""

        from .metal_trunk import MetalTrunk

        self.trunk.metal = MetalTrunk(
            self.trunk_config, max_length=max_length, eps=self.trunk.final_norm.eps
        )
