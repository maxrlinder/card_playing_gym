"""Causal transformer trunk shared by every game's actor and critic.

A game supplies a token embedder; the trunk supplies everything after it:

    x     = RMSNorm(embedding sum)                       optional input norm
    ple   = (game PLE lookup + W_c x) / sqrt(2)          per-layer conditioning
    for each block:
        x = x + attn(LN(x), value embedding at selected blocks)   GQA
        x = x + MLP(LN(x))
        x = x + R(ple[layer])                            repeat-expanded PLE
    h     = LayerNorm(x)

Per-layer embeddings (PLE) are a packed ``[layers * ple_dim]`` lookup from
the game's token fields plus a learned projection of the normalized input,
added after every block. Value embeddings are a packed lookup added directly
to attention V (before the cache write) at alternating or all blocks. Both
start at zero, so enabling them leaves an existing function unchanged.

Three execution paths share the weights: a full causal forward (training and
replay), a prefill that writes fresh cache rows, and a ragged cached step in
which every row sits at its own length (merged rollout waves).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from .kv_cache import KVCache

ValueEmbeddingMode = Literal["off", "alternating", "all"]


@dataclass(frozen=True)
class TrunkConfig:
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: int = 2
    d_ff: int = 768
    per_layer_dim: int = 0
    value_embeddings: ValueEmbeddingMode = "off"
    input_norm: bool = True

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def kv_width(self) -> int:
        return self.n_kv_heads * self.head_dim

    @property
    def value_layers(self) -> tuple[int, ...]:
        """Blocks whose V receives a value embedding.

        ``alternating`` always includes the final block and every other block
        working backward (nanochat's placement).
        """

        if self.value_embeddings == "off":
            return ()
        if self.value_embeddings == "all":
            return tuple(range(self.n_layers))
        return tuple(
            layer for layer in range(self.n_layers) if layer % 2 == (self.n_layers - 1) % 2
        )

    @property
    def ple_width(self) -> int:
        return self.n_layers * self.per_layer_dim

    @property
    def value_width(self) -> int:
        return len(self.value_layers) * self.kv_width

    def validate(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads.")
        if self.n_kv_heads > self.n_heads or self.n_heads % self.n_kv_heads:
            raise ValueError("n_kv_heads must divide n_heads.")
        if self.per_layer_dim < 0:
            raise ValueError("per_layer_dim must be non-negative.")
        if self.value_embeddings not in ("off", "alternating", "all"):
            raise ValueError(f"Unknown value_embeddings mode {self.value_embeddings!r}.")


def init_weights(module: nn.Module) -> None:
    """GPT-2 style N(0, 0.02) weights and zero biases."""

    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)


class RMSNorm(nn.Module):
    """RMSNorm with a gain only."""

    def __init__(self, d: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # One fused kernel (the op-by-op form is six launches per call, paid
        # on every rollout wave); the same function up to fp32 rounding.
        return F.rms_norm(x.float(), (x.shape[-1],), eps=self.eps).to(x.dtype) * self.weight


def _cuda_decode_supported(head_dim: int, span: int, queries: int) -> bool:
    from .cuda_decode import supported

    return supported(head_dim, span, queries)


class DecoderBlock(nn.Module):
    def __init__(self, config: TrunkConfig) -> None:
        super().__init__()
        d = config.d_model
        self.n_heads = config.n_heads
        self.kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.head_group = config.n_heads // config.n_kv_heads
        self.scale = 1.0 / math.sqrt(config.head_dim)
        self.ln_attn = nn.LayerNorm(d)
        # One fused projection: decode issues a forward per game event, so
        # kernel-launch count dominates its cost.
        self.qkv_proj = nn.Linear(d, (self.n_heads + 2 * self.kv_heads) * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.n_heads * self.head_dim, d, bias=False)
        self.ln_mlp = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, config.d_ff), nn.GELU(), nn.Linear(config.d_ff, d))

    def _qkv(self, h: torch.Tensor, value_embedding: torch.Tensor | None):
        batch, length, _ = h.shape
        q, k, v = self.qkv_proj(h).split(
            [self.n_heads * self.head_dim, self.kv_heads * self.head_dim, self.kv_heads * self.head_dim],
            dim=-1,
        )
        q = q.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, length, self.kv_heads, self.head_dim)
        if value_embedding is not None:
            v = v + value_embedding.view_as(v)
        return q, k, v.transpose(1, 2)

    def _finish(self, x: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        batch, _, length, _ = attn.shape
        x = x + self.out_proj(attn.transpose(1, 2).reshape(batch, length, -1))
        return x + self.mlp(self.ln_mlp(x))

    def forward_full(self, x: torch.Tensor, value_embedding: torch.Tensor | None = None) -> torch.Tensor:
        q, k, v = self._qkv(self.ln_attn(x), value_embedding)
        attn = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, enable_gqa=self.kv_heads != self.n_heads
        )
        return self._finish(x, attn)

    def forward_prefill(
        self,
        x: torch.Tensor,
        cache: KVCache,
        layer: int,
        rows: torch.Tensor | None,
        value_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fresh rows: write positions 0..T-1 and attend causally among them."""

        q, k, v = self._qkv(self.ln_attn(x), value_embedding)
        cache.write_range(layer, rows, 0, k, v)
        attn = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, enable_gqa=self.kv_heads != self.n_heads
        )
        return self._finish(x, attn)

    def forward_ragged(
        self,
        x: torch.Tensor,
        cache: KVCache,
        layer: int,
        write_positions: torch.Tensor,
        key_lengths: torch.Tensor,
        span: int,
        value_embedding: torch.Tensor | None = None,
        rows: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Cache rows (0..B-1, or ``rows``) each append T tokens at their own offset.

        ``key_lengths[b]`` is row b's length including its first new token;
        query t of row b may see keys below ``key_lengths[b] + t``. Explicit
        ``rows`` gather each row's keys (a copy); the dense default reads a
        zero-copy prefix of the pool.
        """

        q, k, v = self._qkv(self.ln_attn(x), value_embedding)
        if (
            x.is_cuda
            and cache.dtype == torch.float16
            and _cuda_decode_supported(self.head_dim, span, self.head_group * x.shape[1])
        ):
            # Attention straight from the pool, no per-layer gather (cardgym.nn.cuda_decode).
            from .cuda_decode import decode_attention

            if rows is None:
                rows = torch.arange(x.shape[0], device=x.device)
            cache.write_step(layer, rows, write_positions, k, v)
            attn = decode_attention(q, cache.k[layer], cache.v[layer], rows, key_lengths, span, self.scale)
            return self._finish(x, attn.to(x.dtype))
        if rows is None:
            cache.write_step(layer, torch.arange(x.shape[0], device=x.device), write_positions, k, v)
            k_all, v_all = cache.read(layer, None, span, count=x.shape[0])
        else:
            cache.write_step(layer, rows, write_positions, k, v)
            k_all, v_all = cache.read(layer, rows, span)
        attn = self._grouped_attention(q.to(cache.dtype), k_all, v_all, key_lengths)
        return self._finish(x, attn.to(x.dtype))

    def _grouped_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Ragged GQA attention for T new queries against the cached prefix.

        Written out rather than delegated to SDPA: on MPS the fused kernel is
        markedly slower at short query lengths and its GQA path slower still.
        Grouping the query heads keeps K/V at ``kv_heads`` (merging the
        group and time axes keeps this a 4-D matmul; broadcasting a group
        axis makes MPS materialize expanded K/V, measured 3x slower). K/V are
        read to the batch maximum and masked, which measured far cheaper than
        splitting the batch by length. Softmax runs in fp32.
        """

        batch, _, length, _ = q.shape
        grouped = q.reshape(batch, self.kv_heads, self.head_group * length, self.head_dim)
        scores = (grouped @ k.transpose(-1, -2)) * self.scale
        keys = torch.arange(k.shape[2], device=q.device)
        if length == 1:
            blocked = keys[None, :] >= key_lengths[:, None]
            scores = scores.masked_fill(blocked[:, None, None, :], float("-inf"))
        else:
            limits = key_lengths[:, None] + torch.arange(length, device=q.device)[None, :]
            blocked = keys[None, None, :] >= limits[:, :, None]
            # Row g * length + t of the merged axis is query t of group g.
            scores = scores.masked_fill(blocked.repeat(1, self.head_group, 1)[:, None], float("-inf"))
        weights = scores.float().softmax(dim=-1).to(v.dtype)
        return (weights @ v).view(batch, self.n_heads, length, self.head_dim)


class Trunk(nn.Module):
    def __init__(self, config: TrunkConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        d = config.d_model
        self.input_norm = RMSNorm(d) if config.input_norm else None
        self.ple_context = (
            nn.Linear(d, config.ple_width, bias=False) if config.per_layer_dim else None
        )
        self.blocks = nn.ModuleList(DecoderBlock(config) for _ in range(config.n_layers))
        self.final_norm = nn.LayerNorm(d)
        self._value_index = {layer: index for index, layer in enumerate(config.value_layers)}
        # Optional fused Metal implementation (cardgym.nn.metal_trunk); a plain
        # attribute because it holds no parameters of its own.
        self.metal = None
        self.blocks.apply(init_weights)
        residual_scale = 1.0 / math.sqrt(2 * config.n_layers)
        for block in self.blocks:
            block.out_proj.weight.data.mul_(residual_scale)
            block.mlp[2].weight.data.mul_(residual_scale)
        if self.ple_context is not None:
            nn.init.zeros_(self.ple_context.weight)

    # -- conditioning ----------------------------------------------------

    def condition(
        self,
        x: torch.Tensor,
        ple_lookup: torch.Tensor | None,
        value_lookup: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Normalize the embedding sum and shape the per-layer streams.

        Returns ``x``, ``ple`` [..., layers, ple_dim] and ``values``
        [..., value_layers, kv_width].
        """

        if self.input_norm is not None:
            x = self.input_norm(x)
        lead = x.shape[:-1]
        ple = None
        if self.ple_context is not None:
            if ple_lookup is None:
                raise ValueError("This trunk expects a per-layer embedding lookup.")
            ple = ((ple_lookup + self.ple_context(x)) * (1.0 / math.sqrt(2.0))).view(
                *lead, self.config.n_layers, self.config.per_layer_dim
            )
        values = None
        if self._value_index:
            if value_lookup is None:
                raise ValueError("This trunk expects a value-embedding lookup.")
            values = value_lookup.view(*lead, len(self._value_index), self.config.kv_width)
        return x, ple, values

    def _value(self, values: torch.Tensor | None, layer: int) -> torch.Tensor | None:
        index = self._value_index.get(layer)
        return None if index is None or values is None else values[..., index, :]

    def _add_ple(self, x: torch.Tensor, ple: torch.Tensor | None, layer: int) -> torch.Tensor:
        if ple is None:
            return x
        update = ple[..., layer, :]
        if update.shape[-1] == x.shape[-1]:
            return x + update
        if x.shape[-1] % update.shape[-1] == 0:
            repeats = x.shape[-1] // update.shape[-1]
            # Broadcast in [repeat, ple] form rather than materializing the
            # repeated [layers, d_model] tensor.
            shaped = x.view(*x.shape[:-1], repeats, update.shape[-1])
            shaped = shaped + update.unsqueeze(-2) * (1.0 / math.sqrt(repeats))
            return shaped.reshape_as(x)
        repeats = math.ceil(x.shape[-1] / update.shape[-1])
        expanded = torch.cat([update] * repeats, dim=-1)[..., : x.shape[-1]]
        return x + expanded * (1.0 / math.sqrt(repeats))

    # -- execution paths ---------------------------------------------------

    def uses_metal(self, device: torch.device, length: int) -> bool:
        """The fused kernels hard-code bf16 GEMMs: MPS under bf16 autocast only."""

        return (
            self.metal is not None
            and device.type == "mps"
            and length <= self.metal.max_length
            and torch.is_autocast_enabled("mps")
            and torch.get_autocast_dtype("mps") == torch.bfloat16
        )

    def forward(
        self,
        x: torch.Tensor,
        ple: torch.Tensor | None,
        values: torch.Tensor | None,
        bounds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Full causal forward over [B, T, d]; returns final-norm hidden states.

        With ``bounds`` the input is packed rows [N, d] instead (fused trunk
        only; see :class:`cardgym.nn.packing.Packing`).
        """

        if bounds is not None:
            if self.metal is None:
                raise ValueError("Packed rows need the fused Metal trunk.")
            from .metal_trunk import run_trunk

            return run_trunk(self, x, ple, values, bounds)
        if self.uses_metal(x.device, x.shape[1]):
            from .metal_trunk import run_trunk

            return run_trunk(self, x, ple, values)
        for layer, block in enumerate(self.blocks):
            x = block.forward_full(x, self._value(values, layer))
            x = self._add_ple(x, ple, layer)
        return self.final_norm(x)

    def forward_prefill(
        self,
        x: torch.Tensor,
        ple: torch.Tensor | None,
        values: torch.Tensor | None,
        cache: KVCache,
        rows: torch.Tensor | None,
    ) -> torch.Tensor:
        """Encode fresh prefixes into cache ``rows``; hidden state at the last position."""

        for layer, block in enumerate(self.blocks):
            x = block.forward_prefill(x, cache, layer, rows, self._value(values, layer))
            x = self._add_ple(x, ple, layer)
        return self.final_norm(x[:, -1])

    def forward_ragged(
        self,
        x: torch.Tensor,
        ple: torch.Tensor | None,
        values: torch.Tensor | None,
        cache: KVCache,
        positions: torch.Tensor,
        readout_offsets: torch.Tensor,
        span: int | None = None,
        rows: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Append a padded run of tokens to rows 0..B-1 (or ``rows``), each at its own length.

        ``positions[b]`` is row b's current length (its next write offset).
        Rows shorter than the run are padded; the surplus K/V lands on
        positions the row has not reached, is overwritten on a later wave,
        and is never attended to meanwhile. Writes past the pool are clamped
        onto its last slot, which callers reserve as padding-only. Returns
        the final-norm hidden state of token ``readout_offsets[b]`` per row.
        ``span`` (the longest row after the append) lets callers that know it
        avoid reading ``positions`` back from the device, which would stall.
        """

        length = x.shape[1]
        batch = torch.arange(x.shape[0], device=x.device)
        if self.uses_metal(x.device, length) and self.metal.decode_supported(cache):
            normed = self.metal.decode(x, ple, values, cache, positions, self, rows)
            return normed[batch, readout_offsets]
        steps = torch.arange(length, device=positions.device)[None, :]
        write_positions = (positions[:, None] + steps).clamp(max=cache.max_len - 1)
        if span is None:
            span = int(positions.max()) + length
        span = min(span, cache.max_len)
        for layer, block in enumerate(self.blocks):
            x = block.forward_ragged(
                x, cache, layer, write_positions, positions + 1, span, self._value(values, layer), rows
            )
            x = self._add_ple(x, ple, layer)
        return self.final_norm(x[batch, readout_offsets])
