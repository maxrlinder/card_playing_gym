"""Preallocated key/value cache pool for batched autoregressive decoding.

One pool serves one set of model weights. Row ``r`` holds one observer's
stream; the rollout engine owns the row assignment. Storage is a single
stacked ``[n_layers, rows, kv_heads, max_len, head_dim]`` tensor when that fits
inside the INT_MAX element limit MPSGraph imposes on one tensor, and one tensor
per layer otherwise. ``self.k[layer]`` is a view either way, so every read and
write path is layout-independent.
"""

from __future__ import annotations

import torch

# Elements, not bytes: the MPSGraph limit is on element count.
MAX_TENSOR_ELEMENTS = 2**31 - 1


class KVCache:
    # Cap on the gather temporary one ``copy_rows`` chunk may materialize.
    COPY_CHUNK_BYTES = 256 * 1024 * 1024

    def __init__(
        self,
        *,
        n_layers: int,
        kv_heads: int,
        head_dim: int,
        capacity: int,
        max_len: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.n_layers = n_layers
        self.capacity = capacity
        self.max_len = max_len
        self.device = torch.device(device)
        self.dtype = dtype
        self._row_shape = (kv_heads, max_len, head_dim)
        elements = n_layers * capacity * kv_heads * max_len * head_dim
        self.stacked = elements <= MAX_TENSOR_ELEMENTS
        # Zero-filled: ragged reads span unwritten (masked) positions, and a
        # masked NaN would still poison the weighted sum as 0 * NaN.
        fill = 0.0
        if self.stacked:
            self._base_k = torch.full(
                (n_layers, capacity, *self._row_shape), fill, device=self.device, dtype=dtype
            )
            self._base_v = torch.full_like(self._base_k, fill)
            # unbind returns views, so writes through k[layer] reach the base.
            self.k = list(self._base_k.unbind(0))
            self.v = list(self._base_v.unbind(0))
        else:
            self._base_k = self._base_v = None
            self.k = [
                torch.full((capacity, *self._row_shape), fill, device=self.device, dtype=dtype)
                for _ in range(n_layers)
            ]
            self.v = [torch.full_like(k, fill) for k in self.k]

    @staticmethod
    def bytes_per_row(n_layers: int, kv_heads: int, head_dim: int, max_len: int, dtype: torch.dtype) -> int:
        return n_layers * kv_heads * max_len * head_dim * 2 * dtype.itemsize

    def write_range(
        self,
        layer: int,
        rows: torch.Tensor | None,
        start: int,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> None:
        """Write [B, kv_heads, T, head_dim] at a shared offset; ``rows=None`` is 0..B-1."""

        stop = start + k_new.shape[2]
        if rows is None:
            count = k_new.shape[0]
            self.k[layer][:count, :, start:stop] = k_new.to(self.dtype)
            self.v[layer][:count, :, start:stop] = v_new.to(self.dtype)
        else:
            self.k[layer][rows, :, start:stop] = k_new.to(self.dtype)
            self.v[layer][rows, :, start:stop] = v_new.to(self.dtype)

    def write_step(
        self,
        layer: int,
        rows: torch.Tensor,
        positions: torch.Tensor,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> None:
        """Write K/V at a per-row offset: ``positions`` is [B] or [B, T]."""

        length = k_new.shape[2]
        if positions.dim() == 1:
            if length != 1:
                raise ValueError("A [B] position vector writes exactly one position per row.")
            self.k[layer][rows, :, positions] = k_new[:, :, 0].to(self.dtype)
            self.v[layer][rows, :, positions] = v_new[:, :, 0].to(self.dtype)
            return
        if positions.shape[1] != length:
            raise ValueError("positions must carry one offset per written step.")
        index = rows[:, None]
        self.k[layer][index, :, positions] = k_new.transpose(1, 2).to(self.dtype)
        self.v[layer][index, :, positions] = v_new.transpose(1, 2).to(self.dtype)

    def read(
        self,
        layer: int,
        rows: torch.Tensor | None,
        length: int,
        count: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cached K/V [B, kv_heads, length, head_dim]; ``rows=None`` is a zero-copy 0..count-1 view."""

        if rows is None:
            if count is None:
                raise ValueError("A dense read requires an explicit row count.")
            return self.k[layer][:count, :, :length], self.v[layer][:count, :, :length]
        return self.k[layer][rows, :, :length], self.v[layer][rows, :, :length]

    def copy_rows(self, sources: torch.Tensor, targets: torch.Tensor, length: int) -> None:
        """Copy the first ``length`` positions of ``sources`` rows onto ``targets``."""

        row_bytes = max(self._row_shape[0] * length * self._row_shape[2] * self.dtype.itemsize, 1)
        if self.stacked:
            row_bytes *= self.n_layers
        chunk = max(self.COPY_CHUNK_BYTES // row_bytes, 1)
        for start in range(0, sources.shape[0], chunk):
            src = sources[start : start + chunk]
            dst = targets[start : start + chunk]
            if self.stacked:
                self._base_k[:, dst, :, :length] = self._base_k[:, src, :, :length]
                self._base_v[:, dst, :, :length] = self._base_v[:, src, :, :length]
            else:
                for layer in range(self.n_layers):
                    self.k[layer][dst, :, :length] = self.k[layer][src, :, :length]
                    self.v[layer][dst, :, :length] = self.v[layer][src, :, :length]
