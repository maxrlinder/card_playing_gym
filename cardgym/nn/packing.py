"""Packed rows: the real tokens of padded sequences, concatenated.

Training batches pad each stream to its length bucket. Under the fused Metal
trunk, whose attention takes per-row sequence bounds, a microbatch runs on
its real tokens only: [N, W] packed rows instead of [B, L, W], and every
per-position output (hidden states, the embedding's per-card features) is
[N, ...]. Callers address a (sequence, position) pair through :meth:`flat`;
the padded layout's counterpart is ``sequence * L + position``.
"""

from __future__ import annotations

import numpy as np
import torch


class Packing:
    """Sequences of the given lengths, back to back.

    ``upload`` moves host arrays to the device (default: a blocking copy to
    ``device``).
    """

    def __init__(self, lengths, device: torch.device | None = None, upload=None) -> None:
        if upload is None:
            upload = lambda array: torch.from_numpy(array).to(device)  # noqa: E731
        self.lengths = np.asarray(lengths, dtype=np.int64)
        if self.lengths.ndim != 1 or (self.lengths < 1).any():
            raise ValueError("Packed sequences need positive lengths.")
        self.starts = np.zeros(self.lengths.shape[0], dtype=np.int64)
        np.cumsum(self.lengths[:-1], out=self.starts[1:])
        self.rows = int(self.lengths.sum())
        self.longest = int(self.lengths.max())
        first = np.repeat(self.starts, self.lengths)
        # Each row's sequence as [start, end) rows, for the attention kernels.
        self.bounds = upload(
            np.stack([first, first + np.repeat(self.lengths, self.lengths)], axis=1).astype(np.int32)
        )

    def pack(self, padded: np.ndarray) -> np.ndarray:
        """Host [B, L, ...] -> [N, ...]: each sequence's first ``length`` positions, in order."""

        real = np.arange(padded.shape[1])[None, :] < self.lengths[:, None]
        return padded[real]

    def flat(self, sequence: np.ndarray, position: np.ndarray) -> np.ndarray:
        return self.starts[sequence] + position

    def per_row(self, values: np.ndarray) -> np.ndarray:
        """A per-sequence array repeated for each of its rows."""

        return np.repeat(values, self.lengths, axis=0)
