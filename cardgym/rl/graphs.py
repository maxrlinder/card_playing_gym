"""CUDA graphs of a rollout decode wave: one replay per wave instead of thousands of launches.

A continuous Piquet collection decodes ~190 waves per update, each a ragged
16-layer forward plus the policy heads: about 2,000 small kernels whose
Python dispatch measured ~19 ms per wave against ~8 ms of device time. A
CUDA graph per wave shape (rows, tokens per row, attention span; about 30
occur) replays them with one launch.

Each graph computes, for every row, the hidden state at its readout, every
policy head's logits from it (the heads are fixed-width and cheap, so all
of them run on every row and the host keeps the acting head's), and, with
``values``, the per-seat values: one fp32 [rows, columns] output that goes
to the host in one copy. Only models that declare ``graphable_heads`` (every
head computable from the hidden state and ``pop_context``, no candidate
features) take this path.

What stays valid between replays:

* weights: ``load_state_dict`` copies into the same parameter storage, and
  everything derived from them (autocast's fp16 copies, the card grid's
  Toeplitz matrices) is computed inside the graph, with autocast's cache off
  while capturing;
* the KV cache: its storage is fixed for the collector's layout; the
  collector drops its graphs with it.

The wave's body is compiled with dynamic shapes (cardgym.nn.compile): Inductor
fuses the norms, casts, and residual adds between the GEMMs (4.9 -> ~2.7 ms
a wave of 384 rows by 3 tokens). A graph is captured on the first wave of
its shape: a warm-up run on a side stream with that wave's real inputs (it
compiles the shape if needed and writes the same K/V the replay writes
again), the capture, then the replay. Graphs share one memory pool;
they run one at a time on one stream and each output is copied to the host
before the next replay can overwrite it.
"""

from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch

from cardgym.core.devices import DTYPES
from cardgym.nn.compile import compiled


def graphs_enabled(model, device: torch.device) -> bool:
    """CUDA, a model whose heads need no candidate features, and not disabled by CARDGYM_CUDA_GRAPHS=0."""

    return (
        device.type == "cuda"
        and getattr(model, "graphable_heads", False)
        and os.environ.get("CARDGYM_CUDA_GRAPHS", "1") != "0"
    )


@dataclass
class _Graph:
    graph: torch.cuda.CUDAGraph
    inputs: torch.Tensor  # int64 [rows * run * width + 3 * rows]: tokens, positions, offsets, cache rows
    output: torch.Tensor  # fp32 [rows, columns]


class DecodeGraphs:
    def __init__(self, model, cache, *, precision: str, values: bool) -> None:
        self.model = model
        self.cache = cache
        self.precision = precision
        self.values = values
        self.heads: tuple[str, ...] = tuple(model.policy_heads)
        self.pool = torch.cuda.graph_pool_handle()
        # Compiled on CUDA (cardgym.nn.compile): its warm-up run compiles any new shape before the capture.
        self._forward = compiled(self._outputs, cache.device)
        self._graphs: dict[tuple[int, int, int, int], _Graph] = {}
        # Output columns per head (and of the values), known after the first capture.
        self.columns: dict[str, slice] = {}

    def __len__(self) -> int:
        return len(self._graphs)

    def _autocast(self):
        if self.precision == "fp32":
            return nullcontext()
        return torch.autocast("cuda", dtype=DTYPES[self.precision], cache_enabled=False)

    def _outputs(self, inputs: torch.Tensor, rows: int, run: int, width: int, span: int) -> list[torch.Tensor]:
        """Every head's logits (then the values) at each row's readout, fp32."""

        cells = rows * run * width
        tokens = inputs[:cells].view(rows, run, width)
        positions, offsets, cache_rows = inputs[cells:].view(3, rows)
        with self._autocast():
            hidden = self.model.forward_ragged(tokens, positions, self.cache, offsets, span=span, rows=cache_rows)
            context = self.model.pop_context()
            if context is not None:
                context = context[torch.arange(rows, device=inputs.device), offsets]
            outputs = [self.model.policy_logits(hidden, head, None, None, context) for head in self.heads]
            if self.values:
                outputs.append(self.model.player_values(hidden))
        return [output.float() for output in outputs]

    def _body(self, inputs: torch.Tensor, rows: int, run: int, width: int, span: int) -> torch.Tensor:
        outputs = self._forward(inputs, rows, run, width, span)
        if not self.columns:
            names = [*self.heads, "values"] if self.values else list(self.heads)
            start = 0
            for name, output in zip(names, outputs):
                self.columns[name] = slice(start, start + output.shape[-1])
                start += output.shape[-1]
        return torch.cat(outputs, dim=-1)

    def run(
        self,
        tokens: np.ndarray,
        positions: np.ndarray,
        offsets: np.ndarray,
        cache_rows: np.ndarray,
        span: int,
    ) -> torch.Tensor:
        """Decode one wave; returns the device output [rows, columns] (copy it before the next run)."""

        rows, run, width = tokens.shape
        span = min(span, self.cache.max_len)
        key = (rows, run, width, span)
        host = torch.from_numpy(
            np.concatenate([tokens.reshape(-1), positions, offsets, cache_rows]).astype(np.int64, copy=False)
        ).pin_memory()
        entry = self._graphs.get(key)
        if entry is None:
            return self._capture(key, host)
        entry.inputs.copy_(host, non_blocking=True)
        entry.graph.replay()
        return entry.output

    def _capture(self, key: tuple[int, int, int, int], host: torch.Tensor) -> torch.Tensor:
        rows, run, width, span = key
        inputs = torch.empty(host.shape, dtype=host.dtype, device=self.cache.device)
        inputs.copy_(host, non_blocking=True)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            self._body(inputs, rows, run, width, span)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool):
            output = self._body(inputs, rows, run, width, span)
        self._graphs[key] = _Graph(graph, inputs, output)
        graph.replay()
        return output


@dataclass
class GraphedReadout:
    """A replayed wave's output on its way to the host, and whose decision each row is."""

    states: list  # the _Live states with a pending decision
    rows: list[int]  # each state's row in the output
    output: torch.Tensor  # host fp32 [rows, columns], valid after the wave's fence
    columns: dict[str, slice]

    def deliver(self) -> None:
        table = self.output.numpy()
        values = self.columns.get("values")
        for state, row in zip(self.states, self.rows):
            state.logits = table[row, self.columns[state.pending.head]]
            state.value = None if values is None else table[row, values].copy()
