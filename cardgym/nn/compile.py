"""torch.compile on CUDA: Inductor-fused kernels for the trunk, card grid, and field lookups.

The model is small (d 256) and its streams short, so an eager forward and
backward is mostly memory-bound elementwise work between the GEMMs: norms,
casts, GELU, residual adds, the card grid's 32-wide rows. Inductor fuses
those into Triton kernels and leaves the GEMMs to cuBLAS. On an RTX 2070
Super a 15k-position forward and backward of the Piquet actor measured
131 -> 88 ms.

Shapes are dynamic (microbatches differ in rows, decode waves in rows and
tokens), so a function compiles once per process; Inductor's on-disk cache
makes later processes faster. ``CARDGYM_COMPILE=0`` turns it off.
"""

from __future__ import annotations

import os

import torch


def compile_enabled(device: torch.device) -> bool:
    return device.type == "cuda" and os.environ.get("CARDGYM_COMPILE", "1") != "0"


def compiled(fn, device: torch.device):
    """``fn`` compiled with dynamic shapes on CUDA (unless disabled); ``fn`` itself elsewhere."""

    if not compile_enabled(device):
        return fn
    import torch._dynamo.config
    import torch._functorch.config
    import torch._inductor.config

    # The auxiliary gradient budget differentiates one graph twice
    # (retain_graph), which donated backward buffers forbid.
    torch._functorch.config.donated_buffer = False
    # Compile in this process: a parallel compile pool stays alive with the
    # process (~770 MB RSS each, in the trainer and every rollout worker). It
    # costs a slower start (the first Piquet iteration: ~210 s against ~60 s),
    # paid once per process.
    torch._inductor.config.compile_threads = 1
    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
    return torch.compile(fn, dynamic=True)
