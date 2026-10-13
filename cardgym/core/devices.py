"""Device selection, mixed precision, and accelerator memory hygiene."""

from __future__ import annotations

import ctypes
import sys
from contextlib import contextmanager, nullcontext
from functools import lru_cache
from typing import Iterator

import torch

PRECISIONS = ("fp32", "fp16", "bf16")
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def best_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_device(value: str | None) -> torch.device:
    if value in (None, "", "auto"):
        return best_device()
    return torch.device(value)


def autocast_context(device: torch.device, precision: str):
    """Autocast eligible ops to ``precision``; master weights stay fp32.

    Probabilities, ratios, entropy, KL, returns, and advantages are cast to
    fp32 explicitly by their callers. KV storage has its own dtype setting.
    """

    if precision == "fp32":
        return nullcontext()
    if precision not in PRECISIONS:
        raise ValueError(f"Unknown precision {precision!r}.")
    return torch.autocast(device_type=device.type, dtype=DTYPES[precision])


def release_device_memory(device: torch.device) -> None:
    """Return cached allocator blocks; MPS otherwise keeps its high-water mark."""

    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def device_allocated_bytes(device: torch.device) -> int:
    if device.type == "mps":
        return int(torch.mps.driver_allocated_memory())
    if device.type == "cuda":
        return int(torch.cuda.memory_allocated(device))
    return 0


@lru_cache(maxsize=1)
def _libobjc() -> ctypes.CDLL | None:
    if sys.platform != "darwin":
        return None
    try:
        library = ctypes.CDLL("/usr/lib/libobjc.dylib")
    except OSError:
        return None
    library.objc_autoreleasePoolPush.argtypes = []
    library.objc_autoreleasePoolPush.restype = ctypes.c_void_p
    library.objc_autoreleasePoolPop.argtypes = [ctypes.c_void_p]
    library.objc_autoreleasePoolPop.restype = None
    return library


@contextmanager
def autorelease_pool() -> Iterator[None]:
    """Drain Objective-C objects autoreleased inside the block (macOS only).

    PyTorch's MPS backend returns autoreleased Metal objects from some
    per-operation paths. A Cocoa app drains them every run-loop turn; a plain
    Python training loop has no run loop, so they accumulate as native heap
    (measured at ~3.9 MB per iteration on a 16-layer model). Wrapping each
    training iteration in a pool removes most of it; recycling the process
    (``--recycle-after``) bounds the rest. A no-op elsewhere.
    """

    library = _libobjc()
    if library is None:
        yield
        return
    token = library.objc_autoreleasePoolPush()
    try:
        yield
    finally:
        library.objc_autoreleasePoolPop(token)
