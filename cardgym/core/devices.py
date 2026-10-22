"""Device selection, mixed precision, and accelerator memory hygiene."""

from __future__ import annotations

import ctypes
import sys
from contextlib import contextmanager, nullcontext
from functools import lru_cache
from typing import Iterator

import numpy as np
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


class Fence:
    """A point in the device's work queue that the host can wait for.

    On MPS a blocking copy (``.cpu()``, ``.to(device)``, ``torch.tensor(...,
    device=...)``) waits for *all* queued work, so a host that must not stall
    behind unrelated kernels uploads and downloads with ``non_blocking=True``
    and waits on a fence recorded right after the work it needs.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._event = None
        if device.type == "mps":
            self._event = torch.mps.Event()
            self._event.record()
        elif device.type == "cuda":
            self._event = torch.cuda.Event()
            self._event.record()

    def wait(self) -> None:
        if self._event is not None:
            self._event.synchronize()


def upload_packed(upload, arrays: list[np.ndarray]) -> list[torch.Tensor]:
    """Upload several host arrays with one copy per dtype; returns device views in order.

    Each upload is its own blit (and on MPS its own staging copy); a rollout
    wave uploads a dozen small arrays, so packing them saves the launches.
    The views are contiguous slices of one flat buffer.
    """

    groups: dict[np.dtype, list[int]] = {}
    for index, array in enumerate(arrays):
        groups.setdefault(np.asarray(array).dtype, []).append(index)
    result: list[torch.Tensor | None] = [None] * len(arrays)
    for members in groups.values():
        parts = [np.ascontiguousarray(arrays[i]).reshape(-1) for i in members]
        flat = upload(parts[0] if len(parts) == 1 else np.concatenate(parts))
        offset = 0
        for i, part in zip(members, parts):
            result[i] = flat[offset : offset + part.size].view(np.shape(arrays[i]))
            offset += part.size
    return result


def to_device(array, device: torch.device) -> torch.Tensor:
    """Upload a numpy array without waiting for queued device work.

    The caller keeps ``array`` alive until a later fence has been waited on.
    """

    return torch.from_numpy(array).to(device, non_blocking=True)


class Uploader:
    """Non-blocking uploads of host arrays, kept alive until :meth:`release`.

    A blocking upload on MPS waits for all queued work, so an update that
    uploads its labels and decision rows between kernels drains the queue each
    time and the device idles while Python issues the next kernels. Release
    only after a synchronization point that follows the last upload.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._keep: list[np.ndarray] = []

    def __call__(self, array) -> torch.Tensor:
        array = np.ascontiguousarray(array)
        self._keep.append(array)
        return to_device(array, self.device)

    def release(self) -> None:
        self._keep = []


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
