"""Fused field lookups for MPS: the sum of one row per field, from several tables at once.

A token of F categorical fields looks each field up in a packed table (field
f's values occupy their own rows) and sums the F rows; a model may keep
several such tables (the token embedding, per-layer embeddings, value
embeddings). MPS's ``embedding_bag`` measured 2-6 ms per table at 20k tokens
forward and more in backward, where it scatters every token's gradient into
the table. Here:

* forward: one kernel per table, a thread per (token, 4 columns), summing in
  fp32 (exact, like ``embedding_bag``);
* backward: the tables' gradient is ``onehot^T @ d_out`` with ``onehot``
  [N, rows] holding each token's F ones (fields never share a row). One bf16
  GEMM with fp32 accumulation serves every table, split along N because MPS
  runs long-K GEMMs slowly. The output gradient is rounded to bf16 for it,
  as autocast does for every linear layer's weight gradient.
"""

from __future__ import annotations

import torch

_SOURCE = """
#include <metal_stdlib>
using namespace metal;

// out[n, :] = sum_f table[ids[n, f], :]. Thread (token n, column quad q).
kernel void bag_fwd(
    device const long* ids [[buffer(0)]],
    device const float4* table [[buffer(1)]],
    device float4* out [[buffer(2)]],
    constant int& N [[buffer(3)]],
    constant int& F [[buffer(4)]],
    constant int& Q [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    const int q = gid % Q;
    const long n = gid / Q;
    if (n >= N) return;
    float4 acc = 0.0f;
    for (int f = 0; f < F; ++f) acc += table[ids[n * F + f] * Q + q];
    out[n * Q + q] = acc;
}

// onehot[n, ids[n, f]] = 1 for every field (onehot zeroed beforehand).
kernel void bag_onehot(
    device const long* ids [[buffer(0)]],
    device bfloat* onehot [[buffer(1)]],
    constant int& N [[buffer(2)]],
    constant int& F [[buffer(3)]],
    constant int& rows [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    const long n = gid / F;
    if (n >= N) return;
    onehot[n * rows + ids[gid]] = bfloat(1.0f);
}
"""

_LIBRARY = None
_SPLITS = 8


def _library():
    global _LIBRARY
    if _LIBRARY is None:
        _LIBRARY = torch.mps.compile_shader(_SOURCE)
    return _LIBRARY


def supported(tables) -> bool:
    return all(table.shape[1] % 4 == 0 and table.dtype == torch.float32 for table in tables)


class FieldBags(torch.autograd.Function):
    """ids [N, F] int64 (rows of every table), tables [rows, C_i] fp32 -> sums [N, C_i] fp32."""

    @staticmethod
    def forward(ctx, ids, *tables):
        ids = ids.contiguous()
        count, fields = ids.shape
        lib = _library()
        outputs = []
        for table in tables:
            quads = table.shape[1] // 4
            out = torch.empty(count, table.shape[1], device=ids.device)
            lib.bag_fwd(ids, table.contiguous(), out, count, fields, quads, threads=count * quads, group_size=256)
            outputs.append(out)
        ctx.save_for_backward(ids)
        ctx.shapes = [table.shape for table in tables]
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *grads):
        (ids,) = ctx.saved_tensors
        count, fields = ids.shape
        rows = ctx.shapes[0][0]
        live = [index for index, grad in enumerate(grads) if grad is not None]
        result: list[torch.Tensor | None] = [None] * len(grads)
        if not live:
            return (None, *result)
        onehot = torch.zeros(count, rows, device=ids.device, dtype=torch.bfloat16)
        _library().bag_onehot(ids, onehot, count, fields, rows, threads=count * fields, group_size=256)
        stacked = torch.cat([grads[index].reshape(count, -1).to(torch.bfloat16) for index in live], dim=1)
        main = (count // _SPLITS) * _SPLITS
        table = torch.zeros(rows, stacked.shape[1], device=ids.device)
        if main:
            table += torch.bmm(
                onehot[:main].view(_SPLITS, main // _SPLITS, rows).transpose(1, 2),
                stacked[:main].view(_SPLITS, main // _SPLITS, -1),
            ).sum(0, dtype=torch.float32)
        if main < count:
            table += (onehot[main:].t() @ stacked[main:]).float()
        offset = 0
        for index in live:
            width = ctx.shapes[index][1]
            result[index] = table[:, offset : offset + width].contiguous()
            offset += width
        return (None, *result)
