"""One kernel for everything after attention in a cached decode step (MPS).

A rollout wave decodes a few hundred rows; per layer the unfused path runs
the output projection, add + LayerNorm, up projection, GELU, down projection
and add + LayerNorm as six kernels, each too small to fill the GPU. Here one
threadgroup takes 16 rows through all of it, with the matrix products on the
GPU's matrix units (Metal 4 tensor ops):

    x1 = x + o @ Wout^T                 (fp32, kept in device memory)
    m  = LN2(x1)                        (bf16, threadgroup memory)
    for each 256-wide chunk of the hidden layer:
        g     = GELU(m @ W1[chunk]^T + b1[chunk])     (bf16, threadgroup)
        down += g @ W2[:, chunk]^T                    (fp32, registers)
    x2 = x1 + down + b2 (+ the layer's per-layer embedding)
    a  = LN(x2) with the next layer's (or the final) norm

Measured at 336 rows of one token (12 layers, d_model 256): the decode takes
2.1 ms against 2.5 ms unfused. Larger tiles lose: each threadgroup streams
the layer's weights, and fewer threadgroups leave cores idle.

Same function as the unfused path; GEMM outputs stay fp32 until the next
GEMM operand is rounded to bf16 (the unfused path rounds them to bf16 as
autocast does), so the two agree to bf16 rounding.
"""

from __future__ import annotations

import torch

_ROWS = 16
_CHUNK = 256
_THREADS = 128

_SOURCE = r"""
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

constant constexpr int DM = __DM__;
constant constexpr int FF = __FF__;
constant constexpr int PLE = __PLE__;
constant constexpr int TM = __ROWS__;
constant constexpr int CH = __CHUNK__;  // hidden-layer chunk
constant constexpr int PER_LANE = DM / 32;

typedef dextents<int32_t, 2> ext;
typedef tensor<device bfloat, ext, tensor_inline> dev_bf;
typedef tensor<threadgroup bfloat, ext, tensor_inline> tg_bf;

inline float erf_approx(float x) {
    const float t = 1.0f / (1.0f + 0.3275911f * fabs(x));
    const float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t
                     - 0.284496736f) * t + 0.254829592f) * t * exp(-x * x);
    return copysign(y, x);
}

// LayerNorm of row values v (PER_LANE per lane, column lane + 32 i).
inline void layer_norm(thread float* v, device const float* w, device const float* b, float eps, uint lane) {
    float sum = 0.0f;
    for (int i = 0; i < PER_LANE; ++i) sum += v[i];
    const float mean = simd_sum(sum) / DM;
    float sq = 0.0f;
    for (int i = 0; i < PER_LANE; ++i) { v[i] -= mean; sq += v[i] * v[i]; }
    const float rstd = rsqrt(simd_sum(sq) / DM + eps);
    for (int i = 0; i < PER_LANE; ++i) { const int e = lane + 32 * i; v[i] = v[i] * rstd * w[e] + b[e]; }
}

// o [M, DM] bf16; x [M, DM] fp32 residual; weights in nn.Linear layout ([out, in]) bf16, biases and
// norms fp32. Writes x1 (scratch) and x_out [M, DM] fp32, and the next norm's output: a_bf bf16, or
// a_f fp32 when flags & 2 (the final norm). flags & 1 adds ple_scale * ple[row, ple_off + e % PLE].
kernel void decode_post(
    device const bfloat* o [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device const bfloat* wout [[buffer(2)]],
    device const float* ln2_w [[buffer(3)]],
    device const float* ln2_b [[buffer(4)]],
    device const bfloat* w1 [[buffer(5)]],
    device const float* b1 [[buffer(6)]],
    device const bfloat* w2 [[buffer(7)]],
    device const float* b2 [[buffer(8)]],
    device const float* ple [[buffer(9)]],
    device const float* next_w [[buffer(10)]],
    device const float* next_b [[buffer(11)]],
    device float* x1 [[buffer(12)]],
    device float* x_out [[buffer(13)]],
    device bfloat* a_bf [[buffer(14)]],
    device float* a_f [[buffer(15)]],
    constant int& M [[buffer(16)]],
    constant int& ple_stride [[buffer(17)]],
    constant int& ple_off [[buffer(18)]],
    constant float& ple_scale [[buffer(19)]],
    constant int& flags [[buffer(20)]],
    constant float& eps [[buffer(21)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup bfloat m[TM * DM];
    threadgroup bfloat g[TM * CH];
    const int row0 = tg * TM;
    const int rows = min(TM, M - row0);
    const long base = (long)row0 * DM;

    constexpr auto out_desc = matmul2d_descriptor(TM, DM, DM, false, true, false);
    constexpr auto up_desc = matmul2d_descriptor(TM, CH, DM, false, true, false);
    constexpr auto down_desc = matmul2d_descriptor(TM, DM, CH, false, true, false,
                                                   matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<out_desc, execution_simdgroups<4>> out_op;
    matmul2d<up_desc, execution_simdgroups<4>> up_op;
    matmul2d<down_desc, execution_simdgroups<4>> down_op;

    {
        auto a = dev_bf((device bfloat*)o + base, ext(DM, rows));
        auto w = dev_bf((device bfloat*)wout, ext(DM, DM));
        auto d = out_op.get_destination_cooperative_tensor<decltype(a), decltype(w), float>();
        for (uint16_t i = 0; i < d.get_capacity(); ++i) if (d.is_valid_element(i)) d[i] = 0.0f;
        out_op.run(a, w, d);
        for (uint16_t i = 0; i < d.get_capacity(); ++i) {
            if (!d.is_valid_element(i)) continue;
            const auto at = d.get_multidimensional_index(i);
            if (at[1] < rows) {
                const long e = base + at[1] * DM + at[0];
                x1[e] = x[e] + d[i];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
    for (int r = sg; r < rows; r += 4) {
        float v[PER_LANE];
        for (int i = 0; i < PER_LANE; ++i) v[i] = x1[base + r * DM + lane + 32 * i];
        layer_norm(v, ln2_w, ln2_b, eps, lane);
        for (int i = 0; i < PER_LANE; ++i) m[r * DM + lane + 32 * i] = bfloat(v[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto ma = tg_bf(m, ext(DM, rows));
    auto ga = tg_bf(g, ext(CH, rows));
    auto w2a = dev_bf((device bfloat*)w2, ext(CH, DM), array<int32_t, 2>({1, FF}));
    auto down = down_op.get_destination_cooperative_tensor<decltype(ga), decltype(w2a), float>();
    for (uint16_t i = 0; i < down.get_capacity(); ++i) if (down.is_valid_element(i)) down[i] = 0.0f;
    for (int c = 0; c < FF / CH; ++c) {
        auto w = dev_bf((device bfloat*)w1 + (long)c * CH * DM, ext(DM, CH));
        auto u = up_op.get_destination_cooperative_tensor<decltype(ma), decltype(w), float>();
        for (uint16_t i = 0; i < u.get_capacity(); ++i) if (u.is_valid_element(i)) u[i] = 0.0f;
        up_op.run(ma, w, u);
        for (uint16_t i = 0; i < u.get_capacity(); ++i) {
            if (!u.is_valid_element(i)) continue;
            const auto at = u.get_multidimensional_index(i);
            const float h = u[i] + b1[c * CH + at[0]];
            g[at[1] * CH + at[0]] = bfloat(0.5f * h * (1.0f + erf_approx(h * M_SQRT1_2_F)));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto w2c = dev_bf((device bfloat*)w2 + c * CH, ext(CH, DM), array<int32_t, 2>({1, FF}));
        down_op.run(ga, w2c, down);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint16_t i = 0; i < down.get_capacity(); ++i) {
        if (!down.is_valid_element(i)) continue;
        const auto at = down.get_multidimensional_index(i);
        if (at[1] >= rows) continue;
        const int col = at[0];
        const long e = base + at[1] * DM + col;
        float v = x1[e] + down[i] + b2[col];
        if (flags & 1) v += ple_scale * ple[(long)(row0 + at[1]) * ple_stride + ple_off + (col % PLE)];
        x_out[e] = v;
    }
    threadgroup_barrier(mem_flags::mem_device);
    for (int r = sg; r < rows; r += 4) {
        float v[PER_LANE];
        for (int i = 0; i < PER_LANE; ++i) v[i] = x_out[base + r * DM + lane + 32 * i];
        layer_norm(v, next_w, next_b, eps, lane);
        for (int i = 0; i < PER_LANE; ++i) {
            const long e = base + r * DM + lane + 32 * i;
            if (flags & 2) a_f[e] = v[i]; else a_bf[e] = bfloat(v[i]);
        }
    }
}
"""

_LIBRARIES: dict[tuple[int, int, int], object | None] = {}


def library(d_model: int, d_ff: int, ple: int):
    """The compiled kernel for these widths, or None where it cannot run (no Metal 4 tensor ops)."""

    key = (d_model, d_ff, ple)
    if key not in _LIBRARIES:
        library = None
        if d_model % 64 == 0 and d_model <= 256 and d_ff % _CHUNK == 0:
            source = (
                _SOURCE.replace("__DM__", str(d_model)).replace("__FF__", str(d_ff))
                .replace("__PLE__", str(max(ple, 1))).replace("__ROWS__", str(_ROWS)).replace("__CHUNK__", str(_CHUNK))
            )
            try:
                library = torch.mps.compile_shader(source)
            except Exception:  # noqa: BLE001 - older OS / GPU without tensor ops
                library = None
        _LIBRARIES[key] = library
    return _LIBRARIES[key]


def decode_post(
    lib, o, x, wout, ln2_w, ln2_b, w1, b1, w2, b2, ple, ple_stride, ple_off, ple_scale, next_w, next_b,
    *, last: bool, eps: float,
):
    """Returns (x_out fp32, a): a is the next norm's output, bf16 (fp32 when ``last``)."""

    rows, d_model = x.shape
    x1 = torch.empty_like(x)
    x_out = torch.empty_like(x)
    a = torch.empty(rows, d_model, device=x.device, dtype=torch.float32 if last else torch.bfloat16)
    flags = (1 if ple is not None else 0) | (2 if last else 0)
    lib.decode_post(
        o, x, wout, ln2_w, ln2_b, w1, b1, w2, b2, x if ple is None else ple, next_w, next_b,
        x1, x_out, a, a, rows, ple_stride, ple_off, ple_scale, flags, eps,
        threads=-(-rows // _ROWS) * _THREADS, group_size=_THREADS,
    )
    return x_out, a
