"""Fused Metal kernels for a suit-by-rank card grid (MPS, bf16 autocast).

Shared by Rummy (4 x 13, ranks wrap: runs are cyclic) and Piquet (4 x 8,
zero-padded: sequences stop at the Ace); ``ranks`` and ``circular`` select
the variant, which compiles once per shape. A grid module exposes ``width``,
``hidden``, ``eps``, ``ranks``, ``circular``, ``code``, ``rank``, ``norm``,
and ``blocks`` (each with ``kernel``, ``norm``, ``along_rank``,
``across_suits``, ``out``).

Each position's grid (4 x ranks cards x 32 features) lives in threadgroup memory
while a kernel works on it, and every matrix product runs on the GPU's
matrix units through Metal 4 tensor ops (MetalPerformancePrimitives
``matmul2d``):

    forward    one kernel for the whole grid: embed, then per block norm,
               suit pooling, across projection, circular rank convolution,
               GELU, output projection, residual; final norm
    backward   final-norm backward, then one kernel per block (in reverse)
               that recomputes the block's activations from its saved input
               and produces the input gradient and its weight gradients

The convolution is a GEMM over a sliding window: each suit's ranks sit in
threadgroup memory with ``kernel - 1`` extra rows, wrapped (row i holds rank
(i - kernel // 2) mod ranks) or zero, and a [16 x kernel * 32] tensor view with row
stride 32 over them is the im2col matrix without materializing it. Its data
gradient is the same window over the padded activation gradient against the
tap-flipped weights; its weight gradient is the window transposed against
that gradient. Weight gradients accumulate per threadgroup in registers
(cooperative tensors) across the positions it handles and are summed over
threadgroups at the end.

Precision follows autocast: GEMM operands are bf16 (the convolution input,
the GELU output, the projection weights), accumulation and everything else
fp32 (the residual stream, norm statistics, pooling, the across projection).
The forward saves only each block's fp32 input; the backward recomputes the
rest, which is cheaper than storing and reloading it.

Layouts: card c = suit * ranks + rank; per-position tensors are [N, 4 * ranks, W].
Ranks are padded to 16 rows for the tensor-op tiles, so 8 ranks waste half of
each tile; still a fraction of the unfused graph.
"""

from __future__ import annotations

import torch

_THREADS = 128  # four simdgroups: one per suit where the work splits by suit
_FORWARD_GROUPS = 4096
_BACKWARD_GROUPS = 512


def _source(width: int, hidden: int, kernel: int, ranks: int, circular: bool) -> str:
    source = r"""
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

constant constexpr int W = __W__;
constant constexpr int H = __H__;
constant constexpr int KT = __K__;
constant constexpr int HALF = KT / 2;
constant constexpr int S = 4;
constant constexpr int R = __R__;
constant constexpr int C = S * R;
constant constexpr bool CIRCULAR = __CIRCULAR__;
constant constexpr int PR = 16 + KT - 1;  // padded rows per suit
constant constexpr int TG = __THREADS__;
constant constexpr int DYN = static_cast<int>(dynamic_extent);

typedef tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline> tg_bf;
typedef tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tg_f;
typedef tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> dev_bf;
typedef tensor<device float, dextents<int32_t, 2>, tensor_inline> dev_f;
typedef dextents<int32_t, 2> ext;

inline float erf_approx(float x) {
    const float t = 1.0f / (1.0f + 0.3275911f * fabs(x));
    const float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t
                     - 0.284496736f) * t + 0.254829592f) * t * exp(-x * x);
    return copysign(y, x);
}
inline float gelu(float x) { return 0.5f * x * (1.0f + erf_approx(x * M_SQRT1_2_F)); }
inline float gelu_grad(float x) {
    return 0.5f * (1.0f + erf_approx(x * M_SQRT1_2_F)) + x * exp(-0.5f * x * x) * 0.3989422804014327f;
}

// Write rank r of suit s (one feature) into a padded suit block: row i holds
// rank (i - HALF) mod R when circular, else rank i - HALF (other rows stay zero).
inline void put_wrapped(threadgroup bfloat* pad, int stride, int s, int r, int f, bfloat v) {
    threadgroup bfloat* block = pad + s * PR * stride;
    if (!CIRCULAR) {
        block[(r + HALF) * stride + f] = v;
        return;
    }
    for (int i = r + HALF - R; i < PR; i += R) {
        if (i >= 0) block[i * stride + f] = v;
    }
}

// Zero padding rows are written once per kernel, before any position.
inline void zero_pad(threadgroup bfloat* pad, int count, uint tid) {
    if (CIRCULAR) return;
    for (int i = tid; i < count; i += TG) pad[i] = bfloat(0.0f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// The grid forward. codes [N, C] int64; gains [B + 1, W] (blocks, then the
// final norm); wc [B, KT * W, H] bf16 (tap-major conv weights); bc [B, H];
// wat [B, W, H] (across weight, transposed); wot [B, H, W] bf16 (output
// weight, transposed); bo [B, W]. Writes cards [N, 52, W] bf16 and, with
// save, each block's input and the final residual to saved [B + 1, N, 52, W].
kernel void grid_fwd(
    device const long* codes [[buffer(0)]],
    device const float* code_w [[buffer(1)]],
    device const float* rank_w [[buffer(2)]],
    device const float* gains [[buffer(3)]],
    device const bfloat* wc [[buffer(4)]],
    device const float* bc [[buffer(5)]],
    device const float* wat [[buffer(6)]],
    device const bfloat* wot [[buffer(7)]],
    device const float* bo [[buffer(8)]],
    device bfloat* cards [[buffer(9)]],
    device float* saved [[buffer(10)]],
    constant int& N [[buffer(11)]],
    constant int& B [[buffer(12)]],
    constant int& save [[buffer(13)]],
    constant float& eps [[buffer(14)]],
    uint tg [[threadgroup_position_in_grid]],
    uint ntg [[threadgroups_per_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float xs[C * W];
    threadgroup bfloat ypad[S * PR * W];
    threadgroup float pooled[16 * W];
    threadgroup float across[16 * H];
    threadgroup bfloat act[S * 16 * H];

    constexpr auto conv_desc = matmul2d_descriptor(16, H, KT * W);
    constexpr auto across_desc = matmul2d_descriptor(16, H / 4, W);
    constexpr auto out_desc = matmul2d_descriptor(16, W, H);
    matmul2d<conv_desc, execution_simdgroup> conv_op;
    matmul2d<across_desc, execution_simdgroup> across_op;
    matmul2d<out_desc, execution_simdgroup> out_op;
    const long plane = (long)N * C * W;
    const int s = sg;  // the suit this simdgroup convolves and projects
    zero_pad(ypad, S * PR * W, tid);

    for (int n = tg; n < N; n += ntg) {
        const long base = (long)n * C * W;
        for (int i = tid; i < C * W; i += TG) {
            const int c = i / W, f = i % W;
            xs[i] = code_w[codes[(long)n * C + c] * W + f] + rank_w[(c % R) * W + f];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int b = 0; b < B; ++b) {
            if (save) {
                for (int i = tid; i < C * W; i += TG) saved[b * plane + base + i] = xs[i];
            }
            for (int c = sg; c < C; c += 4) {
                const float v = xs[c * W + lane];
                const float rs = rsqrt(simd_sum(v * v) / W + eps);
                put_wrapped(ypad, W, c / R, c % R, lane, bfloat(v * rs * gains[b * W + lane]));
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int i = tid; i < 16 * W; i += TG) {
                const int r = i / W, f = i % W;
                float sum = 0.0f;
                if (r < R) {
                    for (int t = 0; t < S; ++t) sum += float(ypad[(t * PR + r + HALF) * W + f]);
                }
                pooled[i] = 0.25f * sum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            {
                const int q = H / 4;
                auto a = tg_f(pooled, ext(W, 16));
                auto w = dev_f((device float*)wat + b * W * H + sg * q, ext(q, W), array<int32_t, 2>({1, H}));
                auto o = tg_f(across + sg * q, ext(q, 16), array<int32_t, 2>({1, H}));
                across_op.run(a, w, o);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            {
                auto a = tg_bf(ypad + s * PR * W, ext(KT * W, 16), array<int32_t, 2>({1, W}));
                auto w = dev_bf((device bfloat*)wc + (long)b * KT * W * H, ext(H, KT * W));
                auto along = conv_op.get_destination_cooperative_tensor<decltype(a), decltype(w), float>();
                for (uint16_t i = 0; i < along.get_capacity(); ++i) if (along.is_valid_element(i)) along[i] = 0.0f;
                conv_op.run(a, w, along);
                for (uint16_t i = 0; i < along.get_capacity(); ++i) {
                    if (!along.is_valid_element(i)) continue;
                    const auto at = along.get_multidimensional_index(i);
                    const int o = at[0], r = at[1];
                    const float pre = along[i] + bc[b * H + o] + across[r * H + o];
                    act[(s * 16 + r) * H + o] = bfloat(r < R ? gelu(pre) : 0.0f);
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            {
                auto a = tg_bf(act + s * 16 * H, ext(H, 16));
                auto w = dev_bf((device bfloat*)wot + (long)b * H * W, ext(W, H));
                auto out = out_op.get_destination_cooperative_tensor<decltype(a), decltype(w), float>();
                for (uint16_t i = 0; i < out.get_capacity(); ++i) if (out.is_valid_element(i)) out[i] = 0.0f;
                out_op.run(a, w, out);
                for (uint16_t i = 0; i < out.get_capacity(); ++i) {
                    if (!out.is_valid_element(i)) continue;
                    const auto at = out.get_multidimensional_index(i);
                    const int f = at[0], r = at[1];
                    if (r < R) xs[(s * R + r) * W + f] += out[i] + bo[b * W + f];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (save) {
            for (int i = tid; i < C * W; i += TG) saved[B * plane + base + i] = xs[i];
        }
        for (int c = sg; c < C; c += 4) {
            const float v = xs[c * W + lane];
            const float rs = rsqrt(simd_sum(v * v) / W + eps);
            cards[base + c * W + lane] = bfloat(v * rs * gains[B * W + lane]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Final norm backward: dx = d(norm(x) * gain) / dx for cards [N * 52, W];
// one simdgroup per card. d_cards is bf16 (flags & 1) or fp32. Per
// threadgroup gain-gradient partials [ntg, W].
kernel void grid_final_bwd(
    device const bfloat* dy_bf [[buffer(0)]],
    device const float* dy_f [[buffer(1)]],
    device const float* x [[buffer(2)]],
    device const float* gain [[buffer(3)]],
    device float* dx [[buffer(4)]],
    device float* partial [[buffer(5)]],
    constant int& cards [[buffer(6)]],
    constant int& flags [[buffer(7)]],
    constant float& eps [[buffer(8)]],
    uint tg [[threadgroup_position_in_grid]],
    uint ntg [[threadgroups_per_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint nsg [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float acc[32 * W];
    float dg = 0.0f;
    const float g = gain[lane];
    for (int card = tg * nsg + sg; card < cards; card += ntg * nsg) {
        const long at = (long)card * W + lane;
        const float d = (flags & 1) ? float(dy_bf[at]) : dy_f[at];
        const float v = x[at];
        const float rs = rsqrt(simd_sum(v * v) / W + eps);
        const float xh = v * rs;
        dg += d * xh;
        const float gd = d * g;
        dx[at] = rs * (gd - xh * simd_sum(gd * xh) / W);
    }
    acc[sg * W + lane] = dg;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float sum = 0.0f;
        for (uint k = 0; k < nsg; ++k) sum += acc[k * W + lane];
        partial[(long)tg * W + lane] = sum;
    }
}

// One block's backward. x: the block's saved input [N, 52, W]; dxn: the
// gradient of its output [N, 52, W]; writes dxp, the gradient of its input
// (residual included). Recomputes the norm, pooling, across projection and
// convolution from x. Weights: gain [W]; wc [KT * W, H] bf16; wcf [KT * H, W]
// bf16 (wcf[j * H + o, w] = conv[o, w, KT - 1 - j]); bc [H]; wat [W, H]; wa
// [H, W]; wo [W, H] bf16. Per threadgroup partial gradients: conv weight
// [KT * W, H] (tap-major), output weight [W, H], across weight [H, W], and
// the vectors (conv bias [H], output bias [W], gain [W]) in pvec [ntg, H + 2W].
kernel void grid_block_bwd(
    device const float* x [[buffer(0)]],
    device const float* dxn [[buffer(1)]],
    device float* dxp [[buffer(2)]],
    device const float* gain [[buffer(3)]],
    device const bfloat* wc [[buffer(4)]],
    device const bfloat* wcf [[buffer(5)]],
    device const float* bc [[buffer(6)]],
    device const float* wat [[buffer(7)]],
    device const float* wa [[buffer(8)]],
    device const bfloat* wo [[buffer(9)]],
    device float* pconv [[buffer(10)]],
    device float* pout [[buffer(11)]],
    device float* pacross [[buffer(12)]],
    device float* pvec [[buffer(13)]],
    constant int& N [[buffer(14)]],
    constant float& eps [[buffer(15)]],
    uint tg [[threadgroup_position_in_grid]],
    uint ntg [[threadgroups_per_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup bfloat ypad[S * PR * W];
    threadgroup bfloat dpad[S * PR * H];
    threadgroup float pooled[16 * W];      // pooled, later its gradient
    threadgroup float across[16 * H];      // across, later its gradient
    threadgroup float scratch[S * 16 * H / 2];  // pre-activations, then GELU, then dy [52, W]
    threadgroup bfloat* act = (threadgroup bfloat*)scratch;
    threadgroup float* dy = scratch;

    constexpr auto conv_desc = matmul2d_descriptor(16, H, KT * W);
    constexpr auto across_desc = matmul2d_descriptor(16, H / 4, W);
    constexpr auto dact_desc = matmul2d_descriptor(16, H, W);
    constexpr auto dout_desc = matmul2d_descriptor(W, H / 4, DYN, true, false, false,
                                                   matmul2d_descriptor::mode::multiply_accumulate);
    constexpr auto dconv_desc = matmul2d_descriptor(KT * W, H / 4, DYN, true, false, false,
                                                    matmul2d_descriptor::mode::multiply_accumulate);
    constexpr auto dacross_desc = matmul2d_descriptor(H / 4, W, 16, true, false, false,
                                                      matmul2d_descriptor::mode::multiply_accumulate);
    constexpr auto dpool_desc = matmul2d_descriptor(16, W / 4, H);
    constexpr auto dy_desc = matmul2d_descriptor(16, W, KT * H);
    matmul2d<conv_desc, execution_simdgroup> conv_op;
    matmul2d<across_desc, execution_simdgroup> across_op;
    matmul2d<dact_desc, execution_simdgroup> dact_op;
    matmul2d<dout_desc, execution_simdgroup> dout_op;
    matmul2d<dconv_desc, execution_simdgroup> dconv_op;
    matmul2d<dacross_desc, execution_simdgroup> dacross_op;
    matmul2d<dpool_desc, execution_simdgroup> dpool_op;
    matmul2d<dy_desc, execution_simdgroup> dy_op;

    const int s = sg;
    const int q = H / 4;  // this simdgroup's columns of the H-wide weight gradients
    // Persistent weight-gradient accumulators (this simdgroup's column slices).
    auto dout_a = dev_f((device float*)dxn, ext(W, R));
    auto dout_b = tg_bf(act + q * sg, ext(q, R), array<int32_t, 2>({1, H}));
    auto g_out = dout_op.get_destination_cooperative_tensor<decltype(dout_a), decltype(dout_b), float>();
    auto dconv_a = tg_bf(ypad, ext(KT * W, R), array<int32_t, 2>({1, W}));
    auto dconv_b = tg_bf(dpad + HALF * H + q * sg, ext(q, R), array<int32_t, 2>({1, H}));
    auto g_conv = dconv_op.get_destination_cooperative_tensor<decltype(dconv_a), decltype(dconv_b), float>();
    auto dacross_a = tg_f(across + q * sg, ext(q, 16), array<int32_t, 2>({1, H}));
    auto dacross_b = tg_f(pooled, ext(W, 16));
    auto g_across = dacross_op.get_destination_cooperative_tensor<decltype(dacross_a), decltype(dacross_b), float>();
    for (uint16_t i = 0; i < g_out.get_capacity(); ++i) if (g_out.is_valid_element(i)) g_out[i] = 0.0f;
    for (uint16_t i = 0; i < g_conv.get_capacity(); ++i) if (g_conv.is_valid_element(i)) g_conv[i] = 0.0f;
    for (uint16_t i = 0; i < g_across.get_capacity(); ++i) if (g_across.is_valid_element(i)) g_across[i] = 0.0f;
    float g_bias = 0.0f;  // conv bias, column tid % H
    float g_obias = 0.0f;  // output bias, feature lane
    float g_gain = 0.0f;  // norm gain, feature lane
    const float g = gain[lane];
    zero_pad(ypad, S * PR * W, tid);
    zero_pad(dpad, S * PR * H, tid);

    for (int n = tg; n < N; n += ntg) {
        const long base = (long)n * C * W;
        // Recompute the block's forward up to its pre-activations.
        for (int c = sg; c < C; c += 4) {
            const float v = x[base + c * W + lane];
            const float rs = rsqrt(simd_sum(v * v) / W + eps);
            put_wrapped(ypad, W, c / R, c % R, lane, bfloat(v * rs * g));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = tid; i < 16 * W; i += TG) {
            const int r = i / W, f = i % W;
            float sum = 0.0f;
            if (r < R) {
                for (int t = 0; t < S; ++t) sum += float(ypad[(t * PR + r + HALF) * W + f]);
            }
            pooled[i] = 0.25f * sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            auto a = tg_f(pooled, ext(W, 16));
            auto w = dev_f((device float*)wat + sg * q, ext(q, W), array<int32_t, 2>({1, H}));
            auto o = tg_f(across + sg * q, ext(q, 16), array<int32_t, 2>({1, H}));
            across_op.run(a, w, o);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            auto a = tg_bf(ypad + s * PR * W, ext(KT * W, 16), array<int32_t, 2>({1, W}));
            auto w = dev_bf((device bfloat*)wc, ext(H, KT * W));
            auto along = conv_op.get_destination_cooperative_tensor<decltype(a), decltype(w), float>();
            for (uint16_t i = 0; i < along.get_capacity(); ++i) if (along.is_valid_element(i)) along[i] = 0.0f;
            conv_op.run(a, w, along);
            for (uint16_t i = 0; i < along.get_capacity(); ++i) {
                if (!along.is_valid_element(i)) continue;
                const auto at = along.get_multidimensional_index(i);
                const int o = at[0], r = at[1];
                act[(s * 16 + r) * H + o] = bfloat(along[i] + bc[o] + across[r * H + o]);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        // d(GELU input) = (dxn @ wo) * GELU'(pre); act becomes GELU(pre).
        {
            auto a = dev_f((device float*)dxn + base + s * R * W, ext(W, R));
            auto w = dev_bf((device bfloat*)wo, ext(H, W));
            auto dact = dact_op.get_destination_cooperative_tensor<decltype(a), decltype(w), float>();
            for (uint16_t i = 0; i < dact.get_capacity(); ++i) if (dact.is_valid_element(i)) dact[i] = 0.0f;
            dact_op.run(a, w, dact);
            for (uint16_t i = 0; i < dact.get_capacity(); ++i) {
                if (!dact.is_valid_element(i)) continue;
                const auto at = dact.get_multidimensional_index(i);
                const int o = at[0], r = at[1];
                if (r >= R) continue;
                const float pre = float(act[(s * 16 + r) * H + o]);
                act[(s * 16 + r) * H + o] = bfloat(gelu(pre));
                const bfloat d = bfloat(dact[i] * gelu_grad(pre));
                if (!CIRCULAR) {
                    dpad[(s * PR + r + HALF) * H + o] = d;
                    continue;
                }
                for (int row = r + HALF - R; row < PR; row += R) {
                    if (row >= 0) dpad[(s * PR + row) * H + o] = d;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Output and convolution weight gradients, this simdgroup's columns, every suit.
        for (int t = 0; t < S; ++t) {
            auto a = dev_f((device float*)dxn + base + t * R * W, ext(W, R));
            auto b2 = tg_bf(act + t * 16 * H + q * sg, ext(q, R), array<int32_t, 2>({1, H}));
            dout_op.run(a, b2, g_out);
            auto a3 = tg_bf(ypad + t * PR * W, ext(KT * W, R), array<int32_t, 2>({1, W}));
            auto b3 = tg_bf(dpad + (t * PR + HALF) * H + q * sg, ext(q, R), array<int32_t, 2>({1, H}));
            dconv_op.run(a3, b3, g_conv);
        }
        // Across gradient: the activation gradient summed over suits (and the conv bias's).
        for (int i = tid; i < 16 * H; i += TG) {
            const int r = i / H, o = i % H;
            float sum = 0.0f;
            if (r < R) {
                for (int t = 0; t < S; ++t) sum += float(dpad[(t * PR + r + HALF) * H + o]);
            }
            across[i] = sum;
            g_bias += sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            auto a = tg_f(across + q * sg, ext(q, 16), array<int32_t, 2>({1, H}));
            auto b2 = tg_f(pooled, ext(W, 16));
            dacross_op.run(a, b2, g_across);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            const int p = W / 4;
            auto a = tg_f(across, ext(H, 16));
            auto w = dev_f((device float*)wa + sg * p, ext(p, H), array<int32_t, 2>({1, W}));
            auto o = tg_f(pooled + sg * p, ext(p, 16), array<int32_t, 2>({1, W}));
            dpool_op.run(a, w, o);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Convolution data gradient: the window over the padded gradient, flipped taps.
        {
            auto a = tg_bf(dpad + s * PR * H, ext(KT * H, 16), array<int32_t, 2>({1, H}));
            auto w = dev_bf((device bfloat*)wcf, ext(W, KT * H));
            auto d = dy_op.get_destination_cooperative_tensor<decltype(a), decltype(w), float>();
            for (uint16_t i = 0; i < d.get_capacity(); ++i) if (d.is_valid_element(i)) d[i] = 0.0f;
            dy_op.run(a, w, d);
            for (uint16_t i = 0; i < d.get_capacity(); ++i) {
                if (!d.is_valid_element(i)) continue;
                const auto at = d.get_multidimensional_index(i);
                const int f = at[0], r = at[1];
                if (r < R) dy[(s * R + r) * W + f] = d[i] + 0.25f * pooled[r * W + f];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Norm backward plus the residual.
        for (int c = sg; c < C; c += 4) {
            const long at = base + c * W + lane;
            const float v = x[at];
            const float rs = rsqrt(simd_sum(v * v) / W + eps);
            const float xh = v * rs;
            const float d = dy[c * W + lane];
            const float residual = dxn[at];
            g_gain += d * xh;
            g_obias += residual;
            const float gd = d * g;
            dxp[at] = rs * (gd - xh * simd_sum(gd * xh) / W) + residual;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    g_out.store(dev_f(pout + (long)tg * W * H + q * sg, ext(q, W), array<int32_t, 2>({1, H})));
    g_conv.store(dev_f(pconv + (long)tg * KT * W * H + q * sg, ext(q, KT * W), array<int32_t, 2>({1, H})));
    g_across.store(dev_f(pacross + (long)tg * H * W + q * sg * W, ext(W, q)));
    // Vectors: reduce the per-thread sums over the threadgroup.
    threadgroup float* red = scratch;
    red[tid] = g_bias;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float* vec = pvec + (long)tg * (H + 2 * W);
    if (tid < H) {
        float sum = 0.0f;
        for (int k = tid; k < TG; k += H) sum += red[k];
        vec[tid] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    red[tid] = g_obias;
    red[TG + tid] = g_gain;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < W) {
        float ob = 0.0f, gg = 0.0f;
        for (int k = tid; k < TG; k += W) { ob += red[k]; gg += red[TG + k]; }
        vec[H + tid] = ob;
        vec[H + W + tid] = gg;
    }
}
"""
    return (
        source.replace("__W__", str(width))
        .replace("__H__", str(hidden))
        .replace("__K__", str(kernel))
        .replace("__THREADS__", str(_THREADS))
        .replace("__R__", str(ranks))
        .replace("__CIRCULAR__", "true" if circular else "false")
    )


_LIBRARIES: dict[tuple[int, int, int, int, bool], object] = {}


def _library(width: int, hidden: int, kernel: int, ranks: int = 13, circular: bool = True):
    key = (width, hidden, kernel, ranks, circular)
    if key not in _LIBRARIES:
        _LIBRARIES[key] = torch.mps.compile_shader(_source(*key))
    return _LIBRARIES[key]


def _grid_library(grid):
    return _library(grid.width, grid.hidden, grid.blocks[0].kernel, grid.ranks, grid.circular)


def supported(width: int, hidden: int, kernel: int = 5, ranks: int = 13, circular: bool = True) -> bool:
    """Whether these shapes can run here: threadgroup memory bounds the hidden
    width, tensor-op tiles need these multiples (ranks fit one 16-row tile),
    and the GPU and OS must have Metal 4 tensor ops (else the grid runs as
    PyTorch operations)."""

    if not (width == 32 and hidden in (32, 64) and kernel % 2 == 1 and kernel <= 7 and 3 <= ranks <= 16):
        return False
    try:
        _library(width, hidden, kernel, ranks, circular)
    except Exception:  # noqa: BLE001 - no tensor ops on this GPU / OS
        return False
    return True


def operands(grid) -> dict[str, torch.Tensor]:
    """The kernels' weight layouts, rebuilt only when a weight changes.

    The key needs both parts: version counters catch ``load_state_dict`` and
    ordinary in-place writes, but fused MPS Adam does not bump them, so every
    optimizer step also counts (as for the trunk's cached bf16 weights).
    """

    from cardgym.nn.metal_trunk import optimizer_steps

    params = list(grid.parameters())
    key = (tuple(p._version for p in params), optimizer_steps(), params[0].device)
    if grid._fused_key == key:
        return grid._fused_operands
    blocks = grid.blocks
    with torch.no_grad():
        conv = torch.stack([b.along_rank.weight.detach().float() for b in blocks])  # [B, H, W, K]
        kernel = conv.shape[-1]
        packed = {
            "gains": torch.stack([b.norm.weight.detach().float() for b in blocks] + [grid.norm.weight.detach().float()]),
            # wc[b, t * W + w, o] = conv[b, o, w, t]
            "wc": conv.permute(0, 3, 2, 1).reshape(len(blocks), -1, conv.shape[1]).to(torch.bfloat16).contiguous(),
            # wcf[b, j * H + o, w] = conv[b, o, w, K - 1 - j]
            "wcf": conv.flip(-1).permute(0, 3, 1, 2).reshape(len(blocks), kernel * conv.shape[1], -1)
            .to(torch.bfloat16).contiguous(),
            "bc": torch.stack([b.along_rank.bias.detach().float() for b in blocks]),
            "wa": torch.stack([b.across_suits.weight.detach().float() for b in blocks]).contiguous(),
            "wat": torch.stack([b.across_suits.weight.detach().float().t() for b in blocks]).contiguous(),
            "wo": torch.stack([b.out.weight.detach() for b in blocks]).to(torch.bfloat16).contiguous(),
            "wot": torch.stack([b.out.weight.detach().t() for b in blocks]).to(torch.bfloat16).contiguous(),
            "bo": torch.stack([b.out.bias.detach().float() for b in blocks]),
        }
    grid._fused_key, grid._fused_operands = key, packed
    return packed


class FusedCardGrid(torch.autograd.Function):
    """ledger [rows, C] int64 -> per-card features [rows, C, W] bf16 (C = 4 * ranks).

    ``params``: code embedding, rank embedding, then per block (norm gain,
    conv weight, conv bias, across weight, out weight, out bias), then the
    final norm gain; the kernels read them through :func:`operands`.
    ``save`` keeps what the backward needs: each block's fp32 input.
    """

    @staticmethod
    def forward(ctx, grid, ledger, save, *params):
        width = grid.width
        lib = _grid_library(grid)
        ops = operands(grid)
        rows = ledger.shape[0]
        blocks = len(grid.blocks)
        dev = ledger.device
        count = 4 * grid.ranks
        cards = torch.empty(rows, count, width, device=dev, dtype=torch.bfloat16)
        saved = torch.empty((blocks + 1, rows, count, width) if save else (1,), device=dev)
        if rows:
            lib.grid_fwd(
                ledger, params[0], params[1], ops["gains"], ops["wc"], ops["bc"], ops["wat"], ops["wot"],
                ops["bo"], cards, saved, rows, blocks, int(save), grid.eps,
                threads=min(rows, _FORWARD_GROUPS) * _THREADS, group_size=_THREADS,
            )
        if save:
            ctx.grid = grid
            ctx.save_for_backward(ledger, saved, *params)
        return cards

    @staticmethod
    def backward(ctx, d_cards):
        grid = ctx.grid
        ledger, saved, *params = ctx.saved_tensors
        width, hidden = grid.width, grid.hidden
        kernel = grid.blocks[0].kernel
        lib = _grid_library(grid)
        ops = operands(grid)
        rows = ledger.shape[0]
        blocks = len(grid.blocks)
        dev = ledger.device
        grads: list[torch.Tensor | None] = [None] * len(params)
        if not rows:
            return (None, None, None, *[torch.zeros_like(p) for p in params])
        count = 4 * grid.ranks
        cards = rows * count
        d_cards = d_cards.reshape(cards, width).contiguous()
        bf16 = d_cards.dtype == torch.bfloat16
        if not bf16:
            d_cards = d_cards.float()
        groups = min(256, -(-cards // 8))
        partial = torch.empty(groups, width, device=dev)
        dx = torch.empty(rows, count, width, device=dev)
        lib.grid_final_bwd(
            d_cards, d_cards, saved[blocks], params[-1], dx, partial, cards, int(bf16), grid.eps,
            threads=groups * 256, group_size=256,
        )
        grads[-1] = partial.sum(0)
        groups = min(rows, _BACKWARD_GROUPS)
        for index in reversed(range(blocks)):
            pconv = torch.empty(groups, kernel * width, hidden, device=dev)
            pout = torch.empty(groups, width, hidden, device=dev)
            pacross = torch.empty(groups, hidden, width, device=dev)
            pvec = torch.empty(groups, hidden + 2 * width, device=dev)
            dx_prev = torch.empty_like(dx)
            lib.grid_block_bwd(
                saved[index], dx, dx_prev, params[2 + 6 * index], ops["wc"][index], ops["wcf"][index],
                ops["bc"][index], ops["wat"][index], ops["wa"][index], ops["wo"][index],
                pconv, pout, pacross, pvec, rows, grid.eps,
                threads=groups * _THREADS, group_size=_THREADS,
            )
            base = 2 + 6 * index
            vec = pvec.sum(0)
            grads[base] = vec[hidden + width :].clone()
            # pconv[t * W + w, o] -> conv weight [o, w, t]
            grads[base + 1] = pconv.sum(0).view(kernel, width, hidden).permute(2, 1, 0).contiguous()
            grads[base + 2] = vec[:hidden].clone()
            grads[base + 3] = pacross.sum(0)
            grads[base + 4] = pout.sum(0)
            grads[base + 5] = vec[hidden : hidden + width].clone()
            dx = dx_prev
        flat = dx.view(cards, width)
        grads[0] = torch.zeros_like(params[0]).index_add_(0, ledger.reshape(-1), flat)
        grads[1] = dx.view(rows * 4, grid.ranks, width).sum(0)
        return (None, None, None, *grads)
