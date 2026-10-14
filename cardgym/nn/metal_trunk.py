"""Fused Metal implementation of the decoder trunk for MPS (bf16) training.

The PyTorch trunk spends most of its update time outside the matrix
multiplications: SDPA's grouped-query path expands K/V and reduces their
gradients back with slow strided sums, autocast inserts a cast around every
LayerNorm, and each residual add, GELU and LayerNorm is its own pass over a
[rows, d_model] fp32 tensor. Measured on the plump rl-v6 actor update (156k
positions, 16 layers, bf16) the trunk ran at about 1.3 TFLOPS against the ~20
TFLOPS MPS reaches on the same GEMMs; this module halved the iteration time.

The GEMMs stay on MPS and everything between them is fused:

    forward, per layer   add+LN | QKV | attention | out | add+LN | up | GELU | down
    backward, per layer  down/up/out/qkv data GEMMs and split-K weight GEMMs,
                         one GELU-backward kernel (+ up bias), two LayerNorm
                         backward kernels (+ down bias, per-layer-embedding
                         gradient, the bf16 copy the next GEMM reads), and two
                         attention-backward kernels (dq, then dk/dv)

Rows are either padded sequences [B, L] or packed ones [N] with per-row
sequence bounds (cardgym.nn.packing): every kernel but attention is row-wise,
and attention reads each row's bounds. A few SIMD lanes share one head row
(each holding a slice of its dims), so a thread keeps a few float4s and needs
no threadgroup memory; the dk/dv thread of a key loops over the query heads
sharing its KV head, so the GQA reduction needs no cross-thread step.

The backward accepts several cotangents at once, interleaved per row. The PPO
actor takes the auxiliary and policy gradients separately (each has its own
norm budget); the autograd path pays for that with two full trunk backwards,
here both streams share every saved-activation read and every GEMM launch.

Precision follows PyTorch bf16 autocast wherever it rounds: fp32 residual
stream and LayerNorm statistics, bf16 GEMM operands and outputs, the value
embedding added to v in fp32 and rounded once to bf16. Results match the
autocast path within bf16 rounding, not bit-for-bit.

Enable with ``CARDGYM_METAL_TRUNK=1``. It is a runtime switch rather than a
config key because it changes how the trunk is computed, not what is trained.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.optim.optimizer import register_optimizer_step_post_hook

from .trunk import TrunkConfig

ENV_FLAG = "CARDGYM_METAL_TRUNK"
# Sequences of any length: attention streams keys from device memory.
MAX_LENGTH = 1 << 16
_LN_THREADGROUPS = 128
_LN_SIMDGROUPS = 8
_GELU_ROWS = 32
_WGRAD_SPLITS = 8


def metal_trunk_requested() -> bool:
    return os.environ.get(ENV_FLAG, "").strip().lower() in ("1", "true", "yes", "on")


def _lanes(head_dim: int) -> int:
    """SIMD lanes sharing one attention head row (each holds head_dim / lanes dims).

    Eight dims per lane where the split is a power of two; every lane's share
    must be whole float4s.
    """

    for lanes in (8, 4, 2, 1):
        share = head_dim // lanes
        if head_dim % lanes == 0 and share % 4 == 0 and share >= 8:
            return lanes
    return 1


def unsupported_reason(config: TrunkConfig, max_length: int | None = None) -> str | None:
    """Why this trunk cannot use the fused kernels, or None when it can.

    Any sequence length is supported; ``max_length`` is accepted for callers
    that still pass it.
    """

    if config.per_layer_dim not in (0, 64):
        return "per-layer embeddings must be 64 wide (or off)"
    if config.per_layer_dim and config.d_model < 64:
        return "d_model must be at least the per-layer embedding width"
    if config.head_dim > 64:
        return "head_dim above 64"
    if config.head_dim % 4:
        return "head_dim must be a multiple of 4"
    return None


def _source(dm: int, n_heads: int, kv_heads: int, head_dim: int, ple: int) -> str:
    return f"""
#include <metal_stdlib>
using namespace metal;

constant constexpr int DM = {dm};
constant constexpr int PLE = {max(ple, 1)};
constant constexpr int PER_LANE = (DM + 31) / 32;
constant constexpr int D = {head_dim};
constant constexpr int KVH = {kv_heads};
constant constexpr int NH = {n_heads};
constant constexpr int G = NH / KVH;
constant constexpr int QKV_W = (NH + 2 * KVH) * D;
// Attention: DL lanes share one head row, each holding D / DL dims (V4 float4s).
constant constexpr int DL = {_lanes(head_dim)};
constant constexpr int V4 = D / DL / 4;
constant constexpr int K_OFF = NH * D;
constant constexpr int V_OFF = NH * D + KVH * D;

// Abramowitz & Stegun 7.1.26, |error| < 1.5e-7: Metal has no erf.
inline float erf_approx(float x) {{
    const float t = 1.0f / (1.0f + 0.3275911f * fabs(x));
    const float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t
                     - 0.284496736f) * t + 0.254829592f) * t * exp(-x * x);
    return copysign(y, x);
}}

// Residual add + LayerNorm, one simdgroup per row.
// flags: 1 add x, 2 add delta (bf16), 4 add ple, 8 write x_out, 16 fp32 output.
kernel void add_ln_fwd(
    device const float* x [[buffer(0)]],
    device const bfloat* delta [[buffer(1)]],
    device const float* ple [[buffer(2)]],
    device const float* gamma [[buffer(3)]],
    device const float* beta [[buffer(4)]],
    device float* x_out [[buffer(5)]],
    device bfloat* y_bf [[buffer(6)]],
    device float* y_f [[buffer(7)]],
    device float* stats [[buffer(8)]],
    constant int& R [[buffer(9)]],
    constant int& flags [[buffer(10)]],
    constant int& ple_stride [[buffer(11)]],
    constant int& ple_off [[buffer(12)]],
    constant float& ple_scale [[buffer(13)]],
    constant float& eps [[buffer(14)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{{
    const int row = gid / 32;
    if (row >= R) return;
    const long off = (long)row * DM;
    float v[PER_LANE];
    float sum = 0.0f;
    for (int i = 0; i < PER_LANE; ++i) {{
        const int e = lane + 32 * i;
        float val = 0.0f;
        if (e < DM) {{
            if (flags & 1) val = x[off + e];
            if (flags & 2) val += float(delta[off + e]);
            if (flags & 4) val += ple_scale * ple[(long)row * ple_stride + ple_off + (e % PLE)];
            if (flags & 8) x_out[off + e] = val;
        }}
        v[i] = val;
        sum += val;
    }}
    const float mean = simd_sum(sum) / DM;
    float sq = 0.0f;
    for (int i = 0; i < PER_LANE; ++i) {{
        const int e = lane + 32 * i;
        if (e < DM) {{ const float c = v[i] - mean; sq += c * c; }}
    }}
    const float rstd = rsqrt(simd_sum(sq) / DM + eps);
    for (int i = 0; i < PER_LANE; ++i) {{
        const int e = lane + 32 * i;
        if (e < DM) {{
            const float y = (v[i] - mean) * rstd * gamma[e] + beta[e];
            if (flags & 16) y_f[off + e] = y; else y_bf[off + e] = bfloat(y);
        }}
    }}
    if (lane == 0) {{ stats[2 * row] = mean; stats[2 * row + 1] = rstd; }}
}}

// LayerNorm backward over S interleaved streams: row r of stream s sits at
// r * S + s in dy/dres/dx/dx_bf. One simdgroup per (row, stream); the grid
// stride is a multiple of S, so each simdgroup keeps one stream throughout.
// Per-threadgroup partials per stream: dgamma, dbeta, and the column sum of
// the bf16 dx copy (the bias gradient of the GEMM that produced the residual
// this gradient flows back into). flags: 1 dy bf16, 2 add dres, 4 write
// dx_bf (and its column sum), 8 write the per-layer-embedding gradient.
kernel void ln_bwd(
    device const bfloat* dy_bf [[buffer(0)]],
    device const float* dy_f [[buffer(1)]],
    device const float* x [[buffer(2)]],
    device const float* stats [[buffer(3)]],
    device const float* gamma [[buffer(4)]],
    device const float* dres [[buffer(5)]],
    device float* dx [[buffer(6)]],
    device bfloat* dx_bf [[buffer(7)]],
    device float* dple [[buffer(8)]],
    device float* partial [[buffer(9)]],
    constant int& R [[buffer(10)]],
    constant int& S [[buffer(11)]],
    constant int& flags [[buffer(12)]],
    constant int& ple_stride [[buffer(13)]],
    constant int& ple_off [[buffer(14)]],
    constant float& ple_scale [[buffer(15)]],
    constant int& n_tg [[buffer(16)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint n_sg [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{{
    threadgroup float acc[2 * 3 * DM];
    const int gsg = tg * n_sg + sg;
    const int stream = gsg % S;
    float dg[PER_LANE], db[PER_LANE], dc[PER_LANE];
    for (int i = 0; i < PER_LANE; ++i) {{ dg[i] = 0.0f; db[i] = 0.0f; dc[i] = 0.0f; }}
    const int total = R * S;
    for (int idx = gsg; idx < total; idx += n_tg * n_sg) {{
        const int row = idx / S;
        const long off = (long)row * DM;
        const long soff = (long)idx * DM;
        const float mean = stats[2 * row], rstd = stats[2 * row + 1];
        float xh[PER_LANE], g[PER_LANE];
        float s1 = 0.0f, s2 = 0.0f;
        for (int i = 0; i < PER_LANE; ++i) {{
            const int e = lane + 32 * i;
            xh[i] = 0.0f; g[i] = 0.0f;
            if (e < DM) {{
                const float dyv = (flags & 1) ? float(dy_bf[soff + e]) : dy_f[soff + e];
                xh[i] = (x[off + e] - mean) * rstd;
                g[i] = dyv * gamma[e];
                dg[i] += dyv * xh[i];
                db[i] += dyv;
                s1 += g[i];
                s2 += g[i] * xh[i];
            }}
        }}
        s1 = simd_sum(s1) / DM;
        s2 = simd_sum(s2) / DM;
        // With PLE = 64, column e feeds embedding column e % 64, which is
        // `lane` for even i and `lane + 32` for odd i.
        float pa = 0.0f, pb = 0.0f;
        for (int i = 0; i < PER_LANE; ++i) {{
            const int e = lane + 32 * i;
            if (e < DM) {{
                float val = rstd * (g[i] - s1 - xh[i] * s2);
                if (flags & 2) val += dres[soff + e];
                dx[soff + e] = val;
                if (flags & 4) {{
                    const bfloat vb = bfloat(val);
                    dx_bf[soff + e] = vb;
                    dc[i] += float(vb);
                }}
                if (i % 2 == 0) pa += val; else pb += val;
            }}
        }}
        if (flags & 8) {{
            const long poff = ((long)stream * R + row) * ple_stride + ple_off;
            dple[poff + lane] = ple_scale * pa;
            dple[poff + lane + 32] = ple_scale * pb;
        }}
    }}
    for (int i = sg * 32 + lane; i < S * 3 * DM; i += n_sg * 32) acc[i] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 0; k < n_sg; ++k) {{
        if (sg == k) {{
            threadgroup float* a = acc + stream * 3 * DM;
            for (int i = 0; i < PER_LANE; ++i) {{
                const int e = lane + 32 * i;
                if (e < DM) {{ a[e] += dg[i]; a[DM + e] += db[i]; a[2 * DM + e] += dc[i]; }}
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    for (int i = sg * 32 + lane; i < S * 3 * DM; i += n_sg * 32) {{
        partial[(long)tg * S * 3 * DM + i] = acc[i];
    }}
}}

// du = dgl * gelu'(u) (exact erf GELU) over S <= 2 interleaved streams, plus
// per-threadgroup column sums of the bf16 du (the up-projection bias
// gradient). dgl/du [R, S, F] bf16, u [R, F] bf16, partial [n_tg, S, F].
kernel void gelu_bwd(
    device const bfloat* dgl [[buffer(0)]],
    device const bfloat* u [[buffer(1)]],
    device bfloat* du [[buffer(2)]],
    device float* partial [[buffer(3)]],
    constant int& R [[buffer(4)]],
    constant int& S [[buffer(5)]],
    constant int& F [[buffer(6)]],
    constant int& rows_per_tg [[buffer(7)]],
    uint tg [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]])
{{
    const int r0 = tg * rows_per_tg;
    const int r1 = min(R, r0 + rows_per_tg);
    for (int c = tid; c < F; c += tgsize) {{
        float acc0 = 0.0f, acc1 = 0.0f;
        for (int r = r0; r < r1; ++r) {{
            const float x = float(u[(long)r * F + c]);
            const float cdf = 0.5f * (1.0f + erf_approx(x * M_SQRT1_2_F));
            const float pdf = exp(-0.5f * x * x) * 0.3989422804014327f;
            const float gp = cdf + x * pdf;
            const long i0 = ((long)r * S) * F + c;
            const bfloat v0 = bfloat(float(dgl[i0]) * gp);
            du[i0] = v0;
            acc0 += float(v0);
            if (S > 1) {{
                const bfloat v1 = bfloat(float(dgl[i0 + F]) * gp);
                du[i0 + F] = v1;
                acc1 += float(v1);
            }}
        }}
        partial[((long)tg * S) * F + c] = acc0;
        if (S > 1) partial[((long)tg * S + 1) * F + c] = acc1;
    }}
}}

// Attention over the rows of a batch, each row in one sequence: row t sees
// rows bounds[t].x .. t, and its keys are seen by rows up to bounds[t].y - 1.
// Packed batches pass bounds per row; padded [B, L] batches pass seg_len = L
// (and any buffer as bounds). DL lanes of one SIMD group share a head row,
// each holding D / DL of its dims, so a thread keeps a few float4s, no
// threadgroup memory is needed, and every thread has a row to work on: the
// sequences are short (tens of rows) and attention is a small share of the
// trunk's FLOPs, so occupancy and memory traffic are what count.
inline int2 row_bounds(device const int2* bounds, int seg_len, int t) {{
    if (seg_len > 0) {{ const int s = (t / seg_len) * seg_len; return int2(s, s + seg_len); }}
    return bounds[t];
}}

inline float lane_sum(float x) {{
    for (ushort off = 1; off < DL; off <<= 1) x += simd_shuffle_xor(x, off);
    return x;
}}

inline float2 lane_sum2(float2 x) {{
    for (ushort off = 1; off < DL; off <<= 1) x += simd_shuffle_xor(x, off);
    return x;
}}

// v += value embedding, rounded once to bf16 (autocast adds the fp32
// embedding to the bf16 projection and SDPA reads v in bf16). In place: the
// attention kernels and their backward only ever read this v.
kernel void add_value_embedding(
    device bfloat* qkv [[buffer(0)]],
    device const float* vemb [[buffer(1)]],
    constant int& N [[buffer(2)]],
    constant int& ve_stride [[buffer(3)]],
    constant int& ve_off [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{{
    const int t = gid / (KVH * D);
    const int c = gid % (KVH * D);
    if (t >= N) return;
    const long at = (long)t * QKV_W + V_OFF + c;
    qkv[at] = bfloat(float(qkv[at]) + vemb[(long)t * ve_stride + ve_off + c]);
}}

// Forward. Thread (row t, query head h, lane). qkv [N, QKV_W] bf16;
// out [N, NH*D] bf16; lse [N, NH] fp32 in log2 units.
kernel void attn_fwd(
    device const bfloat* qkv [[buffer(0)]],
    device const int2* bounds [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    device float* lse [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& seg_len [[buffer(5)]],
    constant float& scale_log2 [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{{
    const int lane = gid % DL;
    const int h = (gid / DL) % NH;
    const int t = gid / (DL * NH);
    if (t >= N) return;
    const int j = h / G;
    const int d0 = lane * (D / DL);
    const int2 b = row_bounds(bounds, seg_len, t);
    device const bfloat4* qp = (device const bfloat4*)(qkv + (long)t * QKV_W + h * D + d0);
    float4 q[V4], acc[V4];
    for (int i = 0; i < V4; ++i) {{ q[i] = float4(qp[i]) * scale_log2; acc[i] = 0.0f; }}
    float m = -INFINITY, l = 0.0f;
    for (int s = b.x; s <= t; ++s) {{
        device const bfloat4* kp = (device const bfloat4*)(qkv + (long)s * QKV_W + K_OFF + j * D + d0);
        device const bfloat4* vp = (device const bfloat4*)(qkv + (long)s * QKV_W + V_OFF + j * D + d0);
        float part = 0.0f;
        for (int i = 0; i < V4; ++i) part += dot(q[i], float4(kp[i]));
        const float score = lane_sum(part);
        const float m_new = max(m, score);
        const float corr = exp2(m - m_new);
        const float p = exp2(score - m_new);
        l = l * corr + p;
        for (int i = 0; i < V4; ++i) acc[i] = fma(float4(p), float4(vp[i]), acc[i] * corr);
        m = m_new;
    }}
    const float inv = 1.0f / l;
    device bfloat4* op = (device bfloat4*)(out + (long)t * (NH * D) + h * D + d0);
    for (int i = 0; i < V4; ++i) op[i] = bfloat4(acc[i] * inv);
    if (lane == 0) lse[(long)t * NH + h] = m + log2(l);
}}

// Backward, part 1: dq and delta = dout . out. Thread (row t, stream, query
// head, lane); S cotangent streams are interleaved per row in
// dout [N, S, NH*D] and dqkv [N, S, QKV_W]; delta [S, N, NH] feeds part 2.
kernel void attn_bwd_dq(
    device const bfloat* qkv [[buffer(0)]],
    device const int2* bounds [[buffer(1)]],
    device const bfloat* out [[buffer(2)]],
    device const float* lse [[buffer(3)]],
    device const bfloat* dout [[buffer(4)]],
    device bfloat* dqkv [[buffer(5)]],
    device float* delta [[buffer(6)]],
    constant int& N [[buffer(7)]],
    constant int& S [[buffer(8)]],
    constant int& seg_len [[buffer(9)]],
    constant float& scale_log2 [[buffer(10)]],
    constant float& scale [[buffer(11)]],
    uint gid [[thread_position_in_grid]])
{{
    const int lane = gid % DL;
    const int h = (gid / DL) % NH;
    const int so = (gid / (DL * NH)) % S;
    const int t = gid / (DL * NH * S);
    if (t >= N) return;
    const int j = h / G;
    const int d0 = lane * (D / DL);
    const int2 b = row_bounds(bounds, seg_len, t);
    device const bfloat4* qp = (device const bfloat4*)(qkv + (long)t * QKV_W + h * D + d0);
    device const bfloat4* gp = (device const bfloat4*)(dout + ((long)t * S + so) * (NH * D) + h * D + d0);
    device const bfloat4* op = (device const bfloat4*)(out + (long)t * (NH * D) + h * D + d0);
    float4 q[V4], dov[V4], dq[V4];
    float part = 0.0f;
    for (int i = 0; i < V4; ++i) {{
        q[i] = float4(qp[i]) * scale_log2;
        dov[i] = float4(gp[i]);
        part += dot(dov[i], float4(op[i]));
        dq[i] = 0.0f;
    }}
    const float dl = lane_sum(part);
    const float lse_t = lse[(long)t * NH + h];
    for (int s = b.x; s <= t; ++s) {{
        device const bfloat4* kp = (device const bfloat4*)(qkv + (long)s * QKV_W + K_OFF + j * D + d0);
        device const bfloat4* vp = (device const bfloat4*)(qkv + (long)s * QKV_W + V_OFF + j * D + d0);
        float2 part2 = 0.0f;
        for (int i = 0; i < V4; ++i) {{
            part2.x += dot(q[i], float4(kp[i]));
            part2.y += dot(dov[i], float4(vp[i]));
        }}
        const float2 r = lane_sum2(part2);
        const float ds = exp2(r.x - lse_t) * (r.y - dl);
        for (int i = 0; i < V4; ++i) dq[i] = fma(float4(ds), float4(kp[i]), dq[i]);
    }}
    device bfloat4* dp = (device bfloat4*)(dqkv + ((long)t * S + so) * QKV_W + h * D + d0);
    for (int i = 0; i < V4; ++i) dp[i] = bfloat4(dq[i] * scale);
    if (lane == 0) delta[((long)so * N + t) * NH + h] = dl;
}}

// Backward, part 2: dk and dv. Thread (key row s, stream, KV head, lane)
// sums over the rows that see key s and over the G query heads sharing its KV
// head, so no reduction across threads is needed. The v gradient also lands
// in dvemb [N, S, dve_stride] at ve_off (the value-embedding gradient).
kernel void attn_bwd_dkv(
    device const bfloat* qkv [[buffer(0)]],
    device const int2* bounds [[buffer(1)]],
    device const float* lse [[buffer(2)]],
    device const bfloat* dout [[buffer(3)]],
    device const float* delta [[buffer(4)]],
    device bfloat* dqkv [[buffer(5)]],
    device float* dvemb [[buffer(6)]],
    constant int& N [[buffer(7)]],
    constant int& S [[buffer(8)]],
    constant int& seg_len [[buffer(9)]],
    constant int& ve_off [[buffer(10)]],
    constant int& dve_stride [[buffer(11)]],
    constant float& scale_log2 [[buffer(12)]],
    constant float& scale [[buffer(13)]],
    uint gid [[thread_position_in_grid]])
{{
    const int lane = gid % DL;
    const int j = (gid / DL) % KVH;
    const int so = (gid / (DL * KVH)) % S;
    const int s = gid / (DL * KVH * S);
    if (s >= N) return;
    const int d0 = lane * (D / DL);
    const int2 b = row_bounds(bounds, seg_len, s);
    device const bfloat4* kp = (device const bfloat4*)(qkv + (long)s * QKV_W + K_OFF + j * D + d0);
    device const bfloat4* vp = (device const bfloat4*)(qkv + (long)s * QKV_W + V_OFF + j * D + d0);
    float4 k[V4], v[V4], dk[V4], dv[V4];
    for (int i = 0; i < V4; ++i) {{ k[i] = float4(kp[i]); v[i] = float4(vp[i]); dk[i] = 0.0f; dv[i] = 0.0f; }}
    for (int t = s; t < b.y; ++t) {{
        for (int g = 0; g < G; ++g) {{
            const int h = j * G + g;
            device const bfloat4* qp = (device const bfloat4*)(qkv + (long)t * QKV_W + h * D + d0);
            device const bfloat4* gp = (device const bfloat4*)(dout + ((long)t * S + so) * (NH * D) + h * D + d0);
            float2 part2 = 0.0f;
            for (int i = 0; i < V4; ++i) {{
                part2.x += dot(float4(qp[i]), k[i]);
                part2.y += dot(float4(gp[i]), v[i]);
            }}
            const float2 r = lane_sum2(part2);
            const float p = exp2(r.x * scale_log2 - lse[(long)t * NH + h]);
            const float ds = p * (r.y - delta[((long)so * N + t) * NH + h]);
            for (int i = 0; i < V4; ++i) {{
                dv[i] = fma(float4(p), float4(gp[i]), dv[i]);
                dk[i] = fma(float4(ds), float4(qp[i]), dk[i]);
            }}
        }}
    }}
    // dk was accumulated against the unscaled q; its true factor is scale.
    const long row = ((long)s * S + so) * QKV_W;
    device bfloat4* dkp = (device bfloat4*)(dqkv + row + K_OFF + j * D + d0);
    device bfloat4* dvp = (device bfloat4*)(dqkv + row + V_OFF + j * D + d0);
    for (int i = 0; i < V4; ++i) {{ dkp[i] = bfloat4(dk[i] * scale); dvp[i] = bfloat4(dv[i]); }}
    if (ve_off >= 0) {{
        device float4* ep = (device float4*)(dvemb + ((long)s * S + so) * dve_stride + ve_off + j * D + d0);
        for (int i = 0; i < V4; ++i) ep[i] = dv[i];
    }}
}}

// Cached decode attention for a merged wave: every row appends T tokens at
// its own position and attends to its own prefix. One thread per (row, KV
// head, query head in group, new token). Keys before the row's position come
// from the fp16 cache; the T new ones come straight from qkv, rounded to fp16
// exactly as the cache would hold them, and the g == 0 thread writes them
// back. No thread reads a slot another thread of this dispatch writes.
// qkv [B*T, QKV_W] bf16; vemb [B*T, ve_stride] fp32 at ve_off (< 0: none);
// kc/vc [capacity, KVH, max_len, D] half (one layer); positions [B] int64;
// out [B*T, NH*D] bf16; with ``gathered``, row b reads and writes cache row
// rows[b] instead of b.
kernel void attn_decode(
    device const bfloat* qkv [[buffer(0)]],
    device const float* vemb [[buffer(1)]],
    device half* kc [[buffer(2)]],
    device half* vc [[buffer(3)]],
    device const long* positions [[buffer(4)]],
    device bfloat* out [[buffer(5)]],
    constant int& B [[buffer(6)]],
    constant int& T [[buffer(7)]],
    constant int& max_len [[buffer(8)]],
    constant int& ve_stride [[buffer(9)]],
    constant int& ve_off [[buffer(10)]],
    constant float& scale_log2 [[buffer(11)]],
    device const long* rows [[buffer(12)]],
    constant int& gathered [[buffer(13)]],
    uint gid [[thread_position_in_grid]])
{{
    const int t = gid % T;
    const int g = (gid / T) % G;
    const int j = (gid / (T * G)) % KVH;
    const int b = gid / (T * G * KVH);
    if (b >= B) return;
    const int pos0 = (int)positions[b];
    const long row_b = gathered ? rows[b] : (long)b;
    const long cache_row = (row_b * KVH + j) * max_len;
    const long tok0 = (long)b * T;
    if (g == 0) {{
        const long src = (tok0 + t) * QKV_W;
        const int slot = min(pos0 + t, max_len - 1);
        for (int d = 0; d < D; ++d) {{
            kc[(cache_row + slot) * D + d] = half(float(qkv[src + K_OFF + j * D + d]));
            float v = float(qkv[src + V_OFF + j * D + d]);
            if (ve_off >= 0) v += vemb[(tok0 + t) * ve_stride + ve_off + j * D + d];
            vc[(cache_row + slot) * D + d] = half(v);
        }}
    }}
    const int h = j * G + g;
    float q[D];
    const long qrow = (tok0 + t) * QKV_W + h * D;
    for (int d = 0; d < D; ++d) q[d] = float(qkv[qrow + d]) * scale_log2;
    float m = -INFINITY, l = 0.0f;
    float acc[D];
    for (int d = 0; d < D; ++d) acc[d] = 0.0f;
    const int keys = pos0 + t + 1;
    for (int s = 0; s < keys; ++s) {{
        float k[D], v[D];
        if (s < pos0) {{
            const long at = (cache_row + s) * D;
            for (int d = 0; d < D; ++d) {{ k[d] = float(kc[at + d]); v[d] = float(vc[at + d]); }}
        }} else {{
            const long src = (tok0 + (s - pos0)) * QKV_W;
            for (int d = 0; d < D; ++d) {{
                k[d] = float(half(float(qkv[src + K_OFF + j * D + d])));
                float vv = float(qkv[src + V_OFF + j * D + d]);
                if (ve_off >= 0) vv += vemb[(tok0 + (s - pos0)) * ve_stride + ve_off + j * D + d];
                v[d] = float(half(vv));
            }}
        }}
        float score = 0.0f;
        for (int d = 0; d < D; ++d) score = fma(q[d], k[d], score);
        const float m_new = max(m, score);
        const float corr = exp2(m - m_new);
        const float p = exp2(score - m_new);
        l = l * corr + p;
        for (int d = 0; d < D; ++d) acc[d] = fma(p, v[d], acc[d] * corr);
        m = m_new;
    }}
    const float inv = 1.0f / l;
    const long orow = (tok0 + t) * (NH * D) + h * D;
    for (int d = 0; d < D; ++d) out[orow + d] = bfloat(acc[d] * inv);
}}
"""



_LIBRARIES: dict[tuple[int, ...], object] = {}

# Every optimizer step in the process, counted by a global post-step hook
# installed with the first MetalTrunk; see MetalTrunk._bf16_weights.
_OPTIMIZER_STEPS = [0]
_STEP_HOOK = None


def optimizer_steps() -> int:
    """Optimizer steps taken in this process (counting starts once a cache asks)."""

    _count_optimizer_steps()
    return _OPTIMIZER_STEPS[0]


def _count_optimizer_steps() -> None:
    global _STEP_HOOK
    if _STEP_HOOK is None:

        def bump(optimizer, args, kwargs):
            _OPTIMIZER_STEPS[0] += 1

        _STEP_HOOK = register_optimizer_step_post_hook(bump)


def _library(key: tuple[int, ...]):
    library = _LIBRARIES.get(key)
    if library is None:
        library = torch.mps.compile_shader(_source(*key))
        _LIBRARIES[key] = library
    return library


def _wgrad(dy: torch.Tensor, x: torch.Tensor, streams: int) -> torch.Tensor:
    """dy [R, S*N] bf16, x [R, K] bf16 -> [S, N, K] fp32.

    MPS runs a long-K GEMM at a ~0.7 ms floor whatever the output size; a
    split-K batch restores the parallelism (2.5-4.7x at these shapes).
    """

    rows, width = dy.shape
    depth = x.shape[1]
    main = (rows // _WGRAD_SPLITS) * _WGRAD_SPLITS
    out = torch.bmm(
        dy[:main].view(_WGRAD_SPLITS, main // _WGRAD_SPLITS, width).transpose(1, 2),
        x[:main].view(_WGRAD_SPLITS, main // _WGRAD_SPLITS, depth),
    ).sum(0, dtype=torch.float32)
    if main < rows:
        out += (dy[main:].t() @ x[main:]).float()
    return out.view(streams, width // streams, depth)


# Parameter order within one layer of trunk_parameters().
_LN1_W, _LN1_B, _WQKV, _WOUT, _LN2_W, _LN2_B, _W1, _B1, _W2, _B2 = range(10)
PER_LAYER = 10


def trunk_parameters(trunk) -> list[torch.nn.Parameter]:
    """Every parameter the fused trunk reads, in the order MetalTrunk expects."""

    params: list[torch.nn.Parameter] = []
    for block in trunk.blocks:
        params += [
            block.ln_attn.weight, block.ln_attn.bias,
            block.qkv_proj.weight, block.out_proj.weight,
            block.ln_mlp.weight, block.ln_mlp.bias,
            block.mlp[0].weight, block.mlp[0].bias,
            block.mlp[2].weight, block.mlp[2].bias,
        ]
    params += [trunk.final_norm.weight, trunk.final_norm.bias]
    return params


@dataclass
class _Saved:
    lead: tuple[int, ...]  # the input's leading shape: [B, L] padded or [N] packed
    rows: int
    bounds: torch.Tensor  # int32 [rows, 2] sequence bounds per row (packed) or a placeholder
    seg_len: int  # padded: every sequence is seg_len rows; packed: 0
    layers: list[dict]
    weights: list[tuple[torch.Tensor, ...]]
    x_final: torch.Tensor
    stats_final: torch.Tensor
    values: torch.Tensor | None
    value_width: int
    ple_width: int
    has_ple: bool


class MetalTrunk:
    """The fused trunk for one trunk configuration."""

    max_length = MAX_LENGTH

    def __init__(self, config: TrunkConfig, *, max_length: int | None = None, eps: float = 1e-5) -> None:
        reason = unsupported_reason(config, max_length)
        if reason is not None:
            raise ValueError(f"Metal trunk unsupported: {reason}.")
        self.config = config
        self.eps = eps
        self.layers = config.n_layers
        self.dm = config.d_model
        self.head_dim = config.head_dim
        self.kv_heads = config.n_kv_heads
        self.n_heads = config.n_heads
        self.ple_dim = config.per_layer_dim
        self.ple_scale = (
            1.0 / math.sqrt(math.ceil(self.dm / self.ple_dim)) if self.ple_dim else 0.0
        )
        self.value_index = {layer: index for index, layer in enumerate(config.value_layers)}
        self.kv_width = self.kv_heads * self.head_dim
        self.key = (self.dm, self.n_heads, self.kv_heads, self.head_dim, self.ple_dim)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self._weight_key: tuple | None = None
        self._weights: list[tuple[torch.Tensor, ...]] = []
        _count_optimizer_steps()

    @property
    def lib(self):
        return _library(self.key)

    def _bf16_weights(self, params: Sequence[torch.Tensor]) -> list[tuple[torch.Tensor, ...]]:
        """Per layer (wqkv, wout, w1, w2, b1, b2) in bf16, reused while valid.

        Casting is ~1.25 ms of dispatch per call, paid on every rollout wave
        and every microbatch, so the copy is cached. The key needs both parts:
        version counters catch ``load_state_dict`` and any ordinary in-place
        write, but the fused MPS Adam step does not bump them, so every
        optimizer step also advances ``_OPTIMIZER_STEPS``. (A ``.data`` write
        would evade both; nothing writes trunk weights that way.)
        """

        layer_params = [
            params[PER_LAYER * layer + i]
            for layer in range(self.layers)
            for i in (_WQKV, _WOUT, _W1, _W2, _B1, _B2)
        ]
        key = (_OPTIMIZER_STEPS[0], tuple((id(p), p._version) for p in layer_params))
        if self._weight_key != key:
            cast = [p.detach().to(torch.bfloat16) for p in layer_params]
            self._weights = [tuple(cast[6 * layer : 6 * layer + 6]) for layer in range(self.layers)]
            self._weight_key = key
        return self._weights

    def _layout(self, lead: tuple[int, ...], bounds: torch.Tensor | None, device) -> tuple[torch.Tensor, int]:
        """(bounds, seg_len) for a padded [B, L] input or a packed [N] one with bounds."""

        if bounds is not None:
            if bounds.dtype != torch.int32 or bounds.shape != (math.prod(lead), 2):
                raise ValueError("Packed rows need int32 bounds [rows, 2].")
            return bounds.contiguous(), 0
        if len(lead) != 2:
            raise ValueError("Without bounds the input must be padded [B, L, d].")
        return torch.empty(1, 2, device=device, dtype=torch.int32), lead[1]

    # -- kernels ---------------------------------------------------------

    def _add_ln(self, rows, gamma, beta, *, x=None, delta=None, ple=None,
                layer=0, write_x=False, fp32_out=False):
        dev = gamma.device
        flags = 0
        dummy_b = torch.empty(1, device=dev, dtype=torch.bfloat16)
        if x is not None:
            flags |= 1
        if delta is not None:
            flags |= 2
        if ple is not None:
            flags |= 4
        x_out = torch.empty(rows, self.dm, device=dev) if write_x else gamma
        if write_x:
            flags |= 8
        if fp32_out:
            flags |= 16
            y = torch.empty(rows, self.dm, device=dev)
            y_bf, y_f = dummy_b, y
        else:
            y = torch.empty(rows, self.dm, device=dev, dtype=torch.bfloat16)
            y_bf, y_f = y, gamma
        stats = torch.empty(rows, 2, device=dev)
        self.lib.add_ln_fwd(
            x if x is not None else gamma,
            delta if delta is not None else dummy_b,
            ple if ple is not None else gamma,
            gamma, beta, x_out, y_bf, y_f, stats,
            rows, flags, ple.shape[-1] if ple is not None else 0,
            layer * self.ple_dim, float(self.ple_scale), float(self.eps),
            threads=rows * 32, group_size=256,
        )
        return (x_out if write_x else None), y, stats

    def _ln_backward(self, dy, x, stats, gamma, streams, *, dres=None,
                     want_bf=False, dple=None, layer=0):
        total = dy.shape[0]
        rows = total // streams
        dev = x.device
        flags = 0
        dummy_b = torch.empty(1, device=dev, dtype=torch.bfloat16)
        if dy.dtype == torch.bfloat16:
            flags |= 1
            dy_bf, dy_f = dy, stats
        else:
            dy_bf, dy_f = dummy_b, dy
        if dres is not None:
            flags |= 2
        dx = torch.empty(total, self.dm, device=dev)
        dx_bf = (
            torch.empty(total, self.dm, device=dev, dtype=torch.bfloat16)
            if want_bf
            else dummy_b
        )
        if want_bf:
            flags |= 4
        if dple is not None:
            flags |= 8
        n_tg = min(_LN_THREADGROUPS, max(1, (total + _LN_SIMDGROUPS - 1) // _LN_SIMDGROUPS))
        partial = torch.empty(n_tg, streams, 3, self.dm, device=dev)
        self.lib.ln_bwd(
            dy_bf, dy_f, x, stats, gamma,
            dres if dres is not None else stats,
            dx, dx_bf, dple if dple is not None else stats, partial,
            rows, streams, flags, dple.shape[-1] if dple is not None else 0,
            layer * self.ple_dim, float(self.ple_scale), n_tg,
            threads=n_tg * _LN_SIMDGROUPS * 32, group_size=_LN_SIMDGROUPS * 32,
        )
        return dx, (dx_bf if want_bf else None), partial.sum(0)

    def _gelu_backward(self, dgl, u, streams):
        rows, width = u.shape
        du = torch.empty_like(dgl)
        n_tg = (rows + _GELU_ROWS - 1) // _GELU_ROWS
        partial = torch.empty(n_tg, streams, width, device=u.device)
        self.lib.gelu_bwd(
            dgl, u, du, partial, rows, streams, width, _GELU_ROWS,
            threads=n_tg * 256, group_size=256,
        )
        return du, partial.sum(0)

    def _attention(self, qkv, values, layer, bounds, seg_len):
        """qkv [rows, QKV_W] -> out [rows, NH*D] bf16, lse [rows, NH] fp32.

        A layer with a value embedding adds it into qkv's V columns first (in
        place); everything downstream, the backward included, reads that V.
        """

        rows = qkv.shape[0]
        index = self.value_index.get(layer)
        ve_off = -1 if index is None or values is None else index * self.kv_width
        if ve_off >= 0:
            self.lib.add_value_embedding(
                qkv, values, rows, values.shape[-1], ve_off,
                threads=rows * self.kv_width, group_size=256,
            )
        out = torch.empty(rows, self.n_heads * self.head_dim, device=qkv.device, dtype=torch.bfloat16)
        lse = torch.empty(rows, self.n_heads, device=qkv.device)
        self.lib.attn_fwd(
            qkv, bounds, out, lse, rows, seg_len, self.scale * math.log2(math.e),
            threads=rows * self.n_heads * _lanes(self.head_dim), group_size=256,
        )
        return out, lse, ve_off

    def _attention_backward(self, qkv, out, lse, dout, ve_off, dvalues, bounds, seg_len):
        """dout [rows, S, NH*D] -> dqkv [rows, S, QKV_W]; the V part also lands in dvalues."""

        rows, streams, _ = dout.shape
        lanes = _lanes(self.head_dim)
        dqkv = torch.empty(rows, streams, qkv.shape[-1], device=qkv.device, dtype=torch.bfloat16)
        delta = torch.empty(streams, rows, self.n_heads, device=qkv.device)
        scale_log2 = self.scale * math.log2(math.e)
        self.lib.attn_bwd_dq(
            qkv, bounds, out, lse, dout, dqkv, delta, rows, streams, seg_len, scale_log2, self.scale,
            threads=rows * streams * self.n_heads * lanes, group_size=256,
        )
        use_values = ve_off >= 0
        self.lib.attn_bwd_dkv(
            qkv, bounds, lse, dout, delta, dqkv, dvalues if use_values else lse,
            rows, streams, seg_len, ve_off, dvalues.shape[-1] if use_values else 0,
            scale_log2, self.scale,
            threads=rows * streams * self.kv_heads * lanes, group_size=256,
        )
        return dqkv

    # -- trunk -----------------------------------------------------------

    def forward(self, x0, ple, values, params: Sequence[torch.Tensor], *, save: bool, bounds=None):
        """x0 [*lead, DM], ple [*lead, layers, 64] | None, values [*lead, nv, kv] | None.

        ``lead`` is [B, L] for padded sequences, or [N] for packed rows with
        ``bounds`` (int32 [N, 2], each row's sequence as [start, end) rows).
        Returns the final-norm output [*lead, DM] fp32 and, with ``save``, the
        context ``backward`` needs.
        """

        lead = tuple(x0.shape[:-1])
        rows = math.prod(lead)
        bounds, seg_len = self._layout(lead, bounds, x0.device)
        x = x0.reshape(rows, self.dm).float().contiguous()
        ple_flat = (
            ple.reshape(rows, -1).float().contiguous() if ple is not None and self.ple_dim else None
        )
        ve = (
            values.reshape(rows, -1).float().contiguous()
            if values is not None and self.value_index
            else None
        )
        layers: list[dict] = []
        weights: list[tuple[torch.Tensor, ...]] = []
        cast = self._bf16_weights(params)
        _, a, stats1 = self._add_ln(rows, params[_LN1_W], params[_LN1_B], x=x)
        for layer in range(self.layers):
            p = params[PER_LAYER * layer : PER_LAYER * (layer + 1)]
            wqkv, wout, w1, w2, b1, b2 = cast[layer]
            qkv = F.linear(a, wqkv)
            o, lse, ve_off = self._attention(qkv, ve, layer, bounds, seg_len)
            x1, m, stats2 = self._add_ln(
                rows, p[_LN2_W], p[_LN2_B], x=x, delta=F.linear(o, wout), write_x=True
            )
            u = F.linear(m, w1, b1)
            gl = F.gelu(u)
            down = F.linear(gl, w2, b2)
            last = layer == self.layers - 1
            base = PER_LAYER * (self.layers if last else layer + 1)
            x_next, a_next, stats_next = self._add_ln(
                rows, params[base], params[base + 1], x=x1, delta=down,
                ple=ple_flat, layer=layer, write_x=True, fp32_out=last,
            )
            if save:
                layers.append(dict(x=x, a=a, stats1=stats1, qkv=qkv, o=o, lse=lse,
                                   ve_off=ve_off, x1=x1, m=m, stats2=stats2, u=u, gl=gl))
                weights.append((wqkv, wout, w1, w2))
            x, a, stats1 = x_next, a_next, stats_next
        hidden = a.view(*lead, self.dm)
        if not save:
            return hidden, None
        saved = _Saved(
            lead=lead, rows=rows, bounds=bounds, seg_len=seg_len, layers=layers, weights=weights,
            x_final=x, stats_final=stats1, values=ve,
            value_width=ve.shape[-1] if ve is not None else 0,
            ple_width=ple_flat.shape[-1] if ple_flat is not None else 0,
            has_ple=ple_flat is not None,
        )
        return hidden, saved

    def backward(self, saved: _Saved, dh: torch.Tensor, params: Sequence[torch.Tensor]):
        """dh [*lead, S, DM] (S cotangent streams interleaved per row).

        Returns dx0 [*lead, S, DM], dple [S, *lead, layers, 64] | None,
        dvalues [*lead, S, nv, kv] | None, and per-parameter [S, ...] fp32
        gradients aligned with ``params``.
        """

        lead, rows = saved.lead, saved.rows
        streams = dh.shape[-2]
        dev = dh.device
        grads: list[torch.Tensor | None] = [None] * len(params)
        dple = (
            torch.empty(streams, rows, saved.ple_width, device=dev) if saved.has_ple else None
        )
        dvalues = (
            torch.zeros(rows, streams, saved.value_width, device=dev)
            if saved.values is not None
            else None
        )
        final = PER_LAYER * self.layers
        dx, dx_bf, sums = self._ln_backward(
            dh.reshape(rows * streams, self.dm).float().contiguous(),
            saved.x_final, saved.stats_final, params[final], streams,
            want_bf=True, dple=dple, layer=self.layers - 1,
        )
        grads[final], grads[final + 1] = sums[:, 0], sums[:, 1]
        down_bias = sums[:, 2]
        for layer in reversed(range(self.layers)):
            sv = saved.layers[layer]
            wqkv, wout, w1, w2 = saved.weights[layer]
            p0 = PER_LAYER * layer
            # MLP: dx_bf is bf16(d residual), the down projection's output grad.
            dgl = dx_bf @ w2
            grads[p0 + _W2] = _wgrad(dx_bf.view(rows, -1), sv["gl"], streams)
            grads[p0 + _B2] = down_bias
            du, grads[p0 + _B1] = self._gelu_backward(dgl, sv["u"], streams)
            dm = du @ w1
            grads[p0 + _W1] = _wgrad(du.view(rows, -1), sv["m"], streams)
            dx1, dx1_bf, sums = self._ln_backward(
                dm, sv["x1"], sv["stats2"], params[p0 + _LN2_W], streams,
                dres=dx, want_bf=True,
            )
            grads[p0 + _LN2_W], grads[p0 + _LN2_B] = sums[:, 0], sums[:, 1]
            # Attention.
            do = dx1_bf @ wout
            grads[p0 + _WOUT] = _wgrad(dx1_bf.view(rows, -1), sv["o"], streams)
            dqkv = self._attention_backward(
                sv["qkv"], sv["o"], sv["lse"], do.view(rows, streams, -1),
                sv["ve_off"], dvalues, saved.bounds, saved.seg_len,
            ).view(rows * streams, -1)
            da = dqkv @ wqkv
            grads[p0 + _WQKV] = _wgrad(dqkv.view(rows, -1), sv["a"], streams)
            first = layer == 0
            dx, dx_bf, sums = self._ln_backward(
                da, sv["x"], sv["stats1"], params[p0 + _LN1_W], streams,
                dres=dx1, want_bf=not first,
                dple=None if first else dple, layer=layer - 1,
            )
            grads[p0 + _LN1_W], grads[p0 + _LN1_B] = sums[:, 0], sums[:, 1]
            down_bias = sums[:, 2]
        return (
            dx.view(*lead, streams, self.dm),
            None if dple is None else dple.view(streams, *lead, self.layers, self.ple_dim),
            None if dvalues is None else dvalues.view(
                *lead, streams, len(self.value_index), self.kv_width
            ),
            grads,
        )

    # -- cached decode (rollout) ----------------------------------------

    def decode(self, x0, ple, values, cache, positions, trunk, cache_rows=None):
        """One merged-wave decode step through every layer.

        x0 [B, T, DM] embeddings of the appended tokens, ple/values their
        conditioning, ``positions`` [B] each row's write position for token 0.
        Writes the new K/V into ``cache`` (fp16; rows 0..B-1, or ``cache_rows``
        [B] int64) and returns the final-norm output for every appended token,
        [B, T, DM] fp32.
        """

        params = trunk_parameters(trunk)
        batch, steps, _ = x0.shape
        rows = batch * steps
        x = x0.reshape(rows, self.dm).float().contiguous()
        ple_flat = (
            ple.reshape(rows, -1).float().contiguous() if ple is not None and self.ple_dim else None
        )
        ve = (
            values.reshape(rows, -1).float().contiguous()
            if values is not None and self.value_index
            else None
        )
        positions = positions.to(torch.int64).contiguous()
        if cache_rows is not None:
            cache_rows = cache_rows.to(torch.int64).contiguous()
        cast = self._bf16_weights(params)
        _, a, _ = self._add_ln(rows, params[_LN1_W], params[_LN1_B], x=x)
        for layer in range(self.layers):
            p = params[PER_LAYER * layer : PER_LAYER * (layer + 1)]
            wqkv, wout, w1, w2, b1, b2 = cast[layer]
            qkv = F.linear(a, wqkv)
            index = self.value_index.get(layer)
            ve_off = -1 if index is None or ve is None else index * self.kv_width
            o = torch.empty(rows, self.n_heads * self.head_dim, device=x.device,
                            dtype=torch.bfloat16)
            group = self.n_heads // self.kv_heads
            self.lib.attn_decode(
                qkv, ve if ve_off >= 0 else x, cache.k[layer], cache.v[layer],
                positions, o, batch, steps, cache.max_len,
                ve.shape[-1] if ve_off >= 0 else 0, ve_off,
                self.scale * math.log2(math.e),
                positions if cache_rows is None else cache_rows, int(cache_rows is not None),
                threads=batch * self.kv_heads * group * steps, group_size=256,
            )
            x1, m, _ = self._add_ln(
                rows, p[_LN2_W], p[_LN2_B], x=x, delta=F.linear(o, wout), write_x=True
            )
            down = F.linear(F.gelu(F.linear(m, w1, b1)), w2, b2)
            last = layer == self.layers - 1
            base = PER_LAYER * (self.layers if last else layer + 1)
            x, a, _ = self._add_ln(
                rows, params[base], params[base + 1], x=x1, delta=down,
                ple=ple_flat, layer=layer, write_x=True, fp32_out=last,
            )
        return a.view(batch, steps, self.dm)

    def decode_supported(self, cache) -> bool:
        return (
            cache.dtype == torch.float16
            and all(k.is_contiguous() for k in cache.k)
            and all(v.is_contiguous() for v in cache.v)
        )


class _MetalTrunkFunction(torch.autograd.Function):
    """Single-cotangent autograd wrapper: a drop-in for the PyTorch trunk."""

    @staticmethod
    def forward(ctx, trunk: MetalTrunk, bounds, x0, ple, values, *params):
        hidden, saved = trunk.forward(x0, ple, values, params, save=True, bounds=bounds)
        ctx.trunk = trunk
        ctx.saved = saved
        ctx.params = params
        ctx.has_ple = ple is not None
        ctx.has_values = values is not None
        return hidden

    @staticmethod
    def backward(ctx, dh):
        dx0, dple, dvalues, grads = ctx.trunk.backward(ctx.saved, dh.unsqueeze(-2), ctx.params)
        lead = len(ctx.saved.lead)
        ctx.saved = None
        return (
            None,
            None,
            dx0.select(-2, 0),
            dple[0] if ctx.has_ple and dple is not None else None,
            dvalues.select(lead, 0) if ctx.has_values and dvalues is not None else None,
            *(g[0] for g in grads),
        )


def run_trunk(trunk, x: torch.Tensor, ple, values, bounds: torch.Tensor | None = None) -> torch.Tensor:
    """``Trunk.forward``'s fused branch: conditioned inputs in, hidden out.

    ``x`` is padded [B, L, d], or packed [N, d] with ``bounds`` (see
    :class:`cardgym.nn.packing.Packing`).
    """

    params = trunk_parameters(trunk)
    if torch.is_grad_enabled() and (x.requires_grad or any(p.requires_grad for p in params)):
        return _MetalTrunkFunction.apply(trunk.metal, bounds, x, ple, values, *params)
    with torch.no_grad():
        hidden, _ = trunk.metal.forward(x, ple, values, params, save=False, bounds=bounds)
    return hidden


class DualTrunkPass:
    """One trunk forward whose backward serves several losses at once.

    ``hidden`` is a leaf: losses built on it reach the heads through ordinary
    autograd, the trunk backward runs once for every loss together, and the
    embedding graph below the trunk is replayed per loss. ``gradients``
    returns exactly what one ``torch.autograd.grad(loss, parameters,
    allow_unused=True)`` per loss would, up to bf16 rounding.

    With ``bounds`` the tokens are packed rows [N, W] (see
    cardgym.nn.packing.Packing), and ``hidden`` and ``context`` are [N, ...].
    """

    def __init__(self, model, tokens: torch.Tensor, bounds: torch.Tensor | None = None) -> None:
        self.trunk = model.trunk
        self.x0, self.ple, self.values = model.prepare(tokens)
        # A policy that reads per-position features of the embedding (the
        # model's pop_context) gets them as a leaf too, so each loss's
        # embedding gradient is one replay rather than a head pass plus a replay.
        context = model.pop_context() if hasattr(model, "pop_context") else None
        self.embedded_context = context
        self.context = None if context is None else context.detach().requires_grad_()
        self.params = trunk_parameters(self.trunk)
        with torch.no_grad():
            hidden, self.saved = self.trunk.metal.forward(
                self.x0, self.ple, self.values, self.params, save=True, bounds=bounds
            )
        self.hidden = hidden.requires_grad_()

    def gradients(
        self,
        losses: Sequence[torch.Tensor | None],
        parameters: Sequence[torch.Tensor],
    ) -> list[list[torch.Tensor | None] | None]:
        trunk_slot = {id(p): index for index, p in enumerate(self.params)}
        others = [p for p in parameters if id(p) not in trunk_slot]
        active = [
            index for index, loss in enumerate(losses)
            if loss is not None and loss.requires_grad
        ]
        results: list[list[torch.Tensor | None] | None] = [None] * len(losses)
        if not active:
            return results
        leaves = [self.hidden] + ([self.context] if self.context is not None else [])
        head = []
        for position, index in enumerate(active):
            head.append(torch.autograd.grad(
                losses[index], [*leaves, *others],
                retain_graph=position + 1 < len(active), allow_unused=True,
            ))
        dcontext = [g[1] if self.context is not None else None for g in head]
        head = [(g[0], *g[len(leaves):]) for g in head]
        dh = torch.stack(
            [g[0] if g[0] is not None else torch.zeros_like(self.hidden) for g in head],
            dim=-2,
        )
        lead = len(self.saved.lead)
        dx0, dple, dvalues, trunk_grads = self.trunk.metal.backward(self.saved, dh, self.params)
        self.saved = None
        outputs = [self.x0]
        if self.ple is not None and dple is not None:
            outputs.append(self.ple)
        if self.values is not None and dvalues is not None:
            outputs.append(self.values)
        reachable = [o.requires_grad for o in outputs]
        for position, index in enumerate(active):
            cotangents = [dx0.select(-2, position)]
            if self.ple is not None and dple is not None:
                cotangents.append(dple[position])
            if self.values is not None and dvalues is not None:
                cotangents.append(dvalues.select(lead, position))
            pairs = [(o, c) for o, c, r in zip(outputs, cotangents, reachable) if r]
            if dcontext[position] is not None and self.embedded_context.requires_grad:
                pairs.append((self.embedded_context, dcontext[position]))
            embedding = (
                torch.autograd.grad(
                    [o for o, _ in pairs], others, [c for _, c in pairs],
                    retain_graph=position + 1 < len(active), allow_unused=True,
                )
                if pairs and others
                else (None,) * len(others)
            )
            other_grads = {}
            for p, from_head, from_embedding in zip(others, head[position][1:], embedding):
                if from_head is None:
                    other_grads[id(p)] = from_embedding
                elif from_embedding is None:
                    other_grads[id(p)] = from_head
                else:
                    other_grads[id(p)] = from_head + from_embedding
            row: list[torch.Tensor | None] = []
            for p in parameters:
                slot = trunk_slot.get(id(p))
                row.append(trunk_grads[slot][position] if slot is not None else other_grads[id(p)])
            results[index] = row
        return results
