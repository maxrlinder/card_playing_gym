"""Fused Metal kernels for the card grid (MPS, bf16 autocast).

The PyTorch grid spends its time in narrow per-card operations: RMSNorm over
32 features, GELU, and a 64 -> 32 projection over [N * 52] rows, each a full
pass over a large tensor (a 32-wide RMSNorm forward+backward measured 20-30 ms
per 16k positions; the block-circulant GEMM that does the actual convolution
takes 2 ms). Here the GEMMs stay on MPS and everything between them is fused:

    forward    embed + norm | circulant GEMM | pool, across, GELU, project,
               residual, next norm | circulant GEMM | ... | final norm
    backward   final-norm bwd | per block: tail bwd (projection, GELU,
               across; weight partials per threadgroup) | GEMM data and
               weight grads | norm bwd (+ pooled grad, + residual)

Precision follows the autocast path where it rounds: the residual stream and
norm statistics are fp32, GEMM operands bf16. The fused tail keeps GELU and
the projection in fp32 (autocast rounds both to bf16), so it matches an fp32
reference slightly better than PyTorch's own bf16 run does.

Layouts: card c = suit * 13 + rank; per-position tensors are [N, 52, W]
(row-major), which is also [N * 4, 13 * W] suit rows for the GEMM.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_TAIL_THREADS = 256
_BWD_THREADGROUPS = 512
_NORM_THREADGROUPS = 256
_NORM_SIMDGROUPS = 8


def _source(width: int, hidden: int) -> str:
    return f"""
#include <metal_stdlib>
using namespace metal;

constant constexpr int W = {width};
constant constexpr int H = {hidden};
constant constexpr int C = 52;
constant constexpr int R = 13;
constant constexpr int S = 4;
constant constexpr int TT = {_TAIL_THREADS};
constant constexpr int WO_PER = (W * H + TT - 1) / TT;

inline float erf_approx(float x) {{
    const float t = 1.0f / (1.0f + 0.3275911f * fabs(x));
    const float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t
                     - 0.284496736f) * t + 0.254829592f) * t * exp(-x * x);
    return copysign(y, x);
}}
inline float gelu(float x) {{ return 0.5f * x * (1.0f + erf_approx(x * M_SQRT1_2_F)); }}
inline float gelu_grad(float x) {{
    return 0.5f * (1.0f + erf_approx(x * M_SQRT1_2_F)) + x * exp(-0.5f * x * x) * 0.3989422804014327f;
}}

// Card embedding + RMSNorm: one simdgroup per card, lane = feature.
// tokens [rows, stride] int64, ledger codes at columns offset .. offset + 51.
kernel void grid_embed(
    device const long* tokens [[buffer(0)]],
    device const float* code_w [[buffer(1)]],
    device const float* rank_w [[buffer(2)]],
    device const float* gain [[buffer(3)]],
    device float* x [[buffer(4)]],
    device bfloat* y [[buffer(5)]],
    device float* rstd [[buffer(6)]],
    constant int& cards [[buffer(7)]],
    constant int& stride [[buffer(8)]],
    constant int& offset [[buffer(9)]],
    constant float& eps [[buffer(10)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{{
    const int card = gid / 32;
    if (card >= cards) return;
    const long row = card / C;
    const int c = card % C;
    const long code = tokens[row * stride + offset + c];
    const float v = code_w[code * W + lane] + rank_w[(c % R) * W + lane];
    const float rs = rsqrt(simd_sum(v * v) / W + eps);
    x[(long)card * W + lane] = v;
    y[(long)card * W + lane] = bfloat(v * rs * gain[lane]);
    if (lane == 0) rstd[card] = rs;
}}

// Block tail, one threadgroup per position: pool y over suits, project
// across, GELU(along + across), project back, add the residual, and normalize
// for the next stage (bf16 for the next GEMM, or fp32 cards when final).
kernel void grid_tail(
    device const bfloat* along [[buffer(0)]],
    device const bfloat* y [[buffer(1)]],
    device const float* x [[buffer(2)]],
    device const float* wa [[buffer(3)]],
    device const float* wo [[buffer(4)]],
    device const float* bo [[buffer(5)]],
    device const float* gain [[buffer(6)]],
    device float* x_out [[buffer(7)]],
    device bfloat* y_bf [[buffer(8)]],
    device float* y_f [[buffer(9)]],
    device float* rstd [[buffer(10)]],
    constant int& N [[buffer(11)]],
    constant int& final_out [[buffer(12)]],
    constant float& eps [[buffer(13)]],
    uint tg [[threadgroup_position_in_grid]],
    uint ntg [[threadgroups_per_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{{
    threadgroup float pooled[R * W];
    threadgroup float across[R * H];
    threadgroup float act[C * H];
    threadgroup float wot[H * W];
    for (int i = tid; i < W * H; i += TT) {{ const int j = i / H, o = i % H; wot[o * W + j] = wo[i]; }}
    for (int n = tg; n < N; n += ntg) {{
        const long cbase = (long)n * C;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = tid; i < R * W; i += TT) {{
            const int r = i / W, f = i % W;
            float sum = 0.0f;
            for (int s = 0; s < S; ++s) sum += float(y[(cbase + s * R + r) * W + f]);
            pooled[i] = sum * (1.0f / S);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = tid; i < R * H; i += TT) {{
            const int r = i / H, o = i % H;
            float acc = 0.0f;
            for (int f = 0; f < W; ++f) acc = fma(wa[o * W + f], pooled[r * W + f], acc);
            across[i] = acc;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = tid; i < C * H; i += TT) {{
            const int c = i / H, o = i % H;
            act[i] = gelu(float(along[(cbase + c) * H + o]) + across[(c % R) * H + o]);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int c = sg; c < C; c += TT / 32) {{
            float acc = bo[lane];
            for (int o = 0; o < H; ++o) acc = fma(act[c * H + o], wot[o * W + lane], acc);
            const long at = (cbase + c) * W + lane;
            const float v = x[at] + acc;
            x_out[at] = v;
            const float rs = rsqrt(simd_sum(v * v) / W + eps);
            const float yv = v * rs * gain[lane];
            if (final_out) y_f[at] = yv; else y_bf[at] = bfloat(yv);
            if (lane == 0) rstd[cbase + c] = rs;
        }}
    }}
}}

// Block tail backward. dxo [N, 52, W] fp32 is the gradient at the block's
// output (residual stream). Writes d_along [N, 52, H] bf16 (the GEMM's output
// gradient) and d_pooled [N, 13, W] fp32, and per-threadgroup partials of
// dWo [W, H], dbo [W], dWa [H, W] and the convolution bias gradient [H].
kernel void grid_tail_bwd(
    device const float* dxo [[buffer(0)]],
    device const bfloat* along [[buffer(1)]],
    device const bfloat* y [[buffer(2)]],
    device const float* wa [[buffer(3)]],
    device const float* wo [[buffer(4)]],
    device bfloat* d_along [[buffer(5)]],
    device float* d_pooled [[buffer(6)]],
    device float* partial [[buffer(7)]],
    constant int& N [[buffer(8)]],
    uint tg [[threadgroup_position_in_grid]],
    uint ntg [[threadgroups_per_grid]],
    uint tid [[thread_position_in_threadgroup]])
{{
    threadgroup float dx_s[C * W];
    threadgroup bfloat act[C * H];
    threadgroup float wot[H * W];
    threadgroup float pooled[R * W];
    threadgroup float across[R * H];  // across, then its gradient
    for (int i = tid; i < W * H; i += TT) {{ const int j = i / H, o = i % H; wot[o * W + j] = wo[i]; }}
    float dwo[WO_PER], dwa[WO_PER];
    for (int k = 0; k < WO_PER; ++k) {{ dwo[k] = 0.0f; dwa[k] = 0.0f; }}
    float dbo = 0.0f, dbias = 0.0f;
    for (int n = tg; n < N; n += ntg) {{
        const long cbase = (long)n * C;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = tid; i < C * W; i += TT) dx_s[i] = dxo[cbase * W + i];
        for (int i = tid; i < R * W; i += TT) {{
            const int r = i / W, f = i % W;
            float sum = 0.0f;
            for (int s = 0; s < S; ++s) sum += float(y[(cbase + s * R + r) * W + f]);
            pooled[i] = sum * (1.0f / S);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = tid; i < R * H; i += TT) {{
            const int r = i / H, o = i % H;
            float acc = 0.0f;
            for (int f = 0; f < W; ++f) acc = fma(wa[o * W + f], pooled[r * W + f], acc);
            across[i] = acc;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = tid; i < C * H; i += TT) {{
            const int c = i / H, o = i % H;
            act[i] = bfloat(gelu(float(along[(cbase + c) * H + o]) + across[(c % R) * H + o]));
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Output projection: weight and bias partials.
        for (int k = 0; k < WO_PER; ++k) {{
            const int e = tid + k * TT;
            if (e < W * H) {{
                const int j = e / H, o = e % H;
                float acc = 0.0f;
                for (int c = 0; c < C; ++c) acc = fma(dx_s[c * W + j], float(act[c * H + o]), acc);
                dwo[k] += acc;
            }}
        }}
        if (tid < W) {{
            float acc = 0.0f;
            for (int c = 0; c < C; ++c) acc += dx_s[c * W + tid];
            dbo += acc;
        }}
        // GELU input gradient per (rank, channel), summed over suits into the
        // across gradient; each (r, o) belongs to one thread.
        float dacc[(R * H + TT - 1) / TT];
        for (int k = 0; k * TT + (int)tid < R * H; ++k) {{
            const int i = tid + k * TT;
            const int r = i / H, o = i % H;
            const float ac = across[i];
            float sum = 0.0f;
            for (int s = 0; s < S; ++s) {{
                const int c = s * R + r;
                float da = 0.0f;
                for (int j = 0; j < W; ++j) da = fma(wot[o * W + j], dx_s[c * W + j], da);
                const float dh = da * gelu_grad(float(along[(cbase + c) * H + o]) + ac);
                d_along[(cbase + c) * H + o] = bfloat(dh);
                sum += dh;
            }}
            dacc[k] = sum;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int k = 0; k * TT + (int)tid < R * H; ++k) across[tid + k * TT] = dacc[k];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Across projection: weight partials, bias (sum of every GELU-input
        // gradient), and the pooled gradient.
        for (int k = 0; k < WO_PER; ++k) {{
            const int e = tid + k * TT;
            if (e < W * H) {{
                const int o = e / W, f = e % W;
                float acc = 0.0f;
                for (int r = 0; r < R; ++r) acc = fma(across[r * H + o], pooled[r * W + f], acc);
                dwa[k] += acc;
            }}
        }}
        if (tid < H) {{
            float acc = 0.0f;
            for (int r = 0; r < R; ++r) acc += across[r * H + tid];
            dbias += acc;
        }}
        for (int i = tid; i < R * W; i += TT) {{
            const int r = i / W, f = i % W;
            float acc = 0.0f;
            for (int o = 0; o < H; ++o) acc = fma(wa[o * W + f], across[r * H + o], acc);
            d_pooled[((long)n * R + r) * W + f] = acc;
        }}
    }}
    device float* out = partial + (long)tg * (2 * W * H + W + H);
    for (int k = 0; k < WO_PER; ++k) {{
        const int e = tid + k * TT;
        if (e < W * H) {{ out[e] = dwo[k]; out[W * H + e] = dwa[k]; }}
    }}
    if (tid < W) out[2 * W * H + tid] = dbo;
    if (tid < H) out[2 * W * H + W + tid] = dbias;
}}

// RMSNorm backward, one simdgroup per card (lane = feature), grid-stride.
// dy is bf16 (flags & 1) or fp32; flags & 2 adds d_pooled [N, 13, W] / 4;
// flags & 4 adds the residual gradient dres. Per-threadgroup gain partials.
kernel void grid_norm_bwd(
    device const bfloat* dy_bf [[buffer(0)]],
    device const float* dy_f [[buffer(1)]],
    device const float* d_pooled [[buffer(2)]],
    device const float* x [[buffer(3)]],
    device const float* rstd [[buffer(4)]],
    device const float* gain [[buffer(5)]],
    device const float* dres [[buffer(6)]],
    device float* dx [[buffer(7)]],
    device float* partial [[buffer(8)]],
    constant int& cards [[buffer(9)]],
    constant int& flags [[buffer(10)]],
    uint tg [[threadgroup_position_in_grid]],
    uint ntg [[threadgroups_per_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint nsg [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{{
    threadgroup float acc[32 * W];
    float dg = 0.0f;
    const float g = gain[lane];
    for (int card = tg * nsg + sg; card < cards; card += ntg * nsg) {{
        const long at = (long)card * W + lane;
        float d = (flags & 1) ? float(dy_bf[at]) : dy_f[at];
        if (flags & 2) {{
            const long n = card / C;
            d += 0.25f * d_pooled[(n * R + (card % C) % R) * W + lane];
        }}
        const float rs = rstd[card];
        const float xh = x[at] * rs;
        dg += d * xh;
        const float gd = d * g;
        const float m = simd_sum(gd * xh) / W;
        float v = rs * (gd - xh * m);
        if (flags & 4) v += dres[at];
        dx[at] = v;
    }}
    acc[sg * W + lane] = dg;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {{
        float sum = 0.0f;
        for (uint k = 0; k < nsg; ++k) sum += acc[k * W + lane];
        partial[(long)tg * W + lane] = sum;
    }}
}}
"""


_LIBRARIES: dict[tuple[int, int], object] = {}


def _library(width: int, hidden: int):
    key = (width, hidden)
    if key not in _LIBRARIES:
        _LIBRARIES[key] = torch.mps.compile_shader(_source(width, hidden))
    return _LIBRARIES[key]


def supported(width: int, hidden: int) -> bool:
    return width == 32 and hidden % 32 == 0 and hidden <= 128


def _wgrad(dy: torch.Tensor, x: torch.Tensor, splits: int = 8) -> torch.Tensor:
    """dy [R, M]^T @ x [R, K] -> [M, K] fp32 as a split-K batch (MPS long-K GEMMs are slow)."""

    rows = dy.shape[0]
    main = (rows // splits) * splits
    out = torch.bmm(
        dy[:main].view(splits, main // splits, -1).transpose(1, 2),
        x[:main].view(splits, main // splits, -1),
    ).sum(0, dtype=torch.float32)
    if main < rows:
        out += (dy[main:].t() @ x[main:]).float()
    return out


class FusedCardGrid(torch.autograd.Function):
    """ledger [rows, 52] int64 -> per-card features [rows, 52, W] bf16.

    ``params``: code embedding, rank embedding, then per block (norm gain,
    conv weight, conv bias, across weight, out weight, out bias), then the
    final norm gain. ``grid`` (the CardGrid) builds the circulant matrices.
    """

    @staticmethod
    def forward(ctx, grid, ledger, *params):
        lib = _library(grid.width, grid.hidden)
        code_w, rank_w = params[0], params[1]
        # Convolution weights and biases enter through grid.blocks[i].gemm_operands.
        blocks = [params[2 + 6 * b : 8 + 6 * b] for b in range(len(grid.blocks))]
        final_gain = params[-1]
        rows = ledger.shape[0]
        cards = rows * 52
        width, hidden, eps = grid.width, grid.hidden, grid.eps
        dev = ledger.device
        x = torch.empty(cards, width, device=dev)
        y = torch.empty(cards, width, device=dev, dtype=torch.bfloat16)
        rstd = torch.empty(cards, device=dev)
        lib.grid_embed(
            ledger, code_w, rank_w, blocks[0][0], x, y, rstd, cards, 52, 0, eps,
            threads=cards * 32, group_size=256,
        )
        saved, circulants = [], []
        cards_out = None
        for index, (_, _, _, wa, wo, bo) in enumerate(blocks):
            circulant, bias = grid.blocks[index].gemm_operands(torch.bfloat16)
            along = F.linear(y.view(rows * 4, 13 * width), circulant.t(), bias).view(cards, hidden)
            last = index == len(blocks) - 1
            next_gain = final_gain if last else blocks[index + 1][0]
            x_out = torch.empty_like(x)
            rstd_out = torch.empty_like(rstd)
            # The final norm's output is the cards in bf16: every consumer is a
            # bf16 GEMM under autocast, which would round them the same way.
            y_out = torch.empty_like(y)
            lib.grid_tail(
                along, y, x, wa, wo, bo, next_gain, x_out, y_out, x, rstd_out, rows, 0, eps,
                threads=min(rows, 4096) * _TAIL_THREADS, group_size=_TAIL_THREADS,
            )
            saved.append((x, y, rstd, along))
            circulants.append(circulant)
            x, rstd, y = x_out, rstd_out, y_out
            if last:
                cards_out = y_out
        ctx.grid = grid
        ctx.save_for_backward(ledger, *params)
        ctx.blocks_saved = saved
        ctx.circulants = circulants
        ctx.final = (x, rstd)
        return cards_out.view(rows, 52, width)

    @staticmethod
    def backward(ctx, d_cards):
        grid = ctx.grid
        ledger, *params = ctx.saved_tensors
        lib = _library(grid.width, grid.hidden)
        width, hidden = grid.width, grid.hidden
        rows = ledger.shape[0]
        cards = rows * 52
        dev = ledger.device
        blocks = [params[2 + 6 * b : 8 + 6 * b] for b in range(len(grid.blocks))]
        grads: list[torch.Tensor | None] = [None] * len(params)
        norm_threads = dict(threads=_NORM_THREADGROUPS * _NORM_SIMDGROUPS * 32, group_size=_NORM_SIMDGROUPS * 32)
        x_final, rstd_final = ctx.final
        d_cards = d_cards.reshape(cards, width).float().contiguous()
        dx = torch.empty(cards, width, device=dev)
        partial = torch.empty(_NORM_THREADGROUPS, width, device=dev)
        lib.grid_norm_bwd(
            d_cards, d_cards, d_cards, x_final, rstd_final, params[-1], d_cards, dx, partial, cards, 0,
            **norm_threads,
        )
        grads[-1] = partial.sum(0)
        stride = 2 * width * hidden + width + hidden
        for index in reversed(range(len(blocks))):
            gain, _, _, wa, wo, _ = blocks[index]
            x, y, rstd, along = ctx.blocks_saved[index]
            d_along = torch.empty(cards, hidden, device=dev, dtype=torch.bfloat16)
            d_pooled = torch.empty(rows, 13, width, device=dev)
            groups = min(rows, _BWD_THREADGROUPS)
            tail = torch.empty(groups, stride, device=dev)
            lib.grid_tail_bwd(
                dx, along, y, wa, wo, d_along, d_pooled, tail, rows,
                threads=groups * _TAIL_THREADS, group_size=_TAIL_THREADS,
            )
            sums = tail.sum(0)
            base = 2 + 6 * index
            grads[base + 4] = sums[: width * hidden].view(width, hidden)
            grads[base + 3] = sums[width * hidden : 2 * width * hidden].view(hidden, width)
            grads[base + 5] = sums[2 * width * hidden : 2 * width * hidden + width]
            grads[base + 2] = sums[2 * width * hidden + width :]
            # along = y_rows @ circulant (+ bias): data and weight gradients.
            along_rows = d_along.view(rows * 4, 13 * hidden)
            y_rows = y.view(rows * 4, 13 * width)
            dy = (along_rows @ ctx.circulants[index].t()).contiguous()
            grads[base + 1] = grid.blocks[index].conv_grad(_wgrad(along_rows, y_rows).t())
            dx_prev = torch.empty(cards, width, device=dev)
            partial = torch.empty(_NORM_THREADGROUPS, width, device=dev)
            lib.grid_norm_bwd(
                dy, d_cards, d_pooled, x, rstd, gain, dx, dx_prev, partial, cards, 1 | 2 | 4,
                **norm_threads,
            )
            grads[base] = partial.sum(0)
            dx = dx_prev
        grads[0] = torch.zeros_like(params[0]).index_add_(0, ledger.reshape(-1), dx)
        ones = torch.ones(1, rows * 4, device=dev)
        grads[1] = (ones @ dx.view(rows * 4, 13 * width)).view(13, width)
        return (None, None, *grads)
