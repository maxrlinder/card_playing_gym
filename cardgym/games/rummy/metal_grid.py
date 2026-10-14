"""Fused Metal kernels for the card grid (MPS, bf16 autocast).

The PyTorch grid spends its time in narrow per-card operations: RMSNorm over
32 features, GELU, and a 64 -> 32 projection over [N * 52] rows, each a full
pass over a large tensor (a 32-wide RMSNorm forward+backward measured 20-30 ms
per 16k positions; the block-circulant GEMM that does the actual convolution
takes 2 ms). Here the GEMMs stay on MPS and everything between them is fused:

    forward    embed + norm | circulant GEMM | pool, across, GELU, project,
               residual, next norm | circulant GEMM | ... | final norm
    backward   final-norm bwd | per block: GELU recompute, output and across
               projection GEMMs around a GELU-backward kernel | circulant
               GEMM data and weight grads | norm bwd (+ pooled grad, + residual)

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
// With save, pooled and across [N, 13, H] also go out in bf16 for the
// backward's GEMMs; pooled rows carry W + 4 columns, the last four 1, 0, 0, 0,
// so the across projection's weight-gradient GEMM also yields the
// convolution bias gradient (its sum over rows).
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
    device bfloat* pooled_out [[buffer(14)]],
    device bfloat* across_out [[buffer(15)]],
    constant int& save [[buffer(16)]],
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
            if (save) pooled_out[((long)n * R + r) * (W + 4) + f] = bfloat(pooled[i]);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = tid; i < R * H; i += TT) {{
            const int r = i / H, o = i % H;
            float acc = 0.0f;
            for (int f = 0; f < W; ++f) acc = fma(wa[o * W + f], pooled[r * W + f], acc);
            across[i] = acc;
            if (save) across_out[(long)n * R * H + i] = bfloat(acc);
        }}
        if (save) {{
            for (int i = tid; i < R * 4; i += TT) {{
                pooled_out[((long)n * R + i / 4) * (W + 4) + W + i % 4] = bfloat(i % 4 == 0 ? 1.0f : 0.0f);
            }}
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

// The block's activation act = GELU(along + across) in bf16, recomputed in
// the backward for the output projection's weight gradient. Rows carry H + 4
// columns, the last four 1, 0, 0, 0: the same GEMM then also yields the
// output bias gradient (the column sums of its output gradient).
kernel void grid_act(
    device const bfloat* along [[buffer(0)]],
    device const bfloat* across [[buffer(1)]],
    device bfloat* act [[buffer(2)]],
    constant int& cards [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{{
    const long card = gid / (H + 4);
    if (card >= cards) return;
    const int o = gid % (H + 4);
    if (o >= H) {{ act[gid] = bfloat(o == H ? 1.0f : 0.0f); return; }}
    const long n = card / C;
    const int r = (card % C) % R;
    act[gid] = bfloat(gelu(float(along[card * H + o]) + float(across[(n * R + r) * H + o])));
}}

// GELU backward on one (position, rank, channel) for all four suits:
// d_along = d_act * GELU'(along + across) [N * 52, H] bf16 (the circulant
// GEMM's output gradient), and its sum over suits, d_across [N * 13, H] bf16,
// the across projection's output gradient.
kernel void grid_act_bwd(
    device const bfloat* d_act [[buffer(0)]],
    device const bfloat* along [[buffer(1)]],
    device const bfloat* across [[buffer(2)]],
    device bfloat* d_along [[buffer(3)]],
    device bfloat* d_across_bf [[buffer(4)]],
    constant int& N [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{{
    const long nr = gid / H;
    if (nr >= (long)N * R) return;
    const int o = gid % H;
    const long n = nr / R;
    const int r = nr % R;
    const float ac = float(across[gid]);
    float sum = 0.0f;
    for (int s = 0; s < S; ++s) {{
        const long at = (n * C + s * R + r) * H + o;
        const float dh = float(d_act[at]) * gelu_grad(float(along[at]) + ac);
        d_along[at] = bfloat(dh);
        sum += dh;
    }}
    d_across_bf[gid] = bfloat(sum);
}}

// RMSNorm backward, one simdgroup per card (lane = feature), grid-stride.
// dy is bf16 (flags & 1) or fp32; flags & 2 adds d_pooled [N, 13, W] / 4
// (bf16); flags & 4 adds the residual gradient dres; flags & 8 also writes a
// bf16 copy of dx (the next block's GEMMs read it). Per-threadgroup gain partials.
kernel void grid_norm_bwd(
    device const bfloat* dy_bf [[buffer(0)]],
    device const float* dy_f [[buffer(1)]],
    device const bfloat* d_pooled [[buffer(2)]],
    device const float* x [[buffer(3)]],
    device const float* rstd [[buffer(4)]],
    device const float* gain [[buffer(5)]],
    device const float* dres [[buffer(6)]],
    device float* dx [[buffer(7)]],
    device float* partial [[buffer(8)]],
    constant int& cards [[buffer(9)]],
    constant int& flags [[buffer(10)]],
    device bfloat* dx_bf [[buffer(11)]],
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
            d += 0.25f * float(d_pooled[(n * R + (card % C) % R) * W + lane]);
        }}
        const float rs = rstd[card];
        const float xh = x[at] * rs;
        dg += d * xh;
        const float gd = d * g;
        const float m = simd_sum(gd * xh) / W;
        float v = rs * (gd - xh * m);
        if (flags & 4) v += dres[at];
        dx[at] = v;
        if (flags & 8) dx_bf[at] = bfloat(v);
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
    final norm gain. ``grid`` (the CardGrid) builds the circulant matrices and
    the bf16 projection weights. ``save`` keeps what the backward needs.

    The backward runs each block's small matrix products as GEMMs on MPS
    (output projection data and weight gradients, the across projection's)
    between two elementwise kernels; the scalar-FMA fused tail backward this
    replaces measured 14 ms per block at 20k positions, the GEMM form ~4 ms.
    """

    @staticmethod
    def forward(ctx, grid, ledger, save, *params):
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
            pooled = torch.empty(rows * 13, width + 4, device=dev, dtype=torch.bfloat16) if save else y
            across = torch.empty(rows * 13, hidden, device=dev, dtype=torch.bfloat16) if save else y
            lib.grid_tail(
                along, y, x, wa, wo, bo, next_gain, x_out, y_out, x, rstd_out, rows, 0, eps,
                pooled, across, int(save),
                threads=min(rows, 4096) * _TAIL_THREADS, group_size=_TAIL_THREADS,
            )
            if save:
                saved.append((x, y, rstd, along, pooled, across))
                circulants.append(circulant)
            x, rstd, y = x_out, rstd_out, y_out
            if last:
                cards_out = y_out
        if save:
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
        dummy = torch.empty(1, device=dev, dtype=torch.bfloat16)
        dx = torch.empty(cards, width, device=dev)
        dx_bf = torch.empty(cards, width, device=dev, dtype=torch.bfloat16)
        partial = torch.empty(_NORM_THREADGROUPS, width, device=dev)
        lib.grid_norm_bwd(
            dummy, d_cards, dummy, x_final, rstd_final, params[-1], d_cards, dx, partial, cards, 8, dx_bf,
            **norm_threads,
        )
        grads[-1] = partial.sum(0)
        for index in reversed(range(len(blocks))):
            gain = blocks[index][0]
            block = grid.blocks[index]
            wa_bf, wo_bf = block.projection_operands(torch.bfloat16)
            x, y, rstd, along, pooled, across = ctx.blocks_saved[index]
            base = 2 + 6 * index
            # Output projection: out = act @ wo^T + bo (act's extra ones column gives bo's gradient).
            act = torch.empty(cards, hidden + 4, device=dev, dtype=torch.bfloat16)
            lib.grid_act(along, across, act, cards, threads=cards * (hidden + 4), group_size=256)
            d_act = dx_bf @ wo_bf
            weights = _wgrad(dx_bf, act)
            grads[base + 4] = weights[:, :hidden].contiguous()
            grads[base + 5] = weights[:, hidden].contiguous()
            # GELU, then the across projection: across = pooled @ wa^T.
            d_along = torch.empty(cards, hidden, device=dev, dtype=torch.bfloat16)
            d_across_bf = torch.empty(rows * 13, hidden, device=dev, dtype=torch.bfloat16)
            lib.grid_act_bwd(
                d_act, along, across, d_along, d_across_bf, rows,
                threads=rows * 13 * hidden, group_size=256,
            )
            # pooled's extra ones column gives the convolution bias gradient
            # (the bias feeds every card and channel of along).
            weights = _wgrad(d_across_bf, pooled)
            grads[base + 3] = weights[:, :width].contiguous()
            grads[base + 2] = weights[:, width].contiguous()
            d_pooled = d_across_bf @ wa_bf
            # along = y_rows @ circulant (+ bias): data and weight gradients.
            along_rows = d_along.view(rows * 4, 13 * hidden)
            y_rows = y.view(rows * 4, 13 * width)
            dy = (along_rows @ ctx.circulants[index].t()).contiguous()
            grads[base + 1] = block.conv_grad(_wgrad(along_rows, y_rows).t())
            dx_prev = torch.empty(cards, width, device=dev)
            dx_bf = torch.empty(cards, width, device=dev, dtype=torch.bfloat16) if index else dummy
            partial = torch.empty(_NORM_THREADGROUPS, width, device=dev)
            lib.grid_norm_bwd(
                dy, d_cards, d_pooled, x, rstd, gain, dx, dx_prev, partial, cards,
                1 | 2 | 4 | (8 if index else 0), dx_bf,
                **norm_threads,
            )
            grads[base] = partial.sum(0)
            dx = dx_prev
        grads[0] = torch.zeros_like(params[0]).index_add_(0, ledger.reshape(-1), dx)
        ones = torch.ones(1, rows * 4, device=dev)
        grads[1] = (ones @ dx.view(rows * 4, 13 * width)).view(13, width)
        return (None, None, None, *grads)
