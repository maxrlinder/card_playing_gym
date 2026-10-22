"""The candidate scorer's inference forward as one Metal kernel (MPS, rollout).

:class:`~cardgym.games.rummy.model.CandidateScorer` runs as ~22 small
PyTorch operations per policy head (scatter, bmm, projections, embeddings,
LayerNorm, MLP). A rollout wave scores a few hundred decisions, so those
launches, not arithmetic, are its cost. Here one threadgroup scores one
decision's whole candidate list: thread ``o`` owns feature ``o`` of the
score width; the decision's query projection is computed once, then per
candidate the slot sums of card features, the projections, identity rows,
kind and size embeddings, LayerNorm, hidden layer, and output.

Inference only (no autograd); everything is fp32 from the bf16 card
features, so it agrees with the autocast PyTorch path to bf16 rounding.
Training keeps the PyTorch path, whose gradients it needs.
"""

from __future__ import annotations

import torch

from .cards import NO_CARD, NUM_CARDS

_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

constant constexpr int DIM = __DIM__;
constant constexpr int NSG = DIM / 32;
constant constexpr int CW = __CW__;  // card feature width
constant constexpr int NC = 52;
constant constexpr int NO_CARD = __NO_CARD__;

inline float erf_approx(float x) {
    const float t = 1.0f / (1.0f + 0.3275911f * fabs(x));
    const float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t
                     - 0.284496736f) * t + 0.254829592f) * t * exp(-x * x);
    return copysign(y, x);
}

inline float block_sum(float v, threadgroup float* red, uint sg, uint lane) {
    const float s = simd_sum(v);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) red[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (int i = 0; i < NSG; ++i) total += red[i];
    return total;
}

// One threadgroup (DIM threads) per decision row. hidden [n, d] fp32; grid
// [n, 52, CW] bf16; kind [n, K] int8; cards / target [n, K, MC] int8 (NO_CARD
// pads); taken [n, K, 52] int8 copies (draw head) when has_taken. Weights fp32:
// wq [DIM, d], bq; wp [DIM, 3 * CW] (moved, target, taken projections); ids
// [3 * 52, DIM]; kind_w; size_w; ln_w, ln_b; wh [DIM, DIM], bh; wout [DIM].
kernel void score(
    device const float* hidden [[buffer(0)]],
    device const bfloat* grid [[buffer(1)]],
    device const char* kind [[buffer(2)]],
    device const char* cards [[buffer(3)]],
    device const char* target [[buffer(4)]],
    device const char* taken [[buffer(5)]],
    device const float* wq [[buffer(6)]],
    device const float* bq [[buffer(7)]],
    device const float* wp [[buffer(8)]],
    device const float* ids [[buffer(9)]],
    device const float* kind_w [[buffer(10)]],
    device const float* size_w [[buffer(11)]],
    device const float* ln_w [[buffer(12)]],
    device const float* ln_b [[buffer(13)]],
    device const float* wh [[buffer(14)]],
    device const float* bh [[buffer(15)]],
    device const float* wout [[buffer(16)]],
    device float* logits [[buffer(17)]],
    constant int& K [[buffer(18)]],
    constant int& MC [[buffer(19)]],
    constant int& d [[buffer(20)]],
    constant int& has_taken [[buffer(21)]],
    constant float& bout [[buffer(22)]],
    constant float& eps [[buffer(23)]],
    uint row [[threadgroup_position_in_grid]],
    uint o [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float g[NC * CW];
    threadgroup float summed[3 * CW];
    threadgroup float mixed[DIM];
    threadgroup float red[NSG];
    const int slots = has_taken ? 3 : 2;
    for (int i = o; i < NC * CW; i += DIM) g[i] = float(grid[(long)row * NC * CW + i]);
    float q = bq[o];
    for (int j = 0; j < d; ++j) q = fma(hidden[(long)row * d + j], wq[o * d + j], q);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int k = 0; k < K; ++k) {
        const long cand = (long)row * K + k;
        device const char* moved = cards + cand * MC;
        device const char* aimed = target + cand * MC;
        device const char* took = taken + cand * NC;
        for (int i = o; i < slots * CW; i += DIM) {
            const int slot = i / CW, w = i % CW;
            float acc = 0.0f;
            if (slot < 2) {
                device const char* list = slot == 0 ? moved : aimed;
                for (int c = 0; c < MC; ++c) {
                    const int card = list[c];
                    if (card != NO_CARD) acc += g[card * CW + w];
                }
            } else {
                for (int c = 0; c < NC; ++c) {
                    const int copies = took[c];
                    if (copies) acc = fma(float(copies), g[c * CW + w], acc);
                }
            }
            summed[i] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float v = q + kind_w[int(kind[cand]) * DIM + o];
        for (int j = 0; j < slots * CW; ++j) v = fma(summed[j], wp[o * 3 * CW + j], v);
        int count = 0;
        for (int c = 0; c < MC; ++c) {
            const int card = moved[c];
            if (card != NO_CARD) { v += ids[card * DIM + o]; ++count; }
            const int aim = aimed[c];
            if (aim != NO_CARD) v += ids[(NC + aim) * DIM + o];
        }
        if (has_taken) {
            for (int c = 0; c < NC; ++c) {
                const int copies = took[c];
                if (copies) v = fma(float(copies), ids[(2 * NC + c) * DIM + o], v);
            }
        }
        v += size_w[count * DIM + o];
        const float mean = block_sum(v, red, sg, lane) / DIM;
        const float centered = v - mean;
        const float var = block_sum(centered * centered, red, sg, lane) / DIM;
        mixed[o] = centered * rsqrt(var + eps) * ln_w[o] + ln_b[o];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float h = bh[o];
        for (int i = 0; i < DIM; ++i) h = fma(mixed[i], wh[o * DIM + i], h);
        h = 0.5f * h * (1.0f + erf_approx(h * M_SQRT1_2_F));
        const float logit = block_sum(h * wout[o], red, sg, lane) + bout;
        if (o == 0) logits[cand] = logit;
    }
}
"""

_LIBRARIES: dict[tuple[int, int], object] = {}


def _library(dim: int, card_width: int):
    key = (dim, card_width)
    if key not in _LIBRARIES:
        source = (
            _SOURCE.replace("__DIM__", str(dim)).replace("__CW__", str(card_width))
            .replace("__NO_CARD__", str(NO_CARD))
        )
        _LIBRARIES[key] = torch.mps.compile_shader(source)
    return _LIBRARIES[key]


def supported(scorer) -> bool:
    dim = scorer.out.in_features
    return dim % 32 == 0 and dim <= 1024 and NUM_CARDS == 52


def _weights(scorer) -> dict[str, torch.Tensor]:
    """fp32 copies in the kernel's layouts, rebuilt only when a weight changes."""

    from cardgym.nn.metal_trunk import optimizer_steps

    params = list(scorer.parameters())
    key = (tuple(p._version for p in params), optimizer_steps(), params[0].device)
    if scorer._fused_key == key:
        return scorer._fused_weights
    with torch.no_grad():
        f = lambda t: t.detach().float().contiguous()  # noqa: E731
        weights = {
            "wq": f(scorer.query.weight), "bq": f(scorer.query.bias),
            "wp": f(torch.cat([scorer.card.weight, scorer.target.weight, scorer.taken.weight], dim=1)),
            "ids": f(torch.cat([scorer.card_id.weight, scorer.target_id.weight, scorer.taken_id.weight])),
            "kind": f(scorer.kind.weight), "size": f(scorer.size.weight),
            "ln_w": f(scorer.norm.weight), "ln_b": f(scorer.norm.bias),
            "wh": f(scorer.hidden.weight), "bh": f(scorer.hidden.bias),
            "wout": f(scorer.out.weight.view(-1)),
        }
        weights["bout"] = float(scorer.out.bias)
    scorer._fused_key, scorer._fused_weights = key, weights
    return weights


def score(scorer, hidden, grid, kind, cards, target, taken=None) -> torch.Tensor:
    """Logits [n, K] fp32, as ``scorer(hidden, grid, kind, cards, target, taken)`` computes them."""

    rows, width = kind.shape
    weights = _weights(scorer)
    dim = scorer.out.in_features
    logits = torch.empty(rows, width, device=hidden.device)
    if rows == 0 or width == 0:
        return logits
    dummy = kind if taken is None else taken
    _library(dim, grid.shape[-1]).score(
        hidden.float().contiguous(), grid.to(torch.bfloat16).contiguous(),
        kind.to(torch.int8).contiguous(), cards.to(torch.int8).contiguous(),
        target.to(torch.int8).contiguous(), dummy.to(torch.int8).contiguous(),
        weights["wq"], weights["bq"], weights["wp"], weights["ids"], weights["kind"], weights["size"],
        weights["ln_w"], weights["ln_b"], weights["wh"], weights["bh"], weights["wout"], logits,
        width, cards.shape[-1], hidden.shape[-1], int(taken is not None), weights["bout"], scorer.norm.eps,
        threads=rows * dim, group_size=dim,
    )
    return logits
