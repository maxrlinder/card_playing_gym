"""Ragged GQA decode attention for CUDA, straight from the KV cache.

A rollout wave appends T tokens to each of B cache rows (each at its own
length) and attends every new query to its row's prefix. The PyTorch path
gathers each row's keys and values out of the pool (a copy per layer), runs
two batched GEMMs of a few dozen rows each, and a masked softmax: on an
RTX 2070 Super 142 us a layer for 384 rows of 3 tokens, even replayed in a
CUDA graph. Triton kernels for it measured 73-86 us (below sm_80 its dot
and reductions go through layout conversions); this one, ~10 us.

One block per (row, KV head): the row's keys (padded rows, conflict-free
reads) and values go to shared memory once, then each warp takes one of the
group's G * T queries at a time. A lane scores two keys (lane, lane + 32),
the softmax max and sum are warp shuffles, the probabilities go through
shared memory, and lane d accumulates output dimension d. Scores, softmax,
and accumulation are fp32; K, V, Q, and the output are fp16. Head width 32
and spans up to 64 keys (Piquet: 32 and 60); other shapes take the PyTorch
path.

The extension builds on first use (torch.utils.cpp_extension, cached under
~/.cache/torch_extensions) and is registered as the custom op
``cardgym::decode_attention`` so that torch.compile treats it as opaque.
"""

from __future__ import annotations

from functools import lru_cache

import torch

HEAD_DIM = 32
MAX_SPAN = 64

_CUDA = r"""
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

namespace {

constexpr int D = 32;          // head width: one output dimension per lane
constexpr int KEYS = 64;       // keys per row: two per lane
constexpr int WARPS = 4;
constexpr int KPAD = D / 2 + 1; // half2 per padded key row: odd stride, conflict-free

__global__ void decode_attention_kernel(
    const half* __restrict__ q, const half* __restrict__ kc, const half* __restrict__ vc, half* __restrict__ out,
    const int64_t* __restrict__ rows, const int64_t* __restrict__ lengths,
    int64_t k_row, int64_t k_head, int64_t k_pos, int64_t v_row, int64_t v_head, int64_t v_pos,
    int heads, int steps, int group, int span, float scale)
{
    __shared__ half2 keys[KEYS * KPAD];
    __shared__ half2 values[KEYS * (D / 2)];
    __shared__ float probs[WARPS][KEYS];
    const int b = blockIdx.x, kv = blockIdx.y;
    const int64_t row = rows[b];
    const int length = static_cast<int>(lengths[b]);
    // Query t sees keys < length + t: none of the row's queries reads past this.
    const int visible = min(span, length + steps - 1);
    const half* kbase = kc + row * k_row + kv * k_head;
    const half* vbase = vc + row * v_row + kv * v_head;
    // 16-byte loads: four per key row of K and of V.
    for (int i = threadIdx.x; i < visible * (D / 8); i += blockDim.x) {
        const int key = i / (D / 8), quad = i % (D / 8);
        const uint4 k4 = reinterpret_cast<const uint4*>(kbase + key * k_pos)[quad];
        const uint4 v4 = reinterpret_cast<const uint4*>(vbase + key * v_pos)[quad];
        const half2* kh = reinterpret_cast<const half2*>(&k4);
        #pragma unroll
        for (int j = 0; j < 4; ++j) keys[key * KPAD + quad * 4 + j] = kh[j];
        reinterpret_cast<uint4*>(values)[key * (D / 8) + quad] = v4;
    }
    __syncthreads();
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int pair = lane & (D / 2 - 1), part = lane >> 4;  // output: dimension pair, key parity
    for (int m = warp; m < group * steps; m += WARPS) {
        const int g = m / steps, t = m % steps, head = kv * group + g;
        const int64_t at = ((static_cast<int64_t>(b) * heads + head) * steps + t) * D;
        float2 query[D / 2];
        #pragma unroll
        for (int quad = 0; quad < D / 8; ++quad) {
            const uint4 q4 = reinterpret_cast<const uint4*>(q + at)[quad];
            const half2* qh = reinterpret_cast<const half2*>(&q4);
            #pragma unroll
            for (int j = 0; j < 4; ++j) query[quad * 4 + j] = __half22float2(qh[j]);
        }
        float a0 = 0.f, a1 = 0.f, b0 = 0.f, b1 = 0.f;
        #pragma unroll
        for (int p2 = 0; p2 < D / 2; p2 += 2) {
            const float2 k0 = __half22float2(keys[lane * KPAD + p2]), k0n = __half22float2(keys[lane * KPAD + p2 + 1]);
            const float2 k1 = __half22float2(keys[(lane + 32) * KPAD + p2]), k1n = __half22float2(keys[(lane + 32) * KPAD + p2 + 1]);
            a0 += query[p2].x * k0.x + query[p2].y * k0.y;
            a1 += query[p2 + 1].x * k0n.x + query[p2 + 1].y * k0n.y;
            b0 += query[p2].x * k1.x + query[p2].y * k1.y;
            b1 += query[p2 + 1].x * k1n.x + query[p2 + 1].y * k1n.y;
        }
        const int limit = min(length + t, span);
        const float s0 = lane < limit ? (a0 + a1) * scale : -INFINITY;
        const float s1 = lane + 32 < limit ? (b0 + b1) * scale : -INFINITY;
        float peak = fmaxf(s0, s1);
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) peak = fmaxf(peak, __shfl_xor_sync(0xffffffff, peak, offset));
        const float p0 = __expf(s0 - peak), p1 = __expf(s1 - peak);
        float total = p0 + p1;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) total += __shfl_xor_sync(0xffffffff, total, offset);
        const float inverse = 1.f / total;
        probs[warp][lane] = p0 * inverse;
        probs[warp][lane + 32] = p1 * inverse;
        __syncwarp();
        // Lane (part, pair) sums keys of its parity for its two dimensions; the halves meet in one shuffle.
        float2 c0 = make_float2(0.f, 0.f), c1 = make_float2(0.f, 0.f);
        int key = part;
        for (; key + 2 < limit; key += 4) {
            const float w0 = probs[warp][key], w1 = probs[warp][key + 2];
            const float2 v0 = __half22float2(values[key * (D / 2) + pair]);
            const float2 v1 = __half22float2(values[(key + 2) * (D / 2) + pair]);
            c0.x += w0 * v0.x; c0.y += w0 * v0.y;
            c1.x += w1 * v1.x; c1.y += w1 * v1.y;
        }
        if (key < limit) {
            const float w0 = probs[warp][key];
            const float2 v0 = __half22float2(values[key * (D / 2) + pair]);
            c0.x += w0 * v0.x; c0.y += w0 * v0.y;
        }
        float x = c0.x + c1.x, y = c0.y + c1.y;
        x += __shfl_xor_sync(0xffffffff, x, 16);
        y += __shfl_xor_sync(0xffffffff, y, 16);
        if (part == 0) reinterpret_cast<half2*>(out + at)[pair] = __floats2half2_rn(x, y);
        __syncwarp();
    }
}

}  // namespace

torch::Tensor decode_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor rows,
                               torch::Tensor lengths, int64_t span, double scale) {
    TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::kHalf && q.is_contiguous() && q.size(3) == D);
    TORCH_CHECK(k.scalar_type() == torch::kHalf && v.scalar_type() == torch::kHalf && k.stride(3) == 1 && v.stride(3) == 1);
    TORCH_CHECK(span <= KEYS && rows.scalar_type() == torch::kLong && lengths.scalar_type() == torch::kLong);
    auto out = torch::empty_like(q);
    const int batch = q.size(0), heads = q.size(1), steps = q.size(2), kv_heads = k.size(1);
    if (batch == 0) return out;
    decode_attention_kernel<<<dim3(batch, kv_heads), 32 * WARPS, 0, c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(q.data_ptr()), reinterpret_cast<const half*>(k.data_ptr()),
        reinterpret_cast<const half*>(v.data_ptr()), reinterpret_cast<half*>(out.data_ptr()),
        rows.data_ptr<int64_t>(), lengths.contiguous().data_ptr<int64_t>(),
        k.stride(0), k.stride(1), k.stride(2), v.stride(0), v.stride(1), v.stride(2),
        heads, steps, heads / kv_heads, static_cast<int>(span), static_cast<float>(scale));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""

_CPP = "torch::Tensor decode_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor rows, torch::Tensor lengths, int64_t span, double scale);"


@lru_cache(maxsize=1)
def _extension():
    from torch.utils.cpp_extension import load_inline

    major, minor = torch.cuda.get_device_capability()
    return load_inline(
        name="cardgym_decode_attention",
        cpp_sources=_CPP,
        cuda_sources=_CUDA,
        functions=["decode_attention"],
        extra_cuda_cflags=["-O3", "--use_fast_math", f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"],
        verbose=False,
    )


@torch.library.custom_op("cardgym::decode_attention", mutates_args=())
def _decode_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, rows: torch.Tensor, lengths: torch.Tensor, span: int, scale: float
) -> torch.Tensor:
    return _extension().decode_attention(q, k, v, rows, lengths, span, scale)


@_decode_attention.register_fake
def _(q, k, v, rows, lengths, span, scale):
    return torch.empty_like(q)


def supported(head_dim: int, span: int, queries: int) -> bool:
    return head_dim == HEAD_DIM and span <= MAX_SPAN


def decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    rows: torch.Tensor,
    key_lengths: torch.Tensor,
    span: int,
    scale: float,
) -> torch.Tensor:
    """q [B, H, T, D] against cache rows ``rows`` [B] of k/v [R, KVH, L, D]; query t of row b sees keys < key_lengths[b] + t.

    Returns [B, H, T, D] in the cache's dtype (fp16).
    """

    return _decode_attention(q.to(k_cache.dtype).contiguous(), k_cache, v_cache, rows, key_lengths, span, scale)
