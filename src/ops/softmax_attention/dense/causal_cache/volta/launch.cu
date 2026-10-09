#include "ops/softmax_attention/dense/causal_cache/volta/small_t_bf16_volta.cuh"
#include "ops/softmax_attention/dense/causal_cache/volta/small_t_i8_volta.cuh"
#include "ops/softmax_attention/common/causal_operands.h"
#include "ops/softmax_attention/dense/causal_cache/bf16/launch.h"
#include "ops/softmax_attention/dense/causal_cache/int8/launch.h"
#include "ops/softmax_attention/dense/causal_cache/bf16/plan.h"
#include "ops/softmax_attention/dense/causal_cache/int8/plan.h"
#include "ops/kv_cache/append/launch.h"
#include "core/device.h"
#include "core/layout.h"
#include <algorithm>
#include "ops/softmax_attention/dense/causal_cache/fp8/launch.h"
#include "ops/softmax_attention/dense/causal_cache/fp8/plan.h"
#include "ops/softmax_attention/dense/causal_cache/volta/flash.h"

namespace infernix::ops::detail {
namespace {
int splits(int heads, int keys) {
    // Reserve for the donor's nonmonotone split policy, including its short-context peaks.
    const int factor = heads == 24 ? 1 : 2;
    return std::min(560 * factor, std::max(128 * factor, (keys + 479) / 480 * factor));
}

std::size_t partial_bytes(int heads, int batch, int width,
                          CausalAttentionExecutionEnvelope envelope) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_causal_partials(layout, heads, std::min(heads == 24 ? 8 : 6, width),
                                   splits(heads, envelope.max_visible_keys), batch);
    return layout.peak_bytes(1);
}

std::size_t bytes(int heads, int batch, int width, CausalAttentionExecutionEnvelope envelope) {
    const auto partial = partial_bytes(heads, batch, width, envelope);
    return width >= 64 && batch == 1
               ? std::max(partial,
                          volta_flash_workspace_bytes(heads, width, envelope.max_visible_keys))
               : partial;
}

template <class G, bool Int8, bool Multi, bool Masked, int Tile, bool Fp8>
void launch_tile(const Tensor& q, const Tensor& positions, const Tensor& valid, const Tensor& rows,
                 const Tensor& tree, float scale, PagedKVBatchLayerView cache, Tensor& out,
                 CausalPartialView partial, int capacity, int tokens, int begin, int keys,
                 cudaStream_t stream) {
    const dim3 grid(G::KVHeads, capacity, q.ne[3]);
    const auto* qp     = static_cast<const __nv_bfloat16*>(q.data);
    const auto* pos    = static_cast<const int*>(positions.data);
    const auto* vp     = static_cast<const int*>(valid.data);
    const auto* rp     = static_cast<const int*>(rows.data);
    const auto* tp     = static_cast<const std::uint32_t*>(tree.data);
    const auto* tables = static_cast<const int*>(cache.block_tables.data);
    if constexpr (Int8) {
        volta::causal_attention_small_t_tc_volta_partial_i8_kernel<G, Tile, 4, Multi, Masked,
                                                                   volta::CausalCachedInput, Fp8>
            <<<grid, 128, 0, stream>>>(qp, {}, pos, static_cast<std::int8_t*>(cache.k_pages.data),
                                       static_cast<std::int8_t*>(cache.v_pages.data),
                                       static_cast<half*>(cache.k_scale_pages.data),
                                       static_cast<half*>(cache.v_scale_pages.data), tables, vp, rp,
                                       cache.block_tables.ne[0], tokens, q.ne[2], begin, keys,
                                       scale, partial.acc, partial.maximum, partial.sum, tp);
    } else {
        volta::causal_attention_small_t_tc_volta_partial_bf16_kernel<G, Tile, 4, Multi, Masked,
                                                                     volta::CausalCachedInput>
            <<<grid, 128, 0, stream>>>(qp, {}, pos, static_cast<__nv_bfloat16*>(cache.k_pages.data),
                                       static_cast<half*>(cache.v_pages.data), tables, vp, rp,
                                       cache.block_tables.ne[0], tokens, q.ne[2], begin, keys,
                                       scale, partial.acc, partial.maximum, partial.sum, tp);
    }
    volta::causal_attention_small_t_reduce_output_kernel<G, 256, Int8, Multi, Masked, true>
        <<<dim3(G::QHeads, 1, tokens * q.ne[3]), 256, 0, stream>>>(
            partial.acc, partial.maximum, partial.sum, pos, vp, tokens, q.ne[2], begin, q.ne[3],
            capacity, static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

template <class G, bool Int8, bool Multi, bool Masked, bool Fp8>
void execute(const Tensor& q, const Tensor& positions, const Tensor& valid, const Tensor& rows,
             const Tensor& tree, float scale, PagedKVBatchLayerView cache,
             CausalAttentionExecutionEnvelope envelope, WorkspaceArena& ws, Tensor& out,
             cudaStream_t stream) {
    auto scope         = ws.scope();
    const int capacity = splits(G::QHeads, envelope.max_visible_keys);
    const int max_tile = G::GroupSize == 6 ? 8 : 6;
    const auto partial =
        allocate_causal_partials(ws, G::QHeads, std::min(max_tile, q.ne[2]), capacity, q.ne[3]);
    for (int begin = 0; begin < q.ne[2]; begin += max_tile) {
        const int tokens = std::min(max_tile, q.ne[2] - begin);
#define TILE(T)                                                                                    \
    case T:                                                                                        \
        launch_tile<G, Int8, Multi, Masked, T, Fp8>(q, positions, valid, rows, tree, scale, cache, \
                                                    out, partial.view(), capacity, tokens, begin,  \
                                                    envelope.max_visible_keys, stream);            \
        break
        switch (tokens) {
            TILE(1);
            TILE(2);
            TILE(3);
            TILE(4);
            TILE(5);
            TILE(6);
        default:
            if constexpr (G::GroupSize == 6)
                launch_tile<G, Int8, Multi, Masked, 8, Fp8>(
                    q, positions, valid, rows, tree, scale, cache, out, partial.view(), capacity,
                    tokens, begin, envelope.max_visible_keys, stream);
        }
#undef TILE
    }
}

template <bool Int8, bool Fp8 = false>
void run(const Tensor& q, const Tensor& positions, const Tensor& valid, const Tensor& rows,
         const Tensor& tree, float scale, PagedKVBatchLayerView cache,
         CausalAttentionExecutionEnvelope envelope, WorkspaceArena& ws, Tensor& out,
         cudaStream_t stream) {
    if (q.ne[2] >= 64 && q.ne[3] == 1 && !valid.data && !tree.data &&
        volta_flash_workspace_bytes(q.ne[1], q.ne[2], envelope.max_visible_keys) >=
            partial_bytes(q.ne[1], q.ne[3], q.ne[2], envelope)) {
        volta_flash_cached(q, positions, rows, scale, cache, envelope, ws, out, stream);
        return;
    }
    const auto geometry = [&]<class G>() {
        const auto batch = [&]<bool Multi>() {
            if (valid.data)
                execute<G, Int8, Multi, true, Fp8>(q, positions, valid, rows, tree, scale, cache,
                                                   envelope, ws, out, stream);
            else
                execute<G, Int8, Multi, false, Fp8>(q, positions, valid, rows, tree, scale, cache,
                                                    envelope, ws, out, stream);
        };
        if (q.ne[3] > 1)
            batch.template operator()<true>();
        else
            batch.template operator()<false>();
    };
    if (q.ne[1] == 24)
        geometry.template operator()<volta::CausalD256H24Kv4>();
    else
        geometry.template operator()<volta::CausalD256H16Kv2>();
}
} // namespace

std::size_t bf16_kv_workspace_bytes(int h, int b, int low, int high,
                                    CausalAttentionExecutionEnvelope e, int) {
    return bytes(h, b, high, e);
}

std::size_t int8_kv_workspace_bytes(int h, int b, int low, int high,
                                    CausalAttentionExecutionEnvelope e, int) {
    return bytes(h, b, high, e);
}

void bf16_kv_append_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& pos,
                              const Tensor& valid, const Tensor& rows, const Tensor& tree,
                              float scale, PagedKVBatchLayerView cache,
                              CausalAttentionExecutionEnvelope e, WorkspaceArena& ws, Tensor& out,
                              DeviceExecutionView execution) {
    kv_cache_append_batch_launch(k, v, pos, valid, rows, cache, execution.stream);
    run<false>(q, pos, valid, rows, tree, scale, cache, e, ws, out, execution.stream);
}

void int8_kv_append_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& pos,
                              const Tensor& valid, const Tensor& rows, const Tensor& tree,
                              float scale, PagedKVBatchLayerView cache,
                              CausalAttentionExecutionEnvelope e, WorkspaceArena& ws, Tensor& out,
                              DeviceExecutionView execution) {
    kv_cache_append_batch_launch(k, v, pos, valid, rows, cache, execution.stream);
    run<true>(q, pos, valid, rows, tree, scale, cache, e, ws, out, execution.stream);
}

void bf16_kv_cached_attention(const Tensor& q, const Tensor& pos, float scale,
                              const PagedKVLayerView& cache, CausalAttentionExecutionEnvelope e,
                              WorkspaceArena& ws, Tensor& out, DeviceExecutionView execution) {
    run<false>(q, pos, {}, {}, {}, scale, single_row_paged_kv_batch_view(cache), e, ws, out,
               execution.stream);
}

void int8_kv_cached_attention(const Tensor& q, const Tensor& pos, float scale,
                              const PagedKVLayerView& cache, CausalAttentionExecutionEnvelope e,
                              WorkspaceArena& ws, Tensor& out, DeviceExecutionView execution) {
    run<true>(q, pos, {}, {}, {}, scale, single_row_paged_kv_batch_view(cache), e, ws, out,
              execution.stream);
}
} // namespace infernix::ops::detail

namespace infernix::ops::detail {
std::size_t fp8_kv_workspace_bytes(int h, int b, int low, int high,
                                   CausalAttentionExecutionEnvelope e, int) {
    return bytes(h, b, high, e);
}

void fp8_kv_append_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& pos,
                             const Tensor& valid, const Tensor& rows, const Tensor& tree,
                             float scale, PagedKVBatchLayerView cache,
                             CausalAttentionExecutionEnvelope e, WorkspaceArena& ws, Tensor& out,
                             DeviceExecutionView execution) {
    kv_cache_append_batch_launch(k, v, pos, valid, rows, cache, execution.stream);
    run<true, true>(q, pos, valid, rows, tree, scale, cache, e, ws, out, execution.stream);
}

void fp8_kv_cached_attention(const Tensor& q, const Tensor& pos, float scale,
                             const PagedKVLayerView& cache, CausalAttentionExecutionEnvelope e,
                             WorkspaceArena& ws, Tensor& out, DeviceExecutionView execution) {
    run<true, true>(q, pos, {}, {}, {}, scale, single_row_paged_kv_batch_view(cache), e, ws, out,
                    execution.stream);
}
} // namespace infernix::ops::detail
