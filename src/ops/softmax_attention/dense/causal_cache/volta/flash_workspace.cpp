#include "ops/softmax_attention/dense/causal_cache/volta/flash.h"
#include "core/layout.h"
#include <algorithm>

namespace infernix::ops::detail {
namespace {
struct Scratch {
    Tensor key, value, mask, query, output, meta;
};

template <class Allocator>
Scratch allocate(Allocator& a, int heads, int width, int keys) {
    const int kv_heads = heads == 24 ? 4 : 2;
    const int padded   = (keys + 255) / 256 * 256;
    const int tokens   = std::min(width, kVoltaFlashQBlockTokens);
    const int rows     = (tokens + 63) / 64 * 64;
    return {
        a.alloc(DType::FP16, {256, kv_heads, padded}),
        a.alloc(DType::FP16, {256, kv_heads, padded}),
        a.alloc(DType::FP16, {padded, rows}),
        a.alloc(DType::FP32, {256, heads, tokens}),
        a.alloc(DType::FP32, {256, heads, tokens}),
        a.alloc(DType::FP32,
                {2, static_cast<int>(causal_attention_volta_flash_meta_elements(heads, tokens))})};
}
} // namespace

std::size_t volta_flash_workspace_bytes(int heads, int width, int keys) {
    WorkspaceLayoutBuilder layout;
    (void)allocate(layout, heads, width, keys);
    return layout.peak_bytes(1);
}

void volta_flash_cached(const Tensor& q, const Tensor& pos, const Tensor& rows, float scale,
                        PagedKVBatchLayerView cache, CausalAttentionExecutionEnvelope envelope,
                        WorkspaceArena& ws, Tensor& out, cudaStream_t stream) {
    auto scope   = ws.scope();
    auto scratch = allocate(ws, q.ne[1], q.ne[2], envelope.max_visible_keys);
    causal_attention_volta_flash_launch(
        q, {}, {}, pos, rows, scale, cache, envelope, kVoltaFlashQBlockTokens, scratch.key,
        scratch.value, scratch.mask, scratch.query, scratch.output, scratch.meta, out, stream);
}
} // namespace infernix::ops::detail
