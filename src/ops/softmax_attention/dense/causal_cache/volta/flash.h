#pragma once
#include "infernix/ops/softmax_attention.h"

namespace infernix::ops::detail {
inline constexpr int kVoltaFlashQBlockTokens = 1024;
inline constexpr int kVoltaFlashMaskRowPad   = 64;
std::size_t causal_attention_volta_flash_meta_elements(int heads, int tokens);
void causal_attention_volta_flash_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                         const Tensor& positions, const Tensor& rows, float scale,
                                         PagedKVBatchLayerView cache,
                                         CausalAttentionExecutionEnvelope envelope,
                                         int block_tokens, Tensor& kg, Tensor& vg, Tensor& mask,
                                         Tensor& qf, Tensor& of, Tensor& meta, Tensor& out,
                                         cudaStream_t stream);
std::size_t volta_flash_workspace_bytes(int heads, int width, int keys);
void volta_flash_cached(const Tensor& q, const Tensor& pos, const Tensor& rows, float scale,
                        PagedKVBatchLayerView cache, CausalAttentionExecutionEnvelope envelope,
                        WorkspaceArena& ws, Tensor& out, cudaStream_t stream);
} // namespace infernix::ops::detail
