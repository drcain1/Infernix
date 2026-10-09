#pragma once
#include "core/arena.h"
#include "core/weight.h"
#include "core/tensor.h"

namespace infernix::ops::detail {
void volta_cast_input_fp16(const Tensor& input, Tensor& output, cudaStream_t stream);
std::size_t volta_gemm_workspace_bytes(int n, int k, int tokens);
void volta_gemm_fp32(const Tensor& x, const Weight& weight, float* output,
                     WorkspaceArena& workspace, cudaStream_t stream);

inline bool volta_q8_wide_supported(int n, int k, int tokens) {
    return n > 0 && n % 32 == 0 && k > 0 && k % 128 == 0 && tokens >= 128;
}

std::size_t volta_q8_workspace_bytes(int n, int k, int tokens);
void volta_q8_gemm(const Tensor& x, const Weight& weight, Tensor& output, WorkspaceArena& workspace,
                   cudaStream_t stream);
} // namespace infernix::ops::detail
