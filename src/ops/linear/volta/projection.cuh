#pragma once
#include "ops/linear/volta/gemm.h"
#include "core/layout.h"
#include "ops/linear/fp8/fp8_volta_qpn_gemm.cuh"
#include "ops/linear/nvfp4/nvfp4_volta_qpn_gemm.cuh"
#include <stdexcept>
#include <algorithm>

namespace infernix::ops::detail {
template <class Output>
struct VoltaOffsetOutput {
    Output output;
    int begin;

    __device__ void store(int row, int token, float value) const {
        output.store(row, begin + token, value);
    }
};

inline std::size_t volta_projection_bytes(int n, int k, int tokens) {
    WorkspaceLayoutBuilder layout;
    if (tokens <= 32) {
        (void)layout.alloc(DType::FP16, {k, tokens});
        return layout.peak_bytes();
    }
    (void)layout.alloc(DType::FP32, {n, tokens});
    (void)layout.alloc_bytes(volta_gemm_workspace_bytes(n, k, tokens));
    return layout.peak_bytes();
}

template <class Output>
__global__ void finish_projection(const float* values, const __nv_bfloat16* scales, int n,
                                  int tokens, Output output) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n * tokens) {
        const int r = i % n;
        output.store(r, i / n, values[i] * (scales ? __bfloat162float(scales[r]) : 1.f));
    }
}

template <class Output>
void volta_projection(const Tensor& x, const Weight& w, Output output, WorkspaceArena* workspace,
                      cudaStream_t stream) {
    if (x.ne[1] <= 32) {
        if (workspace) {
            // Preserve the QPN arithmetic, but convert each activation once
            // instead of repeating BF16 conversion in every output-row CTA.
            auto scope = workspace->scope();
            auto input = workspace->alloc(DType::FP16, {w.k, x.ne[1]});
            volta_cast_input_fp16(x, input, stream);
            const auto* values = static_cast<const half*>(input.data);
            if (w.qtype == QType::NVFP4)
                launch_nvfp4_volta_qpn_with_fp16_activation(
                    x, w, values, output, w.n, 1.f / w.weight_scale_divisor, stream);
            else
                launch_fp8_volta_qpn_with_fp16_activation(x, w, values, output, w.n, stream);
            return;
        }
        if (w.qtype == QType::NVFP4)
            launch_nvfp4_volta_qpn_with_output(x, w, output, w.n, 1.f / w.weight_scale_divisor,
                                               stream);
        else
            launch_fp8_volta_qpn_with_output(x, w, output, w.n, stream);
        return;
    }
    if (!workspace) {
        // The workspace-free A16 overload retains its contract using bounded QPN launches.
        for (int begin = 0; begin < x.ne[1]; begin += 32) {
            Tensor tile(static_cast<__nv_bfloat16*>(x.data) + static_cast<long long>(begin) * w.k,
                        DType::BF16, {w.k, std::min(32, x.ne[1] - begin)});
            const VoltaOffsetOutput<Output> shifted{output, begin};
            if (w.qtype == QType::NVFP4)
                launch_nvfp4_volta_qpn_with_output(tile, w, shifted, w.n,
                                                   1.f / w.weight_scale_divisor, stream);
            else
                launch_fp8_volta_qpn_with_output(tile, w, shifted, w.n, stream);
        }
        return;
    }
    auto scope  = workspace->scope();
    auto result = workspace->alloc(DType::FP32, {w.n, x.ne[1]});
    volta_gemm_fp32(x, w, static_cast<float*>(result.data), *workspace, stream);
    finish_projection<<<(w.n * x.ne[1] + 255) / 256, 256, 0, stream>>>(
        static_cast<float*>(result.data),
        w.qtype == QType::FP8_E4M3FN_ROW_BF16 ? static_cast<const __nv_bfloat16*>(w.scales)
                                              : nullptr,
        w.n, x.ne[1], output);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace infernix::ops::detail
