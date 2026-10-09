#include "ops/linear/volta/projection.cuh"
#include "ops/common/math.cuh"
#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_plan.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"

namespace infernix::ops::detail {
namespace {
constexpr int kRows = 34816, kInput = 5120;

void validate(LinearPolicy p, int low, int high) {
    if (!valid_linear_policy(p) || low <= 0 || high < low)
        throw std::invalid_argument("Volta SwiGLU: invalid policy/token interval");
}

std::size_t capacity(int tokens) {
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::FP32, {kRows, tokens});
    if (tokens > 32)
        (void)layout.alloc_bytes(volta_gemm_workspace_bytes(kRows, kInput, tokens));
    else
        (void)layout.alloc(DType::FP16, {kInput, tokens});
    return layout.peak_bytes();
}

__global__ void stage(const __nv_bfloat16* x, half* out, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = __float2half_rn(__bfloat162float(x[i]));
}

__global__ void combine(const float* gate, const float* up, __nv_bfloat16* out, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = __float2bfloat16_rn(silu(gate[i]) * up[i]);
}

struct SplitOutput {
    float* gate;
    float* up;
    int rows;

    __device__ void store(int row, int token, float value) const {
        (row < rows ? gate : up)[static_cast<long long>(token) * rows +
                                (row < rows ? row : row - rows)] = value;
    }
};

__global__ void wide_combine(const float* values, const __nv_bfloat16* scales, __nv_bfloat16* out,
                             int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int row = i % (kRows / 2), token = i / (kRows / 2);
    float gate = values[static_cast<long long>(token) * kRows + row];
    float up   = values[static_cast<long long>(token) * kRows + row + kRows / 2];
    if (scales) {
        gate *= __bfloat162float(scales[row]);
        up *= __bfloat162float(scales[row + kRows / 2]);
    }
    out[i] = __float2bfloat16_rn(silu(gate) * up);
}

void dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy p, WorkspaceArena& ws,
              cudaStream_t stream) {
    validate(p, x.ne[1], x.ne[1]);
    auto scope      = ws.scope();
    auto scratch    = ws.alloc(DType::FP32, {kRows, x.ne[1]});
    auto* gate      = static_cast<float*>(scratch.data);
    const int count = (kRows / 2) * x.ne[1];
    auto* up        = gate + count;
    if (x.ne[1] <= 32) {
        auto activation = ws.alloc(DType::FP16, {kInput, x.ne[1]});
        auto* input     = static_cast<half*>(activation.data);
        stage<<<(kInput * x.ne[1] + 255) / 256, 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), input, kInput * x.ne[1]);
        // Compute the adjacent gate/up row ranges in one grid. Both retain
        // FP32 outputs until the fused SwiGLU epilogue performs its final cast.
        const SplitOutput output{gate, up, kRows / 2};
        if (w.qtype == QType::NVFP4)
            launch_nvfp4_volta_qpn_with_fp16_activation(
                x, w, input, output, kRows, 1.f / w.weight_scale_divisor, stream);
        else if (x.ne[1] <= 16)
            launch_fp8_volta_qpn_with_fp16_activation(x, w, input, output, kRows, stream);
        else {
            // The wider FP8 tile benefits from separate contiguous output grids.
            for (int half_index = 0; half_index < 2; ++half_index) {
                Weight part = w;
                part.n      = kRows / 2;
                part.qdata  = static_cast<const unsigned char*>(w.qdata) +
                              static_cast<long long>(half_index) * part.n * w.k;
                part.scales = static_cast<const unsigned char*>(w.scales) +
                              static_cast<long long>(half_index) * part.n * 2;
                launch_fp8_volta_qpn_with_fp16_activation(
                    x, part, input, Fp8Fp32ContiguousOutput{half_index ? up : gate, part.n},
                    part.n, stream);
            }
        }
    } else {
        // The generic projection's result plane would overlap the live gate/up planes;
        // use GEMM's contiguous result in scratch, then combine by token stride below.
        volta_gemm_fp32(x, w, gate, ws, stream);
        // Wide outputs remain token-major. finish_wide handles scales and the fused cast.
    }
    if (x.ne[1] <= 32)
        combine<<<(count + 255) / 256, 256, 0, stream>>>(
            gate, up, static_cast<__nv_bfloat16*>(out.data), count);
    else
        wide_combine<<<(count + 255) / 256, 256, 0, stream>>>(
            gate,
            w.qtype == QType::FP8_E4M3FN_ROW_BF16 ? static_cast<const __nv_bfloat16*>(w.scales)
                                                  : nullptr,
            static_cast<__nv_bfloat16*>(out.data), count);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace

std::size_t fp8_linear_swiglu_workspace_capacity_bytes(LinearPolicy p, int low, int high) {
    validate(p, low, high);
    return capacity(high);
}

std::size_t nvfp4_linear_swiglu_workspace_capacity_bytes(LinearPolicy p, int low, int high) {
    validate(p, low, high);
    return capacity(high);
}

void fp8_linear_swiglu_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy p,
                                WorkspaceArena& ws, cudaStream_t stream) {
    dispatch(x, w, out, p, ws, stream);
}

void nvfp4_linear_swiglu_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy p,
                                  WorkspaceArena& ws, cudaStream_t stream) {
    dispatch(x, w, out, p, ws, stream);
}
} // namespace infernix::ops::detail
