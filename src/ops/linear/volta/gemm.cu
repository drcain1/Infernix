#include "ops/linear/volta/gemm.h"
#include "core/device.h"
#include "core/layout.h"
#include "cutlass/gemm/device/gemm.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <type_traits>
#include <algorithm>

namespace infernix::ops::detail {
namespace {
int q8_token_tile(int n, int k) {
    // This merger geometry runs over large vision tiles. Bound its temporary
    // storage more tightly so vision reservation does not crowd out text KV.
    return n == 5120 && k == 4608 ? 512 : 2048;
}

template <int TileN>
using Gemm = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm70,
    cutlass::gemm::GemmShape<128, TileN, 32>, cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<8, 8, 4>,
    cutlass::epilogue::thread::LinearCombination<float, 4, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

// Each thread decodes eight adjacent K values and issues one aligned FP16
// vector store. Packed codes are contiguous within that vector in both layouts.
__device__ float e4m3(unsigned code) {
    const unsigned short bits = ((code & 128) << 8) | ((code & 127) << 7);
    return __half2float(__ushort_as_half(bits)) * 256.f;
}

__device__ float e2m1(unsigned code) {
    const unsigned short bits = ((code & 8) << 12) | ((code & 7) << 9);
    return __half2float(__ushort_as_half(bits)) * 16384.f;
}

template <bool Fp8, bool Packed>
__global__ void decode_weight(Weight w, half* out) {
    const int vector = blockIdx.x * blockDim.x + threadIdx.x;
    if (vector >= w.n * (w.k / 8)) return;
    const int r = vector / (w.k / 8), c = (vector % (w.k / 8)) * 8;
    const int lane    = r % 32 / 8 * 4 + r % 4 + r % 8 / 4 * 16;
    const auto* codes = static_cast<const unsigned char*>(w.qdata);
    __align__(16) half values[8];
    if constexpr (Fp8) {
        const long long offset =
            Packed
                ? (((static_cast<long long>(r / 32) * (w.k / 128) + c / 128) * 8 + c % 128 / 16) *
                       32 +
                   lane) * 16 +
                      c % 16
                : static_cast<long long>(r) * w.k + c;
        const uint2 words = *reinterpret_cast<const uint2*>(codes + offset);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const unsigned word = j < 4 ? words.x : words.y;
            values[j]           = __float2half_rn(e4m3((word >> ((j % 4) * 8)) & 255));
        }
    } else {
        long long ci, si;
        if constexpr (Packed) {
            si = (static_cast<long long>(r / 32) * (w.k / 16) + c / 16) * 32 + lane;
            ci = si * 8 + c % 16 / 2;
        } else {
            ci          = (static_cast<long long>(r) * w.k + c) / 2;
            const int g = c / 16;
            si = (static_cast<long long>(r / 128) * (w.k / 64) + g / 4) * 512 + r % 32 * 16 +
                 r % 128 / 32 * 4 + g % 4;
        }
        const unsigned word = *reinterpret_cast<const unsigned*>(codes + ci);
        const float scale   = e4m3(static_cast<const unsigned char*>(w.scales)[si]);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int nibble = Packed ? j / 2 + j % 2 * 4 : j;
            values[j]        = __float2half_rn(e2m1((word >> (nibble * 4)) & 15) * scale);
        }
    }
    reinterpret_cast<uint4*>(out)[vector] = *reinterpret_cast<const uint4*>(values);
}

__global__ void decode_q8_weight(Weight w, half* out) {
    const int vector = blockIdx.x * blockDim.x + threadIdx.x;
    if (vector >= w.n * (w.k / 8)) return;
    const int row = vector / (w.k / 8), col = (vector % (w.k / 8)) * 8;
    const auto* codes = static_cast<const unsigned char*>(w.qdata) +
                        static_cast<long long>(row) * w.padded_shape[1] + col;
    const uint2 words = *reinterpret_cast<const uint2*>(codes);
    const float scale = __half2float(static_cast<const half*>(
        w.scales)[static_cast<long long>(row) * (w.padded_shape[1] / 32) + col / 32]);
    __align__(16) half values[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const unsigned word = j < 4 ? words.x : words.y;
        const int code      = int((word >> ((j % 4) * 8)) & 255);
        values[j]           = __float2half_rn(float((code ^ 128) - 128) * scale);
    }
    reinterpret_cast<uint4*>(out)[vector] = *reinterpret_cast<const uint4*>(values);
}

__global__ void finish_q8(const float* input, __nv_bfloat16* output, long long count) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) output[i] = __float2bfloat16_rn(input[i]);
}

__global__ void cast_input(const __nv_bfloat16* x, half* out, long long count) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) out[i] = __float2half_rn(__bfloat162float(x[i]));
}
} // namespace

void volta_cast_input_fp16(const Tensor& input, Tensor& output, cudaStream_t stream) {
    cast_input<<<(input.numel() + 255) / 256, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input.data), static_cast<half*>(output.data), input.numel());
    CUDA_CHECK(cudaGetLastError());
}

std::size_t volta_gemm_workspace_bytes(int n, int k, int tokens) {
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::FP16, {k, n});
    (void)layout.alloc(DType::FP16, {k, tokens});
    return layout.peak_bytes();
}

void volta_gemm_fp32(const Tensor& x, const Weight& w, float* output, WorkspaceArena& workspace,
                     cudaStream_t stream) {
    auto scope        = workspace.scope();
    auto weight       = workspace.alloc(DType::FP16, {w.k, w.n});
    auto input        = workspace.alloc(DType::FP16, {w.k, x.ne[1]});
    const int vectors = w.n * (w.k / 8);
    const dim3 grid((vectors + 255) / 256);
    if (w.qtype == QType::Q8_G32_FP16) {
        decode_q8_weight<<<grid, 256, 0, stream>>>(w, static_cast<half*>(weight.data));
    } else if (w.qtype == QType::FP8_E4M3FN_ROW_BF16) {
        if (w.layout == QuantLayout::VoltaQpn)
            decode_weight<true, true><<<grid, 256, 0, stream>>>(w, static_cast<half*>(weight.data));
        else
            decode_weight<true, false>
                <<<grid, 256, 0, stream>>>(w, static_cast<half*>(weight.data));
    } else {
        if (w.layout == QuantLayout::VoltaQpn)
            decode_weight<false, true>
                <<<grid, 256, 0, stream>>>(w, static_cast<half*>(weight.data));
        else
            decode_weight<false, false>
                <<<grid, 256, 0, stream>>>(w, static_cast<half*>(weight.data));
    }
    cast_input<<<(x.numel() + 255) / 256, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<half*>(input.data), x.numel());
    CUDA_CHECK(cudaGetLastError());
    auto launch = [&](auto tile_n) {
        using Kernel = Gemm<decltype(tile_n)::value>;
        typename Kernel::Arguments args{
            {x.ne[1], w.n, w.k},
            {static_cast<cutlass::half_t*>(input.data), w.k},
            {static_cast<cutlass::half_t*>(weight.data), w.k},
            {output, w.n},
            {output, w.n},
            {w.qtype == QType::NVFP4 ? 1.f / w.weight_scale_divisor : 1.f, 0.f},
            1};
        Kernel gemm;
        if (gemm.can_implement(args) != cutlass::Status::kSuccess ||
            Kernel::get_workspace_size(args) != 0)
            throw std::invalid_argument("Volta GEMM: unsupported matrix geometry");
        if (gemm(args, nullptr, stream) != cutlass::Status::kSuccess)
            throw std::runtime_error("Volta GEMM launch failed");
    };
    if (x.ne[1] >= 128)
        launch(std::integral_constant<int, 256>{});
    else
        launch(std::integral_constant<int, 128>{});
}

std::size_t volta_q8_workspace_bytes(int n, int k, int tokens) {
    tokens = std::min(tokens, q8_token_tile(n, k));
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::FP32, {n, tokens});
    (void)layout.alloc_bytes(volta_gemm_workspace_bytes(n, k, tokens));
    return layout.peak_bytes();
}

void volta_q8_gemm(const Tensor& x, const Weight& weight, Tensor& output, WorkspaceArena& workspace,
                   cudaStream_t stream) {
    const int tile_capacity = q8_token_tile(weight.n, weight.k);
    auto scope              = workspace.scope();
    auto values = workspace.alloc(DType::FP32, {weight.n, std::min(x.ne[1], tile_capacity)});
    // Bound staging independently of vision/context capacity. Adjacent tiles
    // reuse the caller's scratch in stream order, including a partial last tile.
    for (int begin = 0; begin < x.ne[1]; begin += tile_capacity) {
        const int columns = std::min(x.ne[1] - begin, tile_capacity);
        Tensor tile       = x.slice(1, begin, columns);
        volta_gemm_fp32(tile, weight, static_cast<float*>(values.data), workspace, stream);
        const long long count = static_cast<long long>(weight.n) * columns;
        finish_q8<<<(count + 255) / 256, 256, 0, stream>>>(
            static_cast<const float*>(values.data),
            static_cast<__nv_bfloat16*>(output.data) + static_cast<long long>(begin) * weight.n,
            count);
    }
    CUDA_CHECK(cudaGetLastError());
}
} // namespace infernix::ops::detail
