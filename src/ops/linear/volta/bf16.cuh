#pragma once
#include "core/device.h"
#include "core/weight.h"
#include "core/tensor.h"
#include <cuda_bf16.h>

namespace infernix::ops::detail {
// BF16 SIMT multiplication preserves the BF16 exponent range on Volta. Each
// block reuses a 32-column weight tile across sixteen query rows.
template <class Output>
__global__ void volta_bf16_kernel(const __nv_bfloat16* x, const __nv_bfloat16* w, int n, int k,
                                  int tokens, Output output) {
    __shared__ float a[16][32];
    __shared__ float b[32][33];
    const int lane = threadIdx.x % 32, group = threadIdx.x / 32;
    const int row = blockIdx.x * 32 + lane, token = blockIdx.y * 16 + group;
    float first = 0, second = 0;
    for (int begin = 0; begin < k; begin += 32) {
        for (int t = group; t < 16; t += 8)
            a[t][lane] =
                (blockIdx.y * 16 + t < tokens && begin + lane < k)
                    ? __bfloat162float(
                          x[(static_cast<long long>(blockIdx.y) * 16 + t) * k + begin + lane])
                    : 0.f;
        for (int r = group; r < 32; r += 8)
            b[r][lane] =
                (blockIdx.x * 32 + r < n && begin + lane < k)
                    ? __bfloat162float(
                          w[(static_cast<long long>(blockIdx.x) * 32 + r) * k + begin + lane])
                    : 0.f;
        __syncthreads();
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            first  = fmaf(a[group][j], b[lane][j], first);
            second = fmaf(a[group + 8][j], b[lane][j], second);
        }
        __syncthreads();
    }
    if (row < n && token < tokens) output.store(row, token, first);
    if (row < n && token + 8 < tokens) output.store(row, token + 8, second);
}

template <class Output>
void volta_bf16_projection(const Tensor& x, const Weight& w, Output output, cudaStream_t stream) {
    volta_bf16_kernel<<<dim3((w.n + 31) / 32, (x.ne[1] + 15) / 16), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(w.qdata), w.n,
        w.k, x.ne[1], output);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace infernix::ops::detail
