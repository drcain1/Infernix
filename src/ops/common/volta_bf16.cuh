#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace infernix::ops {
// BF16 has no native Volta arithmetic. Preserve its exponent range in FP32
// for the BF16 projection leaves that cannot use the quantized FP16 MMA route.
__device__ __forceinline__ unsigned volta_ldmatrix(unsigned address, int matrix, bool transpose) {
    const int lane = threadIdx.x & 31, row = lane / 4, column = 2 * (lane % 4);
    if (!transpose) {
        const unsigned source = __shfl_sync(0xffffffffU, address, matrix * 8 + row);
        unsigned result;
        asm("ld.shared.u32 %0, [%1];" : "=r"(result) : "r"(source + column * 2));
        return result;
    }
    const unsigned first  = __shfl_sync(0xffffffffU, address, matrix * 8 + column);
    const unsigned second = __shfl_sync(0xffffffffU, address, matrix * 8 + column + 1);
    unsigned short low, high;
    asm("ld.shared.u16 %0, [%1];" : "=h"(low) : "r"(first + row * 2));
    asm("ld.shared.u16 %0, [%1];" : "=h"(high) : "r"(second + row * 2));
    return unsigned(low) | (unsigned(high) << 16);
}

template <bool Bf16>
__device__ __forceinline__ float volta_half_word(unsigned value, int half) {
    const unsigned short bits = static_cast<unsigned short>(value >> (half * 16));
    if constexpr (Bf16)
        return __uint_as_float(unsigned(bits) << 16);
    else
        return __half2float(__ushort_as_half(bits));
}

template <bool Bf16>
__device__ __forceinline__ float volta_fragment_dot(float c, unsigned a0, unsigned a1, unsigned b0,
                                                    unsigned b1, int row, int column) {
#pragma unroll 1
    for (int k = 0; k < 16; ++k) {
        const unsigned a = k < 8 ? a0 : a1, b = k < 8 ? b0 : b1;
        const int source = k % 8 / 2;
        const float x = volta_half_word<Bf16>(__shfl_sync(0xffffffffU, a, row * 4 + source), k % 2);
        const float y =
            volta_half_word<Bf16>(__shfl_sync(0xffffffffU, b, column * 4 + source), k % 2);
        c = fmaf(x, y, c);
    }
    return c;
}

template <bool Bf16>
__device__ __forceinline__ void volta_m16n8k16(float& c0, float& c1, float& c2, float& c3,
                                               unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                               unsigned b0, unsigned b1) {
    const int lane = threadIdx.x & 31, group = lane / 4, column = 2 * (lane % 4);
    c0 = volta_fragment_dot<Bf16>(c0, a0, a2, b0, b1, group, column);
    c1 = volta_fragment_dot<Bf16>(c1, a0, a2, b0, b1, group, column + 1);
    c2 = volta_fragment_dot<Bf16>(c2, a1, a3, b0, b1, group, column);
    c3 = volta_fragment_dot<Bf16>(c3, a1, a3, b0, b1, group, column + 1);
}
} // namespace infernix::ops
