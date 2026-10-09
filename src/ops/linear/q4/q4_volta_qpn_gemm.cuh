#pragma once

// Historical tuning figures in comments below are from the donor revision in
// third_party/volta/NOTICE; they are not Infernix performance measurements.

// Volta FP16 tensor-core projection; source and license in third_party/volta/NOTICE.

#include "ops/common/volta_mma.cuh"
#include "ops/linear/q4/q4_rowsplit_storage.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <type_traits>

namespace infernix::ops::detail {

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700

struct Q4VoltaQpnSchedule {
    static constexpr int kWarps       = 4;  // warps per CTA; they split K, not N
    static constexpr int kColsPerCta  = 32; // output rows per CTA (mma's N axis)
    static constexpr int kRowsPerTile = 8;  // tokens per A tile (mma's M axis)
    static constexpr int kThreads     = kWarps * 32;
};

// `kTiles` is the number of 8-row A tiles, so T <= 8 * kTiles. The two-tile form decodes each
// weight group once and feeds both A tiles from it, so T=9..16 costs close to T=8 rather than
// double.
//
// `kBlk` is how many Q4 groups a lane reads before consuming any of them; kBlk * kCodeB bytes.
// It trades registers for read efficiency and the right value depends on n -- see the launcher.
struct Q4DefaultOutput {};

template <int kTiles, int kBlk, class Output = Q4DefaultOutput>
__global__ __launch_bounds__(Q4VoltaQpnSchedule::kThreads, 8) void q4_volta_qpn_gemm_kernel(
    const std::uint8_t* __restrict__ codes, const std::uint8_t* __restrict__ scales,
    const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ out, int n, int k, int t,
    int padded_groups, int out_ld, Output output = {}) {
    using S               = Q4VoltaQpnSchedule;
    constexpr int kGroupK = Q4RowSplitStorage::kGroupK;
    constexpr int kCodeB  = Q4RowSplitStorage::kCodeBytesPerGroup;

    __shared__ float cs[S::kWarps][kTiles * S::kRowsPerTile * S::kColsPerCta];

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    // Quadpair index, and this lane's position inside it. `r` is both the A row (token) and the
    // B column local to the quadpair -- that identity is what makes quadpair siblings share A.
    const int qp = (lane >> 2) & 3;
    const int r  = (lane & 3) + ((lane & 16) != 0 ? 4 : 0);

    const int col  = static_cast<int>(blockIdx.x) * S::kColsPerCta + qp * 8 + r;
    const int good = col < n;

    const int groups = k / kGroupK;
    const int gq     = groups / S::kWarps;
    const int g0     = warp * gq;
    const int gend   = (warp == S::kWarps - 1) ? groups : g0 + gq;

    const std::uint8_t* crow =
        codes + static_cast<std::int64_t>(good ? col : 0) * padded_groups * kCodeB;
    const std::uint16_t* srow = reinterpret_cast<const std::uint16_t*>(
        scales + static_cast<std::int64_t>(good ? col : 0) * padded_groups *
                     Q4RowSplitStorage::kScaleBytesPerGroup);

    float c[kTiles][8];
#    pragma unroll
    for (int tile = 0; tile < kTiles; ++tile) {
#    pragma unroll
        for (int i = 0; i < 8; ++i) { c[tile][i] = 0.0f; }
    }

    const half2 bias = __half2half2(__ushort_as_half(0x6408)); // 1032.0

    // Weight loads are the DRAM traffic and they are scattered: lane L owns output row
    // qp*8 + r, so a warp's 32 lanes read 32 *different* rows at the same group offset, strided
    // by the row pitch. Nothing coalesces, so each is a separate long-latency request -- and with
    // the loads issued immediately before the MMAs that consume them, ncu measured 25.53 of the
    // warp-cycles-per-issued-instruction in long_scoreboard, with the SM retiring 0.25
    // instructions per cycle. So the group loop issues a block of weight reads up front and then
    // runs their MMA slices while the rest are still in flight. It is blocked by
    // kBlk groups for a second reason: a lane streams its own weight row sequentially, but a
    // warp's 32 lanes stream 32 *different* rows at once, and at 640 resident CTAs that is ~82k
    // concurrent streams whose 128B lines would all have to stay resident in a 6 MB L2 to be
    // reused. They cannot -- at n=34816 DRAM read measured 2.44x the weight with L2 at 38%,
    // against 1.05x and 64% at n=6144, which is where 128B x streams stops fitting. Consuming a
    // whole line per lane per iteration means each line is touched once and fully used, so
    // residency stops mattering, and it puts 2*kBlk loads in flight to cover their latency.
    // Double-buffering the blocks on top of this does not work: at kBlk=4 the two live blocks are
    // 16 uint4 = 64 registers of weight data alone, against the 64-register budget
    // __launch_bounds__(128, 8) allows, so it spills and costs 1.6x (n=34816 T=5: 327.7us ->
    // 531.4). The block itself is the latency hiding.
    uint4 cw[2 * kBlk];
    std::uint16_t scv[kBlk];

    for (int gb = g0; gb < gend; gb += kBlk) {
        const int blk  = min(kBlk, gend - gb);
        const int last = gend - 1;
#    pragma unroll
        for (int e = 0; e < kBlk; ++e) {
            const int g           = min(gb + e, last);
            const std::uint8_t* p = crow + static_cast<std::int64_t>(g) * kCodeB;
            cw[2 * e]             = __ldg(reinterpret_cast<const uint4*>(p));
            cw[2 * e + 1]         = __ldg(reinterpret_cast<const uint4*>(p + 16));
            scv[e]                = good ? srow[g] : static_cast<std::uint16_t>(0);
        }

        for (int e = 0; e < blk; ++e) {
            const int g     = gb + e;
            const half2 sc2 = __half2half2(__ushort_as_half(scv[e]));
            // 32 code bytes = one Q4 group = 64 k-elements.
            const std::uint32_t words[8] = {cw[2 * e].x,     cw[2 * e].y,     cw[2 * e].z,
                                            cw[2 * e].w,     cw[2 * e + 1].x, cw[2 * e + 1].y,
                                            cw[2 * e + 1].z, cw[2 * e + 1].w};

#    pragma unroll
            for (int w = 0; w < 8; ++w) {
                // One uint32 = 4 code bytes = 8 adjacent k. Byte j decodes straight into
                // half2(k=2j, k=2j+1), which is already the B-fragment register pair.
                const std::uint32_t p4 = words[w] ^ 0x88888888u;
                half2 b[4];
#    pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const std::uint32_t byte = (p4 >> (8 * j)) & 0xffu;
                    std::uint32_t bits = ((byte & 0x0fu) | ((byte & 0xf0u) << 12)) | 0x64006400u;
                    b[j] = __hmul2(__hsub2(*reinterpret_cast<half2*>(&bits), bias), sc2);
                }
                const unsigned* B = reinterpret_cast<const unsigned*>(b);
                const int kbase   = g * kGroupK + w * 8;

#    pragma unroll
                for (int tile = 0; tile < kTiles; ++tile) {
                    const int row = tile * S::kRowsPerTile + r;
                    half2 a[4];
                    if (row < t) {
                        const __nv_bfloat16* xrow = x + static_cast<std::int64_t>(row) * k + kbase;
                        const uint4 raw           = *reinterpret_cast<const uint4*>(xrow);
                        const auto* src           = reinterpret_cast<const __nv_bfloat16*>(&raw);
                        __half tmp[8];
#    pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            tmp[j] = __float2half(__bfloat162float(src[j]));
                        }
#    pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            a[j] = *reinterpret_cast<const half2*>(tmp + 2 * j);
                        }
                    } else {
#    pragma unroll
                        for (int j = 0; j < 4; ++j) { a[j] = __half2half2(__ushort_as_half(0)); }
                    }
                    const unsigned* A = reinterpret_cast<const unsigned*>(a);
                    volta_mma_qp_n(c[tile], A[0], A[1], B[0], B[1]); // k slice 0
                    volta_mma_qp_n(c[tile], A[2], A[3], B[2], B[3]); // k slice 1
                }
            }
        }
    }

    // C map (v100-skinny mma8_probe.cu, roles swapped): register i of lane L holds
    //   A row  (i & 2) | ((L & 16) ? 4 : 0) | (L & 1)
    //   B col  (i & 1) | (((L >> 1) & 1) << 1) | ((i >> 2) << 2)   -- quadpair-local, + qp * 8
#    pragma unroll
    for (int tile = 0; tile < kTiles; ++tile) {
#    pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = (i & 2) | ((lane & 16) != 0 ? 4 : 0) | (lane & 1);
            const int cl  = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
            cs[warp][(tile * S::kRowsPerTile + row) * S::kColsPerCta + qp * 8 + cl] = c[tile][i];
        }
    }
    __syncthreads(); // the only barrier: cross-warp K reduce

    constexpr int kOut = kTiles * S::kRowsPerTile * S::kColsPerCta;
    for (int e = static_cast<int>(threadIdx.x); e < kOut; e += S::kThreads) {
        const int row  = e / S::kColsPerCta;
        const int cl   = e % S::kColsPerCta;
        const int ocol = static_cast<int>(blockIdx.x) * S::kColsPerCta + cl;
        if (row < t && ocol < n) {
            float v = 0.0f;
#    pragma unroll
            for (int w = 0; w < S::kWarps; ++w) { v += cs[w][e]; }
            if constexpr (std::is_same_v<Output, Q4DefaultOutput>)
                out[static_cast<std::int64_t>(row) * out_ld + ocol] = __float2bfloat16(v);
            else
                output.store(ocol, row, v);
        }
    }
}

#endif // sm_70

} // namespace infernix::ops::detail
