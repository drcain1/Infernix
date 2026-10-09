#include "core/device.h"
#include "ops/linear/q4/q4_launch.h"
#include "ops/linear/q4/q4_volta_qpn_gemm.cuh"

#include <cuda_bf16.h>

namespace infernix::ops::detail {

#ifdef INFERNIX_VOLTA

namespace {
using S = Q4VoltaQpnSchedule;
} // namespace

bool q4_volta_qpn_supported(std::int32_t n, std::int32_t k, std::int32_t t) noexcept {
    if (n <= 0 || t <= 0) { return false; }
    // Whole Q4 groups per lane (the scale is per group and held across it), and one uint4 of
    // activations per mma pair.
    if (k % Q4RowSplitStorage::kGroupK != 0 || k % 8 != 0) { return false; }
    // The CTA's four warps split K by whole groups, so every warp needs at least one.
    if (k / Q4RowSplitStorage::kGroupK < S::kWarps) { return false; }
    return t <= 2 * S::kRowsPerTile;
}

void launch_q4_volta_qpn(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const std::int32_t n             = out.ne[0];
    const std::int32_t k             = x.ne[0];
    const std::int32_t t             = x.ne[1];
    const std::int32_t out_ld        = static_cast<std::int32_t>(out.nb[1] / sizeof(__nv_bfloat16));
    const std::int32_t padded_groups = w.padded_shape[1] / Q4RowSplitStorage::kGroupK;

    const auto* codes  = static_cast<const std::uint8_t*>(w.qdata);
    const auto* scales = static_cast<const std::uint8_t*>(w.scales);

    const dim3 grid(static_cast<unsigned>((n + S::kColsPerCta - 1) / S::kColsPerCta));
    // Four Q4 groups consume a 128-byte cache line per lane per iteration.
#    define INFERNIX_QPN_LAUNCH(TILES, BLK)                                                        \
        q4_volta_qpn_gemm_kernel<TILES, BLK><<<grid, S::kThreads, 0, stream>>>(                    \
            codes, scales, static_cast<const __nv_bfloat16*>(x.data),                              \
            static_cast<__nv_bfloat16*>(out.data), n, k, t, padded_groups, out_ld)
    if (t <= S::kRowsPerTile) {
        INFERNIX_QPN_LAUNCH(1, 4);
    } else {
        INFERNIX_QPN_LAUNCH(2, 4);
    }
#    undef INFERNIX_QPN_LAUNCH
    CUDA_CHECK(cudaGetLastError());
}

#endif
} // namespace infernix::ops::detail
