#include "core/device.h"
#include "ops/linear/q8/q8_launch.h"
#include "ops/linear/q8/q8_volta_qpn_gemm.cuh"

#include <cuda_bf16.h>
#include <algorithm>

namespace infernix::ops::detail {

#ifdef INFERNIX_VOLTA

namespace {
using S = Q8VoltaQpnSchedule;
} // namespace

bool q8_volta_qpn_supported(std::int32_t n, std::int32_t k, std::int32_t t) noexcept {
    if (n <= 0 || t <= 0) { return false; }
    // Whole Q8 groups per lane (the scale is per group and held across it), and one uint4 of
    // activations per mma pair.
    if (k % Q8RowSplitStorage::kGroupK != 0 || k % 8 != 0) { return false; }
    // The CTA's four warps split K by whole groups, so every warp needs at least one.
    if (k / Q8RowSplitStorage::kGroupK < S::kWarps) { return false; }
    return true;
}

void launch_q8_volta_qpn(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const std::int32_t n             = out.ne[0];
    const std::int32_t k             = x.ne[0];
    const std::int32_t t             = x.ne[1];
    const std::int32_t out_ld        = static_cast<std::int32_t>(out.nb[1] / sizeof(__nv_bfloat16));
    const std::int32_t padded_groups = w.padded_shape[1] / Q8RowSplitStorage::kGroupK;

    const auto* codes  = static_cast<const std::uint8_t*>(w.qdata);
    const auto* scales = static_cast<const std::uint8_t*>(w.scales);
    auto* out_data     = static_cast<__nv_bfloat16*>(out.data);

    const dim3 grid(static_cast<unsigned>((n + S::kColsPerCta - 1) / S::kColsPerCta));
    // kBlk = 4 groups = 128 B, one cache line per lane per iteration; see the Q4 sibling for why
    // that, rather than an offline weight prepack, is what keeps the scattered per-lane row reads
    // off DRAM.
    for (int begin = 0; begin < t; begin += S::kRowsPerTile) {
        const int columns = std::min(S::kRowsPerTile, t - begin);
        q8_volta_qpn_gemm_kernel<1, 4><<<grid, S::kThreads, 0, stream>>>(
            codes, scales,
            static_cast<const __nv_bfloat16*>(x.data) + static_cast<std::int64_t>(begin) * k,
            out_data + static_cast<std::int64_t>(begin) * out_ld, n, k, columns, padded_groups,
            out_ld);
    }
    CUDA_CHECK(cudaGetLastError());
}

#endif
} // namespace infernix::ops::detail
