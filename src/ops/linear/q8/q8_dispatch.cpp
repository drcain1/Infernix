#include "ops/linear/q8/q8_dispatch.h"
#include "ops/linear/q8/q8_launch.h"
#include "ops/linear/q8/q8_shapes.h"
#ifdef INFERNIX_VOLTA
#include "ops/linear/volta/gemm.h"
#endif

#include <array>
#include <stdexcept>

namespace infernix::ops::detail {
#ifdef INFERNIX_VOLTA
bool q8_volta_qpn_supported(std::int32_t, std::int32_t, std::int32_t) noexcept;
void launch_q8_volta_qpn(const Tensor&, const Weight&, Tensor&, cudaStream_t);
#endif
#ifndef INFERNIX_VOLTA
namespace {
using ShapeSelector = Q8Launch (*)(std::int32_t);

struct ShapeEntry {
    std::int32_t n;
    std::int32_t k;
    ShapeSelector select;
};

template <class Geometry>
constexpr ShapeEntry shape(ShapeSelector select) {
    return {Geometry::kOutputRows, Geometry::kInputRows, select};
}

constexpr std::array kShapes{
    shape<Q8N1024K2048>(select_q8_n1024_k2048),     shape<Q8N1024K5120>(select_q8_n1024_k5120),
    shape<Q8N2048K4096>(select_q8_n2048_k4096),     shape<Q8N2048K4608>(select_q8_n2048_k4608),
    shape<Q8N2048K16384>(select_q8_n2048_k16384),   shape<Q8N4608K4608>(select_q8_n4608_k4608),
    shape<Q8N5120K4608>(select_q8_n5120_k4608),     shape<Q8N5120K6144>(select_q8_n5120_k6144),
    shape<Q8N5120K10240>(select_q8_n5120_k10240),   shape<Q8N5120K17408>(select_q8_n5120_k17408),
    shape<Q8N5120K25600>(select_q8_n5120_k25600),   shape<Q8N6144K5120>(select_q8_n6144_k5120),
    shape<Q8N9216K2048>(select_q8_n9216_k2048),     shape<Q8N12288K2048>(select_q8_n12288_k2048),
    shape<Q8N14336K5120>(select_q8_n14336_k5120),   shape<Q8N17408K5120>(select_q8_n17408_k5120),
    shape<Q8N34816K5120>(select_q8_n34816_k5120),   shape<Q8N248320K5120>(select_q8_n248320_k5120),
    // Qwen3.8-Flash-Next (N 324 and 320 are not 16-row multiples, so they bypass the geometry type).
    ShapeEntry{16384, 2560, select_q8_n16384_k2560},
    ShapeEntry{12800, 2560, select_q8_n12800_k2560},
    ShapeEntry{13952, 2560, select_q8_n13952_k2560},
    ShapeEntry{2560, 2560, select_q8_n2560_k2560},
    ShapeEntry{1280, 2560, select_q8_n1280_k2560},
    ShapeEntry{2560, 6144, select_q8_n2560_k6144},
    ShapeEntry{324, 10240, select_q8_n324_k10240},
    ShapeEntry{320, 10240, select_q8_n320_k10240},
    ShapeEntry{10240, 320, select_q8_n10240_k320},
    ShapeEntry{2560, 640, select_q8_n2560_k640},
    // NInfer's tunings for its own 2560-wide groupings.
    shape<Q8N6144K2560>(select_q8_n6144_k2560),   shape<Q8N10240K2560>(select_q8_n10240_k2560),
    shape<Q8N12288K2560>(select_q8_n12288_k2560),
};

// Shapes without a tuned entry use the runtime-shape templates: predicated SIMT for decode and
// verification widths, MMA tiles beyond, each covering any row count and any K padded to 128.
// Both SIMT widths of a schedule pair reduce a column identically, so a column's output does not
// depend on how many columns (up to 8) share the call.
Q8Launch select_q8_generic(std::int32_t n, std::int32_t k, std::int32_t t) {
    (void)n;
    (void)k;
    if (t <= 4) return launch_q8_a16_simt_r8_t4;
    if (t <= 8) return launch_q8_a16_simt_r8_t8;
    if (t <= 64) return launch_q8_a16_mma_r32_t64;
    if (t <= 96) return launch_q8_a16_mma_r32_t96;
    return launch_q8_a16_mma_r32_t128;
}
} // namespace
#endif

Q8Launch select_q8_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t) {
    if (t <= 0) throw std::invalid_argument("q8 linear: T must be positive");
    if (n <= 0 || k <= 0) throw std::invalid_argument("q8 linear: unsupported shape");
#ifdef INFERNIX_VOLTA
    // Keep one reduction order across small decode batches. Narrow output
    // matrices retain that order at wider extents as well.
    if (t <= 4) return launch_q8_a16_simt_r8_t4;
    if (t <= 8 || n < 4096) return launch_q8_a16_simt_r8_t8;
    if (q8_volta_qpn_supported(n, k, t)) return launch_q8_volta_qpn;
    // Partial groups use the predicated FP32 path, with bounded token slices.
    return launch_q8_a16_simt_r8_t4;
#else
    for (const auto& entry : kShapes) {
        if (entry.n == n && entry.k == k) {
            return entry.select(t);
        }
    }
    return select_q8_generic(n, k, t);
#endif
}

Q8Launch select_q8_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy) {
    if (!valid_linear_policy(policy)) throw std::invalid_argument("q8 linear: unsupported policy");
    return select_q8_a16_launch(n, k, t);
}

std::size_t q8_linear_workspace_capacity_bytes(int n, int k, LinearPolicy policy,
                                               int min_tokens, int max_tokens) {
    (void)select_q8_launch(n, k, min_tokens, policy);
    (void)select_q8_launch(n, k, max_tokens, policy);
#ifdef INFERNIX_VOLTA
    if (volta_q8_wide_supported(n, k, max_tokens))
        return volta_q8_workspace_bytes(n, k, max_tokens);
#endif
    return 0;
}

void q8_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                 WorkspaceArena* workspace, cudaStream_t stream) {
    const auto launch = select_q8_launch(w.n, w.k, x.ne[1], policy);
#ifdef INFERNIX_VOLTA
    if (workspace && volta_q8_wide_supported(w.n, w.k, x.ne[1])) {
        volta_q8_gemm(x, w, out, *workspace, stream);
        return;
    }
#else
    (void)workspace;
#endif
    launch(x, w, out, stream);
}

} // namespace infernix::ops::detail
