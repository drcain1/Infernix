#include "ops/linear/q4/q4_dispatch.h"
#include "ops/linear/q4/q4_shapes.h"
#include <array>
#include <stdexcept>

namespace infernix::ops::detail {
#ifdef INFERNIX_VOLTA
bool q4_volta_qpn_supported(std::int32_t, std::int32_t, std::int32_t) noexcept;
void launch_q4_volta_qpn(const Tensor&, const Weight&, Tensor&, cudaStream_t);
#endif
namespace {
struct ShapeEntry {
    std::int32_t n, k;
    Q4Launch (*select)(std::int32_t);
};

constexpr std::array kShapes{
    ShapeEntry{1024, 5120, select_q4_n1024_k5120},
    ShapeEntry{4096, 5120, select_q4_n4096_k5120},
    ShapeEntry{5120, 6144, select_q4_n5120_k6144},
    ShapeEntry{6144, 5120, select_q4_n6144_k5120},
    ShapeEntry{7168, 5120, select_q4_n7168_k5120},
    ShapeEntry{34816, 5120, select_q4_n34816_k5120},
    ShapeEntry{131072, 5120, select_q4_n131072_k5120},
    ShapeEntry{131072, 2048, select_q4_n131072_k2048},
    ShapeEntry{3456, 1152, select_q4_n3456_k1152},
    ShapeEntry{4304, 1152, select_q4_n4304_k1152},
};
} // namespace

Q4Launch select_q4_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t) {
    if (t <= 0) throw std::invalid_argument("q4 linear: T must be positive");
    for (const auto& entry : kShapes) {
        if (entry.n == n && entry.k == k) {
#ifdef INFERNIX_VOLTA
            if (t == 1 && n == 131072 && k == 5120)
                return launch_q4_a16_gemv_r4_w1_direct;
            if (q4_volta_qpn_supported(n, k, t)) return launch_q4_volta_qpn;
#endif
            return entry.select(t);
        }
    }
    throw std::invalid_argument("q4 linear: unsupported shape");
}

Q4Launch select_q4_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy) {
    if (!valid_linear_policy(policy)) throw std::invalid_argument("q4 linear: unsupported policy");
    return select_q4_a16_launch(n, k, t);
}

void q4_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                 cudaStream_t stream) {
    select_q4_launch(weight.n, weight.k, x.ne[1], policy)(x, weight, out, stream);
}
} // namespace infernix::ops::detail
