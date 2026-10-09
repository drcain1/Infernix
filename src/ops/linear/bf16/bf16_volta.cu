#include "ops/linear/bf16/bf16_dispatch.h"
#include "ops/linear/volta/bf16.cuh"
#include "ops/linear/common/output.cuh"
#include <stdexcept>

namespace infernix::ops::detail {
namespace {
void launch(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    volta_bf16_projection(x, w, LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
                          stream);
}
} // namespace

Bf16Launch select_bf16_a16_launch(int n, int k, int t) {
    if (n <= 0 || k <= 0 || t <= 0) throw std::invalid_argument("BF16 Volta linear geometry");
    return launch;
}

Bf16Launch select_bf16_launch(int n, int k, int t, LinearPolicy p) {
    if (!valid_linear_policy(p)) throw std::invalid_argument("BF16 Volta linear policy");
    return select_bf16_a16_launch(n, k, t);
}

void bf16_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy p,
                   cudaStream_t stream) {
    select_bf16_launch(w.n, w.k, x.ne[1], p)(x, w, out, stream);
}
} // namespace infernix::ops::detail
