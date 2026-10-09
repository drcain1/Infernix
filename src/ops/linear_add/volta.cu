#include "ops/linear/volta/projection.cuh"
#include "ops/linear/common/output.cuh"
#include "ops/linear_add/fp8/fp8_linear_add_plan.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"

namespace infernix::ops::detail {
namespace {
void validate(LinearPolicy p, int low, int high) {
    if (!valid_linear_policy(p) || low <= 0 || high < low)
        throw std::invalid_argument("Volta projection: invalid policy/token interval");
}
} // namespace

std::size_t fp8_linear_add_workspace_capacity_bytes(int n, int k, LinearPolicy p, int low,
                                                    int high) {
    validate(p, low, high);
    if (n != 5120 || (k != 6144 && k != 17408)) throw std::invalid_argument("Volta residual shape");
    return volta_projection_bytes(n, k, high);
}

void fp8_linear_add_dispatch(const Tensor& x, const Weight& w, Tensor& residual, LinearPolicy p,
                             WorkspaceArena& ws, cudaStream_t stream) {
    validate(p, x.ne[1], x.ne[1]);
    volta_projection(x, w, Fp8ResidualOutput{static_cast<__nv_bfloat16*>(residual.data), w.n}, &ws,
                     stream);
}

std::size_t nvfp4_linear_add_workspace_capacity_bytes(int n, int k, LinearPolicy p, int low,
                                                      int high) {
    validate(p, low, high);
    if (n != 5120 || (k != 6144 && k != 17408)) throw std::invalid_argument("Volta residual shape");
    return volta_projection_bytes(n, k, high);
}

void nvfp4_linear_add_dispatch(const Tensor& x, const Weight& w, Tensor& residual, LinearPolicy p,
                               WorkspaceArena& ws, cudaStream_t stream) {
    validate(p, x.ne[1], x.ne[1]);
    volta_projection(x, w, Fp8ResidualOutput{static_cast<__nv_bfloat16*>(residual.data), w.n}, &ws,
                     stream);
}
} // namespace infernix::ops::detail
