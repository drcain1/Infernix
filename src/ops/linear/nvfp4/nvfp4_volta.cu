#include "ops/linear/nvfp4/nvfp4_dispatch.h"
#include "ops/linear/nvfp4/nvfp4_format.h"
#include "ops/linear/volta/projection.cuh"

namespace infernix::ops::detail {
namespace {
void validate(int n, int k, LinearPolicy policy, int low, int high) {
    if (!valid_linear_policy(policy) || low <= 0 || high < low)
        throw std::invalid_argument("nvfp4 Volta linear: invalid policy/token interval");
    if (!((k == 5120 && (n == 14336 || n == 16384 || n == 34816)) ||
          (n == 5120 && (k == 6144 || k == 17408))))
        throw std::invalid_argument("nvfp4 Volta linear: unsupported shape");
}
} // namespace

std::size_t nvfp4_linear_workspace_capacity_bytes(int n, int k, LinearPolicy p, int low, int high) {
    validate(n, k, p, low, high);
    return volta_projection_bytes(n, k, high);
}

void nvfp4_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy p,
                    WorkspaceArena* workspace, cudaStream_t stream) {
    validate_nvfp4_weight(w, "nvfp4 Volta linear");
    validate(w.n, w.k, p, x.ne[1], x.ne[1]);
    volta_projection(x, w, Nvfp4ContiguousOutput{static_cast<__nv_bfloat16*>(out.data), w.n},
                     workspace, stream);
}
} // namespace infernix::ops::detail
