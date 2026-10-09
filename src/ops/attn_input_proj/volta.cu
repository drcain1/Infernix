#include "ops/linear/volta/projection.cuh"
#include "ops/linear/common/output.cuh"
#include "ops/attn_input_proj/fp8/fp8_attn_input_plan.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"

namespace infernix::ops::detail {
namespace {
void validate(LinearPolicy p, int low, int high) {
    if (!valid_linear_policy(p) || low <= 0 || high < low)
        throw std::invalid_argument("Volta projection: invalid policy/token interval");
}
} // namespace

std::size_t fp8_attn_input_workspace_capacity_bytes(LinearPolicy p, int low, int high) {
    validate(p, low, high);
    return volta_projection_bytes(14336, 5120, high);
}

void fp8_attn_input_dispatch(const Tensor& x, const Weight& w, Tensor& q, Tensor& gate, Tensor& k,
                             Tensor& v, LinearPolicy p, WorkspaceArena* ws, cudaStream_t stream) {
    validate(p, x.ne[1], x.ne[1]);
    volta_projection(
        x, w,
        LinearBf16SegmentedOutput<6144, 1024, 6144, 1024>{
            {static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
             static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(v.data)}},
        ws, stream);
}

std::size_t nvfp4_attn_input_workspace_capacity_bytes(LinearPolicy p, int low, int high) {
    validate(p, low, high);
    return volta_projection_bytes(14336, 5120, high);
}

void nvfp4_attn_input_dispatch(const Tensor& x, const Weight& w, Tensor& q, Tensor& gate, Tensor& k,
                               Tensor& v, LinearPolicy p, WorkspaceArena* ws, cudaStream_t stream) {
    validate(p, x.ne[1], x.ne[1]);
    volta_projection(
        x, w,
        LinearBf16SegmentedOutput<6144, 1024, 6144, 1024>{
            {static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
             static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(v.data)}},
        ws, stream);
}

} // namespace infernix::ops::detail
