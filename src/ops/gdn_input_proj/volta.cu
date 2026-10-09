#include "ops/linear/volta/projection.cuh"
#include "ops/linear/common/output.cuh"
#include "ops/gdn_input_proj/fp8/fp8_gdn_input_plan.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"

namespace infernix::ops::detail {
namespace {
void validate(LinearPolicy p, int low, int high) {
    if (!valid_linear_policy(p) || low <= 0 || high < low)
        throw std::invalid_argument("Volta projection: invalid policy/token interval");
}
} // namespace

std::size_t fp8_gdn_input_workspace_capacity_bytes(LinearPolicy p, int low, int high) {
    validate(p, low, high);
    return volta_projection_bytes(16384, 5120, high);
}

void fp8_gdn_input_dispatch(const Tensor& x, const Weight& w, Tensor& qkv, Tensor& z,
                            LinearPolicy p, WorkspaceArena* ws, cudaStream_t stream) {
    validate(p, x.ne[1], x.ne[1]);
    volta_projection(x, w,
                     LinearBf16SegmentedOutput<10240, 6144>{{static_cast<__nv_bfloat16*>(qkv.data),
                                                             static_cast<__nv_bfloat16*>(z.data)}},
                     ws, stream);
}

std::size_t nvfp4_gdn_input_workspace_capacity_bytes(LinearPolicy p, int low, int high) {
    validate(p, low, high);
    return volta_projection_bytes(16384, 5120, high);
}

void nvfp4_gdn_input_dispatch(const Tensor& x, const Weight& w, Tensor& qkv, Tensor& z,
                              LinearPolicy p, WorkspaceArena* ws, cudaStream_t stream) {
    validate(p, x.ne[1], x.ne[1]);
    volta_projection(x, w,
                     LinearBf16SegmentedOutput<10240, 6144>{{static_cast<__nv_bfloat16*>(qkv.data),
                                                             static_cast<__nv_bfloat16*>(z.data)}},
                     ws, stream);
}
} // namespace infernix::ops::detail

#include "ops/gdn_input_proj/fp8/fp8_gdn_conv_plan.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.h"
#include "ops/gdn_input_proj/gdn_projected_conv.h"

namespace infernix::ops::detail {
namespace {
std::size_t snapshot_bytes(int tokens) {
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {10240, tokens});
    (void)layout.alloc_bytes(volta_projection_bytes(16384, 5120, tokens));
    return layout.peak_bytes();
}

void snapshot(const Tensor& x, const Weight& w, const Tensor& conv_weight, Tensor& conv_states,
              const Tensor& valid, const Tensor& initial, const Tensor& base, Tensor& q, Tensor& k,
              Tensor& v, Tensor& z, LinearPolicy p, WorkspaceArena& ws, cudaStream_t stream) {
    validate(p, x.ne[1], x.ne[1]);
    auto scope      = ws.scope();
    const int count = x.ne[1] * x.ne[2];
    auto projected  = ws.alloc(DType::BF16, {10240, count});
    Tensor flat(x.data, DType::BF16, {5120, count});
    volta_projection(
        flat, w,
        LinearBf16SegmentedOutput<10240, 6144>{
            {static_cast<__nv_bfloat16*>(projected.data), static_cast<__nv_bfloat16*>(z.data)}},
        &ws, stream);
    Tensor sequence(projected.data, DType::BF16, {10240, x.ne[1], x.ne[2]});
    gdn_projected_conv_snapshot_launch(sequence, conv_weight, conv_states, valid, initial, base, q,
                                       k, v, stream);
}
} // namespace

std::size_t fp8_gdn_snapshot_workspace_capacity_bytes(LinearPolicy p, int batch, int low,
                                                      int high) {
    validate(p, low, high);
    if (batch < 1 || batch > 8) throw std::invalid_argument("Volta GDN batch");
    return snapshot_bytes(batch * high);
}

std::size_t fp8_gdn_record_workspace_capacity_bytes(LinearPolicy p, int batch, int low, int high) {
    validate(p, low, high);
    if (batch < 1 || batch > 8 || low < 2 || high > 64)
        throw std::invalid_argument("Volta GDN record geometry");
    return volta_projection_bytes(16384, 5120, batch * high);
}

void fp8_gdn_snapshot_dispatch(const Tensor& x, const Weight& w, const Tensor& cw, Tensor& states,
                               const Tensor& valid, const Tensor& initial, const Tensor& base,
                               Tensor& q, Tensor& k, Tensor& v, Tensor& z, LinearPolicy p,
                               WorkspaceArena& ws, cudaStream_t stream) {
    snapshot(x, w, cw, states, valid, initial, base, q, k, v, z, p, ws, stream);
}

void nvfp4_gdn_snapshot_dispatch(const Tensor& x, const Weight& w, const Tensor& cw, Tensor& states,
                                 const Tensor& valid, const Tensor& initial, const Tensor& base,
                                 Tensor& q, Tensor& k, Tensor& v, Tensor& z, LinearPolicy p,
                                 WorkspaceArena& ws, cudaStream_t stream) {
    snapshot(x, w, cw, states, valid, initial, base, q, k, v, z, p, ws, stream);
}

Nvfp4GdnConvPlan nvfp4_gdn_conv_resolve_plan(LinearPolicy p, int tokens, int batch) {
    validate(p, tokens, tokens);
    if (batch < 1 || batch > 8) throw std::invalid_argument("Volta GDN batch");
    return {Nvfp4GdnConvScheduleId::Materialized};
}

std::size_t nvfp4_gdn_snapshot_workspace_capacity_bytes(LinearPolicy p, int low, int high) {
    validate(p, low, high);
    return snapshot_bytes(high);
}

void fp8_gdn_record_dispatch(const Tensor* tree, const Tensor& x, const Weight& w, const Tensor& cw,
                             const Tensor& states, const Tensor& valid, const Tensor& initial,
                             Tensor& record, Tensor& q, Tensor& k, Tensor& v, Tensor& z,
                             LinearPolicy p, WorkspaceArena& ws, cudaStream_t stream) {
    validate(p, x.ne[1], x.ne[1]);
    Tensor flat(x.data, DType::BF16, {5120, x.ne[1] * x.ne[2]});
    volta_projection(
        flat, w,
        LinearBf16SegmentedOutput<10240, 6144>{
            {static_cast<__nv_bfloat16*>(record.data), static_cast<__nv_bfloat16*>(z.data)}},
        &ws, stream);
    if (tree)
        gdn_projected_conv_record_tree_launch(record, cw, states, valid, initial, *tree, q, k, v,
                                              stream);
    else
        gdn_projected_conv_record_launch(record, cw, states, valid, initial, q, k, v, stream);
}

void nvfp4_gdn_record_post_launch(const Tensor& record, const Tensor& cw, const Tensor& states,
                                  const Tensor& valid, const Tensor& initial, Tensor& q, Tensor& k,
                                  Tensor& v, cudaStream_t stream) {
    gdn_projected_conv_record_launch(record, cw, states, valid, initial, q, k, v, stream);
}
} // namespace infernix::ops::detail
