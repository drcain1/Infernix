#pragma once

#include "core/weight.h"
#include "infernix/ops/linear.h"
#include "ops/linear/q8/q8_launch.h"

#include <cstdint>

namespace infernix::ops::detail {

Q8Launch select_q8_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t);
Q8Launch select_q8_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy);
std::size_t q8_linear_workspace_capacity_bytes(int n, int k, LinearPolicy policy,
                                               int min_tokens, int max_tokens);

void q8_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                 WorkspaceArena* workspace, cudaStream_t stream);

} // namespace infernix::ops::detail
