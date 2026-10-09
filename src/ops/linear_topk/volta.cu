#include "ops/linear_topk/linear_topk_launch.h"
#include "ops/linear/fp8/fp8_volta_qpn_gemm.cuh"
#include "ops/linear/q4/q4_volta_qpn_gemm.cuh"
#include "ops/linear/q8/q8_volta_qpn_gemm.cuh"
#include "ops/common/score_id_order.cuh"
#include "core/device.h"
#include <cub/block/block_merge_sort.cuh>

namespace infernix::ops::detail {
namespace {
__global__ void select_partial(const float* logits, const int* ids, int rows, int valid, int groups,
                               std::uint64_t* partial) {
    using Sort = cub::BlockMergeSort<std::uint64_t, 128, 1>;
    __shared__ typename Sort::TempStorage storage;
    const int row        = blockIdx.x * 128 + threadIdx.x;
    std::uint64_t key[1] = {
        row < valid ? score_id_order_key(logits[static_cast<long long>(blockIdx.y) * rows + row],
                                         ids ? ids[row] : row)
                    : 0};
    Sort(storage).Sort(key, ScoreIdOrderGreater{});
    if (threadIdx.x < 16)
        partial[(static_cast<long long>(blockIdx.y) * groups + blockIdx.x) * 16 + threadIdx.x] =
            key[0];
}

void select(const Weight& w, int valid, const int* ids, const LinearTopKWorkspace& ws,
            cudaStream_t stream) {
    select_partial<<<dim3(ws.producer_groups, ws.columns), 128, 0, stream>>>(
        static_cast<const float*>(ws.logits.data), ids, w.n, valid, ws.producer_groups,
        static_cast<std::uint64_t*>(ws.partial_keys.data));
    CUDA_CHECK(cudaGetLastError());
}
} // namespace

void linear_topk_fp8_launch(const Tensor& x, const Weight& w, int valid,
                            const LinearTopKWorkspace& ws, cudaStream_t stream) {
    launch_fp8_volta_qpn_with_output(
        x, w, Fp8Fp32ContiguousOutput{static_cast<float*>(ws.logits.data), w.n}, w.n, stream);
    select(w, valid, nullptr, ws, stream);
}

void linear_topk_q8_launch(const Tensor& x, const Weight& w, int valid,
                           const LinearTopKWorkspace& ws, cudaStream_t stream) {
    q8_volta_qpn_gemm_kernel<1, 4, Fp8Fp32ContiguousOutput><<<(w.n + 31) / 32, 128, 0, stream>>>(
        static_cast<const unsigned char*>(w.qdata), static_cast<const unsigned char*>(w.scales),
        static_cast<const __nv_bfloat16*>(x.data), nullptr, w.n, w.k, x.ne[1],
        w.padded_shape[1] / 32, 0,
        Fp8Fp32ContiguousOutput{static_cast<float*>(ws.logits.data), w.n});
    CUDA_CHECK(cudaGetLastError());
    select(w, valid, nullptr, ws, stream);
}

void linear_topk_q4_launch(const Tensor& x, const Weight& w, const Tensor& ids,
                           const LinearTopKWorkspace& ws, cudaStream_t stream) {
    q4_volta_qpn_gemm_kernel<1, 4, Fp8Fp32ContiguousOutput><<<(w.n + 31) / 32, 128, 0, stream>>>(
        static_cast<const unsigned char*>(w.qdata), static_cast<const unsigned char*>(w.scales),
        static_cast<const __nv_bfloat16*>(x.data), nullptr, w.n, w.k, x.ne[1],
        w.padded_shape[1] / 64, 0,
        Fp8Fp32ContiguousOutput{static_cast<float*>(ws.logits.data), w.n});
    CUDA_CHECK(cudaGetLastError());
    select(w, w.n, static_cast<const int*>(ids.data), ws, stream);
}
} // namespace infernix::ops::detail
