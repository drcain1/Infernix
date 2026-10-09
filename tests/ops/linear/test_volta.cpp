#include "ops/linear/linear_test_common.h"
#include <iostream>
#include <array>
using namespace infernix;
using namespace infernix::test::linear;

namespace {
test::quantized_weight::PackedWeight fp8(int n, int k, unsigned seed) {
    return test::quantized_weight::pack_volta_fixture(make_fp8_weight(n, k, seed));
}

test::quantized_weight::PackedWeight nvfp4(int n, int k, unsigned seed) {
    return test::quantized_weight::pack_volta_fixture(make_nvfp4_weight(n, k, seed));
}

test::quantized_weight::PackedWeight nvfp4_small_divisor(int n, int k, unsigned seed) {
    auto packed = test::quantized_weight::make_patterned_weight(
        QType::NVFP4, n, k, seed, {.weight_scale_divisor = 1.0e-6F, .input_scale_divisor = 3.5F});
    return test::quantized_weight::pack_volta_fixture(std::move(packed));
}
} // namespace

int main() {
    if (!cuda_available()) return 77;
    try {
        const std::array calls{Invocation{1},
                               Invocation{5},
                               Invocation{8},
                               Invocation{16},
                               Invocation{32},
                               Invocation{33},
                               Invocation{64},
                               Invocation{129},
                               Invocation{5, CallForm::Policy, ops::LinearPolicy::AllowA4, true},
                               Invocation{65, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
                               Invocation{33, CallForm::A16Convenience}};
        int failures = 0;
        for (auto make : {fp8, nvfp4}) {
            for (auto shape :
                 {std::pair{14336, 5120}, std::pair{16384, 5120}, std::pair{34816, 5120},
                  std::pair{5120, 6144}, std::pair{5120, 17408}}) {
                failures += run_shape(
                    make == fp8 ? "Volta FP8" : "Volta NVFP4", ActivationCompute::A16, make,
                    {shape.first, shape.second, 823, Comparison::Sampled, true, calls});
            }
        }
        failures += run_shape("Volta FP8 vocabulary", ActivationCompute::A16, fp8,
                              {248320, 5120, 829, Comparison::Sampled, true, calls});
        const std::array scale_calls{
            Invocation{32}, Invocation{33},
            Invocation{65, CallForm::Policy, ops::LinearPolicy::A16Only, true}};
        failures +=
            run_shape("Volta NVFP4 divisor range", ActivationCompute::A16, nvfp4_small_divisor,
                      {5120, 6144, 839, Comparison::Sampled, true, scale_calls});
        // Offline packing is optional at the Op boundary. Qualify the native
        // storage decoders independently through the wide projection route too.
        for (auto make : {make_fp8_weight, make_nvfp4_weight}) {
            failures += run_shape("Volta native storage", ActivationCompute::A16, make,
                                  {16384, 5120, 853, Comparison::Sampled, true, scale_calls});
        }
        return failures ? 1 : 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
