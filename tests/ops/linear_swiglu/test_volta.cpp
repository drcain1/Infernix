#include "ops/linear_swiglu/linear_swiglu_test_common.h"
#include "ops/op_tester.h"
#include <array>
#include <exception>
#include <iostream>

int main() {
    using namespace infernix;
    using namespace infernix::test::linear_swiglu;
    if (test::cuda_unavailable()) return 77;
    try {
        constexpr std::array tokens{1, 5, 8, 16, 32, 33, 65, 129};
        constexpr std::array graphs{5, 65};
        int failures = 0;
        for (auto format : {QType::FP8_E4M3FN_ROW_BF16, QType::NVFP4})
            failures += run_profile(
                "Volta packed SwiGLU",
                {format, 34816, 5120, 17408, 1839U, ActivationCompute::A16, true}, tokens, graphs);
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
