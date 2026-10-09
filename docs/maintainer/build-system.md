# Build system

Infernix builds from its source tree with CMake 3.28+ and C++/CUDA 20.
The default architecture is `sm_120a`; CUDA 13.1 is the validated Blackwell development toolkit.
The V100 port selects `sm_70` and requires CUDA 12.9 (CUDA 13 removes offline Volta compilation).
Its scope is dense Qwen 27B through `Qwen3_5ForCausalLM`, including Qwen3.6/3.8 artifacts;
MoE and Flash-Next are excluded from that build. See the
[V100 comparison and qualification limits](../performance/qwen3.8-27b.md#v100-comparison-with-ninfer).
Product commands and prerequisites are in the
[README](../../README.md#quick-start-windows); test and measurement workflows live in
[tests](../../tests/README.md) and [benchmarks](../../bench/README.md).

## Configuration

The root `CMakeLists.txt` owns language/toolchain constraints, build options and top-level
composition. Architecture selection and validation happen before `project()` detects CUDA.
The default configuration is Release. Ninja links and archives share the single-slot
`infernix_link` pool; compilation uses the build command's parallelism.

| Cache option | Default | Scope |
|---|---|---|
| `INFERNIX_BUILD_APPS` | ON | CLI, HTTP server and perplexity evaluator |
| `BUILD_TESTING` | OFF | C++ tests and registered Python interoperability tests |
| `INFERNIX_BUILD_BENCHMARKS` | OFF | Op, model, Engine and context-cost benchmarks |

Apps or tests enable the internal product support components: media acquisition, prompt input,
logging and serving. This is one derived condition, not a separate user option. FFmpeg belongs
to model media decoding and remains required when apps are disabled. Curl and spdlog are needed
only for product support. A Python 3 interpreter is found when tests are configured. The larger Python
pytest suites and conversion/evaluation scripts run separately from CMake and CTest.

CUDA 13.1 and Python 3.11 describe the maintained environment, not configuration version gates.
CMake also discovers FFmpeg without imposing library version floors. Actual language/API support
is exercised by compilation and tests. The libcurl 7.85 minimum has a concrete API basis:
media acquisition uses `CURLOPT_PROTOCOLS_STR` and `CURLOPT_REDIR_PROTOCOLS_STR`, introduced in
[that release](https://curl.se/libcurl/c/CURLOPT_PROTOCOLS_STR.html).

The checked-in `CMakePresets.json` provides:

- `release`: Release product binaries, tests and benchmarks disabled;
- `dev`: Release products, tests and benchmarks enabled;
- `v100`: Release products in `build-v100/`, targeting `sm_70`;
- matching build presets and a `dev` test preset with failure output enabled.

The `release` and `dev` configure presets use `build/`. Switching between them explicitly resets all three build
options; it does not create independent build trees. Build with `cmake --build build -j`.
For a separate configuration, override the binary directory with `-B build-<name>` and use that
directory in subsequent build/test commands.

For the V100 target, select the CUDA 12.9 compiler explicitly if several toolkits are installed:

```sh
cmake --preset v100 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc
cmake --build build-v100 -j
```

The toolkit path is local configuration. Volta tensor-core projection and attention headers are
vendored under `third_party/`; their version and license records are included there. FP8 and NVFP4
use FP16 tensor-core arithmetic with FP32 accumulation on this target. Activation A4/A8 permissions
select the available A16 implementation; they do not require hardware FP4/FP8 instructions.
BF16, INT8 and FP8 KV storage have Volta attention routes; other KV formats are not admitted.
For small quantized projections (T <= 32), workspace-taking routes stage FP16 activations
once per Op, avoiding repeated BF16 conversion in every output-row block. Their resource
queries include this temporary buffer; workspace-free calls convert within the kernel.
Wide quantized projections use temporary FP16 operands and FP32 accumulation in the Program's
planned workspace; Q8 staging is bounded to 2,048-token tiles (512 for the vision merger's
`[5120,4608]` geometry). Specialized MoE, Flash-Next and
chunked-GDN implementation benchmarks are omitted from this build. The public Engine inference
benchmark remains available with `INFERNIX_BUILD_BENCHMARKS=ON`.
Convert existing v3 weights to the persistent Volta layout before serving:

```sh
python3.11 -m tools.artifact.pack_volta out/model.infernix out/model-volta.infernix
```

This permutation preserves represented weight values and embedded frontend resources. It does
not upgrade v2 NInfer containers. Packing belongs to the offline artifact workflow; the Engine
uploads the stored representation. See [storage layouts](storage-layouts.md) for its exact codec.

Keep machine paths in the ignored `CMakeUserPresets.json`. For example, replacing the interpreter
placeholder with the interpreter from the selected environment:

```json
{
  "version": 6,
  "configurePresets": [
    {
      "name": "local",
      "inherits": "dev",
      "cacheVariables": {
        "Python3_EXECUTABLE": "/path/to/python3"
      }
    }
  ]
}
```

Configure that environment with `cmake --preset local`. Compiler paths may also be supplied here;
changing compilers requires a fresh build directory. There is no tools option, installed SDK,
package export or configure-time dependency download.

## Targets and dependencies

`cmake/Dependencies.cmake` discovers system CUDA, Threads, FFmpeg and conditional curl, and
exposes the repository-pinned JSON/HTTP headers and conditional spdlog library. External include
requirements follow their consuming targets. `cmake/InfernixTargets.cmake` provides private
project includes and the two CUDA archive policies.

`src/CMakeLists.txt` explicitly enters the component directories. Each component's
`CMakeLists.txt` owns its targets, dependency visibility and compile properties:

| Component | Targets |
|---|---|
| `core/` | `infernix_core` |
| `artifact/` | `infernix_artifact` |
| `text/` | `infernix_text`, including bundled utf8proc C source |
| `media/` | `infernix_media_decode` |
| `ops/` | `infernix_ops`, `infernix_nvfp4_non_rdc` |
| `models/` | `infernix_model_loading`, `infernix_model_runtime` |
| `runtime/` | `infernix_runtime_support`, `infernix_engine` and its `infernix::engine` alias |
| `product/` | media acquisition, prompt input and logging libraries |
| `serve/` | `infernix_serve` |

Component declaration order is not the execution dependency graph: CMake resolves named target
links during generation. Model runtime can refer to the runtime support target declared later;
support links core, while Engine links model runtime. The ownership and dependency contracts are
defined by [Engine architecture](engine-architecture.md) and [Op development](op-development.md).

## Explicit source ownership

Small components list their sources directly. Larger components include local `sources.cmake`
manifests which use `target_sources(<owner> PRIVATE ...)`. Included manifests anchor file paths
with `CMAKE_CURRENT_LIST_DIR`; `include()` does not change `CMAKE_CURRENT_SOURCE_DIR`.

- Every implementation under `src/` has one target owner and one explicit registration.
- Ops register simple horizontal implementations in `basic_sources.cmake` and larger semantic
  families in their own manifests. A family also registers its files under `wrapper/`.
- Linear includes one manifest for each of BF16, FP8, NVFP4, Q4, Q5, Q6 and Q8. Each manifest lists
  all its shape filenames and their actual `.cpp`/`.cu` extensions.
- Models separate loading, frontend, execution and Program source lists while retaining the
  distinct immutable-loading and mutable-runtime targets. Checkpoints/recipes do not define
  additional compile owners.
- New files require an explicit manifest edit. New families require an explicit parent include.
  There is no recursive glob, directory discovery or generated shape/dispatch registry.

Adding a source manifest does not require another library. Add a target only for a meaningful
dependency, reuse or compilation boundary. Shape semantics, supported domains and finite
dispatch remain in C++; CMake only selects translation units.

## CUDA compilation boundaries

`infernix_core` and `infernix_ops` enable separable compilation (RDC) and resolve device symbols in
the static archive build. Keep this device-link boundary explicit.

`infernix_nvfp4_non_rdc` disables separable compilation and device-symbol resolution. It contains
the warp-specialized Linear, LinearSwiGLU and causal-attention NVFP4 sources that depend on
`setmaxnreg` register transfer. Their owning family manifests register these sources into the
non-RDC target; host launchers connect them to the normal Ops. All three CUDA archive targets
retain `-lineinfo`.

Source-list maintenance must preserve language, architecture, RDC mode, device-link ownership
and numerical compiler options. Splitting a manifest does not reduce kernel instantiation work
or justify combining shape translation units or enabling unity builds.

## Consumers and verification

`apps/CMakeLists.txt` explicitly defines the three product executables. Tests and benchmarks
include their domain registrations without creating new CMake subdirectories, preserving
`build/tests/`, `build/bench/` and CTest working directories. The helpers are local to those
consumers. Tests explicitly name their linked libraries; public-header and pure-host checks keep
their isolated include/link requirements.

Linear, LinearAdd and LinearSwiGLU reuse compiled test support libraries. Oracle options apply
to those libraries themselves as well as the test executables. Test variants, skip return codes,
interop commands and special linker options belong with the affected registrations.

For build-organization changes, compare configured target/source ownership, compile flags,
device links and CTest commands/properties; check the relevant option combinations, build the
affected targets and run their behavioral tests. A shape-source edit should rebuild its own
translation unit plus necessary downstream links. Build-time or inference-speed claims require
measurements at that scope.
