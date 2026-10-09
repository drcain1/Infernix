target_sources(infernix_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/causal_softmax_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/packed/packed_softmax_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/packed/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/context/context_softmax_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/context/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/sliding_window/sliding_window_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/sliding_window/launch.cu"
)

if(INFERNIX_VOLTA)
  target_sources(infernix_ops PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/volta/launch.cu"
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/volta/flash.cu"
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/volta/flash_workspace.cpp")
  set_source_files_properties("${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/volta/flash.cu"
    PROPERTIES INCLUDE_DIRECTORIES "${PROJECT_SOURCE_DIR}/third_party/llama_cpp_fattn"
    COMPILE_OPTIONS "--extended-lambda")
else()
  target_sources(infernix_ops PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/bf16/launch.cu"
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/bf16/plan.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/fp8/launch.cu"
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/fp8/tiled_launch.cu"
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/fp8/plan.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/int8/launch.cu"
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/int8/plan.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/nvfp4/launch.cu"
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/nvfp4/fast_prompt_launch.cu"
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/nvfp4/plan.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/k8v4/launch.cu"
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/k8v4/tiled_launch.cu"
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/k8v4/plan.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/vq2/launch.cu"
  )
  target_sources(infernix_nvfp4_non_rdc PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/nvfp4/tiled_launch.cu")
endif()
