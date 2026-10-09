target_sources(infernix_ops PRIVATE
  "$<$<NOT:$<BOOL:${INFERNIX_VOLTA}>>:${CMAKE_CURRENT_LIST_DIR}/fp8/fp8_gdn_input_decode.cu>"
  "$<$<NOT:$<BOOL:${INFERNIX_VOLTA}>>:${CMAKE_CURRENT_LIST_DIR}/fp8/fp8_gdn_input_matrix.cu>"
  "$<$<NOT:$<BOOL:${INFERNIX_VOLTA}>>:${CMAKE_CURRENT_LIST_DIR}/fp8/fp8_gdn_input_a8.cu>"
  "$<$<NOT:$<BOOL:${INFERNIX_VOLTA}>>:${CMAKE_CURRENT_LIST_DIR}/fp8/fp8_gdn_input_plan.cpp>"
  "$<$<NOT:$<BOOL:${INFERNIX_VOLTA}>>:${CMAKE_CURRENT_LIST_DIR}/fp8/fp8_gdn_conv_fused.cu>"
  "$<$<NOT:$<BOOL:${INFERNIX_VOLTA}>>:${CMAKE_CURRENT_LIST_DIR}/fp8/fp8_gdn_conv_plan.cpp>"
  "$<$<NOT:$<BOOL:${INFERNIX_VOLTA}>>:${CMAKE_CURRENT_LIST_DIR}/nvfp4/nvfp4_gdn_input_decode.cu>"
  "$<$<NOT:$<BOOL:${INFERNIX_VOLTA}>>:${CMAKE_CURRENT_LIST_DIR}/nvfp4/nvfp4_gdn_input_small_t.cu>"
  "$<$<NOT:$<BOOL:${INFERNIX_VOLTA}>>:${CMAKE_CURRENT_LIST_DIR}/nvfp4/nvfp4_gdn_input_a16.cu>"
  "$<$<NOT:$<BOOL:${INFERNIX_VOLTA}>>:${CMAKE_CURRENT_LIST_DIR}/nvfp4/nvfp4_gdn_input_a4.cu>"
  "$<$<NOT:$<BOOL:${INFERNIX_VOLTA}>>:${CMAKE_CURRENT_LIST_DIR}/nvfp4/nvfp4_gdn_input_plan.cpp>"
  "$<$<NOT:$<BOOL:${INFERNIX_VOLTA}>>:${CMAKE_CURRENT_LIST_DIR}/nvfp4/nvfp4_gdn_snapshot_decode.cu>"
  "$<$<NOT:$<BOOL:${INFERNIX_VOLTA}>>:${CMAKE_CURRENT_LIST_DIR}/nvfp4/nvfp4_gdn_snapshot_small_t.cu>"
  "$<$<NOT:$<BOOL:${INFERNIX_VOLTA}>>:${CMAKE_CURRENT_LIST_DIR}/nvfp4/nvfp4_gdn_snapshot_post.cu>"
  "$<$<NOT:$<BOOL:${INFERNIX_VOLTA}>>:${CMAKE_CURRENT_LIST_DIR}/nvfp4/nvfp4_gdn_snapshot_plan.cpp>"
  "${CMAKE_CURRENT_LIST_DIR}/q4_q5/q4_q5_gdn_input_gemm_mma.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q4_q5/q4_q5_gdn_input_independent.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q4_q5/q4_q5_gdn_input_conv_snapshot.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q4_q5/q4_q5_gdn_input_plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/gdn_projected_conv.cu"
  "$<$<NOT:$<BOOL:${INFERNIX_VOLTA}>>:${CMAKE_CURRENT_LIST_DIR}/q8/q8_gdn_input_decode.cu>"
  "$<$<NOT:$<BOOL:${INFERNIX_VOLTA}>>:${CMAKE_CURRENT_LIST_DIR}/q8/q8_gdn_input_gemm_mma.cu>"
  "$<$<NOT:$<BOOL:${INFERNIX_VOLTA}>>:${CMAKE_CURRENT_LIST_DIR}/q8/q8_gdn_input_gemm_splitk.cu>"
  "${CMAKE_CURRENT_LIST_DIR}/q8/q8_gdn_input_plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/../wrapper/gdn_input_proj.cpp"
)

if(NOT INFERNIX_VOLTA)
target_sources(infernix_nvfp4_non_rdc PRIVATE
  "$<$<NOT:$<BOOL:${INFERNIX_VOLTA}>>:${CMAKE_CURRENT_LIST_DIR}/nvfp4/nvfp4_gdn_input_a4_tma.cu>"
)
endif()

if(INFERNIX_VOLTA)
  target_sources(infernix_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/volta.cu")
endif()
