target_sources(infernix_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/gated_delta_net/gated_delta_net.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/gated_delta_net/replay.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/gated_delta_net/recurrent.cu"
  "$<$<NOT:$<BOOL:${INFERNIX_VOLTA}>>:${CMAKE_CURRENT_LIST_DIR}/gated_delta_net/chunked/prepare.cu>"
  "$<$<NOT:$<BOOL:${INFERNIX_VOLTA}>>:${CMAKE_CURRENT_LIST_DIR}/gated_delta_net/chunked/recurrence.cu>"
  "$<$<NOT:$<BOOL:${INFERNIX_VOLTA}>>:${CMAKE_CURRENT_LIST_DIR}/kimi_delta_attention/kimi_delta_attention.cpp>"
  "$<$<NOT:$<BOOL:${INFERNIX_VOLTA}>>:${CMAKE_CURRENT_LIST_DIR}/kimi_delta_attention/recurrent.cu>"
  "$<$<NOT:$<BOOL:${INFERNIX_VOLTA}>>:${CMAKE_CURRENT_LIST_DIR}/kimi_delta_attention/chunked/prepare.cu>"
  "$<$<NOT:$<BOOL:${INFERNIX_VOLTA}>>:${CMAKE_CURRENT_LIST_DIR}/kimi_delta_attention/chunked/recurrence.cu>"
)
