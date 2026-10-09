target_sources(infernix_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/linear_topk.cpp"
  "$<$<NOT:$<BOOL:${INFERNIX_VOLTA}>>:${CMAKE_CURRENT_LIST_DIR}/q8.cu>"
  "$<$<NOT:$<BOOL:${INFERNIX_VOLTA}>>:${CMAKE_CURRENT_LIST_DIR}/q8_m64.cu>"
  "$<$<NOT:$<BOOL:${INFERNIX_VOLTA}>>:${CMAKE_CURRENT_LIST_DIR}/fp8.cu>"
  "$<$<NOT:$<BOOL:${INFERNIX_VOLTA}>>:${CMAKE_CURRENT_LIST_DIR}/fp8_m64.cu>"
  "$<$<NOT:$<BOOL:${INFERNIX_VOLTA}>>:${CMAKE_CURRENT_LIST_DIR}/q4.cu>"
  "$<$<NOT:$<BOOL:${INFERNIX_VOLTA}>>:${CMAKE_CURRENT_LIST_DIR}/q4_m64.cu>"
  "${CMAKE_CURRENT_LIST_DIR}/merge.cu"
)

if(INFERNIX_VOLTA)
  target_sources(infernix_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/volta.cu")
endif()
