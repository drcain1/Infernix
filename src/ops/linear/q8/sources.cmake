# The Volta dispatcher uses QPN or predicated SIMT. Keep the generic MMA launcher
# for dynamic grouped convolution, which owns a separate finite composition.
target_sources(infernix_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/q8_mma.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8_simt.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8_dispatch.cpp"
)
if(INFERNIX_VOLTA)
  target_sources(infernix_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/q8_volta_qpn_gemm.cu")
else()
target_sources(infernix_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/q8_sliced_k_mma.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8_gemv.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n1024_k2048.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n1024_k5120.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n10240_k2560.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n12288_k2048.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n12288_k2560.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n14336_k5120.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n17408_k5120.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n2048_k16384.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n2048_k4096.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n2048_k4608.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n248320_k5120.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n2560_k6144.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n16384_k2560.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n34816_k5120.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n4608_k4608.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n5120_k10240.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n5120_k17408.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n5120_k25600.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n5120_k4608.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n5120_k6144.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n6144_k2560.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n6144_k5120.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n9216_k2048.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n16384_k2560.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n12800_k2560.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n13952_k2560.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n2560_k2560.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n1280_k2560.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n2560_k6144.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n324_k10240.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n320_k10240.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n10240_k320.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n2560_k640.cu"
)

endif()
