cmake_minimum_required(VERSION 3.28)

# Select one native implementation. Set and validate the
# architecture before CUDA compiler detection so CMake never injects a generic
# host-toolchain default or accepts an unsupported target.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120a CACHE STRING "CUDA architectures to build")
endif()
if(NOT CMAKE_CUDA_ARCHITECTURES MATCHES "^(70|120a)$")
  message(FATAL_ERROR
    "Infernix supports CMAKE_CUDA_ARCHITECTURES=70 or 120a; got "
    "'${CMAKE_CUDA_ARCHITECTURES}'")
endif()

project(infernix LANGUAGES C CXX CUDA)

if(CMAKE_CUDA_ARCHITECTURES STREQUAL "70")
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.9 OR CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13)
    message(FATAL_ERROR "The V100 target requires CUDA 12.9; CUDA 13 removed sm_70 compilation")
  endif()
  set(INFERNIX_VOLTA ON)
  add_compile_definitions(INFERNIX_VOLTA=1)
endif()

# --- Global settings ---------------------------------------------------------
option(INFERNIX_BUILD_APPS "Build the Infernix CLI, server and perplexity evaluator" ON)
option(BUILD_TESTING "Build Infernix tests" OFF)
option(INFERNIX_BUILD_BENCHMARKS "Build Infernix benchmarks" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)        # clangd / IDE tooling
if(MSVC)
  add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
  # Sources are UTF-8. Without /utf-8 MSVC reads them in the system code page, and the bundled fmt
  # rejects any code page but UTF-8 (a stock English or German Windows uses 1252).
  add_compile_options(
    $<$<COMPILE_LANGUAGE:C,CXX>:/Zc:preprocessor>
    $<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
  # CUDA's static runtime requests LIBCMT even when the application and vcpkg dependencies use
  # the project-wide /MD runtime. Keep one CRT in the final process.
  add_link_options($<HOST_LINK:/NODEFAULTLIB:LIBCMT>)
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_GENERATOR MATCHES "Ninja")
  set_property(GLOBAL PROPERTY JOB_POOLS infernix_link=1)
  set(CMAKE_JOB_POOL_LINK infernix_link)
endif()

# Tests also exercise product input, logging and protocol adapters.
set(INFERNIX_BUILD_PRODUCT_SUPPORT OFF)
if(INFERNIX_BUILD_APPS OR BUILD_TESTING)
  set(INFERNIX_BUILD_PRODUCT_SUPPORT ON)
endif()

include(cmake/Dependencies.cmake)
include(cmake/InfernixTargets.cmake)

# --- Subdirectories ----------------------------------------------------------
add_subdirectory(src)
if(INFERNIX_BUILD_APPS)
  add_subdirectory(apps)
endif()
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
if(INFERNIX_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()
