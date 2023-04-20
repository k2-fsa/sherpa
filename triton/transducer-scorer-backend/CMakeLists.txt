# Copyright 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.17...3.22 FATAL_ERROR)

project(TransducerScorerBackend LANGUAGES C CXX)

#
# Options
#
# Must include options required for this project as well as any
# projects included in this one by FetchContent.
#
option(TRITON_ENABLE_GPU "Enable GPU support in backend" ON)
option(TRITON_ENABLE_STATS "Include statistics collections in backend" ON)

set(TRITON_COMMON_REPO_TAG "r22.11" CACHE STRING "Tag for triton-inference-server/common repo")
set(TRITON_CORE_REPO_TAG "r22.11" CACHE STRING "Tag for triton-inference-server/core repo")
set(TRITON_BACKEND_REPO_TAG "r22.11" CACHE STRING "Tag for triton-inference-server/backend repo")

if(NOT CMAKE_BUILD_TYPE)
 set(CMAKE_BUILD_TYPE Release)
endif()

#
# Dependencies
#
# FetchContent requires us to include the transitive closure of all
# repos that we depend on so that we can override the tags.
#
include(FetchContent)

FetchContent_Declare(
  repo-common
  GIT_REPOSITORY https://github.com/triton-inference-server/common.git
  GIT_TAG ${TRITON_COMMON_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-core
  GIT_REPOSITORY https://github.com/triton-inference-server/core.git
  GIT_TAG ${TRITON_CORE_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-backend
  GIT_REPOSITORY https://github.com/triton-inference-server/backend.git
  GIT_TAG ${TRITON_BACKEND_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_MakeAvailable(repo-common repo-core repo-backend)


#
# CUDA
#
if(${TRITON_ENABLE_GPU})
  find_package(CUDAToolkit REQUIRED)
endif()

find_package(Python3 REQUIRED COMPONENTS Development)
#
# The backend must be built into a shared library. Use an ldscript to
# hide all symbols except for the TRITONBACKEND API.
#
configure_file(src/libtriton_scorer.ldscript libtriton_scorer.ldscript COPYONLY)

add_library(
  triton-transudcer-scorer-backend SHARED
  src/scorer.cc
  src/scorer_utils.h
  src/bls.h
  src/bls.cc
  src/bls_utils.h
  src/bls_utils.cc
  src/symbol-table.cc
  src/symbol-table.h
)

add_library(
  TransducerScorerBackend::triton-transducer-scorer-backend ALIAS triton-transudcer-scorer-backend
)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
find_package(Torch REQUIRED)
message(STATUS "    torch include path: ${TORCH_INCLUDE_DIRS}")
message(STATUS "    torch lib path : ${TORCH_LIBRARIES} ")

target_include_directories(
  triton-transudcer-scorer-backend
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${Python3_INCLUDE_DIRS}
)

target_include_directories(
  triton-transudcer-scorer-backend
  PRIVATE ${TORCH_INCLUDE_DIRS}
)

target_compile_features(triton-transudcer-scorer-backend PRIVATE cxx_std_11)
target_compile_options(
  triton-transudcer-scorer-backend PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall -Wextra -Wno-unused-parameter -Wno-type-limits -Werror>
  $<$<CXX_COMPILER_ID:MSVC>:/Wall /D_WIN32_WINNT=0x0A00 /EHsc>
)

if(${TRITON_ENABLE_GPU})
  target_compile_definitions(
    triton-transudcer-scorer-backend
    PRIVATE TRITON_ENABLE_GPU=1
  )
endif() # TRITON_ENABLE_GPU

target_link_libraries(
  triton-transudcer-scorer-backend
  PRIVATE
    triton-core-serverapi   # from repo-core
    triton-core-backendapi  # from repo-core
    triton-core-serverstub  # from repo-core
    triton-backend-utils    # from repo-backend
    ${TORCH_LIBRARIES}
)

if(${TRITON_ENABLE_GPU})
  target_link_libraries(
    triton-transudcer-scorer-backend
    PRIVATE
      CUDA::cudart
  )
endif() # TRITON_ENABLE_GPU


set_target_properties(
  triton-transudcer-scorer-backend PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  OUTPUT_NAME triton_scorer
  LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_scorer.ldscript
  LINK_FLAGS "-Wl,--version-script libtriton_scorer.ldscript"
)


#
# Install
#
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/TransducerScorerBackend)

install(
  TARGETS
    triton-transudcer-scorer-backend
  EXPORT
    triton-transudcer-scorer-backend-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/scorer
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/scorer
)

install(
  EXPORT
    triton-transudcer-scorer-backend-targets
  FILE
    TransducerScorerBackendTargets.cmake
  NAMESPACE
    TransducerScorerBackend::
  DESTINATION
    ${INSTALL_CONFIGDIR}
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/TransducerScorerBackendConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/TransducerScorerBackendConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

install(
  FILES
  ${CMAKE_CURRENT_BINARY_DIR}/TransducerScorerBackendConfig.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)

#
# Export from build tree
#
export(
  EXPORT triton-transudcer-scorer-backend-targets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/TransducerScorerBackendTargets.cmake
  NAMESPACE TransducerScorerBackend::
)

export(PACKAGE TransducerScorerBackend)