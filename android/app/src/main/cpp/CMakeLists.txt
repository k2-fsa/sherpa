cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(k2_device CXX)

set(K2_ANDROID "0.1")

set(SHERPA_ROOT ${CMAKE_SOURCE_DIR}/../../../../..)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(BUILD_RPATH_USE_ORIGIN TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_RPATH "$ORIGIN")
set(CMAKE_BUILD_RPATH "$ORIGIN")

set(CMAKE_CXX_STANDARD 14 CACHE STRING "The C++ version to be used.")
set(CMAKE_CXX_EXTENSIONS OFF)
message(STATUS "C++ Standard version: ${CMAKE_CXX_STANDARD}")

list(APPEND CMAKE_MODULE_PATH ${SHERPA_ROOT}/cmake/Modules)
list(APPEND CMAKE_MODULE_PATH ${SHERPA_ROOT}/cmake)

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No CMAKE_BUILD_TYPE given, default to Release")
  set(CMAKE_BUILD_TYPE Release)
endif()

option(BUILD_SHARED_LIBS "Whether to build shared or static lib" ON)

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${SHERPA_ROOT})

set(build_DIR ${CMAKE_SOURCE_DIR}/../../../build)

# Pytorch
file(GLOB PYTORCH_INCLUDE_DIRS "${build_DIR}/pytorch_android*.aar/headers")
file(GLOB PYTORCH_LINK_DIRS "${build_DIR}/pytorch_android*.aar/jni/${ANDROID_ABI}")

find_library(PYTORCH_LIBRARIES pytorch_jni
  PATHS ${PYTORCH_LINK_DIRS}
  NO_CMAKE_FIND_ROOT_PATH
)

find_library(FBJNI_LIBRARIES fbjni
  PATHS ${PYTORCH_LINK_DIRS}
  NO_CMAKE_FIND_ROOT_PATH
)

list(APPEND TORCH_LIBRARIES ${PYTORCH_LIBRARIES})
list(APPEND TORCH_LIBRARIES ${FBJNI_LIBRARIES})

include_directories(
  ${PYTORCH_INCLUDE_DIRS}
  ${PYTORCH_INCLUDE_DIRS}/torch/csrc/api/include
)

message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")

# needed by kaldifeat
set(K2_TORCH_VERSION_MAJOR 1)
set(K2_TORCH_VERSION_MINOR 11)
# needed by sherpa
set(SHERPA_TORCH_VERSION_MAJOR 1)
set(SHERPA_TORCH_VERSION_MINOR 11)

link_libraries(log libc++_shared.so android)

include(kaldifeat)

add_subdirectory(${SHERPA_ROOT}/sherpa/csrc ${CMAKE_BINARY_DIR}/lib)

set(bin_dep_libs
  ${TORCH_LIBRARIES}
  sherpa_core
)

add_library(k2_device SHARED jni/k2.cc)
set_property(TARGET k2_device PROPERTY CXX_STANDARD 14)
target_link_libraries(k2_device ${bin_dep_libs})
