cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(sherpa)

set(SHERPA_VERSION "0.6")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(BUILD_RPATH_USE_ORIGIN TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

if(NOT APPLE)
  set(SHERPA_RPATH_ORIGIN "$ORIGIN")
else()
  set(SHERPA_RPATH_ORIGIN "@loader_path")
endif()

set(CMAKE_INSTALL_RPATH ${SHERPA_RPATH_ORIGIN})
set(CMAKE_BUILD_RPATH ${SHERPA_RPATH_ORIGIN})

set(BUILD_SHARED_LIBS ON)
if(WIN32)
  message(STATUS "Set BUILD_SHARED_LIBS to OFF for Windows")
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
endif()

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No CMAKE_BUILD_TYPE given, default to Release")
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 14 CACHE STRING "The C++ version to be used.")
set(CMAKE_CXX_EXTENSIONS OFF)

option(SHERPA_ENABLE_TESTS "Whether to build tests" ON)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

if(ANDROID)
  set(build_DIR ${CMAKE_SOURCE_DIR}/android/app/build)
  # Pytorch
  file(GLOB PYTORCH_INCLUDE_DIRS "${build_DIR}/pytorch_android*.aar/headers")
  file(GLOB PYTORCH_LINK_DIRS "${build_DIR}/pytorch_android*.aar/jni/${ANDROID_ABI}")

  message("build_DiR" ${build_DIR})
  message("pytorch_DiR" ${PYTORCH_LINK_DIRS})
  message("pytorch_DiR" ${PYTORCH_INCLUDE_DIRS})
  
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

  # k2
  file(GLOB K2_INCLUDE_DIRS "${build_DIR}/k2-*.aar/headers")
  file(GLOB K2_LINK_DIRS "${build_DIR}/k2-*.aar/jni/${ANDROID_ABI}")
  
  find_library(K2_LIBRARIES
    NAMES k2_log k2context k2fsa
    PATHS ${K2_LINK_DIRS}
    NO_CMAKE_FIND_ROOT_PATH
  )
  
  include_directories(
    ${K2_INCLUDE_DIRS}
  )

  # needed by kaldifeat
  set(K2_TORCH_VERSION_MAJOR 1)
  set(K2_TORCH_VERSION_MINOR 11)
  # needed by sherpa
  set(SHERPA_TORCH_VERSION_MAJOR 1)
  set(SHERPA_TORCH_VERSION_MINOR 11)

  # needed by decode_stream.h
  include(kaldifeat)
else()
  include(pybind11)
  include(torch)
  include(k2)
endif()

if(SHERPA_ENABLE_TESTS)
  enable_testing()
  include(googletest)
endif()

# TORCH_VERSION is defined in cmake/torch.cmake
configure_file(
  ${CMAKE_SOURCE_DIR}/sherpa/python/sherpa/torch_version.py.in
  ${CMAKE_SOURCE_DIR}/sherpa/python/sherpa/torch_version.py @ONLY
)

if(WIN32)
  # disable various warnings for MSVC
  # 4624: destructor was implicitly defined as deleted because a base class destructor is inaccessible or deleted
  set(disabled_warnings
      /wd4624
  )
  message(STATUS "Disabled warnings: ${disabled_warnings}")
  foreach(w IN LISTS disabled_warnings)
    string(APPEND CMAKE_CXX_FLAGS " ${w} ")
  endforeach()
endif()

include_directories(${CMAKE_SOURCE_DIR})

message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")

add_subdirectory(sherpa)

if (ANDROID)
  add_subdirectory(android/app/src/main/cpp)
endif()

