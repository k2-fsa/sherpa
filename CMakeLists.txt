cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(sherpa)

set(SHERPA_VERSION "0.2")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(BUILD_RPATH_USE_ORIGIN TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_RPATH "$ORIGIN")
set(CMAKE_BUILD_RPATH "$ORIGIN")

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

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

include(pybind11)
include(torch)

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
