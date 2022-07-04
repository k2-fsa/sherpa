
if(DEFINED ENV{K2_INSTALL_PREFIX})
  message(STATUS "Using environment variable K2_INSTALL_PREFIX: $ENV{K2_INSTALL_PREFIX}")
  set(K2_CMAKE_PREFIX_PATH $ENV{K2_INSTALL_PREFIX})
else()
  # PYTHON_EXECUTABLE is set by cmake/pybind11.cmake
  message(STATUS "Python executable: ${PYTHON_EXECUTABLE}")

  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c "import k2; print(k2.cmake_prefix_path)"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE K2_CMAKE_PREFIX_PATH
  )
endif()

message(STATUS "K2_CMAKE_PREFIX_PATH: ${K2_CMAKE_PREFIX_PATH}")
list(APPEND CMAKE_PREFIX_PATH "${K2_CMAKE_PREFIX_PATH}")

find_package(k2 REQUIRED)

message(STATUS "K2_FOUND: ${K2_FOUND}")
message(STATUS "K2_INCLUDE_DIRS: ${K2_INCLUDE_DIRS}")
message(STATUS "K2_LIBRARIES: ${K2_LIBRARIES}")
message(STATUS "K2_CXX_FLAGS: ${K2_CXX_FLAGS}")
message(STATUS "K2_CUDA_FLAGS: ${K2_CUDA_FLAGS}")
message(STATUS "K2_TORCH_VERSION_MAJOR: ${K2_TORCH_VERSION_MAJOR}")
message(STATUS "K2_TORCH_VERSION_MINOR: ${K2_TORCH_VERSION_MINOR}")
message(STATUS "K2_WITH_CUDA: ${K2_WITH_CUDA}")
message(STATUS "K2_CUDA_VERSION: ${K2_CUDA_VERSION}")
message(STATUS "K2_VERSION: ${K2_VERSION}")
message(STATUS "K2_GIT_SHA1: ${K2_GIT_SHA1}")
message(STATUS "K2_GIT_DATE: ${K2_GIT_DATE}")

if((NOT K2_TORCH_VERSION_MAJOR VERSION_EQUAL SHERPA_TORCH_VERSION_MAJOR) OR
  (NOT K2_TORCH_VERSION_MINOR VERSION_EQUAL SHERPA_TORCH_VERSION_MINOR))
  message(FATAL_ERROR "k2 was compiled using "
    "PyTorch ${K2_TORCH_VERSION_MAJOR}.${K2_TORCH_VERSION_MINOR}.\n"
    "But you are using PyTorch ${SHERPA_TORCH_VERSION_MAJOR}.${SHERPA_TORCH_VERSION_MINOR} "
    "to compile sherpa. Please make them the same.".
    )
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${K2_CXX_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
