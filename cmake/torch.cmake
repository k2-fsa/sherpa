# PYTHON_EXECUTABLE is set by cmake/pybind11.cmake
message(STATUS "Python executable: ${PYTHON_EXECUTABLE}")

execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c "import os; import torch; print(os.path.dirname(torch.__file__))"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE TORCH_DIR
)

message(STATUS "PYTHON_EXECUTABLE: ${PYTHON_EXECUTABLE}")
message(STATUS "TORCH_DIR: ${TORCH_DIR}")

list(APPEND CMAKE_PREFIX_PATH "${TORCH_DIR}")
find_package(Torch REQUIRED)
message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")


# set the global CMAKE_CXX_FLAGS so that
# sherpa uses the same ABI flag as PyTorch
string(APPEND CMAKE_CXX_FLAGS " ${TORCH_CXX_FLAGS} ")

execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c "import torch; print(torch.__version__)"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE TORCH_VERSION
)

message(STATUS "PyTorch version: ${TORCH_VERSION}")
