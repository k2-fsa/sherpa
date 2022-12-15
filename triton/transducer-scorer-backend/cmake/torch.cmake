

execute_process(
  COMMAND "python3" -c "import os; import torch; print(os.path.dirname(torch.__file__))"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE TORCH_DIR
)

message(STATUS "TORCH_DIR: ${TORCH_DIR}")

list(APPEND CMAKE_PREFIX_PATH "${TORCH_DIR}")
find_package(Torch REQUIRED)


# set the global CMAKE_CXX_FLAGS so that
# sherpa uses the same ABI flag as PyTorch
string(APPEND CMAKE_CXX_FLAGS " ${TORCH_CXX_FLAGS} ")
