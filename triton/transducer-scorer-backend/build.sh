#!/usr/bin/bash

# wget https://download.pytorch.org/libtorch/cu116/libtorch-cxx11-abi-shared-with-deps-1.13.1%2Bcu116.zip 
# unzip -d /mnt/samsung-t7/yuekai/custom_backend/ libtorch-cxx11-abi-shared-with-deps-1.13.1+cu116.zip
export Torch_DIR=/mnt/samsung-t7/yuekai/custom_backend/libtorch
mkdir -p build && cd build && cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install ..
make install
