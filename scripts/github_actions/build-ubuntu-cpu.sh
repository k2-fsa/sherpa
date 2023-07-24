#!/usr/bin/env bash
#
set -ex

if [ -z $PYTHON_VERSION ]; then
  echo "Please set the environment variable PYTHON_VERSION"
  echo "Example: export PYTHON_VERSION=3.8"
  # Valid values: 3.8, 3.9, 3.10, 3.11
  exit 1
fi

if [ -z $TORCH_VERSION ]; then
  echo "Please set the environment variable TORCH_VERSION"
  echo "Example: export TORCH_VERSION=1.10.0"
  exit 1
fi

echo "Installing ${PYTHON_VERSION}.3"

yum -y install openssl-devel bzip2-devel libffi-devel xz-devel wget redhat-lsb-core

curl -O https://www.python.org/ftp/python/${PYTHON_VERSION}.3/Python-${PYTHON_VERSION}.3.tgz
tar xf Python-${PYTHON_VERSION}.3.tgz
pushd Python-${PYTHON_VERSION}.3

PYTHON_INSTALL_DIR=$PWD/py-${PYTHON_VERSION}

if [[ $PYTHON_VERSION =~ 3.1. ]]; then
  yum install -y openssl11-devel
  sed -i 's/PKG_CONFIG openssl /PKG_CONFIG openssl11 /g' configure
fi

./configure --enable-shared --prefix=$PYTHON_INSTALL_DIR >/dev/null 2>&1
make install >/dev/null 2>&1

popd

export PATH=$PYTHON_INSTALL_DIR/bin:$PATH
export LD_LIBRARY_PATH=$PYTHON_INSTALL_DIR/lib:$LD_LIBRARY_PATH
ls -lh $PYTHON_INSTALL_DIR/lib/

nvcc --version || true
rm -rf /usr/local/cuda*
nvcc --version || true

python3 --version
which python3

if [[ $PYTHON_VERSION != 3.6 ]]; then
  curl -O https://bootstrap.pypa.io/get-pip.py
  python3 get-pip.py
fi

python3 -m pip install scikit-build
python3 -m pip install -U pip cmake
python3 -m pip install wheel twine typing_extensions
python3 -m pip install bs4 requests tqdm auditwheel

echo "Installing torch $TORCH_VERSION"
python3 -m pip install -qq torch==$TORCH_VERSION+cpu -f https://download.pytorch.org/whl/torch_stable.html

echo "Install k2 1.24.3.dev20230719+cpu.torch${TORCH_VERSION}"
pip install k2==1.24.3.dev20230719+cpu.torch${TORCH_VERSION} -f https://k2-fsa.github.io/k2/cpu.html

echo "Installing kaldifeat 1.24.dev20230722+cpu.torch${TORCH_VERSION}"
pip install kaldifeat==1.24.dev20230722+cpu.torch${TORCH_VERSION} -f https://csukuangfj.github.io/kaldifeat/cpu.html

rm -rf ~/.cache/pip
yum clean all

cd /var/www

export CMAKE_CUDA_COMPILER_LAUNCHER=
export KALDIFEAT_CMAKE_ARGS=" -DPYTHON_EXECUTABLE=$PYTHON_INSTALL_DIR/bin/python3 "
export KALDIFEAT_MAKE_ARGS=" -j "

python3 setup.py bdist_wheel

auditwheel --verbose repair \
  --exclude libc10.so \
  --exclude libc10_cuda.so \
  --exclude libcuda.so.1 \
  --exclude libcudart.so.${CUDA_VERSION} \
  --exclude libnvToolsExt.so.1 \
  --exclude libnvrtc.so.${CUDA_VERSION} \
  --exclude libtorch.so \
  --exclude libtorch_cpu.so \
  --exclude libtorch_cuda.so \
  --exclude libtorch_python.so \
  \
  --exclude libcudnn.so.8 \
  --exclude libcublas.so.11 \
  --exclude libcublasLt.so.11 \
  --exclude libcudart.so.11.0 \
  --exclude libnvrtc.so.11.2 \
  --exclude libtorch_cuda_cu.so \
  --exclude libtorch_cuda_cpp.so \
  \
  --exclude libkaldifeat_core.so \
  \
  --plat manylinux_2_17_x86_64 \
  -w /var/www/wheelhouse \
  dist/*.whl

ls -lh  /var/www
