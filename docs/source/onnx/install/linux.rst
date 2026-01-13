.. _install_sherpa_onnx_on_linux:

Linux
=====

This page describes how to build `sherpa-onnx`_ on Linux.


CPU (Linux x64 or Linux arm64)
------------------------------

.. code-block:: bash

  git clone https://github.com/k2-fsa/sherpa-onnx
  cd sherpa-onnx
  mkdir build
  cd build

  # By default, it builds static libaries and uses static link.
  cmake -DCMAKE_BUILD_TYPE=Release ..

  # If you have GCC<=10, e.g., use Ubuntu <= 18.04 or use CentOS<=7, please
  # use the following command to build shared libs; otherwise, you would
  # get link errors from libonnxruntime.a
  #
  # cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON ..
  #
  #
  make -j6

GPU (CUDA 11.8, CUDNN 8, Linux x64)
------------------------------------

.. code-block:: bash

  git clone https://github.com/k2-fsa/sherpa-onnx
  cd sherpa-onnx

  wget https://github.com/csukuangfj/onnxruntime-libs/releases/download/v1.17.1/onnxruntime-linux-x64-gpu-1.17.1-patched.zip
  unzip  onnxruntime-linux-x64-gpu-1.17.1-patched.zip

  export SHERPA_ONNXRUNTIME_LIB_DIR=$PWD/onnxruntime-linux-x64-gpu-1.17.1-patched/lib
  export SHERPA_ONNXRUNTIME_INCLUDE_DIR=$PWD/onnxruntime-linux-x64-gpu-1.17.1-patched/include

  mkdir build
  cd build
  cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DSHERPA_ONNX_ENABLE_GPU=ON ..
  make -j6

.. hint::

    You need to install CUDA toolkit 11.8. Otherwise, you would get
    errors at runtime.

    You can refer to `<https://k2-fsa.github.io/k2/installation/cuda-cudnn.html>`_
    to install CUDA toolkit.

GPU (CUDA 12.8, CUDNN 9, Linux x64)
-----------------------------------

.. code-block:: bash

  git clone https://github.com/k2-fsa/sherpa-onnx
  cd sherpa-onnx
  mkdir build
  cd build
  cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DSHERPA_ONNX_ENABLE_GPU=ON ..
  make -j6

.. hint::

    You need to install CUDA toolkit 12.x with CUDNN 9. Otherwise, you would get
    errors at runtime.

    You can refer to `<https://k2-fsa.github.io/k2/installation/cuda-cudnn.html>`_
    to install CUDA toolkit.

.. note::

    You can download pre-build libraries and executables of sherpa-onnx for CUDA 12.x with CUDNN 9
    at `<https://github.com/k2-fsa/sherpa-onnx/releases>`_. Please always use the latest version.
    For instance, for the version ``1.12.13``, you can use::

      wget https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.12.13/sherpa-onnx-v1.12.13-cuda-12.x-cudnn-9.x-linux-x64-gpu.tar.bz2

GPU (CUDA 10.2, CUDNN8, Linux arm64, e.g., Jetson Nano B01)
-----------------------------------------------------------

.. code-block:: bash

  git clone https://github.com/k2-fsa/sherpa-onnx
  cd sherpa-onnx
  mkdir build
  cd build

  cmake \
    -DSHERPA_ONNX_LINUX_ARM64_GPU_ONNXRUNTIME_VERSION=1.11.0 \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DSHERPA_ONNX_ENABLE_GPU=ON \
    ..

  make

GPU (CUDA 11.4, CUDNN8, Linux arm64, e.g., Jetson Orin NX)
----------------------------------------------------------

.. code-block:: bash

  git clone https://github.com/k2-fsa/sherpa-onnx
  cd sherpa-onnx
  mkdir build
  cd build

  cmake \
    -DSHERPA_ONNX_LINUX_ARM64_GPU_ONNXRUNTIME_VERSION=1.16.0 \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DSHERPA_ONNX_ENABLE_GPU=ON \
    ..

  make

GPU (CUDA 12.6, CUDNN9, Linux arm64, e.g., Jetson Orin Nano Engineering Reference Developer Kit Super Jetpack 6.2)
------------------------------------------------------------------------------------------------------------------

.. code-block:: bash

  git clone https://github.com/k2-fsa/sherpa-onnx
  cd sherpa-onnx
  mkdir build
  cd build

  cmake \
    -DSHERPA_ONNX_LINUX_ARM64_GPU_ONNXRUNTIME_VERSION=1.18.1 \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DSHERPA_ONNX_ENABLE_GPU=ON \
    ..

  make


After building, you will find an executable ``sherpa-onnx`` inside the ``bin`` directory.

That's it!

Please refer to :ref:`sherpa-onnx-pre-trained-models` for a list of pre-trained
models.
