.. _install_sherpa_onnx_on_linux:

Linux
=====

This page describes how to build `sherpa-onnx`_ on Linux.

All you need is to run:

.. tabs::

   .. tab:: CPU

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

   .. tab:: Nvidia GPU (CUDA, x64)

      .. code-block:: bash

        git clone https://github.com/k2-fsa/sherpa-onnx
        cd sherpa-onnx
        mkdir build
        cd build
        cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DSHERPA_ONNX_ENABLE_GPU=ON ..
        make -j6

      .. hint::

          You need to install CUDA toolkit 11.8. Otherwise, you would get
          errors at runtime.

          You can refer to `<https://k2-fsa.github.io/k2/installation/cuda-cudnn.html>`_
          to install CUDA toolkit.

   .. tab:: Nvidia GPU (CUDA 10.2, cudnn8, arm64, e.g., Jetson Nano B01)

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

   .. tab:: Nvidia GPU (CUDA 11.4, cudnn8, arm64, e.g., Jetson Orin NX)

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

   .. tab:: Nvidia GPU (CUDA 12.6, cudnn9, arm64, e.g., Jetson Orin Nano Engineering Reference Developer Kit Super Jetpack 6.2)

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
