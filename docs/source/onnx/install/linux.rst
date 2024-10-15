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

   .. tab:: Nvidia GPU (CUDA)

      .. code-block:: bash

        git clone https://github.com/k2-fsa/sherpa-onnx
        cd sherpa-onnx
        mkdir build
        cd build
        cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DSHERPA_ONNX_ENABLE_GPU=ON ..
        make -j6

      .. hint::

          You need to install CUDA toolkit. Otherwise, you would get
          errors at runtime.

          You can refer to `<https://k2-fsa.github.io/k2/installation/cuda-cudnn.html>`_
          to install CUDA toolkit.

After building, you will find an executable ``sherpa-onnx`` inside the ``bin`` directory.

That's it!

Please refer to :ref:`sherpa-onnx-pre-trained-models` for a list of pre-trained
models.
