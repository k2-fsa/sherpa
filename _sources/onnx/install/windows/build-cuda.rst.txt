Build from source for Windows (With NVIDIA GPU)
====================================================

GPU (CUDA 11.8)
~~~~~~~~~~~~~~~

.. code-block:: bash

  git clone https://github.com/k2-fsa/sherpa-onnx
  cd sherpa-onnx

  curl -SL -O https://github.com/microsoft/onnxruntime/releases/download/v1.17.1/onnxruntime-win-x64-gpu-1.17.1.zip
  unzip onnxruntime-win-x64-gpu-1.17.1.zip

  export SHERPA_ONNXRUNTIME_LIB_DIR=$PWD/onnxruntime-win-x64-gpu-1.17.1/lib
  export SHERPA_ONNXRUNTIME_INCLUDE_DIR=$PWD/onnxruntime-win-x64-gpu-1.17.1/include

  mkdir build
  cd build
  cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DSHERPA_ONNX_ENABLE_GPU=ON ..
  cmake --build . --config Release

.. hint::

    You need to install CUDA toolkit 11.8. Otherwise, you would get
    errors at runtime.

    Caution: Please install cuda toolkit 11.8. Other versions do ``NOT`` work!

    Caution: Please install cuda toolkit 11.8. Other versions do ``NOT`` work!

    Caution: Please install cuda toolkit 11.8. Other versions do ``NOT`` work!

    If it crashes without any errors, please see

      `<https://github.com/k2-fsa/sherpa-onnx/issues/2138>`_

    You need to also install ``cudnn``.

GPU (CUDA 12.x)
~~~~~~~~~~~~~~~


.. code-block:: bash

  # We assume you use git bash commandline to run the following commands for curl, unzip, and export

  git clone https://github.com/k2-fsa/sherpa-onnx
  cd sherpa-onnx
  mkdir build
  cd build
  cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DSHERPA_ONNX_ENABLE_GPU=ON ..
  cmake --build . --config Release

.. note::

    You can download pre-build libraries and executables of sherpa-onnx for CUDA 12.x with CUDNN 9
    at `<https://github.com/k2-fsa/sherpa-onnx/releases>`_. Please always use the latest version.
    For instance, for the version ``1.12.13``, you can use::

      wget https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.12.13/sherpa-onnx-v1.12.13-cuda-12.x-cudnn-9.x-win-x64-cuda.tar.bz2

After building, you will find an executable ``sherpa-onnx.exe`` inside the ``bin/Release`` directory.

That's it!

Please refer to :ref:`sherpa-onnx-pre-trained-models` for a list of pre-trained
models.
