Windows
=======

This page describes how to build `sherpa-onnx`_ on Windows.


.. hint::

   MinGW is known not to work with `sherpa-onnx`_.

   Please install ``Visual Studio 2022`` before you continue.

.. note::

   You can download pre-compiled binaries for both 32-bit and 64-bit Windows
   from the following URL `<https://huggingface.co/csukuangfj/sherpa-onnx-libs/tree/main>`_.

   Please always download the latest version.

   URLs to download the version ``1.12.10`` is given below.

   .. list-table::

     * - 64-bit Windows (static lib)
       - `<https://huggingface.co/csukuangfj/sherpa-onnx-libs/resolve/main/win64/1.12.10/sherpa-onnx-v1.12.10-win-x64-static.tar.bz2>`_
     * - 64-bit Windows (shared lib)
       - `<https://huggingface.co/csukuangfj/sherpa-onnx-libs/resolve/main/win64/1.12.10/sherpa-onnx-v1.12.10-win-x64-shared.tar.bz2>`_
     * - 32-bit Windows (static lib)
       - `<https://huggingface.co/csukuangfj/sherpa-onnx-libs/resolve/main/win32/1.12.10/sherpa-onnx-v1.12.10-win-x86-static.tar.bz2>`_
     * - 32-bit Windows (shared lib)
       - `<https://huggingface.co/csukuangfj/sherpa-onnx-libs/resolve/main/win32/1.12.10/sherpa-onnx-v1.12.10-win-x86-shared.tar.bz2>`_

   If you cannot access ``huggingface.co``, then please replace ``huggingface.co`` with
   ``hf-mirror.com``.



64-bit Windows (x64)
--------------------


CPU
~~~~

.. code-block:: bash

  git clone https://github.com/k2-fsa/sherpa-onnx
  cd sherpa-onnx
  mkdir build
  cd build
  cmake -DCMAKE_BUILD_TYPE=Release ..
  cmake --build . --config Release

GPU (CUDA 11.8)
~~~~~~~~~~~~~~~

.. code-block:: bash

  git clone https://github.com/k2-fsa/sherpa-onnx
  cd sherpa-onnx
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

  curl -SL -O https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-win-x64-gpu-1.22.0.zip
  unzip onnxruntime-win-x64-gpu-1.22.0.zip

  export SHERPA_ONNXRUNTIME_LIB_DIR=$PWD/onnxruntime-win-x64-gpu-1.22.0/lib
  export SHERPA_ONNXRUNTIME_INCLUDE_DIR=$PWD/onnxruntime-win-x64-gpu-1.22.0/include

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

32-bit Windows (x86)
--------------------

.. hint::

   It does not support NVIDIA GPU for ``Win32/x86``.

All you need is to run:

.. code-block:: bash

  git clone https://github.com/k2-fsa/sherpa-onnx
  cd sherpa-onnx
  mkdir build
  cd build

  # Please select one toolset among VS 2015, 2017, 2019, and 2022 below
  # We use VS 2022 as an example.

  # For Visual Studio 2015
  # cmake -T v140,host=x64 -A Win32 -D CMAKE_BUILD_TYPE=Release ..

  # For Visual Studio 2017
  # cmake -T v141,host=x64 -A Win32 -D CMAKE_BUILD_TYPE=Release ..

  # For Visual Studio 2019
  # cmake -T v142,host=x64 -A Win32 -D CMAKE_BUILD_TYPE=Release ..

  # For Visual Studio 2022
  cmake -T v143,host=x64 -A Win32 -D CMAKE_BUILD_TYPE=Release ..

  cmake --build . --config Release

After building, you will find an executable ``sherpa-onnx.exe`` inside the ``bin/Release`` directory.

That's it!

Please refer to :ref:`sherpa-onnx-pre-trained-models` for a list of pre-trained
models.

.. hint::

   By default, it builds static libraries of `sherpa-onnx`_. To get dynamic/shared
   libraries, please pass ``-DBUILD_SHARED_LIBS=ON`` to ``cmake``. That is, use

    .. code-block:: bash

        cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON ..
