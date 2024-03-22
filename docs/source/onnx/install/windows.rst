Windows
=======

This page describes how to build `sherpa-onnx`_ on Windows.


.. hint::

   MinGW is known not to work.
   Please install ``Visual Studio`` before you continue.

.. note::

   You can download pre-compiled binaries for both 32-bit and 64-bit Windows
   from the following URL `<https://huggingface.co/csukuangfj/sherpa-onnx-libs/tree/main>`_.

   Please always download the latest version.

   URLs to download the version ``1.9.12`` is given below.

   .. list-table::

     * - 64-bit Windows (static lib)
       - `<https://huggingface.co/csukuangfj/sherpa-onnx-libs/resolve/main/win64/sherpa-onnx-v1.9.12-win-x64-static.tar.bz2>`_
     * - 64-bit Windows (shared lib)
       - `<https://huggingface.co/csukuangfj/sherpa-onnx-libs/resolve/main/win64/sherpa-onnx-v1.9.12-win-x64-shared.tar.bz2>`_
     * - 32-bit Windows (static lib)
       - `<https://huggingface.co/csukuangfj/sherpa-onnx-libs/resolve/main/win32/sherpa-onnx-v1.9.12-win-x86-static.tar.bz2>`_
     * - 32-bit Windows (shared lib)
       - `<https://huggingface.co/csukuangfj/sherpa-onnx-libs/resolve/main/win32/sherpa-onnx-v1.9.12-win-x86-shared.tar.bz2>`_

   If you cannot access ``huggingface.co``, then please replace ``huggingface.co`` with
   ``hf-mirror.com``.



64-bit Windows (x64)
--------------------

All you need is to run:

.. tabs::

   .. tab:: CPU

      .. code-block:: bash

        git clone https://github.com/k2-fsa/sherpa-onnx
        cd sherpa-onnx
        mkdir build
        cd build
        cmake -DCMAKE_BUILD_TYPE=Release ..
        cmake --build . --config Release

   .. tab:: Nvidia GPU (CUDA)

      .. code-block:: bash

        git clone https://github.com/k2-fsa/sherpa-onnx
        cd sherpa-onnx
        mkdir build
        cd build
        cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DSHERPA_ONNX_ENABLE_GPU=ON ..
        cmake --build . --config Release

      .. hint::

          You need to install CUDA toolkit. Otherwise, you would get
          errors at runtime.

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

  cmake --build . --config Release -- -m:6

After building, you will find an executable ``sherpa-onnx.exe`` inside the ``bin/Release`` directory.

That's it!

Please refer to :ref:`sherpa-onnx-pre-trained-models` for a list of pre-trained
models.

.. hint::

   By default, it builds static libraries of `sherpa-onnx`_. To get dynamic/shared
   libraries, please pass ``-DBUILD_SHARED_LIBS=ON`` to ``cmake``. That is, use

    .. code-block:: bash

        cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON ..
