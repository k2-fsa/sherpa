.. _install_sherpa_onnx_python:

Install the Python Package
==========================

You can select one of the following methods to install the Python package.

Method 1 (From pre-compiled wheels, CPU only)
---------------------------------------------

.. hint::

  This method supports the following platfroms:

    - Linux (``x64``, ``aarch64``, ``armv7l``),
    - macOS (``x64``, ``arm64``)
    - Windows (``x64``, ``x86``)

  Note that this method installs a CPU-only version of `sherpa-onnx`_.

.. code-block:: bash

  pip install sherpa-onnx

To check you have installed `sherpa-onnx`_ successfully, please run

.. code-block:: bash

  python3 -c "import sherpa_onnx; print(sherpa_onnx.__file__)"

  which sherpa-onnx
  sherpa-onnx --help

  ls -lh $(dirname $(which sherpa-onnx))/sherpa-onnx*

.. hint::

   You can find previous releases at
   `<https://k2-fsa.github.io/sherpa/onnx/cpu.html>`_

   For Chinese users and users who have no access to huggingface, please visit
   `<https://k2-fsa.github.io/sherpa/onnx/cpu-cn.html>`_.

   You can use::

    pip install sherpa-onnx -f https://k2-fsa.github.io/sherpa/onnx/cpu.html

   or::

    # For Chinese uers
    pip install sherpa-onnx -f https://k2-fsa.github.io/sherpa/onnx/cpu-cn.html

Method 2 (From pre-compiled wheels, CPU + CUDA)
------------------------------------------------

.. note::

   This method installs a version of `sherpa-onnx`_ supporting both ``CUDA``
   and ``CPU``. You need to pass the argument ``provider=cuda`` to use
   NVIDIA GPU, which always uses GPU 0. Otherwise, it uses ``CPU`` by default.

   Please use the environment variable ``CUDA_VISIBLE_DEVICES`` to control
   which GPU is mapped to GPU 0.

   By default, ``provider`` is set to ``cpu``.

   Remeber to follow `<https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements>`_
   to install CUDA 11.8.

   If you have issues about installing CUDA 11.8, please have a look at
   `<https://k2-fsa.github.io/k2/installation/cuda-cudnn.html#cuda-11-8>`_.

   Note that you don't need to have ``sudo`` permission to install CUDA 11.8

This approach supports only Linux x64 and Windows x64.

Please use the following command to install CUDA-enabled `sherpa-onnx`_::

  # We use 1.10.16 here for demonstration.
  #
  # Please visit https://k2-fsa.github.io/sherpa/onnx/cuda.html
  # to find available versions

  pip install sherpa-onnx==1.10.16+cuda -f https://k2-fsa.github.io/sherpa/onnx/cuda.html

  # For Chinese users, please use
  # pip install sherpa-onnx==1.10.16+cuda -f https://k2-fsa.github.io/sherpa/onnx/cuda-cn.html

The installation logs are given below::

  Looking in links: https://k2-fsa.github.io/sherpa/onnx/cuda.html
  Collecting sherpa-onnx==1.10.16+cuda
    Downloading https://huggingface.co/csukuangfj/sherpa-onnx-wheels/resolve/main/cuda/1.10.16/sherpa_onnx-1.10.16%2Bcuda-cp310-cp310-linux_x86_64.whl (183.3 MB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 183.3/183.3 MB 4.4 MB/s eta 0:00:00
  Installing collected packages: sherpa-onnx
  Successfully installed sherpa-onnx-1.10.16+cuda

To check that you have installed `sherpa-onnx`_ successfully, please run::

  python3 -c "import sherpa_onnx; print(sherpa_onnx.__version__)"

which should print something like below::

  1.10.16+cuda



Method 3 (From source)
----------------------

.. tabs::

   .. tab:: CPU

      .. code-block:: bash

        git clone https://github.com/k2-fsa/sherpa-onnx
        cd sherpa-onnx
        python3 setup.py install

   .. tab:: Nvidia GPU (CUDA)

      .. code-block:: bash

        git clone https://github.com/k2-fsa/sherpa-onnx
        export SHERPA_ONNX_CMAKE_ARGS="-DSHERPA_ONNX_ENABLE_GPU=ON"
        cd sherpa-onnx
        python3 setup.py install

Method 4 (For developers)
-------------------------

.. tabs::

   .. tab:: CPU

    .. code-block:: bash

      git clone https://github.com/k2-fsa/sherpa-onnx
      cd sherpa-onnx
      mkdir build
      cd build

      cmake \
        -DSHERPA_ONNX_ENABLE_PYTHON=ON \
        -DBUILD_SHARED_LIBS=ON \
        -DSHERPA_ONNX_ENABLE_CHECK=OFF \
        -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF \
        -DSHERPA_ONNX_ENABLE_C_API=OFF \
        -DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF \
        ..

      make -j
      export PYTHONPATH=$PWD/../sherpa-onnx/python/:$PWD/lib:$PYTHONPATH

   .. tab:: Nvidia GPU (CUDA)

      .. code-block:: bash

        git clone https://github.com/k2-fsa/sherpa-onnx
        cd sherpa-onnx
        mkdir build
        cd build

        cmake \
          -DSHERPA_ONNX_ENABLE_PYTHON=ON \
          -DBUILD_SHARED_LIBS=ON \
          -DSHERPA_ONNX_ENABLE_CHECK=OFF \
          -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF \
          -DSHERPA_ONNX_ENABLE_C_API=OFF \
          -DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF \
          -DSHERPA_ONNX_ENABLE_GPU=ON \
          ..

        make -j
        export PYTHONPATH=$PWD/../sherpa-onnx/python/:$PWD/lib:$PYTHONPATH

      .. hint::

          You need to install CUDA toolkit. Otherwise, you would get
          errors at runtime.

          You can refer to `<https://k2-fsa.github.io/k2/installation/cuda-cudnn.html>`_
          to install CUDA toolkit.


Check your installation
-----------------------

To check that `sherpa-onnx`_ has been successfully installed, please use:

.. code-block:: bash

  python3 -c "import sherpa_onnx; print(sherpa_onnx.__file__)"

It should print some output like below:

.. code-block:: bash

  /Users/fangjun/py38/lib/python3.8/site-packages/sherpa_onnx/__init__.py

Please refer to:

  `<https://github.com/k2-fsa/sherpa-onnx/tree/master/python-api-examples>`_

for usages.

Please refer to :ref:`sherpa-onnx-pre-trained-models` for a list of pre-trained
models.

