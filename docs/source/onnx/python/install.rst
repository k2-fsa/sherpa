.. _install_sherpa_onnx_python:

Install the Python Package
==========================

You can select one of the following methods to install the Python package.

Method 1 (From pre-compiled wheels)
-----------------------------------

.. hint::

  This method supports ``x86_64``, ``arm64`` (e.g., Mac M1, 64-bit Raspberry Pi),
  and ``arm32`` (e.g., 32-bit Raspberry Pi).

.. code-block:: bash

  pip install sherpa-onnx

To check you have installed `sherpa-onnx`_ successfully, please run

.. code-block:: bash

  python3 -c "import sherpa_onnx; print(sherpa_onnx.__file__)"

  which sherpa-onnx
  sherpa-onnx --help

  ls -lh $(dirname $(which sherpa-onnx))/sherpa-onnx*

Method 2 (From source)
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

Method 3 (For developers)
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

