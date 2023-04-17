Install the Python Package
==========================

You can select one of the following methods to install the Python package.

Method 1 (From pre-compiled wheels)
-----------------------------------

.. hint::

  This method supports only ``x86_64`` machines.

  For other architectures, e.g., Mac M1, Raspberry Pi, etc, please
  use Method 2 or 3.

.. code-block:: bash

  pip install sherpa-onnx

Method 2 (From source)
----------------------

.. code-block:: bash

  git clone https://github.com/k2-fsa/sherpa-onnx
  cd sherpa-onnx
  python3 setup.py install

Method 3 (For developers)
-------------------------

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

