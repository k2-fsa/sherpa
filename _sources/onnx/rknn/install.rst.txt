.. _sherpa-onnx-rknn-install:

Install
=======

You can use any methods below to build and install `sherpa-onnx`_ for RKNPU.

From pre-built wheels using pip install
---------------------------------------

You can find pre-built ``whl`` files at  `<https://k2-fsa.github.io/sherpa/onnx/rk-npu.html>`_.

To install it, you can use:

.. code-block:: bash

   pip install sherpa-onnx -f https://k2-fsa.github.io/sherpa/onnx/rk-npu.html

   # 中国用户
   pip install sherpa-onnx -f https://k2-fsa.github.io/sherpa/onnx/rk-npu-cn.html

Build sherpa-onnx directly on your board
----------------------------------------

.. code-block:: bash

   git clone https://github.com/k2-fsa/sherpa-onnx
   cd sherpa-onnx
   mkdir build
   cd build

   cmake \
     -DSHERPA_ONNX_ENABLE_RKNN=ON \
     -DCMAKE_INSTALL_PREFIX=./install \
     ..

   make
   make install

Cross-compiling
---------------

Please first refer to :ref:`sherpa-onnx-linux-aarch64-cross-compiling`
to install toolchains.

.. warning::

   The toolchains for dynamic linking and static linking are different.

After installing a toolchain by following :ref:`sherpa-onnx-linux-aarch64-cross-compiling`

Dynamic link
~~~~~~~~~~~~

.. code-block:: bash

  git clone https://github.com/k2-fsa/sherpa-onnx
  cd sherpa-onnx
  export BUILD_SHARED_LIBS=ON
  ./build-rknn-linux-aarch64.sh

Static link
~~~~~~~~~~~

.. code-block:: bash

  git clone https://github.com/k2-fsa/sherpa-onnx
  cd sherpa-onnx
  export BUILD_SHARED_LIBS=OFF
  ./build-rknn-linux-aarch64.sh
