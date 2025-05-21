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

   # For Chinese users
   pip install sherpa-onnx -f https://k2-fsa.github.io/sherpa/onnx/rk-npu-cn.html

To check that you have installed `sherpa-onnx`_ with rknn support, please run

.. code-block:: bash

  (py310) orangepi@orangepi5max:~/t$ ldd $(which sherpa-onnx)
    linux-vdso.so.1 (0x0000007f9fd93000)
    librknnrt.so => /lib/librknnrt.so (0x0000007f9f480000)
    libonnxruntime.so => /home/orangepi/py310/bin/../lib/python3.10/site-packages/sherpa_onnx/lib/libonnxruntime.so (0x0000007f9e7f0000)
    libm.so.6 => /lib/aarch64-linux-gnu/libm.so.6 (0x0000007f9e750000)
    libstdc++.so.6 => /lib/aarch64-linux-gnu/libstdc++.so.6 (0x0000007f9e520000)
    libgcc_s.so.1 => /lib/aarch64-linux-gnu/libgcc_s.so.1 (0x0000007f9e4f0000)
    libc.so.6 => /lib/aarch64-linux-gnu/libc.so.6 (0x0000007f9e340000)
    /lib/ld-linux-aarch64.so.1 (0x0000007f9fd5a000)
    libpthread.so.0 => /lib/aarch64-linux-gnu/libpthread.so.0 (0x0000007f9e320000)
    libdl.so.2 => /lib/aarch64-linux-gnu/libdl.so.2 (0x0000007f9e300000)
    librt.so.1 => /lib/aarch64-linux-gnu/librt.so.1 (0x0000007f9e2e0000)

You should check that ``librknnrt.so`` is in the dependency list.

If you cannot find ``librknnrt.so``, it means you have failed to install `sherpa-onnx`_
with rknn support. In that case, please visit

  - `<https://k2-fsa.github.io/sherpa/onnx/rk-npu.html>`_
  - For Chinese users `<https://k2-fsa.github.io/sherpa/onnx/rk-npu-cn.html>`_

and download the ``whl`` file to your board and use ``pip install ./*.whl``
to install from ``whl``. Remember to recheck with ``ldd $(which sherpa-onnx)``.

**Caution**: Please use the following command to check the version of your ``librknnrt.so``:

.. code-block:: bash

  strings /lib/librknnrt.so | grep "librkrnnrt version"

It should print something like below::

  librknnrt version: 2.2.0 (c195366594@2024-09-14T12:18:56)

``librknnrt.so`` ``2.2.0`` is known to work on rk3588.

You can download ``librknnrt.so`` ``2.2.0`` from:

  `<https://github.com/airockchip/rknn-toolkit2/blob/v2.2.0/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so>`_

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
