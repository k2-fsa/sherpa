.. _sherpa-onnx-jni-linux-build:

Build JNI interface (Linux)
===========================

In the following, we describe how to build the JNI interface for Linux.
It is applicable for both Linux x64 and arm64.

For macOS users, please refer to :ref:`sherpa-onnx-jni-macos-build`

.. hint::

   For Windows users, you have to modify the commands by yourself.

Setup the environment
---------------------

Make sure you have the following two items ready:

  - a working C/C++ compiler that supports C++17
  - you are able to run ``java`` and ``javac`` commands in your terminal.

To check your environment, please run:

.. code-block:: bash

   gcc --version
   java -version
   javac -version

The above three commands print the following output on my computer. You don't need
to use the exact versions as I am using.

.. code-block::

    # output of gcc --version

    gcc (Ubuntu 12.3.0-1ubuntu1~23.04) 12.3.0
    Copyright (C) 2022 Free Software Foundation, Inc.
    This is free software; see the source for copying conditions.  There is NO
    warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

    # output of java -version

    java version "17.0.11" 2024-04-16 LTS
    Java(TM) SE Runtime Environment (build 17.0.11+7-LTS-207)
    Java HotSpot(TM) 64-Bit Server VM (build 17.0.11+7-LTS-207, mixed mode, sharing)

    # output of javac -version

    javac 17.0.11

Build sherpa-onnx
-----------------

Please use the following commands to build `sherpa-onnx`_:

.. code-block::

  git clone https://github.com/k2-fsa/sherpa-onnx

  cd sherpa-onnx

  mkdir build

  cd build

  # If you want to enable GPU support, please
  # set OFF to ON
  SHERPA_ONNX_ENABLE_GPU=OFF

  cmake \
    -DSHERPA_ONNX_ENABLE_GPU=$SHERPA_ONNX_ENABLE_GPU \
    -DSHERPA_ONNX_ENABLE_PYTHON=OFF \
    -DSHERPA_ONNX_ENABLE_TESTS=OFF \
    -DSHERPA_ONNX_ENABLE_CHECK=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF \
    -DSHERPA_ONNX_ENABLE_JNI=ON \
    ..

  make -j4

  # Remove unused libs
  rm lib/lib*.a
  rm lib/libcargs.so

  # You don't need it for jni
  rm lib/libsherpa-onnx-c-api.so

  ls -lh lib

You should see the following output for ``ls -lh lib``::

  total 4.0M
  -rwxrwxr-x 1 fangjun fangjun 4.0M Aug 29 00:56 libsherpa-onnx-jni.so

``libsherpa-onnx-jni.so`` contains the JNI interface for `sherpa-onnx`_.

.. hint::

   You can find ``libonnxruntime.so`` by running::

    fangjun@ubuntu23-04:~/sherpa-onnx/build$ ls _deps/onnxruntime-src/lib/
    libonnxruntime.so

Download pre-built JNI libs
---------------------------

If you don't want to build ``JNI`` libs by yourself, please download pre-built ``JNI``
libs from

    `<https://huggingface.co/csukuangfj/sherpa-onnx-libs/tree/main/jni>`_

For Chinese users, please use

  `<https://hf-mirror.com/csukuangfj/sherpa-onnx-libs/tree/main/jni>`_

Please always use the latest version. In the following, we describe how to download
the version ``1.10.23``.

.. code-block:: bash

   wget https://huggingface.co/csukuangfj/sherpa-onnx-libs/resolve/main/jni/sherpa-onnx-v1.10.23-linux-x64-jni.tar.bz2

   # For Chinese users
   # wget https://hf-mirror.com/csukuangfj/sherpa-onnx-libs/resolve/main/jni/sherpa-onnx-v1.10.23-linux-x64-jni.tar.bz2

   tar xf sherpa-onnx-v1.10.23-linux-x64-jni.tar.bz2
   rm sherpa-onnx-v1.10.23-linux-x64-jni.tar.bz2

.. note::

   You can also download it from

    `<https://github.com/k2-fsa/sherpa-onnx/releases>`_

You should find the following files:

.. code-block:: bash

  ls -lh sherpa-onnx-v1.10.23-linux-x64-jni/lib/

  total 19M
  -rw-r--r-- 1 fangjun fangjun  15M Aug 24 22:18 libonnxruntime.so
  -rwxr-xr-x 1 fangjun fangjun 4.2M Aug 24 22:25 libsherpa-onnx-jni.so
