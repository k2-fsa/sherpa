.. _sherpa-onnx-jni-macos-build:

Build JNI interface (macOS)
===========================

In the following, we describe how to build the JNI interface for macOS.
It is applicable for both macOS x64 and arm64.

For Linux users, please refer to :ref:`sherpa-onnx-jni-linux-build`

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

    Apple clang version 14.0.0 (clang-1400.0.29.202)
    Target: x86_64-apple-darwin22.2.0
    Thread model: posix
    InstalledDir: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin

    # output of java -version

    openjdk version "19.0.1" 2022-10-18
    OpenJDK Runtime Environment (build 19.0.1+10-21)
    OpenJDK 64-Bit Server VM (build 19.0.1+10-21, mixed mode, sharing)

    # output of javac -version

    javac 19.0.1

Build sherpa-onnx
-----------------

Please use the following commands to build `sherpa-onnx`_:

.. code-block::

  git clone https://github.com/k2-fsa/sherpa-onnx

  cd sherpa-onnx

  mkdir build

  cd build

  cmake \
    -DSHERPA_ONNX_ENABLE_PYTHON=OFF \
    -DSHERPA_ONNX_ENABLE_TESTS=OFF \
    -DSHERPA_ONNX_ENABLE_CHECK=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF \
    -DSHERPA_ONNX_ENABLE_JNI=ON \
    ..

  make -j4

  # Remove unused files
  rm lib/lib*.a
  rm lib/libcargs.dylib
  rm lib/libsherpa-onnx-c-api.dylib
  rm lib/libsherpa-onnx-cxx-api.dylib

  ls -lh lib

You should see the following output for ``ls -lh lib``::

  total 8024
  -rwxr-xr-x  1 fangjun  staff   3.9M Aug 18 19:34 libsherpa-onnx-jni.dylib

``libsherpa-onnx-jni.dylib`` contains the JNI interface for `sherpa-onnx`_.

.. hint::

   You can find ``libonnxruntime.dylib`` by running::

      fangjuns-MacBook-Pro:build fangjun$ pwd
      /Users/fangjun/open-source/sherpa-onnx/build

      fangjuns-MacBook-Pro:build fangjun$ ls -lh _deps/onnxruntime-src/lib/
      total 51664
      -rwxr-xr-x  1 fangjun  staff    25M Aug 14 14:09 libonnxruntime.1.17.1.dylib
      drwxr-xr-x  3 fangjun  staff    96B Aug 14 14:09 libonnxruntime.1.17.1.dylib.dSYM
      lrwxr-xr-x  1 fangjun  staff    27B Aug 14 14:09 libonnxruntime.dylib -> libonnxruntime.1.17.1.dylib


Download pre-built JNI libs
---------------------------

If you don't want to build ``JNI`` libs by yourself, please download pre-built ``JNI``
libs from

    `<https://github.com/k2-fsa/sherpa-onnx/releases>`_

Please always use the latest version. In the following, we describe how to download
the version `1.11.1 <https://github.com/k2-fsa/sherpa-onnx/releases/tag/v1.11.1>`_.

.. tabs::

   .. tab:: Intel CPU (x86_64)

      .. code-block:: bash

         wget https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.11.1/sherpa-onnx-v1.11.1-osx-x86_64-jni.tar.bz2

         tar xf sherpa-onnx-v1.11.1-osx-x86_64-jni.tar.bz2
         rm sherpa-onnx-v1.11.1-osx-x86_64-jni.tar.bz2

   .. tab:: Apple Silicon (arm64)

      .. code-block:: bash

         wget https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.11.1/sherpa-onnx-v1.11.1-osx-arm64-jni.tar.bz2

         tar xf sherpa-onnx-v1.11.1-osx-arm64-jni.tar.bz2
         rm sherpa-onnx-v1.11.1-osx-arm64-jni.tar.bz2

After downloading, you should see the following files:

.. code-block:: bash

  # For x86_64
  ls -lh sherpa-onnx-v1.11.1-osx-x86_64-jni/lib
  total 30M
  -rw-r--r-- 1 fangjun fangjun  26M Aug 25 00:31 libonnxruntime.1.17.1.dylib
  lrwxrwxrwx 1 fangjun fangjun   27 Aug 25 00:35 libonnxruntime.dylib -> libonnxruntime.1.17.1.dylib
  -rwxr-xr-x 1 fangjun fangjun 3.9M Aug 25 00:35 libsherpa-onnx-jni.dylib

  # For arm64
  ls -lh sherpa-onnx-v1.11.1-osx-arm64-jni/lib/
  total 27M
  -rw-r--r-- 1 fangjun fangjun  23M Aug 24 23:56 libonnxruntime.1.17.1.dylib
  lrwxrwxrwx 1 fangjun fangjun   27 Aug 24 23:59 libonnxruntime.dylib -> libonnxruntime.1.17.1.dylib
  -rwxr-xr-x 1 fangjun fangjun 3.6M Aug 24 23:59 libsherpa-onnx-jni.dylib
