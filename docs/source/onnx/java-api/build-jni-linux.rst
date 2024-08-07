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

  cmake \
    -DSHERPA_ONNX_ENABLE_PYTHON=OFF \
    -DSHERPA_ONNX_ENABLE_TESTS=OFF \
    -DSHERPA_ONNX_ENABLE_CHECK=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF \
    -DSHERPA_ONNX_ENABLE_JNI=ON \
    ..

  make -j4

  ls -lh lib

You should see the following output for ``ls -lh lib``::

  total 7.7M
  -rwxrwxr-x 1 fangjun fangjun  16K May 15 03:53 libcargs.so
  -rwxrwxr-x 1 fangjun fangjun 396K May 15 03:53 libespeak-ng.so
  -rwxrwxr-x 1 fangjun fangjun 652K May 15 03:53 libkaldi-decoder-core.so
  -rwxrwxr-x 1 fangjun fangjun 108K May 15 03:53 libkaldi-native-fbank-core.so
  lrwxrwxrwx 1 fangjun fangjun   23 May 15 03:53 libpiper_phonemize.so -> libpiper_phonemize.so.1
  lrwxrwxrwx 1 fangjun fangjun   27 May 15 03:53 libpiper_phonemize.so.1 -> libpiper_phonemize.so.1.2.0
  -rwxrwxr-x 1 fangjun fangjun 450K May 15 03:53 libpiper_phonemize.so.1.2.0
  -rwxrwxr-x 1 fangjun fangjun 107K May 15 03:55 libsherpa-onnx-c-api.so
  -rwxrwxr-x 1 fangjun fangjun 2.4M May 15 03:54 libsherpa-onnx-core.so
  lrwxrwxrwx 1 fangjun fangjun   26 May 15 03:53 libsherpa-onnx-fstfar.so -> libsherpa-onnx-fstfar.so.7
  -rwxrwxr-x 1 fangjun fangjun  18K May 15 03:53 libsherpa-onnx-fstfar.so.7
  lrwxrwxrwx 1 fangjun fangjun   23 May 15 03:53 libsherpa-onnx-fst.so -> libsherpa-onnx-fst.so.6
  -rwxrwxr-x 1 fangjun fangjun 2.1M May 15 03:53 libsherpa-onnx-fst.so.6
  -rwxrwxr-x 1 fangjun fangjun 134K May 15 03:55 libsherpa-onnx-jni.so
  -rwxrwxr-x 1 fangjun fangjun 1.2M May 15 03:53 libsherpa-onnx-kaldifst-core.so
  -rwxrwxr-x 1 fangjun fangjun 229K May 15 03:53 libucd.so

Note that all these ``*.so`` files are required.

``libsherpa-onnx-jni.so`` contains the JNI interface for `sherpa-onnx`_.


Download pre-built JNI libs
---------------------------

If you don't want to build ``JNI`` libs by yourself, please download pre-built ``JNI``
libs from

    `<https://huggingface.co/csukuangfj/sherpa-onnx-libs/tree/main/jni>`_

For Chinese users, please use

  `<https://hf-mirror.com/csukuangfj/sherpa-onnx-libs/tree/main/jni>`_

Please always use the latest version. In the following, we describe how to download
the version ``1.10.2``.

.. code-block:: bash

   wget https://huggingface.co/csukuangfj/sherpa-onnx-libs/resolve/main/jni/sherpa-onnx-v1.10.2-linux-x64-jni.tar.bz2

   # For Chinese users
   # wget https://hf-mirror.com/csukuangfj/sherpa-onnx-libs/resolve/main/jni/sherpa-onnx-v1.10.2-linux-x64-jni.tar.bz2

   tar xf sherpa-onnx-v1.10.2-linux-x64-jni.tar.bz2
   rm sherpa-onnx-v1.10.2-linux-x64-jni.tar.bz2

You should find the following files:

.. code-block:: bash

  ls -lh sherpa-onnx-v1.10.2-linux-x64-jni/lib/

  -rwxr-xr-x  1 fangjun  staff    16K Jun 25 11:41 libcargs.so
  -rwxr-xr-x  1 fangjun  staff   383K Jun 25 11:41 libespeak-ng.so
  -rwxr-xr-x  1 fangjun  staff   648K Jun 25 11:41 libkaldi-decoder-core.so
  -rwxr-xr-x  1 fangjun  staff   214K Jun 25 11:41 libkaldi-native-fbank-core.so
  -rw-r--r--  1 fangjun  staff    15M Jun 25 11:40 libonnxruntime.so
  -rw-r--r--  1 fangjun  staff    15M Jun 25 11:40 libonnxruntime.so.1.17.1
  lrwxr-xr-x  1 fangjun  staff    23B Jun 25 11:44 libpiper_phonemize.so -> libpiper_phonemize.so.1
  lrwxr-xr-x  1 fangjun  staff    27B Jun 25 11:44 libpiper_phonemize.so.1 -> libpiper_phonemize.so.1.2.0
  -rwxr-xr-x  1 fangjun  staff   535K Jun 25 11:41 libpiper_phonemize.so.1.2.0
  -rwxr-xr-x  1 fangjun  staff   128K Jun 25 11:44 libsherpa-onnx-c-api.so
  -rwxr-xr-x  1 fangjun  staff   2.5M Jun 25 11:43 libsherpa-onnx-core.so
  -rwxr-xr-x  1 fangjun  staff   2.1M Jun 25 11:41 libsherpa-onnx-fst.so
  -rwxr-xr-x  1 fangjun  staff    18K Jun 25 11:41 libsherpa-onnx-fstfar.so
  -rwxr-xr-x  1 fangjun  staff   155K Jun 25 11:44 libsherpa-onnx-jni.so
  -rwxr-xr-x  1 fangjun  staff   1.1M Jun 25 11:41 libsherpa-onnx-kaldifst-core.so
  -rwxr-xr-x  1 fangjun  staff    81K Jun 25 11:41 libsherpa-onnx-portaudio.so
  -rwxr-xr-x  1 fangjun  staff   275K Jun 25 11:41 libssentencepiece_core.so
  -rwxr-xr-x  1 fangjun  staff   224K Jun 25 11:40 libucd.so
