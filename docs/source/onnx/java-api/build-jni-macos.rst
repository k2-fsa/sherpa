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

  ls -lh lib

You should see the following output for ``ls -lh lib``::

  total 11576
  -rwxr-xr-x  1 fangjun  staff    33K May 15 11:24 libcargs.dylib
  -rwxr-xr-x  1 fangjun  staff   350K May 15 11:24 libespeak-ng.dylib
  -rwxr-xr-x  1 fangjun  staff   471K May 15 11:24 libkaldi-decoder-core.dylib
  -rwxr-xr-x  1 fangjun  staff   113K May 15 11:24 libkaldi-native-fbank-core.dylib
  -rwxr-xr-x  1 fangjun  staff   423K May 15 11:24 libpiper_phonemize.1.2.0.dylib
  lrwxr-xr-x  1 fangjun  staff    30B May 15 11:24 libpiper_phonemize.1.dylib -> libpiper_phonemize.1.2.0.dylib
  lrwxr-xr-x  1 fangjun  staff    26B May 15 11:24 libpiper_phonemize.dylib -> libpiper_phonemize.1.dylib
  -rwxr-xr-x  1 fangjun  staff   113K May 15 11:25 libsherpa-onnx-c-api.dylib
  -rwxr-xr-x  1 fangjun  staff   1.8M May 15 11:24 libsherpa-onnx-core.dylib
  -rwxr-xr-x  1 fangjun  staff   1.5M May 15 11:24 libsherpa-onnx-fst.6.dylib
  lrwxr-xr-x  1 fangjun  staff    26B May 15 11:24 libsherpa-onnx-fst.dylib -> libsherpa-onnx-fst.6.dylib
  -rwxr-xr-x  1 fangjun  staff    34K May 15 11:24 libsherpa-onnx-fstfar.7.dylib
  lrwxr-xr-x  1 fangjun  staff    29B May 15 11:24 libsherpa-onnx-fstfar.dylib -> libsherpa-onnx-fstfar.7.dylib
  -rwxr-xr-x  1 fangjun  staff   127K May 15 11:25 libsherpa-onnx-jni.dylib
  -rwxr-xr-x  1 fangjun  staff   933K May 15 11:24 libsherpa-onnx-kaldifst-core.dylib
  -rwxr-xr-x  1 fangjun  staff   187K May 15 11:24 libucd.dylib


Note that all these ``*.dylib`` files are required.

``libsherpa-onnx-jni.dylib`` contains the JNI interface for `sherpa-onnx`_.
