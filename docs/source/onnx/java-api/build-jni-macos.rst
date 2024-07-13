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

Download pre-built JNI libs
---------------------------

If you don't want to build ``JNI`` libs by yourself, please download pre-built ``JNI``
libs from

    `<https://huggingface.co/csukuangfj/sherpa-onnx-libs/tree/main/jni>`_

For Chinese users, please use

  `<https://hf-mirror.com/csukuangfj/sherpa-onnx-libs/tree/main/jni>`_

Please always use the latest version. In the following, we describe how to download
the version ``1.10.2``.

.. tabs::

   .. tab:: Intel CPU (x86_64)

      .. code-block:: bash

         wget https://huggingface.co/csukuangfj/sherpa-onnx-libs/resolve/main/jni/sherpa-onnx-v1.10.2-osx-x86_64-jni.tar.bz2

         # For Chinese users
         # wget https://hf-mirror.com/csukuangfj/sherpa-onnx-libs/resolve/main/jni/sherpa-onnx-v1.10.2-osx-x86_64-jni.tar.bz2

         tar xf sherpa-onnx-v1.10.2-osx-x86_64-jni.tar.bz2
         rm sherpa-onnx-v1.10.2-osx-x86_64-jni.tar.bz2

   .. tab:: Apple Silicon (arm64)

      .. code-block:: bash

         wget https://huggingface.co/csukuangfj/sherpa-onnx-libs/resolve/main/jni/sherpa-onnx-v1.10.2-osx-arm64-jni.tar.bz2

         # For Chinese users
         # wget https://hf-mirror.com/csukuangfj/sherpa-onnx-libs/resolve/main/jni/sherpa-onnx-v1.10.2-osx-arm64-jni.tar.bz2

         tar xf sherpa-onnx-v1.10.2-osx-arm64-jni.tar.bz2
         rm sherpa-onnx-v1.10.2-osx-arm64-jni.tar.bz2

After downloading, you should see the following files:

.. code-block:: bash

  # For x86_64
  ls -lh sherpa-onnx-v1.10.2-osx-x86_64-jni/lib/

  -rwxr-xr-x  1 fangjun  staff    33K Jun 25 11:39 libcargs.dylib
  -rwxr-xr-x  1 fangjun  staff   335K Jun 25 11:39 libespeak-ng.dylib
  -rwxr-xr-x  1 fangjun  staff   484K Jun 25 11:39 libkaldi-decoder-core.dylib
  -rwxr-xr-x  1 fangjun  staff   121K Jun 25 11:39 libkaldi-native-fbank-core.dylib
  -rw-r--r--  1 fangjun  staff    25M Jun 25 11:39 libonnxruntime.1.17.1.dylib
  lrwxr-xr-x  1 fangjun  staff    27B Jun 25 11:39 libonnxruntime.dylib -> libonnxruntime.1.17.1.dylib
  -rwxr-xr-x  1 fangjun  staff   427K Jun 25 11:39 libpiper_phonemize.1.2.0.dylib
  lrwxr-xr-x  1 fangjun  staff    30B Jun 25 11:39 libpiper_phonemize.1.dylib -> libpiper_phonemize.1.2.0.dylib
  lrwxr-xr-x  1 fangjun  staff    26B Jun 25 11:39 libpiper_phonemize.dylib -> libpiper_phonemize.1.dylib
  -rwxr-xr-x  1 fangjun  staff   115K Jun 25 11:39 libsherpa-onnx-c-api.dylib
  -rwxr-xr-x  1 fangjun  staff   1.9M Jun 25 11:39 libsherpa-onnx-core.dylib
  -rwxr-xr-x  1 fangjun  staff   1.5M Jun 25 11:39 libsherpa-onnx-fst.dylib
  -rwxr-xr-x  1 fangjun  staff    40K Jun 25 11:39 libsherpa-onnx-fstfar.dylib
  -rwxr-xr-x  1 fangjun  staff   134K Jun 25 11:39 libsherpa-onnx-jni.dylib
  -rwxr-xr-x  1 fangjun  staff   963K Jun 25 11:39 libsherpa-onnx-kaldifst-core.dylib
  -rwxr-xr-x  1 fangjun  staff   117K Jun 25 11:39 libsherpa-onnx-portaudio.dylib
  -rwxr-xr-x  1 fangjun  staff   150K Jun 25 11:39 libssentencepiece_core.dylib
  -rwxr-xr-x  1 fangjun  staff   187K Jun 25 11:39 libucd.dylib

  # For arm64
  ls -lh sherpa-onnx-v1.10.2-osx-arm64-jni/lib/

  -rwxr-xr-x  1 fangjun  staff    49K Jun 25 11:39 libcargs.dylib
  -rwxr-xr-x  1 fangjun  staff   318K Jun 25 11:39 libespeak-ng.dylib
  -rwxr-xr-x  1 fangjun  staff   436K Jun 25 11:39 libkaldi-decoder-core.dylib
  -rwxr-xr-x  1 fangjun  staff   122K Jun 25 11:39 libkaldi-native-fbank-core.dylib
  -rw-r--r--  1 fangjun  staff    23M Jun 25 11:39 libonnxruntime.1.17.1.dylib
  lrwxr-xr-x  1 fangjun  staff    27B Jun 25 11:39 libonnxruntime.dylib -> libonnxruntime.1.17.1.dylib
  -rwxr-xr-x  1 fangjun  staff   430K Jun 25 11:39 libpiper_phonemize.1.2.0.dylib
  lrwxr-xr-x  1 fangjun  staff    30B Jun 25 11:39 libpiper_phonemize.1.dylib -> libpiper_phonemize.1.2.0.dylib
  lrwxr-xr-x  1 fangjun  staff    26B Jun 25 11:39 libpiper_phonemize.dylib -> libpiper_phonemize.1.dylib
  -rwxr-xr-x  1 fangjun  staff   132K Jun 25 11:39 libsherpa-onnx-c-api.dylib
  -rwxr-xr-x  1 fangjun  staff   1.7M Jun 25 11:39 libsherpa-onnx-core.dylib
  -rwxr-xr-x  1 fangjun  staff   1.4M Jun 25 11:39 libsherpa-onnx-fst.dylib
  -rwxr-xr-x  1 fangjun  staff    56K Jun 25 11:39 libsherpa-onnx-fstfar.dylib
  -rwxr-xr-x  1 fangjun  staff   151K Jun 25 11:39 libsherpa-onnx-jni.dylib
  -rwxr-xr-x  1 fangjun  staff   859K Jun 25 11:39 libsherpa-onnx-kaldifst-core.dylib
  -rwxr-xr-x  1 fangjun  staff   118K Jun 25 11:39 libsherpa-onnx-portaudio.dylib
  -rwxr-xr-x  1 fangjun  staff   149K Jun 25 11:39 libssentencepiece_core.dylib
  -rwxr-xr-x  1 fangjun  staff   188K Jun 25 11:39 libucd.dylib
