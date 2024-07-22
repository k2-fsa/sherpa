C API for SenseVoice
====================

This page describes how to use the C API for `SenseVoice`_.

Please refer to :ref:`sherpa-onnx-c-api` for how to build `sherpa-onnx`_.

The following is a very quick introduction for using the C API of `sherpa-onnx`_
in the form of shared libraries on macOS and Linux.

.. hint::

  We do support static libraries and also support Windows.

If you copy, paste, and run the following commands in your terminal, you should be able
to see the following recognition result:

.. code-block::

   Decoded text: The tribal chieftain called for the boy and presented him with 50 pieces of gold.


.. code-block:: bash

  cd /tmp

  git clone https://github.com/k2-fsa/sherpa-onnx
  cd sherpa-onnx

  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
  tar xvf sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
  rm sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2


  ls -lh sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17

  echo "---"

  ls -lh sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/test_wavs

  mkdir build
  cd build
  cmake \
    -D CMAKE_BUILD_TYPE=Release \
    -D BUILD_SHARED_LIBS=ON \
    -D CMAKE_INSTALL_PREFIX=./install \
    -D SHERPA_ONNX_ENABLE_BINARY=OFF \
    ..

  make -j2 install

  ls -lh install/lib
  ls -lh install/include

  cd ..

  gcc -o sense-voice-c-api ./c-api-examples/sense-voice-c-api.c \
    -I ./build/install/include \
    -L ./build/install/lib/ \
    -l sherpa-onnx-c-api \
    -l onnxruntime

  ls -lh sense-voice-c-api

  export LD_LIBRARY_PATH=$PWD/build/install/lib:$LD_LIBRARY_PATH
  export DYLD_LIBRARY_PATH=$PWD/build/install/lib:$DYLD_LIBRARY_PATH

  ./sense-voice-c-api

Note that we have hard-coded the file paths inside `sense-voice-c-api.c <https://github.com/k2-fsa/sherpa-onnx/blob/master/c-api-examples/sense-voice-c-api.c>`_

.. hint::

  Since we are using shared libraries in the above example, you have to set
  the environemnt variable ``LD_LIBRARY_PATH`` for Linux and ``DYLD_LIBRARY_PATH``
  for macOS. Otherwise, you would get runtime errors when running ``./sense-voice-c-api``.

Explanations
------------

1. Download `sherpa-onnx`_
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  cd /tmp

  git clone https://github.com/k2-fsa/sherpa-onnx
  cd sherpa-onnx

In this example, we download `sherpa-onnx`_ and place it inside the directory
``/tmp/``. You can replace ``/tmp/`` with any directory you like.

Please always download the latest master of `sherpa-onnx`_.

2. Download the model
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash


  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
  tar xvf sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
  rm sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2

Note that we have placed the model in the directory ``/tmp/sherpa-onnx``.

3. Build sherpa-onnx
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  mkdir build
  cd build
  cmake \
    -D CMAKE_BUILD_TYPE=Release \
    -D BUILD_SHARED_LIBS=ON \
    -D CMAKE_INSTALL_PREFIX=./install \
    -D SHERPA_ONNX_ENABLE_BINARY=OFF \
    ..

  make -j2 install

We build a Release version of `sherpa-onnx`_. Also, we use shared libraries here.
The header file ``c-api.h`` and shared libraries are installed into the directory
``./build/install``.

If you are using Linux, you should see the following content::

  Install the project...
  -- Install configuration: "Release"
  -- Installing: /tmp/sherpa-onnx/build/install/lib/libonnxruntime.so
  -- Installing: /tmp/sherpa-onnx/build/install/./sherpa-onnx.pc
  -- Installing: /tmp/sherpa-onnx/build/install/lib/libsherpa-onnx-c-api.so
  -- Set non-toolchain portion of runtime path of "/tmp/sherpa-onnx/build/install/lib/libsherpa-onnx-c-api.so" to "$ORIGIN"
  -- Installing: /tmp/sherpa-onnx/build/install/include/sherpa-onnx/c-api/c-api.h

If you are using macOS, you should see::

  Install the project...
  -- Install configuration: "Release"
  -- Installing: /tmp/sherpa-onnx/build/install/lib/libonnxruntime.1.17.1.dylib
  -- Installing: /tmp/sherpa-onnx/build/install/lib/libonnxruntime.dylib
  -- Installing: /tmp/sherpa-onnx/build/install/./sherpa-onnx.pc
  -- Installing: /tmp/sherpa-onnx/build/install/lib/libsherpa-onnx-c-api.dylib
  -- Installing: /tmp/sherpa-onnx/build/install/include/sherpa-onnx/c-api/c-api.h

4. View the build result
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  ls -lh install/lib
  ls -lh install/include

If you are using Linux, you should see the following content::

  total 19M
  -rw-r--r-- 1 runner docker  15M Jul 22 08:47 libonnxruntime.so
  -rw-r--r-- 1 runner docker 4.1M Jul 22 08:47 libsherpa-onnx-c-api.so
  drwxr-xr-x 2 runner docker 4.0K Jul 22 08:47 pkgconfig
  total 4.0K
  drwxr-xr-x 3 runner docker 4.0K Jul 22 08:47 sherpa-onnx

If you are using macOS, you should see the following content::

  total 53976
  -rw-r--r--  1 runner  staff    23M Jul 22 08:48 libonnxruntime.1.17.1.dylib
  lrwxr-xr-x  1 runner  staff    27B Jul 22 08:48 libonnxruntime.dylib -> libonnxruntime.1.17.1.dylib
  -rwxr-xr-x  1 runner  staff   3.5M Jul 22 08:48 libsherpa-onnx-c-api.dylib
  drwxr-xr-x  3 runner  staff    96B Jul 22 08:48 pkgconfig
  total 0
  drwxr-xr-x  3 runner  staff    96B Jul 22 08:48 sherpa-onnx


5. Build the C API example
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  cd ..

  gcc -o sense-voice-c-api ./c-api-examples/sense-voice-c-api.c \
    -I ./build/install/include \
    -L ./build/install/lib/ \
    -l sherpa-onnx-c-api

  ls -lh sense-voice-c-api

Note that:

  - ``-I ./build/install/include`` is to add the directory ``./build/install/include``
    to the header search path so that ``#include "sherpa-onnx/c-api/c-api.h`` won't throw an error.
  - ``-L ./build/install/lib/`` is to add the directory ``./build/install/lib``
    to the library search path so that it can find ``-l sherpa-onnx-c-api``
  - ``-l sherpa-onnx-c-api`` is to link the library ``libsherpa-onnx-c-api.so`` for Linux
    and ``libsherpa-onnx-c-api.dylib`` for macOS.

6. Run it
^^^^^^^^^

.. code-block:: bash

  export LD_LIBRARY_PATH=$PWD/build/install/lib:$LD_LIBRARY_PATH
  export DYLD_LIBRARY_PATH=$PWD/build/install/lib:$DYLD_LIBRARY_PATH

  ./sense-voice-c-api

Note that we have to use::

  # For Linux
  export LD_LIBRARY_PATH=$PWD/build/install/lib:$LD_LIBRARY_PATH

and::

  # for macOS
  export DYLD_LIBRARY_PATH=$PWD/build/install/lib:$DYLD_LIBRARY_PATH

Otherwise, it cannot find ``libsherpa-onnx-c-api.so`` for Linux
and ``libsherpa-onnx-c-api.dylib`` at ``runtime``.

7. Where to find sense-voice-c-api.c
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can find ``sense-voice-c-api.c`` at the following address:

  `<https://github.com/k2-fsa/sherpa-onnx/blob/master/c-api-examples/sense-voice-c-api.c>`_

