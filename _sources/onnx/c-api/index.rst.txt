.. _sherpa-onnx-c-api:

C API
=====

In this section, we describe how to use the C API of `sherpa-onnx`_.


Specifically, we will describe:

  - How to generate required files
  - How to use ``pkg-config`` with `sherpa-onnx`_

You can find the implementation at

  - `<https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/c-api/c-api.h>`_
  - `<https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/c-api/c-api.cc>`_

Generate required files
-----------------------

Before using the C API of `sherpa-onnx`_, we need to first build required
libraries. You can choose either to build static libraries or shared libraries.

Build shared libraries
^^^^^^^^^^^^^^^^^^^^^^

Assume that we want to put library files and header files in the directory
``/tmp/sherpa-onnx/shared``:

.. code-block:: bash

  git clone https://github.com/k2-fsa/sherpa-onnx
  cd sherpa-onnx
  mkdir build-shared
  cd build-shared

  cmake \
    -DSHERPA_ONNX_ENABLE_C_API=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_INSTALL_PREFIX=/tmp/sherpa-onnx/shared \
    ..

  make -j6
  make install

You should find the following files inside ``/tmp/sherpa-onnx/shared``:

.. tabs::

   .. tab:: macOS

      .. code-block:: bash

          $ tree /tmp/sherpa-onnx/shared/

          /tmp/sherpa-onnx/shared
          ├── bin
          │   ├── sherpa-onnx
          │   ├── sherpa-onnx-keyword-spotter
          │   ├── sherpa-onnx-keyword-spotter-microphone
          │   ├── sherpa-onnx-microphone
          │   ├── sherpa-onnx-microphone-offline
          │   ├── sherpa-onnx-microphone-offline-audio-tagging
          │   ├── sherpa-onnx-microphone-offline-speaker-identification
          │   ├── sherpa-onnx-offline
          │   ├── sherpa-onnx-offline-audio-tagging
          │   ├── sherpa-onnx-offline-language-identification
          │   ├── sherpa-onnx-offline-parallel
          │   ├── sherpa-onnx-offline-punctuation
          │   ├── sherpa-onnx-offline-tts
          │   ├── sherpa-onnx-offline-tts-play
          │   ├── sherpa-onnx-offline-websocket-server
          │   ├── sherpa-onnx-online-punctuation
          │   ├── sherpa-onnx-online-websocket-client
          │   ├── sherpa-onnx-online-websocket-server
          │   ├── sherpa-onnx-vad-microphone
          │   └── sherpa-onnx-vad-microphone-offline-asr
          ├── include
          │   └── sherpa-onnx
          │       └── c-api
          │           └── c-api.h
          ├── lib
          │   ├── libonnxruntime.1.17.1.dylib
          │   ├── libonnxruntime.dylib -> libonnxruntime.1.17.1.dylib
          │   └── libsherpa-onnx-c-api.dylib
          └── sherpa-onnx.pc

          5 directories, 25 files

   .. tab:: Linux

      .. code-block:: bash

          $ tree /tmp/sherpa-onnx/shared/

          /tmp/sherpa-onnx/shared
          ├── bin
          │   ├── sherpa-onnx
          │   ├── sherpa-onnx-alsa
          │   ├── sherpa-onnx-alsa-offline
          │   ├── sherpa-onnx-alsa-offline-audio-tagging
          │   ├── sherpa-onnx-alsa-offline-speaker-identification
          │   ├── sherpa-onnx-keyword-spotter
          │   ├── sherpa-onnx-keyword-spotter-alsa
          │   ├── sherpa-onnx-offline
          │   ├── sherpa-onnx-offline-audio-tagging
          │   ├── sherpa-onnx-offline-language-identification
          │   ├── sherpa-onnx-offline-parallel
          │   ├── sherpa-onnx-offline-punctuation
          │   ├── sherpa-onnx-offline-tts
          │   ├── sherpa-onnx-offline-tts-play-alsa
          │   ├── sherpa-onnx-offline-websocket-server
          │   ├── sherpa-onnx-online-punctuation
          │   ├── sherpa-onnx-online-websocket-client
          │   ├── sherpa-onnx-online-websocket-server
          │   └── sherpa-onnx-vad-alsa
          ├── include
          │   └── sherpa-onnx
          │       └── c-api
          │           └── c-api.h
          ├── lib
          │   ├── libonnxruntime.so
          │   └── libsherpa-onnx-c-api.so
          └── sherpa-onnx.pc

          6 directories, 23 files


Build static libraries
^^^^^^^^^^^^^^^^^^^^^^

Assume that we want to put library files and header files in the directory
``/tmp/sherpa-onnx/static``:

.. code-block:: bash

  git clone https://github.com/k2-fsa/sherpa-onnx
  cd sherpa-onnx
  mkdir build-static
  cd build-static

  cmake \
    -DSHERPA_ONNX_ENABLE_C_API=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_INSTALL_PREFIX=/tmp/sherpa-onnx/static \
    ..

  make -j6
  make install

You should find the following files in ``/tmp/sherpa-onnx/static``:

.. tabs::

   .. tab:: macOS

      .. code-block:: bash

          $ tree /tmp/sherpa-onnx/static/

          /tmp/sherpa-onnx/static
          ├── bin
          │   ├── sherpa-onnx
          │   ├── sherpa-onnx-keyword-spotter
          │   ├── sherpa-onnx-keyword-spotter-microphone
          │   ├── sherpa-onnx-microphone
          │   ├── sherpa-onnx-microphone-offline
          │   ├── sherpa-onnx-microphone-offline-audio-tagging
          │   ├── sherpa-onnx-microphone-offline-speaker-identification
          │   ├── sherpa-onnx-offline
          │   ├── sherpa-onnx-offline-audio-tagging
          │   ├── sherpa-onnx-offline-language-identification
          │   ├── sherpa-onnx-offline-parallel
          │   ├── sherpa-onnx-offline-punctuation
          │   ├── sherpa-onnx-offline-tts
          │   ├── sherpa-onnx-offline-tts-play
          │   ├── sherpa-onnx-offline-websocket-server
          │   ├── sherpa-onnx-online-punctuation
          │   ├── sherpa-onnx-online-websocket-client
          │   ├── sherpa-onnx-online-websocket-server
          │   ├── sherpa-onnx-vad-microphone
          │   └── sherpa-onnx-vad-microphone-offline-asr
          ├── include
          │   └── sherpa-onnx
          │       └── c-api
          │           └── c-api.h
          ├── lib
          │   ├── libespeak-ng.a
          │   ├── libkaldi-decoder-core.a
          │   ├── libkaldi-native-fbank-core.a
          │   ├── libonnxruntime.a
          │   ├── libpiper_phonemize.a
          │   ├── libsherpa-onnx-c-api.a
          │   ├── libsherpa-onnx-core.a
          │   ├── libsherpa-onnx-fst.a
          │   ├── libsherpa-onnx-fstfar.a
          │   ├── libsherpa-onnx-kaldifst-core.a
          │   ├── libsherpa-onnx-portaudio_static.a
          │   ├── libssentencepiece_core.a
          │   └── libucd.a
          └── sherpa-onnx.pc

          5 directories, 35 files

   .. tab:: Linux

      .. code-block:: bash

          $ tree /tmp/sherpa-onnx/static/

          /tmp/sherpa-onnx/static
          ├── bin
          │   ├── sherpa-onnx
          │   ├── sherpa-onnx-alsa
          │   ├── sherpa-onnx-alsa-offline
          │   ├── sherpa-onnx-alsa-offline-audio-tagging
          │   ├── sherpa-onnx-alsa-offline-speaker-identification
          │   ├── sherpa-onnx-keyword-spotter
          │   ├── sherpa-onnx-keyword-spotter-alsa
          │   ├── sherpa-onnx-keyword-spotter-microphone
          │   ├── sherpa-onnx-microphone
          │   ├── sherpa-onnx-microphone-offline
          │   ├── sherpa-onnx-microphone-offline-audio-tagging
          │   ├── sherpa-onnx-microphone-offline-speaker-identification
          │   ├── sherpa-onnx-offline
          │   ├── sherpa-onnx-offline-audio-tagging
          │   ├── sherpa-onnx-offline-language-identification
          │   ├── sherpa-onnx-offline-parallel
          │   ├── sherpa-onnx-offline-punctuation
          │   ├── sherpa-onnx-offline-tts
          │   ├── sherpa-onnx-offline-tts-play
          │   ├── sherpa-onnx-offline-tts-play-alsa
          │   ├── sherpa-onnx-offline-websocket-server
          │   ├── sherpa-onnx-online-punctuation
          │   ├── sherpa-onnx-online-websocket-client
          │   ├── sherpa-onnx-online-websocket-server
          │   ├── sherpa-onnx-vad-alsa
          │   ├── sherpa-onnx-vad-microphone
          │   └── sherpa-onnx-vad-microphone-offline-asr
          ├── include
          │   └── sherpa-onnx
          │       └── c-api
          │           └── c-api.h
          ├── lib
          │   ├── libespeak-ng.a
          │   ├── libkaldi-decoder-core.a
          │   ├── libkaldi-native-fbank-core.a
          │   ├── libonnxruntime.a
          │   ├── libpiper_phonemize.a
          │   ├── libsherpa-onnx-c-api.a
          │   ├── libsherpa-onnx-core.a
          │   ├── libsherpa-onnx-fst.a
          │   ├── libsherpa-onnx-fstfar.a
          │   ├── libsherpa-onnx-kaldifst-core.a
          │   ├── libsherpa-onnx-portaudio_static.a
          │   ├── libssentencepiece_core.a
          │   └── libucd.a
          └── sherpa-onnx.pc

          6 directories, 42 files


Build decode-file-c-api.c with generated files
----------------------------------------------

To build the following file:

  `<https://github.com/k2-fsa/sherpa-onnx/blob/master/c-api-examples/decode-file-c-api.c>`_

We can use:

.. tabs::

   .. tab:: static link

      .. code-block:: bash

          export PKG_CONFIG_PATH=/tmp/sherpa-onnx/static:$PKG_CONFIG_PATH

          cd ./c-api-examples
          gcc -o decode-file-c-api $(pkg-config --cflags sherpa-onnx) ./decode-file-c-api.c $(pkg-config --libs sherpa-onnx)

          ./decode-file-c-api --help

   .. tab:: dynamic link

      .. code-block:: bash

          export PKG_CONFIG_PATH=/tmp/sherpa-onnx/shared:$PKG_CONFIG_PATH

          cd ./c-api-examples
          gcc -o decode-file-c-api $(pkg-config --cflags sherpa-onnx) ./decode-file-c-api.c $(pkg-config --libs sherpa-onnx)

          ./decode-file-c-api --help

.. warning::

   The order of linking the libraries matters. Please see

    - Static link without TTS: `<https://github.com/k2-fsa/sherpa-onnx/blob/master/cmake/sherpa-onnx-static-no-tts.pc.in>`_
    - Static link with TTS: `<https://github.com/k2-fsa/sherpa-onnx/blob/master/cmake/sherpa-onnx-static.pc.in>`_
    - Dynamic link: `<https://github.com/k2-fsa/sherpa-onnx/blob/master/cmake/sherpa-onnx-shared.pc.in>`_

colab
-----

We provide a colab notebook
|Sherpa-onnx c api example colab notebook|
for you to try the C API of `sherpa-onnx`_.

.. |Sherpa-onnx c api example colab notebook| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://github.com/k2-fsa/colab/blob/master/sherpa-onnx/sherpa_onnx_c_api_example.ipynb
