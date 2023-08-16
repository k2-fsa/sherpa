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

          /tmp/sherpa-onnx/shared/
          ├── bin
          │   ├── sherpa-onnx
          │   ├── sherpa-onnx-microphone
          │   ├── sherpa-onnx-microphone-offline
          │   ├── sherpa-onnx-offline
          │   ├── sherpa-onnx-offline-websocket-server
          │   ├── sherpa-onnx-online-websocket-client
          │   └── sherpa-onnx-online-websocket-server
          ├── include
          │   ├── cargs.h
          │   └── sherpa-onnx
          │       └── c-api
          │           └── c-api.h
          ├── lib
          │   ├── cargs.h
          │   ├── libcargs.dylib
          │   ├── libkaldi-native-fbank-core.dylib
          │   ├── libonnxruntime.1.15.1.dylib
          │   ├── libonnxruntime.dylib -> libonnxruntime.1.15.1.dylib
          │   ├── libsherpa-onnx-c-api.dylib
          │   ├── libsherpa-onnx-core.dylib
          │   └── libsherpa-onnx-portaudio.dylib
          └── sherpa-onnx.pc

          5 directories, 18 files

   .. tab:: Linux

      .. code-block:: bash

          $ tree /tmp/sherpa-onnx/shared/

          /tmp/sherpa-onnx/shared/
          |-- bin
          |   |-- sherpa-onnx
          |   |-- sherpa-onnx-alsa
          |   |-- sherpa-onnx-microphone
          |   |-- sherpa-onnx-microphone-offline
          |   |-- sherpa-onnx-offline
          |   |-- sherpa-onnx-offline-websocket-server
          |   |-- sherpa-onnx-online-websocket-client
          |   `-- sherpa-onnx-online-websocket-server
          |-- include
          |   |-- cargs.h
          |   `-- sherpa-onnx
          |       `-- c-api
          |           `-- c-api.h
          |-- lib
          |   |-- cargs.h
          |   |-- libcargs.so
          |   |-- libkaldi-native-fbank-core.so
          |   |-- libonnxruntime.so -> libonnxruntime.so.1.15.1
          |   |-- libonnxruntime.so.1.15.1
          |   |-- libsherpa-onnx-c-api.so
          |   |-- libsherpa-onnx-core.so
          |   `-- libsherpa-onnx-portaudio.so
          `-- sherpa-onnx.pc

          5 directories, 19 files


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

          /tmp/sherpa-onnx//static/
          ├── bin
          │   ├── sherpa-onnx
          │   ├── sherpa-onnx-microphone
          │   ├── sherpa-onnx-microphone-offline
          │   ├── sherpa-onnx-offline
          │   ├── sherpa-onnx-offline-websocket-server
          │   ├── sherpa-onnx-online-websocket-client
          │   └── sherpa-onnx-online-websocket-server
          ├── include
          │   ├── cargs.h
          │   └── sherpa-onnx
          │       └── c-api
          │           └── c-api.h
          ├── lib
          │   ├── cargs.h
          │   ├── libcargs.a
          │   ├── libkaldi-native-fbank-core.a
          │   ├── libonnxruntime.1.15.1.dylib
          │   ├── libonnxruntime.dylib -> libonnxruntime.1.15.1.dylib
          │   ├── libsherpa-onnx-c-api.a
          │   ├── libsherpa-onnx-core.a
          │   └── libsherpa-onnx-portaudio_static.a
          └── sherpa-onnx.pc

          5 directories, 18 files


   .. tab:: Linux

      .. code-block:: bash

          $ tree /tmp/sherpa-onnx/static/

          /tmp/sherpa-onnx/static/
          |-- bin
          |   |-- sherpa-onnx
          |   |-- sherpa-onnx-alsa
          |   |-- sherpa-onnx-microphone
          |   |-- sherpa-onnx-microphone-offline
          |   |-- sherpa-onnx-offline
          |   |-- sherpa-onnx-offline-websocket-server
          |   |-- sherpa-onnx-online-websocket-client
          |   `-- sherpa-onnx-online-websocket-server
          |-- include
          |   |-- cargs.h
          |   `-- sherpa-onnx
          |       `-- c-api
          |           `-- c-api.h
          |-- lib
          |   |-- cargs.h
          |   |-- libcargs.a
          |   |-- libkaldi-native-fbank-core.a
          |   |-- libonnxruntime.so -> libonnxruntime.so.1.15.1
          |   |-- libonnxruntime.so.1.15.1
          |   |-- libsherpa-onnx-c-api.a
          |   |-- libsherpa-onnx-core.a
          |   `-- libsherpa-onnx-portaudio_static.a
          `-- sherpa-onnx.pc

          5 directories, 19 files

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

colab
-----

We provide a colab notebook
|Sherpa-onnx c api example colab notebook|
for you to try the C API of `sherpa-onnx`_.

.. |Sherpa-onnx c api example colab notebook| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://github.com/k2-fsa/colab/blob/master/sherpa-onnx/sherpa_onnx_c_api_example.ipynb
