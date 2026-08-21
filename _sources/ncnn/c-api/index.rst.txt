.. _sherpa-ncnn-c-api:

C API
=====

In this section, we describe how to use the C API of `sherpa-ncnn`_.

Specifically, we will describe:

  - How to generate required files
  - How to use ``pkg-config`` with `sherpa-ncnn`_

Generate required files
-----------------------

Before using the C API of `sherpa-ncnn`_, we need to first build required
libraries. You can choose either to build static libraries or shared libraries.

Build shared libraries
^^^^^^^^^^^^^^^^^^^^^^

Assume that we want to put library files and header files in the directory
``/tmp/sherpa-ncnn/shared``:

.. code-block:: bash

  git clone https://github.com/k2-fsa/sherpa-ncnn
  cd sherpa-ncnn
  mkdir build-shared
  cd build-shared

  cmake \
    -DSHERPA_NCNN_ENABLE_C_API=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_INSTALL_PREFIX=/tmp/sherpa-ncnn/shared \
    ..

  make -j6
  make install

You should find the following files inside ``/tmp/sherpa-ncnn/shared``:

.. tabs::

   .. tab:: macOS

      .. code-block:: bash

          $ tree /tmp/sherpa-ncnn/shared/
          /tmp/sherpa-ncnn/shared/
          ├── bin
          │   ├── sherpa-ncnn
          │   └── sherpa-ncnn-microphone
          ├── include
          │   └── sherpa-ncnn
          │       └── c-api
          │           └── c-api.h
          ├── lib
          │   ├── libkaldi-native-fbank-core.dylib
          │   ├── libncnn.dylib
          │   ├── libsherpa-ncnn-c-api.dylib
          │   └── libsherpa-ncnn-core.dylib
          └── sherpa-ncnn.pc

          5 directories, 8 files

   .. tab:: Linux

      .. code-block:: bash

          $ tree /tmp/sherpa-ncnn/shared/
          /tmp/sherpa-ncnn/shared/
          ├── bin
          │   ├── sherpa-ncnn
          │   └── sherpa-ncnn-microphone
          ├── include
          │   └── sherpa-ncnn
          │       └── c-api
          │           └── c-api.h
          ├── lib
          │   ├── libkaldi-native-fbank-core.so
          │   ├── libncnn.so
          │   ├── libsherpa-ncnn-c-api.so
          │   └── libsherpa-ncnn-core.so
          └── sherpa-ncnn.pc

          5 directories, 8 files

Build static libraries
^^^^^^^^^^^^^^^^^^^^^^

Assume that we want to put library files and header files in the directory
``/tmp/sherpa-ncnn/static``:

.. code-block:: bash

  git clone https://github.com/k2-fsa/sherpa-ncnn
  cd sherpa-ncnn
  mkdir build-static
  cd build-static

  cmake \
    -DSHERPA_NCNN_ENABLE_C_API=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_INSTALL_PREFIX=/tmp/sherpa-ncnn/static \
    ..

  make -j6
  make install

You should find the following files in ``/tmp/sherpa-ncnn/static``:

.. code-block:: bash

  $ tree /tmp/sherpa-ncnn/static/
  /tmp/sherpa-ncnn/static/
  ├── bin
  │   ├── sherpa-ncnn
  │   └── sherpa-ncnn-microphone
  ├── include
  │   └── sherpa-ncnn
  │       └── c-api
  │           └── c-api.h
  ├── lib
  │   ├── libkaldi-native-fbank-core.a
  │   ├── libncnn.a
  │   ├── libsherpa-ncnn-c-api.a
  │   └── libsherpa-ncnn-core.a
  └── sherpa-ncnn.pc

  5 directories, 8 files

Build decode-file-c-api.c with generated files
----------------------------------------------

To build the following file:

  `<https://github.com/k2-fsa/sherpa-ncnn/blob/master/c-api-examples/decode-file-c-api.c>`_

We can use:

.. tabs::

   .. tab:: static link

      .. code-block:: bash

          export PKG_CONFIG_PATH=/tmp/sherpa-ncnn/static:$PKG_CONFIG_PATH

          cd ./c-api-examples
          gcc -o decode-file-c-api $(pkg-config --cflags sherpa-ncnn) ./decode-file-c-api.c $(pkg-config --libs sherpa-ncnn)

   .. tab:: dynamic link

      .. code-block:: bash

          export PKG_CONFIG_PATH=/tmp/sherpa-ncnn/shared:$PKG_CONFIG_PATH

          cd ./c-api-examples
          gcc -o decode-file-c-api $(pkg-config --cflags sherpa-ncnn) ./decode-file-c-api.c $(pkg-config --libs sherpa-ncnn)
