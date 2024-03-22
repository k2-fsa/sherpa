.. _sherpa-onnx-embedded-linux-arm-install:

Embedded Linux (arm)
====================

This page describes how to build `sherpa-onnx`_ for embedded Linux (arm, 32-bit)
with ``cross-compiling`` on an x86 machine with Ubuntu OS.

.. caution::

   If you want to build `sherpa-onnx`_ directly on your board, please don't
   use this document. Refer to :ref:`install_sherpa_onnx_on_linux` instead.

.. caution::

   If you want to build `sherpa-onnx`_ directly on your board, please don't
   use this document. Refer to :ref:`install_sherpa_onnx_on_linux` instead.

.. caution::

   If you want to build `sherpa-onnx`_ directly on your board, please don't
   use this document. Refer to :ref:`install_sherpa_onnx_on_linux` instead.

.. hint::

   This page is for cross-compiling.

.. note::

   You can download pre-compiled binaries for 32-bit ``ARM`` from the following URL
   `<https://huggingface.co/csukuangfj/sherpa-onnx-libs/tree/main/arm32>`_

   Please always download the latest version.

   Example command to download the version ``1.9.12``:

    .. code-block:: bash

      # binaries built with shared libraries
      wget https://huggingface.co/csukuangfj/sherpa-onnx-libs/resolve/main/arm32/sherpa-onnx-v1.9.12-linux-arm-gnueabihf-shared.tar.bz2

      # binaries built with static link
      wget https://huggingface.co/csukuangfj/sherpa-onnx-libs/resolve/main/arm32/sherpa-onnx-v1.9.12-linux-arm-gnueabihf-static.tar.bz2

      # For users from China
      # 中国国内用户，如果访问不了 huggingface, 请使用

      # binaries built with shared libraries
      wget https://hf-mirror.com/csukuangfj/sherpa-onnx-libs/resolve/main/arm32/sherpa-onnx-v1.9.12-linux-arm-gnueabihf-shared.tar.bz2

      # binaries built with static link
      wget https://hf-mirror.com/csukuangfj/sherpa-onnx-libs/resolve/main/arm32/sherpa-onnx-v1.9.12-linux-arm-gnueabihf-static.tar.bz2

.. hint::

   We provide two colab notebooks
   for you to try this section step by step.

    .. list-table::

     * - Build with ``shared`` libraries
       - Build with ``static`` libraries
     * - |build sherpa-onnx for arm shared colab notebook|
       - |build sherpa-onnx for arm static colab notebook|

   If you are using Windows/macOS or you don't want to setup your local environment
   for cross-compiling, please use the above colab notebooks.

.. |build sherpa-onnx for arm shared colab notebook| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://github.com/k2-fsa/colab/blob/master/sherpa-onnx/sherpa_onnx_arm_cross_compiling_shared_libs.ipynb

.. |build sherpa-onnx for arm static colab notebook| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://github.com/k2-fsa/colab/blob/master/sherpa-onnx/sherpa_onnx_arm_cross_compiling_static_libs.ipynb

Install toolchain
-----------------

The first step is to install a toolchain for cross-compiling.

.. warning::

  You can use any toolchain that is suitable for your platform. The toolchain
  we use below is just an example.

Visit `<https://developer.arm.com/downloads/-/arm-gnu-toolchain-downloads>`_ to download the toolchain:

We are going to download ``gcc-arm-10.3-2021.07-x86_64-arm-none-linux-gnueabihf.tar.xz``,
which has been uploaded to `<https://huggingface.co/csukuangfj/sherpa-ncnn-toolchains>`_.

Assume you want to install it in the folder ``$HOME/software``:

.. code-block:: bash

   mkdir -p $HOME/software
   cd $HOME/software
   wget -q https://huggingface.co/csukuangfj/sherpa-ncnn-toolchains/resolve/main/gcc-arm-10.3-2021.07-x86_64-arm-none-linux-gnueabihf.tar.xz

   # For users from China
   # 中国国内用户，如果访问不了 huggingface, 请使用
   # wget -q https://hf-mirror.com/csukuangfj/sherpa-ncnn-toolchains/resolve/main/gcc-arm-10.3-2021.07-x86_64-arm-none-linux-gnueabihf.tar.xz

   tar xf gcc-arm-10.3-2021.07-x86_64-arm-none-linux-gnueabihf.tar.xz

Next, we need to set the following environment variable:

.. code-block:: bash

   export PATH=$HOME/software/gcc-arm-10.3-2021.07-x86_64-arm-none-linux-gnueabihf/bin:$PATH

To check that we have installed the cross-compiling toolchain successfully, please
run:

.. code-block:: bash

  arm-none-linux-gnueabihf-gcc --version

which should print the following log:

.. code-block::

  arm-none-linux-gnueabihf-gcc (GNU Toolchain for the A-profile Architecture 10.3-2021.07 (arm-10.29)) 10.3.1 20210621
  Copyright (C) 2020 Free Software Foundation, Inc.
  This is free software; see the source for copying conditions.  There is NO
  warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

Congratulations! You have successfully installed a toolchain for cross-compiling
`sherpa-onnx`_.

Build sherpa-onnx
-----------------

Finally, let us build `sherpa-onnx`_.

.. code-block:: bash

  git clone https://github.com/k2-fsa/sherpa-onnx
  cd sherpa-onnx
  export BUILD_SHARED_LIBS=ON
  ./build-arm-linux-gnueabihf.sh

After building, you will get the following binaries:

.. code-block:: bash

  $ ls -lh  build-arm-linux-gnueabihf/install/bin/

  total 1.2M
  -rwxr-xr-x 1 kuangfangjun root 395K Jul  7 16:28 sherpa-onnx
  -rwxr-xr-x 1 kuangfangjun root 391K Jul  7 16:28 sherpa-onnx-alsa
  -rwxr-xr-x 1 kuangfangjun root 351K Jul  7 16:28 sherpa-onnx-offline

That's it!

.. hint::

  - ``sherpa-onnx`` is for decoding a single file using a streaming model
  - ``sherpa-onnx-offline`` is for decoding a single file using a non-streaming model
  - ``sherpa-onnx-alsa`` is for real-time speech recongition using a streaming model by reading
    the microphone with `ALSA <https://en.wikipedia.org/wiki/Advanced_Linux_Sound_Architecture>`_

.. caution::

  We recommend that you use ``sherpa-onnx-alsa`` on embedded systems such
  as Raspberry pi.

  You need to provide a ``device_name`` when invoking ``sherpa-onnx-alsa``.
  We describe below how to find the device name for your microphone.

  Run the following command:

      .. code-block:: bash

        arecord -l

  to list all avaliable microphones for recording. If it complains that
  ``arecord: command not found``, please use ``sudo apt-get install alsa-utils``
  to install it.

  If the above command gives the following output:

    .. code-block:: bash

      **** List of CAPTURE Hardware Devices ****
      card 0: Audio [Axera Audio], device 0: 49ac000.i2s_mst-es8328-hifi-analog es8328-hifi-analog-0 []
        Subdevices: 1/1
        Subdevice #0: subdevice #0

  In this case, I only have 1 microphone. It is ``card 0`` and that card
  has only ``device 0``. To select ``card 0`` and ``device 0`` on that card,
  we need to pass ``plughw:0,0`` to ``sherpa-onnx-alsa``. (Note: It has the format
  ``plughw:card_number,device_index``.)

  For instance, you have to use

    .. code-block:: bash

      # Note: We use int8 models below.
      ./bin/sherpa-onnx-alsa \
        ./sherpa-onnx-streaming-zipformer-en-2023-06-26/tokens.txt \
        ./sherpa-onnx-streaming-zipformer-en-2023-06-26/encoder-epoch-99-avg-1-chunk-16-left-64.int8.onnx \
        ./sherpa-onnx-streaming-zipformer-en-2023-06-26/decoder-epoch-99-avg-1-chunk-16-left-64.int8.onnx \
        ./sherpa-onnx-streaming-zipformer-en-2023-06-26/joiner-epoch-99-avg-1-chunk-16-left-64.int8.onnx \
        "plughw:0,0"

  Please change the card number and also the device index on the selected card
  accordingly in your own situation. Otherwise, you won't be able to record
  with your microphone.

Please read :ref:`sherpa-onnx-pre-trained-models` for usages about
the generated binaries.

Read below if you want to learn more.

.. hint::

  By default, all external dependencies are statically linked. That means,
  the generated binaries are self-contained (except that it requires the
  onnxruntime shared library at runtime).

  You can use the following commands to check that and you will find
  they depend only on system libraries.

    .. code-block:: bash

      $ readelf -d build-arm-linux-gnueabihf/install/bin/sherpa-onnx

        Dynamic section at offset 0x61ee8 contains 30 entries:
          Tag        Type                         Name/Value
         0x00000001 (NEEDED)                     Shared library: [libonnxruntime.so.1.14.0]
         0x00000001 (NEEDED)                     Shared library: [libstdc++.so.6]
         0x00000001 (NEEDED)                     Shared library: [libm.so.6]
         0x00000001 (NEEDED)                     Shared library: [libgcc_s.so.1]
         0x00000001 (NEEDED)                     Shared library: [libc.so.6]
         0x0000000f (RPATH)                      Library rpath: [$ORIGIN:$ORIGIN/../lib:$ORIGIN/../../../sherpa_onnx/lib]

      $ readelf -d build-arm-linux-gnueabihf/install/bin/sherpa-onnx-alsa

        Dynamic section at offset 0x60ee0 contains 31 entries:
          Tag        Type                         Name/Value
         0x00000001 (NEEDED)                     Shared library: [libasound.so.2]
         0x00000001 (NEEDED)                     Shared library: [libonnxruntime.so.1.14.0]
         0x00000001 (NEEDED)                     Shared library: [libstdc++.so.6]
         0x00000001 (NEEDED)                     Shared library: [libm.so.6]
         0x00000001 (NEEDED)                     Shared library: [libgcc_s.so.1]
         0x00000001 (NEEDED)                     Shared library: [libc.so.6]
         0x0000000f (RPATH)                      Library rpath: [$ORIGIN]


Please create an issue at `<https://github.com/k2-fsa/sherpa-onnx/issues>`_
if you have any problems.

How to build static libraries and static linked binaries
--------------------------------------------------------

If you want to build static libraries and static linked binaries, please first
download a cross compile toolchain with GCC >= 9.0. The following is an example:

.. code-block:: bash

   mkdir -p $HOME/software
   cd $HOME/software
   wget -q https://huggingface.co/csukuangfj/sherpa-ncnn-toolchains/resolve/main/gcc-arm-10.3-2021.07-x86_64-arm-none-linux-gnueabihf.tar.xz

   # For users from China
   # 中国国内用户，如果访问不了 huggingface, 请使用
   wget -q https://hf-mirror.com/csukuangfj/sherpa-ncnn-toolchains/resolve/main/gcc-arm-10.3-2021.07-x86_64-arm-none-linux-gnueabihf.tar.xz

   tar xf gcc-arm-10.3-2021.07-x86_64-arm-none-linux-gnueabihf.tar.xz

Next, we need to set the following environment variable:

.. code-block:: bash

   export PATH=$HOME/software/gcc-arm-10.3-2021.07-x86_64-arm-none-linux-gnueabihf/bin:$PATH


To check that we have installed the cross-compiling toolchain successfully, please
run:

.. code-block:: bash

  arm-none-linux-gnueabihf-gcc --version

which should print the following log:

.. code-block::

  arm-none-linux-gnueabihf-gcc (GNU Toolchain for the A-profile Architecture 10.3-2021.07 (arm-10.29)) 10.3.1 20210621
  Copyright (C) 2020 Free Software Foundation, Inc.
  This is free software; see the source for copying conditions.  There is NO
  warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

Now you can build static libraries and static linked binaries with the following commands:

.. code-block:: bash

  git clone https://github.com/k2-fsa/sherpa-onnx
  cd sherpa-onnx
  export BUILD_SHARED_LIBS=OFF
  ./build-arm-linux-gnueabihf.sh

You can use the following commands to check that the generated binaries are indeed static linked:

.. code-block:: bash

    $ cd build-arm-linux-gnueabihf/bin

    $ ldd sherpa-onnx-alsa
        not a dynamic executable

    $ readelf -d sherpa-onnx-alsa

    Dynamic section at offset 0xa68eb4 contains 31 entries:
      Tag        Type                         Name/Value
     0x00000001 (NEEDED)                     Shared library: [libasound.so.2]
     0x00000001 (NEEDED)                     Shared library: [libdl.so.2]
     0x00000001 (NEEDED)                     Shared library: [libm.so.6]
     0x00000001 (NEEDED)                     Shared library: [libpthread.so.0]
     0x00000001 (NEEDED)                     Shared library: [libc.so.6]
     0x00000001 (NEEDED)                     Shared library: [ld-linux-armhf.so.3]
     0x0000000f (RPATH)                      Library rpath: [$ORIGIN:/star-fj/fangjun/open-source/sherpa-onnx/build-arm-linux-gnueabihf/_deps/espeak_ng-src/lib:/star-fj/fangjun/open-source/sherpa-onnx/build-arm-linux-gnueabihf/_deps/onnxruntime-src/lib:]
     0x0000000c (INIT)                       0x13550
