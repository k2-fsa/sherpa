Embedded Linux (aarch64)
========================

This page describes how to build `sherpa-onnx`_ for embedded Linux (aarch64, 64-bit)
with cross-compiling on an x64 machine with Ubuntu OS.

.. warning::

  By cross-compiling we mean that you do the compilation on a ``x86_64`` machine.
  And you copy the generated binaries from a ``x86_64`` machine and run them on
  an ``aarch64`` machine.

  If you want to compile `sherpa-onnx`_ on an ``aarch64`` machine directly,
  please see :ref:`install_sherpa_onnx_on_linux`.

.. note::

   You can download pre-compiled binaries for ``aarch64`` from the following URL
   `<https://huggingface.co/csukuangfj/sherpa-onnx-libs/tree/main/aarch64>`_

   Please always download the latest version.

   Example command to download the version ``1.9.12``:

    .. code-block:: bash

      # binaries built with shared libraries
      wget https://huggingface.co/csukuangfj/sherpa-onnx-libs/resolve/main/aarch64/sherpa-onnx-v1.9.12-linux-aarch64-shared.tar.bz2

      # binaries built with static link
      wget https://huggingface.co/csukuangfj/sherpa-onnx-libs/resolve/main/aarch64/sherpa-onnx-v1.9.12-linux-aarch64-static.tar.bz2

      # For users from China
      # 中国国内用户，如果访问不了 huggingface, 请使用

      # binaries built with shared libraries
      wget https://hf-mirror.com/csukuangfj/sherpa-onnx-libs/resolve/main/aarch64/sherpa-onnx-v1.9.12-linux-aarch64-shared.tar.bz2

      # binaries built with static link
      wget https://hf-mirror.com/csukuangfj/sherpa-onnx-libs/resolve/main/aarch64/sherpa-onnx-v1.9.12-linux-aarch64-static.tar.bz2

.. hint::

   We provide two colab notebooks
   for you to try this section step by step.

    .. list-table::

     * - Build with ``shared`` libraries
       - Build with ``static`` libraries
     * - |build sherpa-onnx for aarch64 shared colab notebook|
       - |build sherpa-onnx for aarch64 static colab notebook|

   If you are using Windows/macOS or you don't want to setup your local environment
   for cross-compiling, please use the above colab notebooks.

.. |build sherpa-onnx for aarch64 shared colab notebook| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://github.com/k2-fsa/colab/blob/master/sherpa-onnx/sherpa_onnx_aarch64_cross_compiling_shared_libs.ipynb

.. |build sherpa-onnx for aarch64 static colab notebook| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://github.com/k2-fsa/colab/blob/master/sherpa-onnx/sherpa_onnx_aarch64_cross_compiling_static_libs.ipynb

.. _sherpa_onnx_install_for_aarch64_embedded_linux:

Install toolchain
-----------------

The first step is to install a toolchain for cross-compiling.

.. warning::

  You can use any toolchain that is suitable for your platform. The toolchain
  we use below is just an example.

Visit `<https://releases.linaro.org/components/toolchain/binaries/latest-7/aarch64-linux-gnu/>`_

We are going to download ``gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu.tar.xz``,
which has been uploaded to `<https://huggingface.co/csukuangfj/sherpa-ncnn-toolchains>`_.

Assume you want to install it in the folder ``$HOME/software``:

.. code-block:: bash

   mkdir -p $HOME/software
   cd $HOME/software
   wget https://huggingface.co/csukuangfj/sherpa-ncnn-toolchains/resolve/main/gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu.tar.xz

   # For users from China
   # 中国国内用户，如果访问不了 huggingface, 请使用
   # wget https://hf-mirror.com/csukuangfj/sherpa-ncnn-toolchains/resolve/main/gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu.tar.xz

   tar xvf gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu.tar.xz

Next, we need to set the following environment variable:

.. code-block:: bash

   export PATH=$HOME/software/gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu/bin:$PATH

To check that we have installed the cross-compiling toolchain successfully, please
run:

.. code-block:: bash

  aarch64-linux-gnu-gcc --version

which should print the following log:

.. code-block::

  aarch64-linux-gnu-gcc (Linaro GCC 7.5-2019.12) 7.5.0
  Copyright (C) 2017 Free Software Foundation, Inc.
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
  ./build-aarch64-linux-gnu.sh

After building, you will get two binaries:

.. code-block:: bash

  sherpa-onnx$ ls -lh build-aarch64-linux-gnu/install/bin/
  total 378K
  -rwxr-xr-x 1 kuangfangjun root 187K Feb 21 21:55 sherpa-onnx
  -rwxr-xr-x 1 kuangfangjun root 191K Feb 21 21:55 sherpa-onnx-alsa

.. note::

  Please also copy the ``onnxruntime`` lib to your embedded systems and put it
  into the same directory as ``sherpa-onnx`` and ``sherpa-onnx-alsa``.


  .. code-block:: bash

      sherpa-onnx$ ls -lh build-aarch64-linux-gnu/install/lib/*onnxruntime*
      lrw-r--r-- 1 kuangfangjun root  24 Feb 21 21:38 build-aarch64-linux-gnu/install/lib/libonnxruntime.so -> libonnxruntime.so.1.14.0
      -rw-r--r-- 1 kuangfangjun root 15M Feb 21 21:38 build-aarch64-linux-gnu/install/lib/libonnxruntime.so.1.14.0


That's it!

.. hint::

  - ``sherpa-onnx`` is for decoding a single file
  - ``sherpa-onnx-alsa`` is for real-time speech recongition by reading
    the microphone with `ALSA <https://en.wikipedia.org/wiki/Advanced_Linux_Sound_Architecture>`_

.. _sherpa-onnx-alsa:

sherpa-onnx-alsa
----------------

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
      card 3: UACDemoV10 [UACDemoV1.0], device 0: USB Audio [USB Audio]
        Subdevices: 1/1
        Subdevice #0: subdevice #0

  In this case, I only have 1 microphone. It is ``card 3`` and that card
  has only ``device 0``. To select ``card 3`` and ``device 0`` on that card,
  we need to pass ``hw:3,0`` to ``sherpa-onnx-alsa``. (Note: It has the format
  ``hw:card_number,device_index``.)

  For instance, you have to use

    .. code-block:: bash

      ./sherpa-onnx-alsa \
        ./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt \
        ./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.onnx \
        ./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx \
        ./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.onnx \
        hw:3,0

  Please change the card number and also the device index on the selected card
  accordingly in your own situation. Otherwise, you won't be able to record
  with your microphone.

Please read :ref:`sherpa-onnx-pre-trained-models` for usages about
the generated binaries.

.. hint::

  If you want to select a pre-trained model for Raspberry that can be
  run on real-time, we recommend you to
  use :ref:`sherpa_onnx_zipformer_transducer_models`.


Please create an issue at `<https://github.com/k2-fsa/sherpa-onnx/issues>`_
if you have any problems.

How to build static libraries and static linked binaries
--------------------------------------------------------

If you want to build static libraries and static linked binaries, please first
download a cross compile toolchain with GCC >= 9.0. The following is an example:

.. code-block:: bash

   mkdir -p $HOME/software
   cd $HOME/software
   wget -q https://huggingface.co/csukuangfj/sherpa-ncnn-toolchains/resolve/main/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu.tar.xz

   # For users from China
   # 中国国内用户，如果访问不了 huggingface, 请使用
   # wget -q https://hf-mirror.com/csukuangfj/sherpa-ncnn-toolchains/resolve/main/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu.tar.xz

   tar xf gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu.tar.xz

Next, we need to set the following environment variable:

.. code-block:: bash

   export PATH=$HOME/software/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu/bin:$PATH

To check that we have installed the cross-compiling toolchain successfully, please
run:

.. code-block:: bash

  aarch64-none-linux-gnu-gcc --version

which should print the following log:

.. code-block::

  aarch64-none-linux-gnu-gcc (GNU Toolchain for the A-profile Architecture 10.3-2021.07 (arm-10.29)) 10.3.1 20210621
  Copyright (C) 2020 Free Software Foundation, Inc.
  This is free software; see the source for copying conditions.  There is NO
  warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

Now you can build static libraries and static linked binaries with the following commands:

.. code-block:: bash

  git clone https://github.com/k2-fsa/sherpa-onnx
  cd sherpa-onnx
  export BUILD_SHARED_LIBS=OFF
  ./build-aarch64-linux-gnu.sh

You can use the following commands to check that the generated binaries are indeed static linked:

.. code-block:: bash

    $ cd build-aarch64-linux-gnu/bin

    $ ldd sherpa-onnx-alsa
        not a dynamic executable

    $ readelf -d sherpa-onnx-alsa

    Dynamic section at offset 0xed9950 contains 30 entries:
      Tag        Type                         Name/Value
     0x0000000000000001 (NEEDED)             Shared library: [libasound.so.2]
     0x0000000000000001 (NEEDED)             Shared library: [libdl.so.2]
     0x0000000000000001 (NEEDED)             Shared library: [libm.so.6]
     0x0000000000000001 (NEEDED)             Shared library: [libpthread.so.0]
     0x0000000000000001 (NEEDED)             Shared library: [libc.so.6]
     0x000000000000000f (RPATH)              Library rpath: [$ORIGIN:/star-fj/fangjun/open-source/sherpa-onnx/build-aarch64-linux-gnu/_deps/onnxruntime-sr
    c/lib:]
     0x000000000000000c (INIT)               0x404218
