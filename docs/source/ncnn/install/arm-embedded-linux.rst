.. _sherpa-ncnn-embedded-linux-arm-install:

Embedded Linux (arm)
====================

This page describes how to build `sherpa-ncnn`_ for embedded Linux (arm, 32-bit)
with cross-compiling on an x86 machine with Ubuntu OS.

.. caution::

   If you want to build `sherpa-ncnn`_ directly on your board, please don't
   use this document. Refer to :ref:`install_sherpa_ncnn_on_linux` instead.

.. caution::

   If you want to build `sherpa-ncnn`_ directly on your board, please don't
   use this document. Refer to :ref:`install_sherpa_ncnn_on_linux` instead.

.. caution::

   If you want to build `sherpa-ncnn`_ directly on your board, please don't
   use this document. Refer to :ref:`install_sherpa_ncnn_on_linux` instead.

.. hint::

   This page is for cross-compiling.

Install toolchain
-----------------

The first step is to install a toolchain for cross-compiling.

.. warning::

  You can use any toolchain that is suitable for your platform. The toolchain
  we use below is just an example.

Visit `<https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-a/downloads/8-3-2019-03>`_ to download the toolchain:

We are going to download ``gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf.tar.xz``,
which has been uploaded to `<https://huggingface.co/csukuangfj/sherpa-ncnn-toolchains>`_.

Assume you want to install it in the folder ``$HOME/software``:

.. code-block:: bash

   mkdir -p $HOME/software
   cd $HOME/software
   wget https://huggingface.co/csukuangfj/sherpa-ncnn-toolchains/resolve/main/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf.tar.xz
   tar xvf gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf.tar.xz

Next, we need to set the following environment variable:

.. code-block:: bash

   export PATH=$HOME/software/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf/bin:$PATH

To check that we have installed the cross-compiling toolchain successfully, please
run:

.. code-block:: bash

  arm-linux-gnueabihf-gcc --version

which should print the following log:

.. code-block::

  arm-linux-gnueabihf-gcc (GNU Toolchain for the A-profile Architecture 8.3-2019.03 (arm-rel-8.36)) 8.3.0
  Copyright (C) 2018 Free Software Foundation, Inc.
  This is free software; see the source for copying conditions.  There is NO
  warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

Congratulations! You have successfully installed a toolchain for cross-compiling
`sherpa-ncnn`_.

Build sherpa-ncnn
-----------------

Finally, let us build `sherpa-ncnn`_.

.. code-block:: bash

  git clone https://github.com/k2-fsa/sherpa-ncnn
  cd sherpa-ncnn
  ./build-arm-linux-gnueabihf.sh

After building, you will get two binaries:

.. code-block:: bash

  $ ls -lh  build-arm-linux-gnueabihf/install/bin/

  total 6.6M
  -rwxr-xr-x 1 kuangfangjun root 2.2M Jan 14 21:46 sherpa-ncnn
  -rwxr-xr-x 1 kuangfangjun root 2.2M Jan 14 21:46 sherpa-ncnn-alsa

That's it!

.. hint::

  - ``sherpa-ncnn`` is for decoding a single file
  - ``sherpa-ncnn-alsa`` is for real-time speech recongition by reading
    the microphone with `ALSA <https://en.wikipedia.org/wiki/Advanced_Linux_Sound_Architecture>`_

.. caution::

  We recommend that you use ``sherpa-ncnn-alsa`` on embedded systems such
  as Raspberry pi.

  You need to provide a ``device_name`` when invoking ``sherpa-ncnn-alsa``.
  We describe below how to find the device name for you microphone.

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
  we need to pass ``hw:0,0`` to ``sherpa-ncnn-alsa``. (Note: It has the format
  ``hw:card_number,device_index``.)

  For instance, you have to use

    .. code-block:: bash

      # Note: We use int8 models for encoder and joiner below.
      ./bin/sherpa-ncnn-alsa \
        ./sherpa-ncnn-conv-emformer-transducer-small-2023-01-09/tokens.txt \
        ./sherpa-ncnn-conv-emformer-transducer-small-2023-01-09/encoder_jit_trace-pnnx.ncnn.int8.param \
        ./sherpa-ncnn-conv-emformer-transducer-small-2023-01-09/encoder_jit_trace-pnnx.ncnn.int8.bin \
        ./sherpa-ncnn-conv-emformer-transducer-small-2023-01-09/decoder_jit_trace-pnnx.ncnn.param \
        ./sherpa-ncnn-conv-emformer-transducer-small-2023-01-09/decoder_jit_trace-pnnx.ncnn.bin \
        ./sherpa-ncnn-conv-emformer-transducer-small-2023-01-09/joiner_jit_trace-pnnx.ncnn.int8.param \
        ./sherpa-ncnn-conv-emformer-transducer-small-2023-01-09/joiner_jit_trace-pnnx.ncnn.int8.bin \
        "hw:0,0"

  Please change the card number and also the device index on the selected card
  accordingly in your own situation. Otherwise, you won't be able to record
  with your microphone.

Please read :ref:`sherpa-ncnn-pre-trained-models` for usages about
the generated binaries.

Read below if you want to learn more.

.. hint::

  By default, all external dependencies are statically linked. That means,
  the generated binaries are self-contained.

  You can use the following commands to check that and you will find
  they depend only on system libraries.

    .. code-block:: bash

      $ readelf -d build-arm-linux-gnueabihf/install/bin/sherpa-ncnn

      Dynamic section at offset 0x1c7ee8 contains 30 entries:
        Tag        Type                         Name/Value
       0x00000001 (NEEDED)                     Shared library: [libstdc++.so.6]
       0x00000001 (NEEDED)                     Shared library: [libm.so.6]
       0x00000001 (NEEDED)                     Shared library: [libgcc_s.so.1]
       0x00000001 (NEEDED)                     Shared library: [libpthread.so.0]
       0x00000001 (NEEDED)                     Shared library: [libc.so.6]
       0x0000000f (RPATH)                      Library rpath: [$ORIGIN]

      $ readelf -d build-arm-linux-gnueabihf/install/bin/sherpa-ncnn-alsa

      Dynamic section at offset 0x22ded8 contains 32 entries:
        Tag        Type                         Name/Value
       0x00000001 (NEEDED)                     Shared library: [libasound.so.2]
       0x00000001 (NEEDED)                     Shared library: [libgomp.so.1]
       0x00000001 (NEEDED)                     Shared library: [libpthread.so.0]
       0x00000001 (NEEDED)                     Shared library: [libstdc++.so.6]
       0x00000001 (NEEDED)                     Shared library: [libm.so.6]
       0x00000001 (NEEDED)                     Shared library: [libgcc_s.so.1]
       0x00000001 (NEEDED)                     Shared library: [libc.so.6]
       0x0000000f (RPATH)                      Library rpath: [$ORIGIN]


Please create an issue at `<https://github.com/k2-fsa/sherpa-ncnn/issues>`_
if you have any problems.
