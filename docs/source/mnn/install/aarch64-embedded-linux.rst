Embedded Linux (aarch64)
========================

This page describes how to build `sherpa-mnn`_ for embedded Linux (aarch64, 64-bit)
with cross-compiling on an x86 machine with Ubuntu OS.

.. caution::

   If you want to build `sherpa-mnn`_ directly on your board, please don't
   use this document. Refer to :ref:`install_sherpa_mnn_on_linux` instead.

.. caution::

   If you want to build `sherpa-mnn`_ directly on your board, please don't
   use this document. Refer to :ref:`install_sherpa_mnn_on_linux` instead.

.. caution::

   If you want to build `sherpa-mnn`_ directly on your board, please don't
   use this document. Refer to :ref:`install_sherpa_mnn_on_linux` instead.

.. hint::

   This page is for cross-compiling.

.. _sherpa_mnn_install_for_aarch64_embedded_linux:

Install toolchain
-----------------

The first step is to install a toolchain for cross-compiling.

.. warning::

  You can use any toolchain that is suitable for your platform. The toolchain
  we use below is just an example.

Visit `<https://releases.linaro.org/components/toolchain/binaries/latest-7/aarch64-linux-gnu/>`_
to download the toolchain.

We are going to download ``gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu.tar.xz``,
which has been uploaded to `<https://huggingface.co/csukuangfj/sherpa-ncnn-toolchains>`_.

Assume you want to install it in the folder ``$HOME/software``:

.. code-block:: bash

   mkdir -p $HOME/software
   cd $HOME/software
   wget https://huggingface.co/csukuangfj/sherpa-ncnn-toolchains/resolve/main/gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu.tar.xz
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
`sherpa-mnn`_.

Build sherpa-mnn
-----------------

Finally, let us build `sherpa-mnn`_.

.. code-block:: bash

  git clone https://github.com/k2-fsa/sherpa-mnn
  cd sherpa-mnn
  ./build-aarch64-linux-gnu.sh

After building, you will get two binaries:

.. code-block:: bash

  $ ls -lh build-aarch64-linux-gnu/install/bin/
  total 5.0M
  -rwxr-xr-x 1 meixu meixu 2.5M Oct 15 20:17 sherpa-mnn
  -rwxr-xr-x 1 meixu meixu 2.5M Oct 15 20:17 sherpa-mnn-alsa

That's it!

.. hint::

  - ``sherpa-mnn`` is for decoding a single file
  - ``sherpa-mnn-alsa`` is for real-time speech recongition by reading
    the microphone with `ALSA <https://en.wikipedia.org/wiki/Advanced_Linux_Sound_Architecture>`_

.. _sherpa-mnn-alsa:

sherpa-mnn-alsa
----------------

.. caution::

  We recommend that you use ``sherpa-mnn-alsa`` on embedded systems such
  as Raspberry pi.

  You need to provide a ``device_name`` when invoking ``sherpa-mnn-alsa``.
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
  we need to pass ``hw:3,0`` to ``sherpa-mnn-alsa``. (Note: It has the format
  ``hw:card_number,device_index``.)

  For instance, you have to use

    .. code-block:: bash

      ./bin/sherpa-mnn-alsa \
        --tokens=k2fsa-zipformer-bilingual-zh-en-t/data/lang_char_bpe/tokens.txt \
        --encoder=k2fsa-zipformer-bilingual-zh-en-t/exp/encoder-epoch-99-avg-1.mnn \
        --decoder=k2fsa-zipformer-bilingual-zh-en-t/exp/decoder-epoch-99-avg-1.mnn \
        --joiner=k2fsa-zipformer-bilingual-zh-en-t/exp/joiner-epoch-99-avg-1.mnn \
        "hw:3,0"

  Please change the card number and also the device index on the selected card
  accordingly in your own situation. Otherwise, you won't be able to record
  with your microphone.

Please read :ref:`sherpa-mnn-pre-trained-models` for usages about
the generated binaries.

Read below if you want to learn more.

.. hint::

  By default, all external dependencies are statically linked. That means,
  the generated binaries are self-contained.

  You can use the following commands to check that and you will find
  they depend only on system libraries.

    .. code-block:: bash

      $ readelf -d build-aarch64-linux-gnu/install/bin/sherpa-mnn

      Dynamic section at offset 0x2718d0 contains 30 entries:
        Tag        Type                         Name/Value
       0x0000000000000001 (NEEDED)             Shared library: [libdl.so.2]
       0x0000000000000001 (NEEDED)             Shared library: [libstdc++.so.6]
       0x0000000000000001 (NEEDED)             Shared library: [libm.so.6]
       0x0000000000000001 (NEEDED)             Shared library: [libgcc_s.so.1]
       0x0000000000000001 (NEEDED)             Shared library: [libpthread.so.0]
       0x0000000000000001 (NEEDED)             Shared library: [libc.so.6]
       0x000000000000000f (RPATH)              Library rpath: [$ORIGIN]

      $ readelf -d build-aarch64-linux-gnu/install/bin/sherpa-mnn-alsa

      Dynamic section at offset 0x2718c0 contains 31 entries:
        Tag        Type                         Name/Value
       0x0000000000000001 (NEEDED)             Shared library: [libasound.so.2]nd.so.2]
       0x0000000000000001 (NEEDED)             Shared library: [libdl.so.2].so.1]
       0x0000000000000001 (NEEDED)             Shared library: [libstdc++.so.6]ead.so.0]
       0x0000000000000001 (NEEDED)             Shared library: [libm.so.6]++.so.6]
       0x0000000000000001 (NEEDED)             Shared library: [libgcc_s.so.1].6]
       0x0000000000000001 (NEEDED)             Shared library: [libpthread.so.0]s.so.1]
       0x0000000000000001 (NEEDED)             Shared library: [libc.so.6].6]
       0x000000000000000f (RPATH)              Library rpath: [$ORIGIN]


Please create an issue at `<https://github.com/k2-fsa/sherpa-mnn/issues>`_
if you have any problems.
