Embedded Linux (riscv64)
========================

This page describes how to build `sherpa-ncnn`_ for embedded Linux (RISC-V, 64-bit)
with cross-compiling on an x86 machine with Ubuntu OS.


Install toolchain
-----------------

The first step is to install a toolchain for cross-compiling.

.. code-block:: bash

   sudo apt-get install gcc-riscv64-linux-gnu
   sudo apt-get install g++-riscv64-linux-gnu

To check that you have installed the toolchain successfully, please run

.. code-block:: bash

  $ riscv64-linux-gnu-gcc --version
  riscv64-linux-gnu-gcc (Ubuntu 7.5.0-3ubuntu1~18.04) 7.5.0
  Copyright (C) 2017 Free Software Foundation, Inc.
  This is free software; see the source for copying conditions.  There is NO
  warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

  $ riscv64-linux-gnu-g++ --version
  riscv64-linux-gnu-g++ (Ubuntu 7.5.0-3ubuntu1~18.04) 7.5.0
  Copyright (C) 2017 Free Software Foundation, Inc.
  This is free software; see the source for copying conditions.  There is NO
  warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.


Build sherpa-ncnn
-----------------

Next, let us build `sherpa-ncnn`_.

.. code-block:: bash

  git clone https://github.com/k2-fsa/sherpa-ncnn
  cd sherpa-ncnn
  ./build-riscv64-linux-gnu.sh

After building, you will get two binaries:

.. code-block:: bash

  $ ls -lh build-riscv64-linux-gnu/install/bin/
  total 3.8M
  -rwxr-xr-x 1 kuangfangjun root 1.9M May 23 22:12 sherpa-ncnn
  -rwxr-xr-x 1 kuangfangjun root 1.9M May 23 22:12 sherpa-ncnn-alsa

That's it!

.. hint::

  - ``sherpa-ncnn`` is for decoding a single file
  - ``sherpa-ncnn-alsa`` is for real-time speech recongition by reading
    the microphone with `ALSA <https://en.wikipedia.org/wiki/Advanced_Linux_Sound_Architecture>`_

.. _sherpa-ncnn-alsa:

Please read :ref:`sherpa-ncnn-pre-trained-models` for usages about
the generated binaries.

.. hint::

  If you want to select a pre-trained model for `VisionFive 2 <https://www.starfivetech.com/en/site/boards>`_
  that can be run on real-time, we recommend you to use
  :ref:`sherpa_ncnn_streaming_zipformer_small_bilingual_zh_en_2023_02_16`.

  You can use the following command with the above model:

    .. code-block:: bash

      ./sherpa-ncnn \
        ./sherpa-ncnn-streaming-zipformer-small-bilingual-zh-en-2023-02-16/tokens.txt \
        ./sherpa-ncnn-streaming-zipformer-small-bilingual-zh-en-2023-02-16/64/encoder_jit_trace-pnnx.ncnn.param \
        ./sherpa-ncnn-streaming-zipformer-small-bilingual-zh-en-2023-02-16/64/encoder_jit_trace-pnnx.ncnn.bin \
        ./sherpa-ncnn-streaming-zipformer-small-bilingual-zh-en-2023-02-16/64/decoder_jit_trace-pnnx.ncnn.param \
        ./sherpa-ncnn-streaming-zipformer-small-bilingual-zh-en-2023-02-16/64/decoder_jit_trace-pnnx.ncnn.bin \
        ./sherpa-ncnn-streaming-zipformer-small-bilingual-zh-en-2023-02-16/64/joiner_jit_trace-pnnx.ncnn.param \
        ./sherpa-ncnn-streaming-zipformer-small-bilingual-zh-en-2023-02-16/64/joiner_jit_trace-pnnx.ncnn.bin \
        ./sherpa-ncnn-streaming-zipformer-small-bilingual-zh-en-2023-02-16/test_wavs/5.wav \
        4 \
        greedy_search

Read below if you want to learn more.

.. hint::

  By default, all external dependencies are statically linked. That means,
  the generated binaries are self-contained.

  You can use the following commands to check that and you will find
  they depend only on system libraries.

    .. code-block:: bash

      $ readelf -d build-riscv64-linux-gnu/install/bin/sherpa-ncnn

      Dynamic section at offset 0x1d6dc0 contains 31 entries:
        Tag        Type                         Name/Value
       0x0000000000000001 (NEEDED)             Shared library: [libgomp.so.1]
       0x0000000000000001 (NEEDED)             Shared library: [libpthread.so.0]
       0x0000000000000001 (NEEDED)             Shared library: [libstdc++.so.6]
       0x0000000000000001 (NEEDED)             Shared library: [libm.so.6]
       0x0000000000000001 (NEEDED)             Shared library: [libgcc_s.so.1]
       0x0000000000000001 (NEEDED)             Shared library: [libc.so.6]
       0x0000000000000001 (NEEDED)             Shared library: [ld-linux-riscv64-lp64d.so.1]
       0x000000000000001d (RUNPATH)            Library runpath: [$ORIGIN]
       0x0000000000000020 (PREINIT_ARRAY)      0x1e18e0
       0x0000000000000021 (PREINIT_ARRAYSZ)    0x8

      $ readelf -d build-riscv64-linux-gnu/install/bin/sherpa-ncnn-alsa

      Dynamic section at offset 0x1d3db0 contains 32 entries:
        Tag        Type                         Name/Value
       0x0000000000000001 (NEEDED)             Shared library: [libasound.so.2]
       0x0000000000000001 (NEEDED)             Shared library: [libgomp.so.1]
       0x0000000000000001 (NEEDED)             Shared library: [libpthread.so.0]
       0x0000000000000001 (NEEDED)             Shared library: [libstdc++.so.6]
       0x0000000000000001 (NEEDED)             Shared library: [libm.so.6]
       0x0000000000000001 (NEEDED)             Shared library: [libgcc_s.so.1]
       0x0000000000000001 (NEEDED)             Shared library: [libc.so.6]
       0x0000000000000001 (NEEDED)             Shared library: [ld-linux-riscv64-lp64d.so.1]
       0x000000000000001d (RUNPATH)            Library runpath: [$ORIGIN]
       0x0000000000000020 (PREINIT_ARRAY)      0x1de8c8
       0x0000000000000021 (PREINIT_ARRAYSZ)    0x8

Please create an issue at `<https://github.com/k2-fsa/sherpa-ncnn/issues>`_
if you have any problems.
