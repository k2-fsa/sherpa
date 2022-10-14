sherpa-ncnn
===========

.. hint::

  A colab notebook is provided for you so that you can try `sherpa-ncnn`_
  in the browser.

  |sherpa-ncnn colab notebook|

  .. |sherpa-ncnn colab notebook| image:: https://colab.research.google.com/assets/colab-badge.svg
     :target: https://colab.research.google.com/drive/1zdNAdWgV5rh1hLbLDqvLjxTa5tjU7cPa?usp=sharing

We support using `ncnn`_ to replace PyTorch for neural network computation.
The code is put in a separate repository `sherpa-ncnn`_

`sherpa-ncnn`_ is self-contained and everything can be compiled from source.

Please refer to `<https://k2-fsa.github.io/icefall/recipes/librispeech/lstm_pruned_stateless_transducer.html#export-models>`_
for how to export models to `ncnn`_ format.

In the following, we describe how to build `sherpa-ncnn`_ on Linux, macOS,
and Windows. Also, we show how to use it for speech recognition with
pretrained models.

.. hint::

   You can find two YouTube videos below for demonstration, one for English
   and the other for Chinese.

.. caution::

   We only provide support for LSTM transducer at present.
   The work for conformer transducer is still on-going.

Build sherpa-ncnn for Linux and macOS
-------------------------------------

.. code-block:: bash

  git clone https://github.com/k2-fsa/sherpa-ncnn
  cd sherpa-ncnn
  mkdir build
  cd build
  cmake -DCMAKE_BUILD_TYPE=Release ..
  make -j6

It will generate two executables inside ``./bin/``:

  - ``sherpa-ncnn``: For decoding a single wave file
  - ``sherpa-ncnn-microphone``: For real-time speech recognition from a microphone

.. code-block::

  (py38) kuangfangjun:build$ ls -lh bin/
  total 9.5M
  -rwxr-xr-x 1 kuangfangjun root 4.8M Sep 21 20:17 sherpa-ncnn
  -rwxr-xr-x 1 kuangfangjun root 4.8M Sep 21 20:17 sherpa-ncnn-microphone

where ``sherpa-ncnn`` is for decoding a single wave file while
``sherpa-ncnn-microphone`` is for speech recognition with a microphone.

.. code-block::

  (py38) kuangfangjun:build$ readelf -d bin/sherpa-ncnn-microphone | head -n 12

  Dynamic section at offset 0x438858 contains 33 entries:
    Tag        Type                         Name/Value
   0x0000000000000001 (NEEDED)             Shared library: [libpthread.so.0]
   0x0000000000000001 (NEEDED)             Shared library: [libgomp.so.1]
   0x0000000000000001 (NEEDED)             Shared library: [libstdc++.so.6]
   0x0000000000000001 (NEEDED)             Shared library: [libm.so.6]
   0x0000000000000001 (NEEDED)             Shared library: [libgcc_s.so.1]
   0x0000000000000001 (NEEDED)             Shared library: [libc.so.6]
   0x000000000000001d (RUNPATH)            Library runpath: [$ORIGIN]
   0x000000000000000c (INIT)               0x1d4b0
   0x000000000000000d (FINI)               0x3d0f94

You can see that they only depend on system libraries and have no other external
dependencies.

.. note::

   Please read below to see how to use the generated binaries for speech
   recognition with pretrained models.

Build sherpa-ncnn for Windows
-----------------------------

.. code-block:: bash

  git clone https://github.com/k2-fsa/sherpa-ncnn
  cd sherpa-ncnn
  mkdir build
  cd build
  cmake -DCMAKE_BUILD_TYPE=Release ..
  cmake --build . --config Release -- -m:6

It will generate two executables inside ``./bin/Release/``:

  - ``sherpa-ncnn.exe``: For decoding a single wave file.
  - ``sherpa-ncnn-microphone.exe``: For real-time speech recognition from a microphone


.. note::

   Please read below to see how to use the generated binaries for speech
   recognition with pretrained models.

Speech recognition with sherpa-ncnn
-----------------------------------

In the following, we describe how to use the precompiled binary ``sherpa-ncnn``
for offline speech recognition with pre-trained models.

We also show how to use ``sherpa-ncnn-microphone`` for real-time speech
recognition with pretrained models from a microphone.

We provide two examples: One is for English and the other is for Chinese.

English
^^^^^^^

First, let us download the pretrained model:

.. code-block:: bash

  cd /path/to/sherpa-ncnn

  git lfs install
  git clone https://huggingface.co/csukuangfj/sherpa-ncnn-2022-09-05

.. caution::

   You have to use ``git lfs`` to download the pretrained models.

Decode a single wave file with ./build/bin/sherpa-ncnn
::::::::::::::::::::::::::::::::::::::::::::::::::::::

.. hint::

   It supports decoding only wave files with a single channel and the sampling rate
   should be 16 kHz.

.. code-block:: bash

  cd /path/to/sherpa-ncnn

  ./build/bin/sherpa-ncnn \
    ./sherpa-ncnn-2022-09-05/tokens.txt \
    ./sherpa-ncnn-2022-09-05/bar/encoder_jit_trace-iter-468000-avg-16-pnnx.ncnn.param \
    ./sherpa-ncnn-2022-09-05/bar/encoder_jit_trace-iter-468000-avg-16-pnnx.ncnn.bin \
    ./sherpa-ncnn-2022-09-05/bar/decoder_jit_trace-iter-468000-avg-16-pnnx.ncnn.param \
    ./sherpa-ncnn-2022-09-05/bar/decoder_jit_trace-iter-468000-avg-16-pnnx.ncnn.bin \
    ./sherpa-ncnn-2022-09-05/bar/joiner_jit_trace-iter-468000-avg-16-pnnx.ncnn.param \
    ./sherpa-ncnn-2022-09-05/bar/joiner_jit_trace-iter-468000-avg-16-pnnx.ncnn.bin \
    ./sherpa-ncnn-2022-09-05/test_wavs/1089-134686-0001.wav

.. note::

   Please use ``./build/bin/Release/sherpa-ncnn.exe`` for Windows.


Real-time speech recognition from a microphone with build/bin/sherpa-ncnn-microphone
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

.. code-block:: bash

  cd /path/to/sherpa-ncnn

  ./build/bin/sherpa-ncnn-microphone \
    ./sherpa-ncnn-2022-09-05/tokens.txt \
    ./sherpa-ncnn-2022-09-05/bar/encoder_jit_trace-iter-468000-avg-16-pnnx.ncnn.param \
    ./sherpa-ncnn-2022-09-05/bar/encoder_jit_trace-iter-468000-avg-16-pnnx.ncnn.bin \
    ./sherpa-ncnn-2022-09-05/bar/decoder_jit_trace-iter-468000-avg-16-pnnx.ncnn.param \
    ./sherpa-ncnn-2022-09-05/bar/decoder_jit_trace-iter-468000-avg-16-pnnx.ncnn.bin \
    ./sherpa-ncnn-2022-09-05/bar/joiner_jit_trace-iter-468000-avg-16-pnnx.ncnn.param \
    ./sherpa-ncnn-2022-09-05/bar/joiner_jit_trace-iter-468000-avg-16-pnnx.ncnn.bin

.. note::

   Please use ``./build/bin/Release/sherpa-ncnn-microphone.exe`` for Windows.

It will print something like below:

.. code-block::

  Number of threads: 4
  num devices: 4
  Use default device: 2
    Name: MacBook Pro Microphone
    Max input channels: 1
  Started

Speak and it will show you the recognition result in real-time.

You can find a demo below:

..  youtube:: m6ynSxycpX0
   :width: 120%

Chinese
^^^^^^^

First, let us download the pretrained model:

.. code-block:: bash

  cd /path/to/sherpa-ncnn

  git lfs install
  git clone https://huggingface.co/csukuangfj/sherpa-ncnn-2022-09-30

.. caution::

   You have to use ``git lfs`` to download the pretrained models.

Decode a single wave file with ./build/bin/sherpa-ncnn
::::::::::::::::::::::::::::::::::::::::::::::::::::::

.. hint::

   It supports decoding only wave files with a single channel and the sampling rate
   should be 16 kHz.

.. code-block:: bash

   cd /path/to/sherpa-ncnn

   ./build/bin/sherpa-ncnn \
    ./sherpa-ncnn-2022-09-30/tokens.txt \
    ./sherpa-ncnn-2022-09-30/encoder_jit_trace-epoch-11-avg-2-pnnx.ncnn.param \
    ./sherpa-ncnn-2022-09-30/encoder_jit_trace-epoch-11-avg-2-pnnx.ncnn.bin \
    ./sherpa-ncnn-2022-09-30/decoder_jit_trace-epoch-11-avg-2-pnnx.ncnn.param \
    ./sherpa-ncnn-2022-09-30/decoder_jit_trace-epoch-11-avg-2-pnnx.ncnn.bin \
    ./sherpa-ncnn-2022-09-30/joiner_jit_trace-epoch-11-avg-2-pnnx.ncnn.param \
    ./sherpa-ncnn-2022-09-30/joiner_jit_trace-epoch-11-avg-2-pnnx.ncnn.bin \
    ./sherpa-ncnn-2022-09-30/test_wavs/0.wav

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

Real-time speech recognition from a microphone with build/bin/sherpa-ncnn-microphone
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

.. code-block:: bash

   cd /path/to/sherpa-ncnn

   ./build/bin/sherpa-ncnn-microphone \
    ./sherpa-ncnn-2022-09-30/tokens.txt \
    ./sherpa-ncnn-2022-09-30/encoder_jit_trace-epoch-11-avg-2-pnnx.ncnn.param \
    ./sherpa-ncnn-2022-09-30/encoder_jit_trace-epoch-11-avg-2-pnnx.ncnn.bin \
    ./sherpa-ncnn-2022-09-30/decoder_jit_trace-epoch-11-avg-2-pnnx.ncnn.param \
    ./sherpa-ncnn-2022-09-30/decoder_jit_trace-epoch-11-avg-2-pnnx.ncnn.bin \
    ./sherpa-ncnn-2022-09-30/joiner_jit_trace-epoch-11-avg-2-pnnx.ncnn.param \
    ./sherpa-ncnn-2022-09-30/joiner_jit_trace-epoch-11-avg-2-pnnx.ncnn.bin

.. note::

   Please use ``./build/bin/Release/sherpa-ncnn-microphone.exe`` for Windows.

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

You can find a demo below:

..  youtube:: bbQfoRT75oM
   :width: 120%
