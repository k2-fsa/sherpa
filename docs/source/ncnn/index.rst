sherpa-ncnn
===========

We support using `ncnn`_ to replace PyTorch for neural network computation.
The code is put in a separate repository `sherpa-ncnn`_

`sherpa-ncnn`_ is self-contained and everything can be compiled from source.

Please refer to `<https://k2-fsa.github.io/icefall/recipes/librispeech/lstm_pruned_stateless_transducer.html#export-models>`_
for how to export models to `ncnn`_ format.

In the following, we describe how to compile `sherpa-ncnn`_ and how to use it
for speech recognition.

Download the pretrained model
-----------------------------

We provide pretrained models for `sherpa-ncnn`_. You can use the following command
to download it.

.. hint::

   The pretrained model is trained using the `LibriSpeech`_ dataset.
   We will upload models trained using the `WenetSpeech`_ dataset.

.. code-block:: bash

  cd /tmp/
  git lfs install
  git clone https://huggingface.co/csukuangfj/sherpa-ncnn-2022-09-05

Build sherpa-ncnn
-----------------

.. code-block:: bash

  git clone https://github.com/k2-fsa/sherpa-ncnn
  cd sherpa-ncnn
  mkdir build
  cd build
  cmake ..
  make -j6
  ls -lh bin/

You will find two executables:

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

Decode a single wave file
-------------------------

.. hint::

   It supports decoding only wave files with a single channel and the sampling rate
   should be 16 kHz.

.. code-block:: bash

  ./bin/sherpa-ncnn \
    /tmp/sherpa-ncnn-2022-09-05/tokens.txt \
    /tmp/sherpa-ncnn-2022-09-05/bar/encoder_jit_trace-iter-468000-avg-16-pnnx.ncnn.param \
    /tmp/sherpa-ncnn-2022-09-05/bar/encoder_jit_trace-iter-468000-avg-16-pnnx.ncnn.bin \
    /tmp/sherpa-ncnn-2022-09-05/bar/decoder_jit_trace-iter-468000-avg-16-pnnx.ncnn.param \
    /tmp/sherpa-ncnn-2022-09-05/bar/decoder_jit_trace-iter-468000-avg-16-pnnx.ncnn.bin \
    /tmp/sherpa-ncnn-2022-09-05/bar/joiner_jit_trace-iter-468000-avg-16-pnnx.ncnn.param \
    /tmp/sherpa-ncnn-2022-09-05/bar/joiner_jit_trace-iter-468000-avg-16-pnnx.ncnn.bin \
    /tmp/sherpa-ncnn-2022-09-05/test_wavs/1089-134686-0001.wav


Real-time speech recognition from a microphone
----------------------------------------------

.. code-block:: bash

  ./bin/sherpa-ncnn-microphone \
    /tmp/sherpa-ncnn-2022-09-05/tokens.txt \
    /tmp/sherpa-ncnn-2022-09-05/bar/encoder_jit_trace-iter-468000-avg-16-pnnx.ncnn.param \
    /tmp/sherpa-ncnn-2022-09-05/bar/encoder_jit_trace-iter-468000-avg-16-pnnx.ncnn.bin \
    /tmp/sherpa-ncnn-2022-09-05/bar/decoder_jit_trace-iter-468000-avg-16-pnnx.ncnn.param \
    /tmp/sherpa-ncnn-2022-09-05/bar/decoder_jit_trace-iter-468000-avg-16-pnnx.ncnn.bin \
    /tmp/sherpa-ncnn-2022-09-05/bar/joiner_jit_trace-iter-468000-avg-16-pnnx.ncnn.param \
    /tmp/sherpa-ncnn-2022-09-05/bar/joiner_jit_trace-iter-468000-avg-16-pnnx.ncnn.bin

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
