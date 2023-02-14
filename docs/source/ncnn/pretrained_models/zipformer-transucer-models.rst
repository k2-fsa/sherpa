Zipformer-transducer-based Models
=================================

English
-------

This model is converted from `<https://huggingface.co/csukuangfj/sherpa-ncnn-streaming-zipformer-en-2023-02-13>`_,
which supports only English as it is trained on the `LibriSpeech`_ corpus.

In the following, we describe how to download and use it with `sherpa-ncnn`_.

Please use the following commands to download it.

.. code-block:: bash

  cd /path/to/sherpa-ncnn
  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/sherpa-ncnn-streaming-zipformer-en-2023-02-13
  cd sherpa-ncnn-streaming-zipformer-en-2023-02-13
  git lfs pull --include "*.bin"

Please check that the file sizes of the pre-trained models are correct. See
the file sizes of ``*.bin`` files below.

.. code-block:: bash

  # before running `git lfs pull`

  sherpa-ncnn-streaming-zipformer-en-2023-02-13 fangjun$ ls -lh *.bin
  -rw-r--r--  1 fangjun  staff   131B Feb 14 11:51 decoder_jit_trace-pnnx.ncnn.bin
  -rw-r--r--  1 fangjun  staff   134B Feb 14 11:51 encoder_jit_trace-pnnx.ncnn.bin
  -rw-r--r--  1 fangjun  staff   132B Feb 14 11:51 joiner_jit_trace-pnnx.ncnn.bin

  sherpa-ncnn-streaming-zipformer-en-2023-02-13 fangjun$ git lfs pull --include "*.bin"

  # after running `git lfs pull`
  sherpa-ncnn-streaming-zipformer-en-2023-02-13 fangjun$ ls -lh *.bin
  -rw-r--r--  1 fangjun  staff   508K Feb 14 22:19 decoder_jit_trace-pnnx.ncnn.bin
  -rw-r--r--  1 fangjun  staff   132M Feb 14 22:19 encoder_jit_trace-pnnx.ncnn.bin
  -rw-r--r--  1 fangjun  staff   1.4M Feb 14 22:19 joiner_jit_trace-pnnx.ncnn.bin

Decode a single wave file with ./build/bin/sherpa-ncnn
::::::::::::::::::::::::::::::::::::::::::::::::::::::

.. hint::

   It supports decoding only wave files with a single channel and the sampling rate
   should be 16 kHz.

.. code-block:: bash

  cd /path/to/sherpa-ncnn

  for method in greedy_search modified_beam_search; do
    ./build/bin/sherpa-ncnn \
      ./sherpa-ncnn-streaming-zipformer-en-2023-02-13/tokens.txt \
      ./sherpa-ncnn-streaming-zipformer-en-2023-02-13/encoder_jit_trace-pnnx.ncnn.param \
      ./sherpa-ncnn-streaming-zipformer-en-2023-02-13/encoder_jit_trace-pnnx.ncnn.bin \
      ./sherpa-ncnn-streaming-zipformer-en-2023-02-13/decoder_jit_trace-pnnx.ncnn.param \
      ./sherpa-ncnn-streaming-zipformer-en-2023-02-13/decoder_jit_trace-pnnx.ncnn.bin \
      ./sherpa-ncnn-streaming-zipformer-en-2023-02-13/joiner_jit_trace-pnnx.ncnn.param \
      ./sherpa-ncnn-streaming-zipformer-en-2023-02-13/joiner_jit_trace-pnnx.ncnn.bin \
      ./sherpa-ncnn-streaming-zipformer-en-2023-02-13/test_wavs/1221-135766-0002.wav \
      2 \
      $method
  done

You should see the following output:

.. literalinclude:: ./code-zipformer/sherpa-ncnn-streaming-zipformer-en-2023-02-13-sherpa-ncnn.txt

.. note::

   Please use ``./build/bin/Release/sherpa-ncnn.exe`` for Windows.

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
    ./sherpa-ncnn-streaming-zipformer-en-2023-02-13/tokens.txt \
    ./sherpa-ncnn-streaming-zipformer-en-2023-02-13/encoder_jit_trace-pnnx.ncnn.param \
    ./sherpa-ncnn-streaming-zipformer-en-2023-02-13/encoder_jit_trace-pnnx.ncnn.bin \
    ./sherpa-ncnn-streaming-zipformer-en-2023-02-13/decoder_jit_trace-pnnx.ncnn.param \
    ./sherpa-ncnn-streaming-zipformer-en-2023-02-13/decoder_jit_trace-pnnx.ncnn.bin \
    ./sherpa-ncnn-streaming-zipformer-en-2023-02-13/joiner_jit_trace-pnnx.ncnn.param \
    ./sherpa-ncnn-streaming-zipformer-en-2023-02-13/joiner_jit_trace-pnnx.ncnn.bin \
    2 \
    greedy_search


Bilingual (Chinese and English)
-------------------------------

This model is converted from `<https://huggingface.co/pfluo/k2fsa-zipformer-chinese-english-mixed>`_,
which supports both Chinese and English.

In the following, we describe how to download and use it with `sherpa-ncnn`_.

Please use the following commands to download it.

.. code-block:: bash

  cd /path/to/sherpa-ncnn

  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13
  cd sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13
  git lfs pull --include "*.bin"

Please check that the file sizes of the pre-trained models are correct. See
the file sizes of ``*.bin`` files below.

.. code-block::

  sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13 fangjun$ ls -lh *.bin
  -rw-r--r--  1 fangjun  staff   6.1M Feb 14 10:08 decoder_jit_trace-pnnx.ncnn.bin
  -rw-r--r--  1 fangjun  staff   121M Feb 14 10:09 encoder_jit_trace-pnnx.ncnn.bin
  -rw-r--r--  1 fangjun  staff   7.0M Feb 14 10:08 joiner_jit_trace-pnnx.ncnn.bin

Decode a single wave file with ./build/bin/sherpa-ncnn
::::::::::::::::::::::::::::::::::::::::::::::::::::::

.. hint::

   It supports decoding only wave files with a single channel and the sampling rate
   should be 16 kHz.

.. code-block:: bash

  cd /path/to/sherpa-ncnn

  for method in greedy_search modified_beam_search; do
    ./build/bin/sherpa-ncnn \
      ./sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/tokens.txt \
      ./sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/encoder_jit_trace-pnnx.ncnn.param \
      ./sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/encoder_jit_trace-pnnx.ncnn.bin \
      ./sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/decoder_jit_trace-pnnx.ncnn.param \
      ./sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/decoder_jit_trace-pnnx.ncnn.bin \
      ./sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/joiner_jit_trace-pnnx.ncnn.param \
      ./sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/joiner_jit_trace-pnnx.ncnn.bin \
      ./sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/test_wavs/1.wav \
      2 \
      $method
  done

You should see the following output:

.. literalinclude:: ./code-zipformer/sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13-sherpa-ncnn.txt

.. note::

   Please use ``./build/bin/Release/sherpa-ncnn.exe`` for Windows.

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
    ./sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/tokens.txt \
    ./sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/encoder_jit_trace-pnnx.ncnn.param \
    ./sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/encoder_jit_trace-pnnx.ncnn.bin \
    ./sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/decoder_jit_trace-pnnx.ncnn.param \
    ./sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/decoder_jit_trace-pnnx.ncnn.bin \
    ./sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/joiner_jit_trace-pnnx.ncnn.param \
    ./sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/joiner_jit_trace-pnnx.ncnn.bin \
    2 \
    greedy_search

Japanese
--------

TODO: Add it.

- `<https://huggingface.co/csukuangfj/sherpa-ncnn-streaming-zipformer-ja-fluent-2023-02-14>`_
- `<https://huggingface.co/csukuangfj/sherpa-ncnn-streaming-zipformer-ja-disfluent-2023-02-14>`_

