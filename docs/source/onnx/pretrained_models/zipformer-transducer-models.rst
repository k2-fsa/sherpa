.. _sherpa_onnx_zipformer_transducer_models:

Zipformer-transducer-based Models
=================================

.. hint::

   Please refer to :ref:`install_sherpa_onnx` to install `sherpa-onnx`_
   before you read this section.

csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-02-21 (English)
------------------------------------------------------------------

This model is converted from

`<https://huggingface.co/Zengwei/icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29>`_

which supports only English as it is trained on the `LibriSpeech`_ corpus.

You can find the training code at

`<https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/pruned_transducer_stateless7_streaming>`_

In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-02-21
  cd sherpa-onnx-streaming-zipformer-en-2023-02-21
  git lfs pull --include "*.onnx"

Please check that the file sizes of the pre-trained models are correct. See
the file sizes of ``*.onnx`` files below.

.. code-block:: bash

  # before running `git lfs pull`

  sherpa-onnx-streaming-zipformer-en-2023-02-21 fangjun$ ls -lh *.onnx
  -rw-r--r--  1 fangjun  staff   132B Feb 21 20:58 decoder-epoch-99-avg-1.onnx
  -rw-r--r--  1 fangjun  staff   134B Feb 21 20:58 encoder-epoch-99-avg-1.onnx
  -rw-r--r--  1 fangjun  staff   132B Feb 21 20:58 joiner-epoch-99-avg-1.onnx

  # after running `git lfs pull`

  sherpa-onnx-streaming-zipformer-en-2023-02-21 fangjun$ ls -lh *.onnx
  -rw-r--r--  1 fangjun  staff   2.0M Feb 21 21:34 decoder-epoch-99-avg-1.onnx
  -rw-r--r--  1 fangjun  staff   337M Feb 21 21:35 encoder-epoch-99-avg-1.onnx
  -rw-r--r--  1 fangjun  staff   1.0M Feb 21 21:35 joiner-epoch-99-avg-1.onnx

Decode a single wave file
~~~~~~~~~~~~~~~~~~~~~~~~~

.. hint::

   It supports decoding only wave files with a single channel and the sampling rate
   should be 16 kHz.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx \
    ./sherpa-onnx-streaming-zipformer-en-2023-02-21/tokens.txt \
    ./sherpa-onnx-streaming-zipformer-en-2023-02-21/encoder-epoch-99-avg-1.onnx \
    ./sherpa-onnx-streaming-zipformer-en-2023-02-21/decoder-epoch-99-avg-1.onnx \
    ./sherpa-onnx-streaming-zipformer-en-2023-02-21/joiner-epoch-99-avg-1.onnx \
    ./sherpa-onnx-streaming-zipformer-en-2023-02-21/test_wavs/1089-134686-0001.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx.exe`` for Windows.

You should see the following output:

.. literalinclude:: ./code-zipformer/sherpa-onnx-streaming-zipformer-en-2023-02-21.txt

Real-time speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-microphone \
    ./sherpa-onnx-streaming-zipformer-en-2023-02-21/tokens.txt \
    ./sherpa-onnx-streaming-zipformer-en-2023-02-21/encoder-epoch-99-avg-1.onnx \
    ./sherpa-onnx-streaming-zipformer-en-2023-02-21/decoder-epoch-99-avg-1.onnx \
    ./sherpa-onnx-streaming-zipformer-en-2023-02-21/joiner-epoch-99-avg-1.onnx

.. hint::

   If your system is Linux (including embedded Linux), you can also use
   :ref:`sherpa-onnx-alsa` to do real-time speech recognition with your
   microphone if ``sherpa-onnx-microphone`` does not work for you.


.. _sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20:

sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20 (Bilingual, Chinese + English)
-----------------------------------------------------------------------------------------

This model is converted from

`<https://huggingface.co/pfluo/k2fsa-zipformer-chinese-english-mixed>`_

which supports both Chinese and English. The model is contributed by the community
and is trained on tens of thousands of some internal dataset.

In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20
  cd sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20
  git lfs pull --include "*.onnx"

Please check that the file sizes of the pre-trained models are correct. See
the file sizes of ``*.onnx`` files below.

.. code-block:: bash

  # before running `git lfs pull`

  sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20 fangjun$ ls -lh *.onnx
  -rw-r--r--  1 fangjun  staff   133B Feb 21 20:27 decoder-epoch-99-avg-1.onnx
  -rw-r--r--  1 fangjun  staff   134B Feb 21 20:27 encoder-epoch-99-avg-1.onnx
  -rw-r--r--  1 fangjun  staff   133B Feb 21 20:27 joiner-epoch-99-avg-1.onnx

  # after running `git lfs pull`

  sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20 fangjun$ ls -lh *.onnx
  -rw-r--r--  1 fangjun  staff    13M Feb 21 20:30 decoder-epoch-99-avg-1.onnx
  -rw-r--r--  1 fangjun  staff   315M Feb 21 20:32 encoder-epoch-99-avg-1.onnx
  -rw-r--r--  1 fangjun  staff    12M Feb 21 20:30 joiner-epoch-99-avg-1.onnx

Decode a single wave file
~~~~~~~~~~~~~~~~~~~~~~~~~

.. hint::

   It supports decoding only wave files with a single channel and the sampling rate
   should be 16 kHz.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx \
    ./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt \
    ./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.onnx \
    ./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx \
    ./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.onnx \
    ./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/test_wavs/1.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx.exe`` for Windows.

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

You should see the following output:

.. literalinclude:: ./code-zipformer/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.txt

Real-time speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-microphone \
    ./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt \
    ./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.onnx \
    ./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx \
    ./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.onnx

.. hint::

   If your system is Linux (including embedded Linux), you can also use
   :ref:`sherpa-onnx-alsa` to do real-time speech recognition with your
   microphone if ``sherpa-onnx-microphone`` does not work for you.
