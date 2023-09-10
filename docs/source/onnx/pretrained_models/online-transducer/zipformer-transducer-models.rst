.. _sherpa_onnx_zipformer_transducer_models:

Zipformer-transducer-based Models
=================================

.. hint::

   Please refer to :ref:`install_sherpa_onnx` to install `sherpa-onnx`_
   before you read this section.


.. _sherpa-onnx-wenetspeech-2023-06-15-streaming:

pkufool/icefall-asr-zipformer-streaming-wenetspeech-20230615 (Chinese)
----------------------------------------------------------------------

This model is from

`<https://huggingface.co/pkufool/icefall-asr-zipformer-streaming-wenetspeech-20230615>`_

which supports only Chinese as it is trained on the `WenetSpeech`_ corpus.

If you are interested in how the model is trained, please refer to
`<https://github.com/k2-fsa/icefall/pull/1130>`_.

In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/pkufool/icefall-asr-zipformer-streaming-wenetspeech-20230615
  cd icefall-asr-zipformer-streaming-wenetspeech-20230615
  git lfs pull --include "exp/*chunk-16-left-128.*onnx"

Please check that the file sizes of the pre-trained models are correct. See
the file sizes of ``*.onnx`` files below.

.. code-block:: bash

  icefall-asr-zipformer-streaming-wenetspeech-20230615 fangjun$ ls -lh exp/*chunk-16-left-128.*onnx
  -rw-r--r--  1 fangjun  staff    11M Jun 26 15:42 exp/decoder-epoch-12-avg-4-chunk-16-left-128.int8.onnx
  -rw-r--r--  1 fangjun  staff    12M Jun 26 15:42 exp/decoder-epoch-12-avg-4-chunk-16-left-128.onnx
  -rw-r--r--  1 fangjun  staff    68M Jun 26 15:42 exp/encoder-epoch-12-avg-4-chunk-16-left-128.int8.onnx
  -rw-r--r--  1 fangjun  staff   250M Jun 26 15:43 exp/encoder-epoch-12-avg-4-chunk-16-left-128.onnx
  -rw-r--r--  1 fangjun  staff   2.7M Jun 26 15:42 exp/joiner-epoch-12-avg-4-chunk-16-left-128.int8.onnx
  -rw-r--r--  1 fangjun  staff    11M Jun 26 15:42 exp/joiner-epoch-12-avg-4-chunk-16-left-128.onnx

Decode a single wave file
~~~~~~~~~~~~~~~~~~~~~~~~~

.. hint::

   It supports decoding only wave files of a single channel with 16-bit
   encoded samples, while the sampling rate does not need to be 16 kHz.

fp32
^^^^

The following code shows how to use ``fp32`` models to decode a wave file:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx \
    --tokens=./icefall-asr-zipformer-streaming-wenetspeech-20230615/data/lang_char/tokens.txt \
    --encoder=./icefall-asr-zipformer-streaming-wenetspeech-20230615/exp/encoder-epoch-12-avg-4-chunk-16-left-128.onnx \
    --decoder=./icefall-asr-zipformer-streaming-wenetspeech-20230615/exp/decoder-epoch-12-avg-4-chunk-16-left-128.onnx \
    --joiner=./icefall-asr-zipformer-streaming-wenetspeech-20230615/exp/joiner-epoch-12-avg-4-chunk-16-left-128.onnx \
    ./icefall-asr-zipformer-streaming-wenetspeech-20230615/test_wavs/DEV_T0000000000.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx.exe`` for Windows.

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

You should see the following output:

.. literalinclude:: ./code-zipformer/icefall-asr-zipformer-streaming-wenetspeech-20230615.txt

int8
^^^^

The following code shows how to use ``int8`` models to decode a wave file:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx \
    --tokens=./icefall-asr-zipformer-streaming-wenetspeech-20230615/data/lang_char/tokens.txt \
    --encoder=./icefall-asr-zipformer-streaming-wenetspeech-20230615/exp/encoder-epoch-12-avg-4-chunk-16-left-128.int8.onnx \
    --decoder=./icefall-asr-zipformer-streaming-wenetspeech-20230615/exp/decoder-epoch-12-avg-4-chunk-16-left-128.int8.onnx \
    --joiner=./icefall-asr-zipformer-streaming-wenetspeech-20230615/exp/joiner-epoch-12-avg-4-chunk-16-left-128.int8.onnx \
    ./icefall-asr-zipformer-streaming-wenetspeech-20230615/test_wavs/DEV_T0000000000.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx.exe`` for Windows.

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

You should see the following output:

.. literalinclude:: ./code-zipformer/icefall-asr-zipformer-streaming-wenetspeech-20230615-int8.txt

Real-time speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-microphone \
    ./icefall-asr-zipformer-streaming-wenetspeech-20230615/data/lang_char/tokens.txt \
    ./icefall-asr-zipformer-streaming-wenetspeech-20230615/exp/encoder-epoch-12-avg-4-chunk-16-left-128.int8.onnx \
    ./icefall-asr-zipformer-streaming-wenetspeech-20230615/exp/decoder-epoch-12-avg-4-chunk-16-left-128.int8.onnx \
    ./icefall-asr-zipformer-streaming-wenetspeech-20230615/exp/joiner-epoch-12-avg-4-chunk-16-left-128.int8.onnx

.. hint::

   If your system is Linux (including embedded Linux), you can also use
   :ref:`sherpa-onnx-alsa` to do real-time speech recognition with your
   microphone if ``sherpa-onnx-microphone`` does not work for you.


.. _sherpa-onnx-streaming-zipformer-en-2023-06-26-english:

csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26 (English)
------------------------------------------------------------------

This model is converted from

`<https://huggingface.co/Zengwei/icefall-asr-librispeech-streaming-zipformer-2023-05-17>`_

which supports only English as it is trained on the `LibriSpeech`_ corpus.

If you are interested in how the model is trained, please refer to
`<https://github.com/k2-fsa/icefall/pull/1058>`_.

In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26
  cd sherpa-onnx-streaming-zipformer-en-2023-06-26
  git lfs pull --include "*.onnx"

Please check that the file sizes of the pre-trained models are correct. See
the file sizes of ``*.onnx`` files below.

.. code-block:: bash

  sherpa-onnx-streaming-zipformer-en-2023-06-26 fangjun$ ls -lh *.onnx
  -rw-r--r--  1 fangjun  staff   1.2M Jun 26 11:53 decoder-epoch-99-avg-1-chunk-16-left-64.int8.onnx
  -rw-r--r--  1 fangjun  staff   2.0M Jun 26 11:53 decoder-epoch-99-avg-1-chunk-16-left-64.onnx
  -rw-r--r--  1 fangjun  staff    68M Jun 26 11:54 encoder-epoch-99-avg-1-chunk-16-left-64.int8.onnx
  -rw-r--r--  1 fangjun  staff   250M Jun 26 11:55 encoder-epoch-99-avg-1-chunk-16-left-64.onnx
  -rw-r--r--  1 fangjun  staff   253K Jun 26 11:53 joiner-epoch-99-avg-1-chunk-16-left-64.int8.onnx
  -rw-r--r--  1 fangjun  staff   1.0M Jun 26 11:53 joiner-epoch-99-avg-1-chunk-16-left-64.onnx

Decode a single wave file
~~~~~~~~~~~~~~~~~~~~~~~~~

.. hint::

   It supports decoding only wave files of a single channel with 16-bit
   encoded samples, while the sampling rate does not need to be 16 kHz.

fp32
^^^^

The following code shows how to use ``fp32`` models to decode a wave file:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx \
    --tokens=./sherpa-onnx-streaming-zipformer-en-2023-06-26/tokens.txt \
    --encoder=./sherpa-onnx-streaming-zipformer-en-2023-06-26/encoder-epoch-99-avg-1-chunk-16-left-64.onnx \
    --decoder=./sherpa-onnx-streaming-zipformer-en-2023-06-26/decoder-epoch-99-avg-1-chunk-16-left-64.onnx \
    --joiner=./sherpa-onnx-streaming-zipformer-en-2023-06-26/joiner-epoch-99-avg-1-chunk-16-left-64.onnx \
    ./sherpa-onnx-streaming-zipformer-en-2023-06-26/test_wavs/0.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx.exe`` for Windows.

You should see the following output:

.. literalinclude:: ./code-zipformer/sherpa-onnx-streaming-zipformer-en-2023-06-26.txt

int8
^^^^

The following code shows how to use ``int8`` models to decode a wave file:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx \
    --tokens=./sherpa-onnx-streaming-zipformer-en-2023-06-26/tokens.txt \
    --encoder=./sherpa-onnx-streaming-zipformer-en-2023-06-26/encoder-epoch-99-avg-1-chunk-16-left-64.int8.onnx \
    --decoder=./sherpa-onnx-streaming-zipformer-en-2023-06-26/decoder-epoch-99-avg-1-chunk-16-left-64.int8.onnx \
    --joiner=./sherpa-onnx-streaming-zipformer-en-2023-06-26/joiner-epoch-99-avg-1-chunk-16-left-64.int8.onnx \
    ./sherpa-onnx-streaming-zipformer-en-2023-06-26/test_wavs/0.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx.exe`` for Windows.

You should see the following output:

.. literalinclude:: ./code-zipformer/sherpa-onnx-streaming-zipformer-en-2023-06-26-int8.txt

Real-time speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-microphone \
    ./sherpa-onnx-streaming-zipformer-en-2023-06-26/tokens.txt \
    ./sherpa-onnx-streaming-zipformer-en-2023-06-26/encoder-epoch-99-avg-1-chunk-16-left-64.onnx \
    ./sherpa-onnx-streaming-zipformer-en-2023-06-26/decoder-epoch-99-avg-1-chunk-16-left-64.onnx \
    ./sherpa-onnx-streaming-zipformer-en-2023-06-26/joiner-epoch-99-avg-1-chunk-16-left-64.onnx

.. hint::

   If your system is Linux (including embedded Linux), you can also use
   :ref:`sherpa-onnx-alsa` to do real-time speech recognition with your
   microphone if ``sherpa-onnx-microphone`` does not work for you.


csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-21 (English)
------------------------------------------------------------------

This model is converted from

`<https://huggingface.co/marcoyang/icefall-libri-giga-pruned-transducer-stateless7-streaming-2023-04-04>`_

which supports only English as it is trained on the `LibriSpeech`_ and `GigaSpeech`_ corpus.

If you are interested in how the model is trained, please refer to
`<https://github.com/k2-fsa/icefall/pull/984>`_.

In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-21
  cd sherpa-onnx-streaming-zipformer-en-2023-06-21
  git lfs pull --include "*.onnx"

Please check that the file sizes of the pre-trained models are correct. See
the file sizes of ``*.onnx`` files below.

.. code-block:: bash

  sherpa-onnx-streaming-zipformer-en-2023-06-21 fangjun$ ls -lh *.onnx
  -rw-r--r--  1 fangjun  staff   1.2M Jun 21 15:34 decoder-epoch-99-avg-1.int8.onnx
  -rw-r--r--  1 fangjun  staff   2.0M Jun 21 15:34 decoder-epoch-99-avg-1.onnx
  -rw-r--r--  1 fangjun  staff   179M Jun 21 15:36 encoder-epoch-99-avg-1.int8.onnx
  -rw-r--r--  1 fangjun  staff   337M Jun 21 15:37 encoder-epoch-99-avg-1.onnx
  -rw-r--r--  1 fangjun  staff   253K Jun 21 15:34 joiner-epoch-99-avg-1.int8.onnx
  -rw-r--r--  1 fangjun  staff   1.0M Jun 21 15:34 joiner-epoch-99-avg-1.onnx

Decode a single wave file
~~~~~~~~~~~~~~~~~~~~~~~~~

.. hint::

   It supports decoding only wave files of a single channel with 16-bit
   encoded samples, while the sampling rate does not need to be 16 kHz.

fp32
^^^^

The following code shows how to use ``fp32`` models to decode a wave file:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx \
    --tokens=./sherpa-onnx-streaming-zipformer-en-2023-06-21/tokens.txt \
    --encoder=./sherpa-onnx-streaming-zipformer-en-2023-06-21/encoder-epoch-99-avg-1.onnx \
    --decoder=./sherpa-onnx-streaming-zipformer-en-2023-06-21/decoder-epoch-99-avg-1.onnx \
    --joiner=./sherpa-onnx-streaming-zipformer-en-2023-06-21/joiner-epoch-99-avg-1.onnx \
    ./sherpa-onnx-streaming-zipformer-en-2023-06-21/test_wavs/0.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx.exe`` for Windows.

You should see the following output:

.. literalinclude:: ./code-zipformer/sherpa-onnx-streaming-zipformer-en-2023-06-21.txt

int8
^^^^

The following code shows how to use ``int8`` models to decode a wave file:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx \
    --tokens=./sherpa-onnx-streaming-zipformer-en-2023-06-21/tokens.txt \
    --encoder=./sherpa-onnx-streaming-zipformer-en-2023-06-21/encoder-epoch-99-avg-1.int8.onnx \
    --decoder=./sherpa-onnx-streaming-zipformer-en-2023-06-21/decoder-epoch-99-avg-1.int8.onnx \
    --joiner=./sherpa-onnx-streaming-zipformer-en-2023-06-21/joiner-epoch-99-avg-1.int8.onnx \
    ./sherpa-onnx-streaming-zipformer-en-2023-06-21/test_wavs/0.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx.exe`` for Windows.

You should see the following output:

.. literalinclude:: ./code-zipformer/sherpa-onnx-streaming-zipformer-en-2023-06-21-int8.txt

Real-time speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-microphone \
    ./sherpa-onnx-streaming-zipformer-en-2023-06-21/tokens.txt \
    ./sherpa-onnx-streaming-zipformer-en-2023-06-21/encoder-epoch-99-avg-1.onnx \
    ./sherpa-onnx-streaming-zipformer-en-2023-06-21/decoder-epoch-99-avg-1.onnx \
    ./sherpa-onnx-streaming-zipformer-en-2023-06-21/joiner-epoch-99-avg-1.onnx

.. hint::

   If your system is Linux (including embedded Linux), you can also use
   :ref:`sherpa-onnx-alsa` to do real-time speech recognition with your
   microphone if ``sherpa-onnx-microphone`` does not work for you.


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

  sherpa-onnx-streaming-zipformer-en-2023-02-21$ ls -lh *.onnx
  -rw-r--r-- 1 kuangfangjun root  1.3M Mar 31 23:06 decoder-epoch-99-avg-1.int8.onnx
  -rw-r--r-- 1 kuangfangjun root  2.0M Feb 21 20:51 decoder-epoch-99-avg-1.onnx
  -rw-r--r-- 1 kuangfangjun root  180M Mar 31 23:07 encoder-epoch-99-avg-1.int8.onnx
  -rw-r--r-- 1 kuangfangjun root  338M Feb 21 20:51 encoder-epoch-99-avg-1.onnx
  -rw-r--r-- 1 kuangfangjun root  254K Mar 31 23:06 joiner-epoch-99-avg-1.int8.onnx
  -rw-r--r-- 1 kuangfangjun root 1003K Feb 21 20:51 joiner-epoch-99-avg-1.onnx

Decode a single wave file
~~~~~~~~~~~~~~~~~~~~~~~~~

.. hint::

   It supports decoding only wave files of a single channel with 16-bit
   encoded samples, while the sampling rate does not need to be 16 kHz.

fp32
^^^^

The following code shows how to use ``fp32`` models to decode a wave file:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx \
    --tokens=./sherpa-onnx-streaming-zipformer-en-2023-02-21/tokens.txt \
    --encoder=./sherpa-onnx-streaming-zipformer-en-2023-02-21/encoder-epoch-99-avg-1.onnx \
    --decoder=./sherpa-onnx-streaming-zipformer-en-2023-02-21/decoder-epoch-99-avg-1.onnx \
    --joiner=./sherpa-onnx-streaming-zipformer-en-2023-02-21/joiner-epoch-99-avg-1.onnx \
    ./sherpa-onnx-streaming-zipformer-en-2023-02-21/test_wavs/0.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx.exe`` for Windows.

You should see the following output:

.. literalinclude:: ./code-zipformer/sherpa-onnx-streaming-zipformer-en-2023-02-21.txt

int8
^^^^

The following code shows how to use ``int8`` models to decode a wave file:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx \
    --tokens=./sherpa-onnx-streaming-zipformer-en-2023-02-21/tokens.txt \
    --encoder=./sherpa-onnx-streaming-zipformer-en-2023-02-21/encoder-epoch-99-avg-1.int8.onnx \
    --decoder=./sherpa-onnx-streaming-zipformer-en-2023-02-21/decoder-epoch-99-avg-1.int8.onnx \
    --joiner=./sherpa-onnx-streaming-zipformer-en-2023-02-21/joiner-epoch-99-avg-1.int8.onnx \
    ./sherpa-onnx-streaming-zipformer-en-2023-02-21/test_wavs/0.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx.exe`` for Windows.

You should see the following output:

.. literalinclude:: ./code-zipformer/sherpa-onnx-streaming-zipformer-en-2023-02-21-int8.txt

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

csukuangfj/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20 (Bilingual, Chinese + English)
----------------------------------------------------------------------------------------------------

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

  sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20$ ls -lh *.onnx
  -rw-r--r-- 1 kuangfangjun root  13M Mar 31 21:11 decoder-epoch-99-avg-1.int8.onnx
  -rw-r--r-- 1 kuangfangjun root  14M Feb 20 20:13 decoder-epoch-99-avg-1.onnx
  -rw-r--r-- 1 kuangfangjun root 174M Mar 31 21:11 encoder-epoch-99-avg-1.int8.onnx
  -rw-r--r-- 1 kuangfangjun root 315M Feb 20 20:13 encoder-epoch-99-avg-1.onnx
  -rw-r--r-- 1 kuangfangjun root 3.1M Mar 31 21:11 joiner-epoch-99-avg-1.int8.onnx
  -rw-r--r-- 1 kuangfangjun root  13M Feb 20 20:13 joiner-epoch-99-avg-1.onnx


Decode a single wave file
~~~~~~~~~~~~~~~~~~~~~~~~~

.. hint::

   It supports decoding only wave files of a single channel with 16-bit
   encoded samples, while the sampling rate does not need to be 16 kHz.

fp32
^^^^

The following code shows how to use ``fp32`` models to decode a wave file:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx \
    --tokens=./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt \
    --encoder=./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.onnx \
    --decoder=./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx \
    --joiner=./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.onnx \
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

int8
^^^^

The following code shows how to use ``fp32`` models to decode a wave file:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx \
    --tokens=./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt \
    --encoder=./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.int8.onnx \
    --decoder=./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.int8.onnx \
    --joiner=./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.int8.onnx \
    ./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/test_wavs/1.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx.exe`` for Windows.

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

You should see the following output:

.. literalinclude:: ./code-zipformer/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20-int8.txt

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



.. _sherpa_onnx_streaming_zipformer_fr_2023_04_14:

shaojieli/sherpa-onnx-streaming-zipformer-fr-2023-04-14 (French)
----------------------------------------------------------------

This model is converted from

`<https://huggingface.co/shaojieli/icefall-asr-commonvoice-fr-pruned-transducer-stateless7-streaming-2023-04-02>`_

which supports only French as it is trained on the `CommonVoice`_ corpus.
In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

  cd /path/to/sherpa-onnx
  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/shaojieli/sherpa-onnx-streaming-zipformer-fr-2023-04-14
  cd sherpa-onnx-streaming-zipformer-fr-2023-04-14
  git lfs pull --include "*.onnx"


Please check that the file sizes of the pre-trained models are correct. See
the file sizes of ``*.onnx`` files below.

.. code-block:: bash

  sherpa-onnx-streaming-zipformer-fr-2023-04-14 shaojieli$ ls -lh *.bin

  -rw-r--r-- 1 lishaojie Students  1.3M 4月  14 14:09 decoder-epoch-29-avg-9-with-averaged-model.int8.onnx
  -rw-r--r-- 1 lishaojie Students  2.0M 4月  14 14:09 decoder-epoch-29-avg-9-with-averaged-model.onnx
  -rw-r--r-- 1 lishaojie Students  121M 4月  14 14:09 encoder-epoch-29-avg-9-with-averaged-model.int8.onnx
  -rw-r--r-- 1 lishaojie Students  279M 4月  14 14:09 encoder-epoch-29-avg-9-with-averaged-model.onnx
  -rw-r--r-- 1 lishaojie Students  254K 4月  14 14:09 joiner-epoch-29-avg-9-with-averaged-model.int8.onnx
  -rw-r--r-- 1 lishaojie Students 1003K 4月  14 14:09 joiner-epoch-29-avg-9-with-averaged-model.onnx

Decode a single wave file
~~~~~~~~~~~~~~~~~~~~~~~~~

.. hint::

   It supports decoding only wave files of a single channel with 16-bit
   encoded samples, while the sampling rate does not need to be 16 kHz.

fp32
^^^^

The following code shows how to use ``fp32`` models to decode a wave file:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx \
    --tokens=./sherpa-onnx-streaming-zipformer-fr-2023-04-14/tokens.txt \
    --encoder=./sherpa-onnx-streaming-zipformer-fr-2023-04-14/encoder-epoch-29-avg-9-with-averaged-model.onnx \
    --decoder=./sherpa-onnx-streaming-zipformer-fr-2023-04-14/decoder-epoch-29-avg-9-with-averaged-model.onnx \
    --joiner=./sherpa-onnx-streaming-zipformer-fr-2023-04-14/joiner-epoch-29-avg-9-with-averaged-model.onnx \
    ./sherpa-onnx-streaming-zipformer-fr-2023-04-14/test_wavs/common_voice_fr_19364697.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx.exe`` for Windows.

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

You should see the following output:

.. literalinclude:: ./code-zipformer/sherpa-onnx-streaming-zipformer-fr-2023-04-14.txt

int8
^^^^

The following code shows how to use ``fp32`` models to decode a wave file:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx \
    --tokens=./sherpa-onnx-streaming-zipformer-fr-2023-04-14/tokens.txt \
    --encoder=./sherpa-onnx-streaming-zipformer-fr-2023-04-14/encoder-epoch-29-avg-9-with-averaged-model.int8.onnx \
    --decoder=./sherpa-onnx-streaming-zipformer-fr-2023-04-14/decoder-epoch-29-avg-9-with-averaged-model.int8.onnx \
    --joiner=./sherpa-onnx-streaming-zipformer-fr-2023-04-14/joiner-epoch-29-avg-9-with-averaged-model.int8.onnx \
    ./sherpa-onnx-streaming-zipformer-fr-2023-04-14/test_wavs/common_voice_fr_19364697.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx.exe`` for Windows.

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

You should see the following output:

.. literalinclude:: ./code-zipformer/sherpa-onnx-streaming-zipformer-fr-2023-04-14-int8.txt

Real-time speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx
  ./build/bin/sherpa-onnx-microphone \
    ./sherpa-onnx-streaming-zipformer-fr-2023-04-14/tokens.txt \
    ./sherpa-onnx-streaming-zipformer-fr-2023-04-14/encoder-epoch-29-avg-9-with-averaged-model.onnx \
    ./sherpa-onnx-streaming-zipformer-fr-2023-04-14/decoder-epoch-29-avg-9-with-averaged-model.onnx \
    ./sherpa-onnx-streaming-zipformer-fr-2023-04-14/joiner-epoch-29-avg-9-with-averaged-model.onnx \

.. hint::

   If your system is Linux (including embedded Linux), you can also use
   :ref:`sherpa-onnx-alsa` to do real-time speech recognition with your
   microphone if ``sherpa-onnx-microphone`` does not work for you.

.. _sherpa_onnx_streaming_zipformer_zh_14M_2023_02_23:

csukuangfj/sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23 (Chinese)
----------------------------------------------------------------------

.. hint::

   It is a small model.

This model is from

`<https://huggingface.co/marcoyang/sherpa-ncnn-streaming-zipformer-zh-14M-2023-02-23/>`_

which supports only Chinese as it is trained on the `WenetSpeech`_ corpus.

In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23
  cd sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23
  git lfs pull --include ".*onnx"

Please check that the file sizes of the pre-trained models are correct. See
the file sizes of ``*.onnx`` files below.

.. code-block:: bash

  sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23 fangjun$ ls -lh *.onnx
  -rw-r--r--  1 fangjun  staff   1.8M Sep 10 15:31 decoder-epoch-99-avg-1.int8.onnx
  -rw-r--r--  1 fangjun  staff   7.2M Sep 10 15:31 decoder-epoch-99-avg-1.onnx
  -rw-r--r--  1 fangjun  staff    21M Sep 10 15:31 encoder-epoch-99-avg-1.int8.onnx
  -rw-r--r--  1 fangjun  staff    39M Sep 10 15:31 encoder-epoch-99-avg-1.onnx
  -rw-r--r--  1 fangjun  staff   1.7M Sep 10 15:31 joiner-epoch-99-avg-1.int8.onnx
  -rw-r--r--  1 fangjun  staff   6.8M Sep 10 15:31 joiner-epoch-99-avg-1.onnx

Decode a single wave file
~~~~~~~~~~~~~~~~~~~~~~~~~

.. hint::

   It supports decoding only wave files of a single channel with 16-bit
   encoded samples, while the sampling rate does not need to be 16 kHz.

fp32
^^^^

The following code shows how to use ``fp32`` models to decode a wave file:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx \
    --tokens=./sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23/tokens.txt \
    --encoder=./sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23/encoder-epoch-99-avg-1.onnx \
    --decoder=./sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23/decoder-epoch-99-avg-1.onnx \
    --joiner=./sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23/joiner-epoch-99-avg-1.onnx \
    ./sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23/test_wavs/0.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx.exe`` for Windows.

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

You should see the following output:

.. literalinclude:: ./code-zipformer/sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23.txt

int8
^^^^

The following code shows how to use ``int8`` models to decode a wave file:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx \
    --tokens=./sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23/tokens.txt \
    --encoder=./sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23/encoder-epoch-99-avg-1.int8.onnx \
    --decoder=./sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23/decoder-epoch-99-avg-1.onnx \
    --joiner=./sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23/joiner-epoch-99-avg-1.int8.onnx \
    ./sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23/test_wavs/0.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx.exe`` for Windows.

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

You should see the following output:

.. literalinclude:: ./code-zipformer/sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23-int8.txt

Real-time speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-microphone \
    --tokens=./sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23/tokens.txt \
    --encoder=./sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23/encoder-epoch-99-avg-1.onnx \
    --decoder=./sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23/decoder-epoch-99-avg-1.onnx \
    --joiner=./sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23/joiner-epoch-99-avg-1.onnx

.. hint::

   If your system is Linux (including embedded Linux), you can also use
   :ref:`sherpa-onnx-alsa` to do real-time speech recognition with your
   microphone if ``sherpa-onnx-microphone`` does not work for you.


.. _sherpa_onnx_streaming_zipformer_en_20M_2023_02_17:

csukuangfj/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17 (English)
-----------------------------------------------------------------------

.. hint::

   It is a small model.

This model is from

`<https://huggingface.co/desh2608/icefall-asr-librispeech-pruned-transducer-stateless7-streaming-small>`_

which supports only English as it is trained on the `LibriSpeech`_ corpus.

In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/desh2608/icefall-asr-librispeech-pruned-transducer-stateless7-streaming-small
  cd icefall-asr-librispeech-pruned-transducer-stateless7-streaming-small
  git lfs pull --include ".*onnx"

Please check that the file sizes of the pre-trained models are correct. See
the file sizes of ``*.onnx`` files below.

.. code-block:: bash

  sherpa-onnx-streaming-zipformer-en-20M-2023-02-17 fangjun$ ls -lh *.onnx
  -rw-r--r--  1 fangjun  staff   527K Sep 10 17:06 decoder-epoch-99-avg-1.int8.onnx
  -rw-r--r--  1 fangjun  staff   2.0M Sep 10 17:06 decoder-epoch-99-avg-1.onnx
  -rw-r--r--  1 fangjun  staff    41M Sep 10 17:06 encoder-epoch-99-avg-1.int8.onnx
  -rw-r--r--  1 fangjun  staff    85M Sep 10 17:06 encoder-epoch-99-avg-1.onnx
  -rw-r--r--  1 fangjun  staff   253K Sep 10 17:06 joiner-epoch-99-avg-1.int8.onnx
  -rw-r--r--  1 fangjun  staff   1.0M Sep 10 17:06 joiner-epoch-99-avg-1.onnx

Decode a single wave file
~~~~~~~~~~~~~~~~~~~~~~~~~

.. hint::

   It supports decoding only wave files of a single channel with 16-bit
   encoded samples, while the sampling rate does not need to be 16 kHz.

fp32
^^^^

The following code shows how to use ``fp32`` models to decode a wave file:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx \
    --tokens=./sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/tokens.txt \
    --encoder=./sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/encoder-epoch-99-avg-1.onnx \
    --decoder=./sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/decoder-epoch-99-avg-1.onnx \
    --joiner=./sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/joiner-epoch-99-avg-1.onnx \
    ./sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/test_wavs/0.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx.exe`` for Windows.

You should see the following output:

.. literalinclude:: ./code-zipformer/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.txt

int8
^^^^

The following code shows how to use ``int8`` models to decode a wave file:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx \
    --tokens=./sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/tokens.txt \
    --encoder=./sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/encoder-epoch-99-avg-1.int8.onnx \
    --decoder=./sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/decoder-epoch-99-avg-1.onnx \
    --joiner=./sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/joiner-epoch-99-avg-1.int8.onnx \
    ./sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/test_wavs/0.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx.exe`` for Windows.

You should see the following output:

.. literalinclude:: ./code-zipformer/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17-int8.txt

Real-time speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-microphone \
    --tokens=./sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/tokens.txt \
    --encoder=./sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/encoder-epoch-99-avg-1.onnx \
    --decoder=./sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/decoder-epoch-99-avg-1.onnx \
    --joiner=./sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/joiner-epoch-99-avg-1.onnx \

.. hint::

   If your system is Linux (including embedded Linux), you can also use
   :ref:`sherpa-onnx-alsa` to do real-time speech recognition with your
   microphone if ``sherpa-onnx-microphone`` does not work for you.
