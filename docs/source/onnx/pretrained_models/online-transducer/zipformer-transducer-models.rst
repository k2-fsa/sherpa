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
