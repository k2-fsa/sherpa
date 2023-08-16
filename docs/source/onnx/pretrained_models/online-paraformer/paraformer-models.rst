Paraformer models
=================

.. hint::

   Please refer to :ref:`install_sherpa_onnx` to install `sherpa-onnx`_
   before you read this section.

.. _sherpa_onnx_online_paraformer_bilingual_zh_en:

csukuangfj/sherpa-onnx-streaming-paraformer-bilingual-zh-en (Chinese + English)
-------------------------------------------------------------------------------

This model is converted from

`<https://www.modelscope.cn/models/damo/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8404-online/summary>`_

The code for converting can be found at

`<https://huggingface.co/csukuangfj/streaming-paraformer-zh>`_


In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/sherpa-onnx-streaming-paraformer-bilingual-zh-en
  cd sherpa-onnx-streaming-paraformer-bilingual-zh-en
  git lfs pull --include "*.onnx"

Please check that the file sizes of the pre-trained models are correct. See
the file sizes of ``*.onnx`` files below.

.. code-block:: bash

  sherpa-onnx-streaming-paraformer-bilingual-zh-en fangjun$ ls -lh *.onnx
  -rw-r--r--  1 fangjun  staff    68M Aug 14 09:53 decoder.int8.onnx
  -rw-r--r--  1 fangjun  staff   218M Aug 14 09:55 decoder.onnx
  -rw-r--r--  1 fangjun  staff   158M Aug 14 09:54 encoder.int8.onnx
  -rw-r--r--  1 fangjun  staff   607M Aug 14 09:57 encoder.onnx

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
    --tokens=./sherpa-onnx-streaming-paraformer-bilingual-zh-en/tokens.txt \
    --paraformer-encoder=./sherpa-onnx-streaming-paraformer-bilingual-zh-en/encoder.onnx \
    --paraformer-decoder=./sherpa-onnx-streaming-paraformer-bilingual-zh-en/decoder.onnx \
    ./sherpa-onnx-streaming-paraformer-bilingual-zh-en/test_wavs/0.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx.exe`` for Windows.

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

You should see the following output:

.. literalinclude:: ./code-paraformer/sherpa-onnx-streaming-paraformer-bilingual-zh-en.txt

int8
^^^^

The following code shows how to use ``int8`` models to decode a wave file:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx \
    --tokens=./sherpa-onnx-streaming-paraformer-bilingual-zh-en/tokens.txt \
    --paraformer-encoder=./sherpa-onnx-streaming-paraformer-bilingual-zh-en/encoder.int8.onnx \
    --paraformer-decoder=./sherpa-onnx-streaming-paraformer-bilingual-zh-en/decoder.int8.onnx \
    ./sherpa-onnx-streaming-paraformer-bilingual-zh-en/test_wavs/0.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx.exe`` for Windows.

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

You should see the following output:

.. literalinclude:: ./code-paraformer/sherpa-onnx-streaming-paraformer-bilingual-zh-en.int8.txt

Real-time speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-microphone \
    --tokens=./sherpa-onnx-streaming-paraformer-bilingual-zh-en/tokens.txt \
    --paraformer-encoder=./sherpa-onnx-streaming-paraformer-bilingual-zh-en/encoder.int8.onnx \
    --paraformer-decoder=./sherpa-onnx-streaming-paraformer-bilingual-zh-en/decoder.int8.onnx

.. hint::

   If your system is Linux (including embedded Linux), you can also use
   :ref:`sherpa-onnx-alsa` to do real-time speech recognition with your
   microphone if ``sherpa-onnx-microphone`` does not work for you.
