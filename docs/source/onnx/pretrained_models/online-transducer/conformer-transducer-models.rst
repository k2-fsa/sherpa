.. _sherpa_onnx_streaming_conformer_transducer_models:

Conformer-transducer-based Models
=================================

.. hint::

   Please refer to :ref:`install_sherpa_onnx` to install `sherpa-onnx`_
   before you read this section.

csukuangfj/sherpa-onnx-streaming-conformer-zh-2023-05-23 (Chinese)
------------------------------------------------------------------

This model is converted from

`<https://huggingface.co/luomingshuang/icefall_asr_wenetspeech_pruned_transducer_stateless5_streaming>`_


which supports only English as it is trained on the `WenetSpeech`_ corpus.

You can find the training code at

`<https://github.com/k2-fsa/icefall/tree/master/egs/wenetspeech/ASR/pruned_transducer_stateless5>`_

In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/sherpa-onnx-streaming-conformer-zh-2023-05-23
  cd sherpa-onnx-streaming-conformer-zh-2023-05-23
  git lfs pull --include "*.onnx"

Please check that the file sizes of the pre-trained models are correct. See
the file sizes of ``*.onnx`` files below.

.. code-block:: bash

  sherpa-onnx-streaming-conformer-zh-2023-05-23 fangjun$ ls -lh *.onnx
  -rw-r--r--  1 fangjun  staff    11M May 23 14:44 decoder-epoch-99-avg-1.int8.onnx
  -rw-r--r--  1 fangjun  staff    12M May 23 14:44 decoder-epoch-99-avg-1.onnx
  -rw-r--r--  1 fangjun  staff   160M May 23 14:46 encoder-epoch-99-avg-1.int8.onnx
  -rw-r--r--  1 fangjun  staff   345M May 23 14:47 encoder-epoch-99-avg-1.onnx
  -rw-r--r--  1 fangjun  staff   2.7M May 23 14:44 joiner-epoch-99-avg-1.int8.onnx
  -rw-r--r--  1 fangjun  staff    11M May 23 14:44 joiner-epoch-99-avg-1.onnx

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
    --tokens=./sherpa-onnx-streaming-conformer-zh-2023-05-23/tokens.txt \
    --encoder=./sherpa-onnx-streaming-conformer-zh-2023-05-23/encoder-epoch-99-avg-1.onnx \
    --decoder=./sherpa-onnx-streaming-conformer-zh-2023-05-23/decoder-epoch-99-avg-1.onnx \
    --joiner=./sherpa-onnx-streaming-conformer-zh-2023-05-23/joiner-epoch-99-avg-1.onnx \
    ./sherpa-onnx-streaming-conformer-zh-2023-05-23/test_wavs/0.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx.exe`` for Windows.

You should see the following output:

.. literalinclude:: ./code-conformer/sherpa-onnx-streaming-conformer-zh-2023-05-23.txt

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

int8
^^^^

The following code shows how to use ``int8`` models to decode a wave file:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx \
    --tokens=./sherpa-onnx-streaming-conformer-zh-2023-05-23/tokens.txt \
    --encoder=./sherpa-onnx-streaming-conformer-zh-2023-05-23/encoder-epoch-99-avg-1.int8.onnx \
    --decoder=./sherpa-onnx-streaming-conformer-zh-2023-05-23/decoder-epoch-99-avg-1.int8.onnx \
    --joiner=./sherpa-onnx-streaming-conformer-zh-2023-05-23/joiner-epoch-99-avg-1.int8.onnx \
    ./sherpa-onnx-streaming-conformer-zh-2023-05-23/test_wavs/0.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx.exe`` for Windows.

You should see the following output:

.. literalinclude:: ./code-conformer/sherpa-onnx-streaming-conformer-zh-2023-05-23-int8.txt

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

Real-time speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-microphone \
    ./sherpa-onnx-streaming-conformer-zh-2023-05-23/tokens.txt \
    ./sherpa-onnx-streaming-conformer-zh-2023-05-23/encoder-epoch-99-avg-1.onnx \
    ./sherpa-onnx-streaming-conformer-zh-2023-05-23/decoder-epoch-99-avg-1.onnx \
    ./sherpa-onnx-streaming-conformer-zh-2023-05-23/joiner-epoch-99-avg-1.onnx

.. hint::

   If your system is Linux (including embedded Linux), you can also use
   :ref:`sherpa-onnx-alsa` to do real-time speech recognition with your
   microphone if ``sherpa-onnx-microphone`` does not work for you.
