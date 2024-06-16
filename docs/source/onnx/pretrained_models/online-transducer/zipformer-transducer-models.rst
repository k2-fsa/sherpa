.. _sherpa_onnx_zipformer_transducer_models:

Zipformer-transducer-based Models
=================================

.. hint::

   Please refer to :ref:`install_sherpa_onnx` to install `sherpa-onnx`_
   before you read this section.

sherpa-onnx-streaming-zipformer-korean-2024-06-16 (Korean)
----------------------------------------------------------

Training code for this model can be found at `<https://github.com/k2-fsa/icefall/pull/1651>`_.
It supports only Korean.

PyTorch checkpoint with optimizer state can be found at
`<https://huggingface.co/johnBamma/icefall-asr-ksponspeech-pruned-transducer-stateless7-streaming-2024-06-12>`_

In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-korean-2024-06-16.tar.bz2

  # For Chinese users, you can use the following mirror
  # wget https://hub.nuaa.cf/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-korean-2024-06-16.tar.bz2

  tar xvf sherpa-onnx-streaming-zipformer-korean-2024-06-16.tar.bz2
  rm sherpa-onnx-streaming-zipformer-korean-2024-06-16.tar.bz2
  ls -lh sherpa-onnx-streaming-zipformer-korean-2024-06-16

The output is given below:

.. code-block::

  $ ls -lh sherpa-onnx-streaming-zipformer-korean-2024-06-16

  total 907104
  -rw-r--r--  1 fangjun  staff   307K Jun 16 17:36 bpe.model
  -rw-r--r--  1 fangjun  staff   2.7M Jun 16 17:36 decoder-epoch-99-avg-1.int8.onnx
  -rw-r--r--  1 fangjun  staff    11M Jun 16 17:36 decoder-epoch-99-avg-1.onnx
  -rw-r--r--  1 fangjun  staff   121M Jun 16 17:36 encoder-epoch-99-avg-1.int8.onnx
  -rw-r--r--  1 fangjun  staff   279M Jun 16 17:36 encoder-epoch-99-avg-1.onnx
  -rw-r--r--  1 fangjun  staff   2.5M Jun 16 17:36 joiner-epoch-99-avg-1.int8.onnx
  -rw-r--r--  1 fangjun  staff   9.8M Jun 16 17:36 joiner-epoch-99-avg-1.onnx
  drwxr-xr-x  7 fangjun  staff   224B Jun 16 17:36 test_wavs
  -rw-r--r--  1 fangjun  staff    59K Jun 16 17:36 tokens.txt

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
    --tokens=./sherpa-onnx-streaming-zipformer-korean-2024-06-16/tokens.txt \
    --encoder=./sherpa-onnx-streaming-zipformer-korean-2024-06-16/encoder-epoch-99-avg-1.onnx \
    --decoder=./sherpa-onnx-streaming-zipformer-korean-2024-06-16/decoder-epoch-99-avg-1.onnx \
    --joiner=./sherpa-onnx-streaming-zipformer-korean-2024-06-16/joiner-epoch-99-avg-1.onnx \
    ./sherpa-onnx-streaming-zipformer-korean-2024-06-16/test_wavs/0.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx.exe`` for Windows.

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

You should see the following output:

.. literalinclude:: ./code-zipformer/sherpa-onnx-streaming-zipformer-korean-2024-06-16.txt

int8
^^^^

The following code shows how to use ``int8`` models to decode a wave file:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx \
    --tokens=./sherpa-onnx-streaming-zipformer-korean-2024-06-16/tokens.txt \
    --encoder=./sherpa-onnx-streaming-zipformer-korean-2024-06-16/encoder-epoch-99-avg-1.int8.onnx \
    --decoder=./sherpa-onnx-streaming-zipformer-korean-2024-06-16/decoder-epoch-99-avg-1.onnx \
    --joiner=./sherpa-onnx-streaming-zipformer-korean-2024-06-16/joiner-epoch-99-avg-1.int8.onnx \
    ./sherpa-onnx-streaming-zipformer-korean-2024-06-16/test_wavs/0.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx.exe`` for Windows.

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

You should see the following output:

.. literalinclude:: ./code-zipformer/sherpa-onnx-streaming-zipformer-korean-2024-06-16-int8.txt

Real-time speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-microphone \
    --tokens=./sherpa-onnx-streaming-zipformer-korean-2024-06-16/tokens.txt \
    --encoder=./sherpa-onnx-streaming-zipformer-korean-2024-06-16/encoder-epoch-99-avg-1.int8.onnx \
    --decoder=./sherpa-onnx-streaming-zipformer-korean-2024-06-16/decoder-epoch-99-avg-1.onnx \
    --joiner=./sherpa-onnx-streaming-zipformer-korean-2024-06-16/joiner-epoch-99-avg-1.int8.onnx

.. hint::

   If your system is Linux (including embedded Linux), you can also use
   :ref:`sherpa-onnx-alsa` to do real-time speech recognition with your
   microphone if ``sherpa-onnx-microphone`` does not work for you.


sherpa-onnx-streaming-zipformer-multi-zh-hans-2023-12-12 (Chinese)
------------------------------------------------------------------

Training code for this model can be found at `<https://github.com/k2-fsa/icefall/pull/1369>`_.
It supports only Chinese.

Please refer to `<https://github.com/k2-fsa/icefall/tree/master/egs/multi_zh-hans/ASR#included-training-sets>`_
for the detailed information about the training data. In total, there are 14k hours of training data.

In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-multi-zh-hans-2023-12-12.tar.bz2

  # For Chinese users, you can use the following mirror
  # wget https://hub.nuaa.cf/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-multi-zh-hans-2023-12-12.tar.bz2

  tar xf sherpa-onnx-streaming-zipformer-multi-zh-hans-2023-12-12.tar.bz2
  rm sherpa-onnx-streaming-zipformer-multi-zh-hans-2023-12-12.tar.bz2
  ls -lh sherpa-onnx-streaming-zipformer-multi-zh-hans-2023-12-12

The output is given below:

.. code-block::

  $ ls -lh sherpa-onnx-streaming-zipformer-multi-zh-hans-2023-12-12
  total 668864
  -rw-r--r--  1 fangjun  staff    28B Dec 12 18:59 README.md
  -rw-r--r--  1 fangjun  staff   131B Dec 12 18:59 bpe.model
  -rw-r--r--  1 fangjun  staff   1.2M Dec 12 18:59 decoder-epoch-20-avg-1-chunk-16-left-128.int8.onnx
  -rw-r--r--  1 fangjun  staff   4.9M Dec 12 18:59 decoder-epoch-20-avg-1-chunk-16-left-128.onnx
  -rw-r--r--  1 fangjun  staff    67M Dec 12 18:59 encoder-epoch-20-avg-1-chunk-16-left-128.int8.onnx
  -rw-r--r--  1 fangjun  staff   249M Dec 12 18:59 encoder-epoch-20-avg-1-chunk-16-left-128.onnx
  -rw-r--r--  1 fangjun  staff   1.0M Dec 12 18:59 joiner-epoch-20-avg-1-chunk-16-left-128.int8.onnx
  -rw-r--r--  1 fangjun  staff   3.9M Dec 12 18:59 joiner-epoch-20-avg-1-chunk-16-left-128.onnx
  drwxr-xr-x  8 fangjun  staff   256B Dec 12 18:59 test_wavs
  -rw-r--r--  1 fangjun  staff    18K Dec 12 18:59 tokens.txt

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
    --tokens=./sherpa-onnx-streaming-zipformer-multi-zh-hans-2023-12-12/tokens.txt \
    --encoder=./sherpa-onnx-streaming-zipformer-multi-zh-hans-2023-12-12/encoder-epoch-20-avg-1-chunk-16-left-128.onnx \
    --decoder=./sherpa-onnx-streaming-zipformer-multi-zh-hans-2023-12-12/decoder-epoch-20-avg-1-chunk-16-left-128.onnx \
    --joiner=./sherpa-onnx-streaming-zipformer-multi-zh-hans-2023-12-12/joiner-epoch-20-avg-1-chunk-16-left-128.onnx \
    ./sherpa-onnx-streaming-zipformer-multi-zh-hans-2023-12-12/test_wavs/DEV_T0000000000.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx.exe`` for Windows.

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

You should see the following output:

.. literalinclude:: ./code-zipformer/sherpa-onnx-streaming-zipformer-multi-zh-hans-2023-12-12.txt

int8
^^^^

The following code shows how to use ``int8`` models to decode a wave file:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx \
    --tokens=./sherpa-onnx-streaming-zipformer-multi-zh-hans-2023-12-12/tokens.txt \
    --encoder=./sherpa-onnx-streaming-zipformer-multi-zh-hans-2023-12-12/encoder-epoch-20-avg-1-chunk-16-left-128.int8.onnx \
    --decoder=./sherpa-onnx-streaming-zipformer-multi-zh-hans-2023-12-12/decoder-epoch-20-avg-1-chunk-16-left-128.onnx \
    --joiner=./sherpa-onnx-streaming-zipformer-multi-zh-hans-2023-12-12/joiner-epoch-20-avg-1-chunk-16-left-128.int8.onnx \
    ./sherpa-onnx-streaming-zipformer-multi-zh-hans-2023-12-12/test_wavs/DEV_T0000000000.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx.exe`` for Windows.

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

You should see the following output:

.. literalinclude:: ./code-zipformer/sherpa-onnx-streaming-zipformer-multi-zh-hans-2023-12-12-int8.txt

Real-time speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-microphone \
    --tokens=./sherpa-onnx-streaming-zipformer-multi-zh-hans-2023-12-12/tokens.txt \
    --encoder=./sherpa-onnx-streaming-zipformer-multi-zh-hans-2023-12-12/encoder-epoch-20-avg-1-chunk-16-left-128.int8.onnx \
    --decoder=./sherpa-onnx-streaming-zipformer-multi-zh-hans-2023-12-12/decoder-epoch-20-avg-1-chunk-16-left-128.onnx \
    --joiner=./sherpa-onnx-streaming-zipformer-multi-zh-hans-2023-12-12/joiner-epoch-20-avg-1-chunk-16-left-128.int8.onnx

.. hint::

   If your system is Linux (including embedded Linux), you can also use
   :ref:`sherpa-onnx-alsa` to do real-time speech recognition with your
   microphone if ``sherpa-onnx-microphone`` does not work for you.


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

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/icefall-asr-zipformer-streaming-wenetspeech-20230615.tar.bz2

  # For Chinese users, you can use the following mirror
  # wget https://hub.nuaa.cf/k2-fsa/sherpa-onnx/releases/download/asr-models/icefall-asr-zipformer-streaming-wenetspeech-20230615.tar.bz2

  tar xvf icefall-asr-zipformer-streaming-wenetspeech-20230615.tar.bz2
  rm icefall-asr-zipformer-streaming-wenetspeech-20230615.tar.bz2

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
    --decoder=./icefall-asr-zipformer-streaming-wenetspeech-20230615/exp/decoder-epoch-12-avg-4-chunk-16-left-128.onnx \
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
    --tokens=./icefall-asr-zipformer-streaming-wenetspeech-20230615/data/lang_char/tokens.txt \
    --encoder=./icefall-asr-zipformer-streaming-wenetspeech-20230615/exp/encoder-epoch-12-avg-4-chunk-16-left-128.int8.onnx \
    --decoder=./icefall-asr-zipformer-streaming-wenetspeech-20230615/exp/decoder-epoch-12-avg-4-chunk-16-left-128.onnx \
    --joiner=./icefall-asr-zipformer-streaming-wenetspeech-20230615/exp/joiner-epoch-12-avg-4-chunk-16-left-128.int8.onnx

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

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2

  # For Chinese users, you can use the following mirror
  # wget https://hub.nuaa.cf/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2

  tar xvf sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2
  rm sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2

Please check that the file sizes of the pre-trained models are correct. See
the file sizes below.

.. code-block:: bash

    -rw-r--r-- 1 1001 127  240K Apr 23 06:45 bpe.model
    -rw-r--r-- 1 1001 127  1.3M Apr 23 06:45 decoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx
    -rw-r--r-- 1 1001 127  2.0M Apr 23 06:45 decoder-epoch-99-avg-1-chunk-16-left-128.onnx
    -rw-r--r-- 1 1001 127   68M Apr 23 06:45 encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx
    -rw-r--r-- 1 1001 127  250M Apr 23 06:45 encoder-epoch-99-avg-1-chunk-16-left-128.onnx
    -rwxr-xr-x 1 1001 127   814 Apr 23 06:45 export-onnx-zipformer-online.sh
    -rw-r--r-- 1 1001 127  254K Apr 23 06:45 joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx
    -rw-r--r-- 1 1001 127 1003K Apr 23 06:45 joiner-epoch-99-avg-1-chunk-16-left-128.onnx
    -rw-r--r-- 1 1001 127   216 Apr 23 06:45 README.md
    drwxr-xr-x 2 1001 127  4.0K Apr 23 06:45 test_wavs
    -rw-r--r-- 1 1001 127  5.0K Apr 23 06:45 tokens.txt

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
    --encoder=./sherpa-onnx-streaming-zipformer-en-2023-06-26/encoder-epoch-99-avg-1-chunk-16-left-128.onnx \
    --decoder=./sherpa-onnx-streaming-zipformer-en-2023-06-26/decoder-epoch-99-avg-1-chunk-16-left-128.onnx \
    --joiner=./sherpa-onnx-streaming-zipformer-en-2023-06-26/joiner-epoch-99-avg-1-chunk-16-left-128.onnx \
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
    --encoder=./sherpa-onnx-streaming-zipformer-en-2023-06-26/encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx \
    --decoder=./sherpa-onnx-streaming-zipformer-en-2023-06-26/decoder-epoch-99-avg-1-chunk-16-left-128.onnx \
    --joiner=./sherpa-onnx-streaming-zipformer-en-2023-06-26/joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx \
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
    --tokens=./sherpa-onnx-streaming-zipformer-en-2023-06-26/tokens.txt \
    --encoder=./sherpa-onnx-streaming-zipformer-en-2023-06-26/encoder-epoch-99-avg-1-chunk-16-left-128.onnx \
    --decoder=./sherpa-onnx-streaming-zipformer-en-2023-06-26/decoder-epoch-99-avg-1-chunk-16-left-128.onnx \
    --joiner=./sherpa-onnx-streaming-zipformer-en-2023-06-26/joiner-epoch-99-avg-1-chunk-16-left-128.onnx

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

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-21.tar.bz2

  # For Chinese users, you can use the following mirror
  # wget https://hub.nuaa.cf/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-21.tar.bz2

  tar xvf sherpa-onnx-streaming-zipformer-en-2023-06-21.tar.bz2
  rm sherpa-onnx-streaming-zipformer-en-2023-06-21.tar.bz2

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
    --decoder=./sherpa-onnx-streaming-zipformer-en-2023-06-21/decoder-epoch-99-avg-1.onnx \
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
    --tokens=./sherpa-onnx-streaming-zipformer-en-2023-06-21/tokens.txt \
    --encoder=./sherpa-onnx-streaming-zipformer-en-2023-06-21/encoder-epoch-99-avg-1.onnx \
    --decoder=./sherpa-onnx-streaming-zipformer-en-2023-06-21/decoder-epoch-99-avg-1.onnx \
    --joiner=./sherpa-onnx-streaming-zipformer-en-2023-06-21/joiner-epoch-99-avg-1.onnx

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

.. tabs::

   .. tab:: GitHub

      .. code-block:: bash

        cd /path/to/sherpa-onnx

        wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-02-21.tar.bz2

        # For Chinese users, you can use the following mirror
        # wget https://hub.nuaa.cf/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-02-21.tar.bz2

        tar xvf sherpa-onnx-streaming-zipformer-en-2023-02-21.tar.bz2
        rm sherpa-onnx-streaming-zipformer-en-2023-02-21.tar.bz2

   .. tab:: ModelScope

      .. code-block:: bash

        cd /path/to/sherpa-onnx

        GIT_LFS_SKIP_SMUDGE=1 git clone https://www.modelscope.cn/pkufool/sherpa-onnx-streaming-zipformer-en-2023-02-21.git
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
    --decoder=./sherpa-onnx-streaming-zipformer-en-2023-02-21/decoder-epoch-99-avg-1.onnx \
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
    --tokens=./sherpa-onnx-streaming-zipformer-en-2023-02-21/tokens.txt \
    --encoder=./sherpa-onnx-streaming-zipformer-en-2023-02-21/encoder-epoch-99-avg-1.onnx \
    --decoder=./sherpa-onnx-streaming-zipformer-en-2023-02-21/decoder-epoch-99-avg-1.onnx \
    --joiner=./sherpa-onnx-streaming-zipformer-en-2023-02-21/joiner-epoch-99-avg-1.onnx

.. hint::

   If your system is Linux (including embedded Linux), you can also use
   :ref:`sherpa-onnx-alsa` to do real-time speech recognition with your
   microphone if ``sherpa-onnx-microphone`` does not work for you.


.. _sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20:

csukuangfj/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20 (Bilingual, Chinese + English)
----------------------------------------------------------------------------------------------------

This model is converted from

`<https://huggingface.co/csukuangfj/k2fsa-zipformer-chinese-english-mixed>`_

which supports both Chinese and English. The model is contributed by the community
and is trained on tens of thousands of some internal dataset.

In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

    cd /path/to/sherpa-onnx

    wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2

    # For Chinese users, you can use the following mirror
    # wget https://hub.nuaa.cf/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2

    tar xvf sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
    rm sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2

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
    --tokens=./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt \
    --encoder=./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.onnx \
    --decoder=./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx \
    --joiner=./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.onnx

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

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-fr-2023-04-14.tar.bz2

  # For Chinese users, you can use the following mirror
  # wget https://hub.nuaa.cf/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-fr-2023-04-14.tar.bz2

  tar xvf sherpa-onnx-streaming-zipformer-fr-2023-04-14.tar.bz2
  rm sherpa-onnx-streaming-zipformer-fr-2023-04-14.tar.bz2

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
    --decoder=./sherpa-onnx-streaming-zipformer-fr-2023-04-14/decoder-epoch-29-avg-9-with-averaged-model.onnx \
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
    --tokens=./sherpa-onnx-streaming-zipformer-fr-2023-04-14/tokens.txt \
    --encoder=./sherpa-onnx-streaming-zipformer-fr-2023-04-14/encoder-epoch-29-avg-9-with-averaged-model.onnx \
    --decoder=./sherpa-onnx-streaming-zipformer-fr-2023-04-14/decoder-epoch-29-avg-9-with-averaged-model.onnx \
    --joiner=./sherpa-onnx-streaming-zipformer-fr-2023-04-14/joiner-epoch-29-avg-9-with-averaged-model.onnx

.. hint::

   If your system is Linux (including embedded Linux), you can also use
   :ref:`sherpa-onnx-alsa` to do real-time speech recognition with your
   microphone if ``sherpa-onnx-microphone`` does not work for you.

.. _sherpa_onnx_streaming_zipformer_small_bilingual_zh_en_2023_02_16:

sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16 (Bilingual, Chinese + English)
-----------------------------------------------------------------------------------------------

.. hint::

   It is a small model.

This model is converted from

`<https://huggingface.co/csukuangfj/k2fsa-zipformer-bilingual-zh-en-t>`_

which supports both Chinese and English. The model is contributed by the community
and is trained on tens of thousands of some internal dataset.

In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16.tar.bz2

  # For Chinese users, you can use the following mirror
  # wget https://hub.nuaa.cf/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16.tar.bz2

  tar xf sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16.tar.bz2
  rm sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16.tar.bz2

Please check that the file sizes of the pre-trained models are correct. See
the file sizes of ``*.onnx`` files below.

.. code-block:: bash

  sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16 fangjun$ ls -lh *.onnx
  total 158M
  drwxr-xr-x 2 1001 127 4.0K Mar 20 13:11 64
  drwxr-xr-x 2 1001 127 4.0K Mar 20 13:11 96
  -rw-r--r-- 1 1001 127 240K Mar 20 13:11 bpe.model
  -rw-r--r-- 1 1001 127 3.4M Mar 20 13:11 decoder-epoch-99-avg-1.int8.onnx
  -rw-r--r-- 1 1001 127  14M Mar 20 13:11 decoder-epoch-99-avg-1.onnx
  -rw-r--r-- 1 1001 127  41M Mar 20 13:11 encoder-epoch-99-avg-1.int8.onnx
  -rw-r--r-- 1 1001 127  85M Mar 20 13:11 encoder-epoch-99-avg-1.onnx
  -rw-r--r-- 1 1001 127 3.1M Mar 20 13:11 joiner-epoch-99-avg-1.int8.onnx
  -rw-r--r-- 1 1001 127  13M Mar 20 13:11 joiner-epoch-99-avg-1.onnx
  drwxr-xr-x 2 1001 127 4.0K Mar 20 13:11 test_wavs
  -rw-r--r-- 1 1001 127  55K Mar 20 13:11 tokens.txt

.. hint::

   There are two sub-folders in the model directory: ``64`` and ``96``.
   The number represents chunk size. The larger the number, the lower the RTF.
   The default chunk size is 32.

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
    --tokens=./sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16/tokens.txt \
    --encoder=./sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16/encoder-epoch-99-avg-1.onnx \
    --decoder=./sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16/decoder-epoch-99-avg-1.onnx \
    --joiner=./sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16/joiner-epoch-99-avg-1.onnx \
    ./sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16/test_wavs/0.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx.exe`` for Windows.

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

You should see the following output:

.. literalinclude:: ./code-zipformer/sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16.txt

int8
^^^^

The following code shows how to use ``int8`` models to decode a wave file:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx \
    --tokens=./sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16/tokens.txt \
    --encoder=./sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16/encoder-epoch-99-avg-1.int8.onnx \
    --decoder=./sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16/decoder-epoch-99-avg-1.onnx \
    --joiner=./sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16/joiner-epoch-99-avg-1.int8.onnx \
    ./sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16/test_wavs/0.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx.exe`` for Windows.

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

You should see the following output:

.. literalinclude:: ./code-zipformer/sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16.int8.txt

Real-time speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-microphone \
    --tokens=./sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16/tokens.txt \
    --encoder=./sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16/encoder-epoch-99-avg-1.int8.onnx \
    --decoder=./sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16/decoder-epoch-99-avg-1.onnx \
    --joiner=./sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16/joiner-epoch-99-avg-1.int8.onnx

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

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23.tar.bz2

  # For Chinese users, you can use the following mirror
  # wget https://hub.nuaa.cf/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23.tar.bz2

  tar xvf sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23.tar.bz2
  rm sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23.tar.bz2

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

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2

  # For Chinese users, you can use the following mirror
  # wget https://hub.nuaa.cf/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2

  tar xvf sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2
  rm sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2

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
    --joiner=./sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/joiner-epoch-99-avg-1.onnx

.. hint::

   If your system is Linux (including embedded Linux), you can also use
   :ref:`sherpa-onnx-alsa` to do real-time speech recognition with your
   microphone if ``sherpa-onnx-microphone`` does not work for you.
