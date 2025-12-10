.. _sherpa_onnx_zipformer_ctc_models:

Zipformer-CTC-based Models
==========================

.. hint::

   Please refer to :ref:`install_sherpa_onnx` to install `sherpa-onnx`_
   before you read this section.

.. _sherpa-onnx-streaming-zipformer-ctc-zh-xlarge-int8-2025-06-30:

sherpa-onnx-streaming-zipformer-ctc-zh-xlarge-int8-2025-06-30 (Chinese)
------------------------------------------------------------------------

PyTorch checkpoint for this model can be found at
`<https://huggingface.co/yuekai/icefall-asr-multi-zh-hans-zipformer-xl>`_.

The training code can be found at
`<https://github.com/k2-fsa/icefall/blob/master/egs/multi_zh-hans/ASR/RESULTS.md#multi-chinese-datasets-char-based-training-results-streaming-on-zipformer-xl-model>`_

.. note::

   We only show the ``int8`` quantized model here. You can also use

    - ``fp16``: `sherpa-onnx-streaming-zipformer-ctc-zh-xlarge-fp16-2025-06-30.tar.bz2 <https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-ctc-zh-xlarge-fp16-2025-06-30.tar.bz2>`_

Download the model
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-ctc-zh-xlarge-int8-2025-06-30.tar.bz2
  tar xvf sherpa-onnx-streaming-zipformer-ctc-zh-xlarge-int8-2025-06-30.tar.bz2
  rm sherpa-onnx-streaming-zipformer-ctc-zh-xlarge-int8-2025-06-30.tar.bz2

  ls -lh sherpa-onnx-streaming-zipformer-ctc-zh-xlarge-int8-2025-06-30

The output is given below:

.. code-block:: bash

  -rw-r--r--  1 fangjun  staff   311B Jun 30 18:00 README.md
  -rw-r--r--  1 fangjun  staff   258K Jun 30 18:00 bpe.model
  -rw-r--r--  1 fangjun  staff   728M Jun 30 18:00 model.int8.onnx
  drwxr-xr-x  5 fangjun  staff   160B Jun 30 17:59 test_wavs
  -rw-r--r--  1 fangjun  staff    18K Jun 30 18:00 tokens.txt

Decode a single wave file
~~~~~~~~~~~~~~~~~~~~~~~~~

.. hint::

   It supports decoding only wave files of a single channel with 16-bit
   encoded samples, while the sampling rate does not need to be 16 kHz.

The following code shows how to use ``int8`` models to decode a wave file:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx \
    --zipformer2-ctc-model=./sherpa-onnx-streaming-zipformer-ctc-zh-xlarge-int8-2025-06-30/model.int8.onnx \
    --tokens=./sherpa-onnx-streaming-zipformer-ctc-zh-xlarge-int8-2025-06-30/tokens.txt \
    ./sherpa-onnx-streaming-zipformer-ctc-zh-xlarge-int8-2025-06-30/test_wavs/0.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx.exe`` for Windows.

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

You should see the following output:

.. literalinclude:: ./code-zipformer/sherpa-onnx-streaming-zipformer-ctc-zh-xlarge-int8-2025-06-30.txt

Real-time speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-microphone \
    --zipformer2-ctc-model=./sherpa-onnx-streaming-zipformer-ctc-zh-xlarge-int8-2025-06-30/model.int8.onnx \
    --tokens=./sherpa-onnx-streaming-zipformer-ctc-zh-xlarge-int8-2025-06-30/tokens.txt

.. hint::

   If your system is Linux (including embedded Linux), you can also use
   :ref:`sherpa-onnx-alsa` to do real-time speech recognition with your
   microphone if ``sherpa-onnx-microphone`` does not work for you.


.. _sherpa-onnx-streaming-zipformer-ctc-zh-int8-2025-06-30:

sherpa-onnx-streaming-zipformer-ctc-zh-int8-2025-06-30 (Chinese)
-----------------------------------------------------------------

PyTorch checkpoint for this model can be found at
`<https://huggingface.co/yuekai/icefall-asr-multi-zh-hans-zipformer-large>`_.

The training code can be found at
`<https://github.com/k2-fsa/icefall/blob/master/egs/multi_zh-hans/ASR/RESULTS.md#multi-chinese-datasets-char-based-training-results-streaming-on-zipformer-large-model>`_

.. note::

   We only show the ``int8`` quantized model here. You can also use

    - ``fp32``: `sherpa-onnx-streaming-zipformer-ctc-zh-2025-06-30.tar.bz2 <https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-ctc-zh-2025-06-30.tar.bz2>`_
    - ``fp16``: `sherpa-onnx-streaming-zipformer-ctc-zh-fp16-2025-06-30.tar.bz2 <https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-ctc-zh-fp16-2025-06-30.tar.bz2>`_

Download the model
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-ctc-zh-int8-2025-06-30.tar.bz2
  tar xvf sherpa-onnx-streaming-zipformer-ctc-zh-int8-2025-06-30.tar.bz2
  rm sherpa-onnx-streaming-zipformer-ctc-zh-int8-2025-06-30.tar.bz2

  ls -lh sherpa-onnx-streaming-zipformer-ctc-zh-int8-2025-06-30

The output is given below:

.. code-block:: bash

  -rw-r--r--  1 fangjun  staff   317B Jun 30 14:58 README.md
  -rw-r--r--  1 fangjun  staff   155M Jun 30 14:58 model.int8.onnx
  drwxr-xr-x  5 fangjun  staff   160B Jun 30 14:58 test_wavs
  -rw-r--r--  1 fangjun  staff    20K Jun 30 14:58 tokens.txt

Decode a single wave file
~~~~~~~~~~~~~~~~~~~~~~~~~

.. hint::

   It supports decoding only wave files of a single channel with 16-bit
   encoded samples, while the sampling rate does not need to be 16 kHz.

The following code shows how to use ``int8`` models to decode a wave file:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx \
    --zipformer2-ctc-model=./sherpa-onnx-streaming-zipformer-ctc-zh-int8-2025-06-30/model.int8.onnx \
    --tokens=./sherpa-onnx-streaming-zipformer-ctc-zh-int8-2025-06-30/tokens.txt \
    ./sherpa-onnx-streaming-zipformer-ctc-zh-int8-2025-06-30/test_wavs/0.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx.exe`` for Windows.

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

You should see the following output:

.. literalinclude:: ./code-zipformer/sherpa-onnx-streaming-zipformer-ctc-zh-int8-2025-06-30.txt

Real-time speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-microphone \
    --zipformer2-ctc-model=./sherpa-onnx-streaming-zipformer-ctc-zh-int8-2025-06-30/model.int8.onnx \
    --tokens=./sherpa-onnx-streaming-zipformer-ctc-zh-int8-2025-06-30/tokens.txt

.. hint::

   If your system is Linux (including embedded Linux), you can also use
   :ref:`sherpa-onnx-alsa` to do real-time speech recognition with your
   microphone if ``sherpa-onnx-microphone`` does not work for you.


.. _sherpa-onnx-streaming-zipformer-small-ctc-zh-int8-2025-04-01:

sherpa-onnx-streaming-zipformer-small-ctc-zh-int8-2025-04-01 (Chinese)
----------------------------------------------------------------------

PyTorch checkpoint for this model can be found at
`<https://huggingface.co/csukuangfj/icefall-streaming-zipformer-small-ctc-zh-2025-04-01>`_.

It supports only Chinese and uses byte-BPE with vocab size ``1000``.

In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-small-ctc-zh-int8-2025-04-01.tar.bz2

  tar xvf sherpa-onnx-streaming-zipformer-small-ctc-zh-int8-2025-04-01.tar.bz2
  rm sherpa-onnx-streaming-zipformer-small-ctc-zh-int8-2025-04-01.tar.bz2
  ls -lh sherpa-onnx-streaming-zipformer-small-ctc-zh-int8-2025-04-01

The output is given below:

.. code-block:: bash

  ls -lh sherpa-onnx-streaming-zipformer-small-ctc-zh-int8-2025-04-01/
  total 51992
  -rw-r--r--  1 fangjun  staff   249K Apr  1 19:39 bbpe.model
  -rw-r--r--  1 fangjun  staff    25M Apr  1 19:38 model.int8.onnx
  drwxr-xr-x  5 fangjun  staff   160B Jan  3  2024 test_wavs
  -rw-r--r--  1 fangjun  staff    13K Apr  1 19:39 tokens.txt

Decode a single wave file
~~~~~~~~~~~~~~~~~~~~~~~~~

.. hint::

   It supports decoding only wave files of a single channel with 16-bit
   encoded samples, while the sampling rate does not need to be 16 kHz.

The following code shows how to use ``int8`` models to decode a wave file:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx \
    --zipformer2-ctc-model=./sherpa-onnx-streaming-zipformer-small-ctc-zh-int8-2025-04-01/model.int8.onnx \
    --tokens=./sherpa-onnx-streaming-zipformer-small-ctc-zh-int8-2025-04-01/tokens.txt \
    ./sherpa-onnx-streaming-zipformer-small-ctc-zh-int8-2025-04-01/test_wavs/0.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx.exe`` for Windows.

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

You should see the following output:

.. literalinclude:: ./code-zipformer/sherpa-onnx-streaming-zipformer-small-ctc-zh-int8-2025-04-01.txt

Real-time speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-microphone \
    --zipformer2-ctc-model=./sherpa-onnx-streaming-zipformer-small-ctc-zh-int8-2025-04-01/model.int8.onnx \
    --tokens=./sherpa-onnx-streaming-zipformer-small-ctc-zh-int8-2025-04-01/tokens.txt

.. hint::

   If your system is Linux (including embedded Linux), you can also use
   :ref:`sherpa-onnx-alsa` to do real-time speech recognition with your
   microphone if ``sherpa-onnx-microphone`` does not work for you.

.. _sherpa-onnx-streaming-zipformer-small-ctc-zh-2025-04-01:

sherpa-onnx-streaming-zipformer-small-ctc-zh-2025-04-01 (Chinese)
----------------------------------------------------------------------

PyTorch checkpoint for this model can be found at
`<https://huggingface.co/csukuangfj/icefall-streaming-zipformer-small-ctc-zh-2025-04-01>`_.

It supports only Chinese and uses byte-BPE with vocab size ``1000``.

In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-small-ctc-zh-2025-04-01.tar.bz2

  tar xvf sherpa-onnx-streaming-zipformer-small-ctc-zh-2025-04-01.tar.bz2
  rm sherpa-onnx-streaming-zipformer-small-ctc-zh-2025-04-01.tar.bz2
  ls -lh sherpa-onnx-streaming-zipformer-small-ctc-zh-2025-04-01

The output is given below:

.. code-block:: bash

  ls -lh sherpa-onnx-streaming-zipformer-small-ctc-zh-2025-04-01/
  total 179248
  -rw-r--r--  1 fangjun  staff   249K Apr  1 19:39 bbpe.model
  -rw-r--r--  1 fangjun  staff    87M Apr  1 19:39 model.onnx
  drwxr-xr-x  5 fangjun  staff   160B Jan  3  2024 test_wavs
  -rw-r--r--  1 fangjun  staff    13K Apr  1 19:39 tokens.txt

Decode a single wave file
~~~~~~~~~~~~~~~~~~~~~~~~~

.. hint::

   It supports decoding only wave files of a single channel with 16-bit
   encoded samples, while the sampling rate does not need to be 16 kHz.

The following code shows how to use ``fp32`` models to decode a wave file:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx \
    --zipformer2-ctc-model=./sherpa-onnx-streaming-zipformer-small-ctc-zh-2025-04-01/model.onnx \
    --tokens=./sherpa-onnx-streaming-zipformer-small-ctc-zh-2025-04-01/tokens.txt \
    ./sherpa-onnx-streaming-zipformer-small-ctc-zh-2025-04-01/test_wavs/0.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx.exe`` for Windows.

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

You should see the following output:

.. literalinclude:: ./code-zipformer/sherpa-onnx-streaming-zipformer-small-ctc-zh-2025-04-01.txt

Real-time speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-microphone \
    --zipformer2-ctc-model=./sherpa-onnx-streaming-zipformer-small-ctc-zh-2025-04-01/model.onnx \
    --tokens=./sherpa-onnx-streaming-zipformer-small-ctc-zh-2025-04-01/tokens.txt

.. hint::

   If your system is Linux (including embedded Linux), you can also use
   :ref:`sherpa-onnx-alsa` to do real-time speech recognition with your
   microphone if ``sherpa-onnx-microphone`` does not work for you.


sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13 (Chinese)
----------------------------------------------------------------------

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

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13.tar.bz2

  tar xvf sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13.tar.bz2
  rm sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13.tar.bz2
  ls -lh sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13

The output is given below:

.. code-block::

  $ ls -lh sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13
  total 654136
  -rw-r--r--@ 1 fangjun  staff    28B Dec 13 16:19 README.md
  -rw-r--r--@ 1 fangjun  staff   258K Dec 13 16:19 bpe.model
  -rw-r--r--@ 1 fangjun  staff    68M Dec 13 16:19 ctc-epoch-20-avg-1-chunk-16-left-128.int8.onnx
  -rw-r--r--@ 1 fangjun  staff   252M Dec 13 16:19 ctc-epoch-20-avg-1-chunk-16-left-128.onnx
  drwxr-xr-x@ 8 fangjun  staff   256B Dec 13 16:19 test_wavs
  -rw-r--r--@ 1 fangjun  staff    18K Dec 13 16:19 tokens.txt

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
    --zipformer2-ctc-model=./sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13/ctc-epoch-20-avg-1-chunk-16-left-128.onnx \
    --tokens=./sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13/tokens.txt \
    ./sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13/test_wavs/DEV_T0000000000.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx.exe`` for Windows.

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

You should see the following output:

.. literalinclude:: ./code-zipformer/sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13.txt

int8
^^^^

The following code shows how to use ``int8`` models to decode a wave file:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx \
    --zipformer2-ctc-model=./sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13/ctc-epoch-20-avg-1-chunk-16-left-128.int8.onnx \
    --tokens=./sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13/tokens.txt \
    ./sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13/test_wavs/DEV_T0000000000.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx.exe`` for Windows.

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

You should see the following output:

.. literalinclude:: ./code-zipformer/sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13.int8.txt

Real-time speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-microphone \
    --zipformer2-ctc-model=./sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13/ctc-epoch-20-avg-1-chunk-16-left-128.onnx \
    --tokens=./sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13/tokens.txt

.. hint::

   If your system is Linux (including embedded Linux), you can also use
   :ref:`sherpa-onnx-alsa` to do real-time speech recognition with your
   microphone if ``sherpa-onnx-microphone`` does not work for you.
