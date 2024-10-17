.. _sherpa_onnx_zipformer_ctc_models:

Zipformer-CTC-based Models
==========================

.. hint::

   Please refer to :ref:`install_sherpa_onnx` to install `sherpa-onnx`_
   before you read this section.

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

  # For Chinese users, please use the following mirror
  # wget https://hub.nuaa.cf/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13.tar.bz2

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
