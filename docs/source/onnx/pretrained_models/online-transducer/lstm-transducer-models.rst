LSTM-transducer-based Models
============================

.. hint::

   Please refer to :ref:`install_sherpa_onnx` to install `sherpa-onnx`_
   before you read this section.

csukuangfj/sherpa-onnx-lstm-en-2023-02-17 (English)
---------------------------------------------------

This model trained using the `GigaSpeech`_ and the `LibriSpeech`_ dataset.

Please see `<https://github.com/k2-fsa/icefall/pull/558>`_ for how the model
is trained.

You can find the training code at

`<https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/lstm_transducer_stateless2>`_

In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/sherpa-onnx-lstm-en-2023-02-17
  cd sherpa-onnx-lstm-en-2023-02-17
  git lfs pull --include "*.onnx"

Please check that the file sizes of the pre-trained models are correct. See
the file sizes of ``*.onnx`` files below.

.. code-block:: bash

  sherpa-onnx-lstm-en-2023-02-17$ ls -lh *.onnx
  -rw-r--r-- 1 kuangfangjun root  1.3M Mar 31 22:41 decoder-epoch-99-avg-1.int8.onnx
  -rw-r--r-- 1 kuangfangjun root  2.0M Mar 31 22:41 decoder-epoch-99-avg-1.onnx
  -rw-r--r-- 1 kuangfangjun root   80M Mar 31 22:41 encoder-epoch-99-avg-1.int8.onnx
  -rw-r--r-- 1 kuangfangjun root  319M Mar 31 22:41 encoder-epoch-99-avg-1.onnx
  -rw-r--r-- 1 kuangfangjun root  254K Mar 31 22:41 joiner-epoch-99-avg-1.int8.onnx
  -rw-r--r-- 1 kuangfangjun root 1003K Mar 31 22:41 joiner-epoch-99-avg-1.onnx

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
    --tokens=./sherpa-onnx-lstm-en-2023-02-17/tokens.txt \
    --encoder=./sherpa-onnx-lstm-en-2023-02-17/encoder-epoch-99-avg-1.onnx \
    --decoder=./sherpa-onnx-lstm-en-2023-02-17/decoder-epoch-99-avg-1.onnx \
    --joiner=./sherpa-onnx-lstm-en-2023-02-17/joiner-epoch-99-avg-1.onnx \
    ./sherpa-onnx-lstm-en-2023-02-17/test_wavs/0.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx.exe`` for Windows.

You should see the following output:

.. literalinclude:: ./code-lstm/sherpa-onnx-lstm-en-2023-02-17.txt

int8
^^^^

The following code shows how to use ``int8`` models to decode a wave file:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx \
    --tokens=./sherpa-onnx-lstm-en-2023-02-17/tokens.txt \
    --encoder=./sherpa-onnx-lstm-en-2023-02-17/encoder-epoch-99-avg-1.int8.onnx \
    --decoder=./sherpa-onnx-lstm-en-2023-02-17/decoder-epoch-99-avg-1.int8.onnx \
    --joiner=./sherpa-onnx-lstm-en-2023-02-17/joiner-epoch-99-avg-1.int8.onnx \
    ./sherpa-onnx-lstm-en-2023-02-17/test_wavs/0.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx.exe`` for Windows.

You should see the following output:

.. literalinclude:: ./code-lstm/sherpa-onnx-lstm-en-2023-02-17-int8.txt

Real-time speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-microphone \
    ./sherpa-onnx-lstm-en-2023-02-17/tokens.txt \
    ./sherpa-onnx-lstm-en-2023-02-17/encoder-epoch-99-avg-1.onnx \
    ./sherpa-onnx-lstm-en-2023-02-17/decoder-epoch-99-avg-1.onnx \
    ./sherpa-onnx-lstm-en-2023-02-17/joiner-epoch-99-avg-1.onnx

.. hint::

   If your system is Linux (including embedded Linux), you can also use
   :ref:`sherpa-onnx-alsa` to do real-time speech recognition with your
   microphone if ``sherpa-onnx-microphone`` does not work for you.


csukuangfj/sherpa-onnx-lstm-zh-2023-02-20 (Chinese)
---------------------------------------------------

This is a model trained using the `WenetSpeech`_ dataset.

Please see `<https://github.com/k2-fsa/icefall/pull/595>`_ for how the model
is trained.

In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/sherpa-onnx-lstm-zh-2023-02-20
  cd sherpa-onnx-lstm-zh-2023-02-20
  git lfs pull --include "*.onnx"

Please check that the file sizes of the pre-trained models are correct. See
the file sizes of ``*.onnx`` files below.

.. code-block:: bash

  sherpa-onnx-lstm-zh-2023-02-20$ ls -lh *.onnx
  -rw-r--r-- 1 kuangfangjun root  12M Mar 31 20:55 decoder-epoch-11-avg-1.int8.onnx
  -rw-r--r-- 1 kuangfangjun root  12M Mar 31 20:55 decoder-epoch-11-avg-1.onnx
  -rw-r--r-- 1 kuangfangjun root  80M Mar 31 20:55 encoder-epoch-11-avg-1.int8.onnx
  -rw-r--r-- 1 kuangfangjun root 319M Mar 31 20:55 encoder-epoch-11-avg-1.onnx
  -rw-r--r-- 1 kuangfangjun root 2.8M Mar 31 20:55 joiner-epoch-11-avg-1.int8.onnx
  -rw-r--r-- 1 kuangfangjun root  11M Mar 31 20:55 joiner-epoch-11-avg-1.onnx

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
    --tokens=./sherpa-onnx-lstm-zh-2023-02-20/tokens.txt \
    --encoder=./sherpa-onnx-lstm-zh-2023-02-20/encoder-epoch-11-avg-1.onnx \
    --decoder=./sherpa-onnx-lstm-zh-2023-02-20/decoder-epoch-11-avg-1.onnx \
    --joiner=./sherpa-onnx-lstm-zh-2023-02-20/joiner-epoch-11-avg-1.onnx \
    ./sherpa-onnx-lstm-zh-2023-02-20/test_wavs/0.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx.exe`` for Windows.

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

You should see the following output:

.. literalinclude:: ./code-lstm/sherpa-onnx-lstm-zh-2023-02-20.txt

int8
^^^^

The following code shows how to use ``int8`` models to decode a wave file:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx \
    --tokens=./sherpa-onnx-lstm-zh-2023-02-20/tokens.txt \
    --encoder=./sherpa-onnx-lstm-zh-2023-02-20/encoder-epoch-11-avg-1.int8.onnx \
    --decoder=./sherpa-onnx-lstm-zh-2023-02-20/decoder-epoch-11-avg-1.int8.onnx \
    --joiner=./sherpa-onnx-lstm-zh-2023-02-20/joiner-epoch-11-avg-1.int8.onnx \
    ./sherpa-onnx-lstm-zh-2023-02-20/test_wavs/0.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx.exe`` for Windows.

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

You should see the following output:

.. literalinclude:: ./code-lstm/sherpa-onnx-lstm-zh-2023-02-20-int8.txt

Real-time speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-microphone \
    ./sherpa-onnx-lstm-zh-2023-02-20/tokens.txt \
    ./sherpa-onnx-lstm-zh-2023-02-20/encoder-epoch-11-avg-1.onnx \
    ./sherpa-onnx-lstm-zh-2023-02-20/decoder-epoch-11-avg-1.onnx \
    ./sherpa-onnx-lstm-zh-2023-02-20/joiner-epoch-11-avg-1.onnx

.. hint::

   If your system is Linux (including embedded Linux), you can also use
   :ref:`sherpa-onnx-alsa` to do real-time speech recognition with your
   microphone if ``sherpa-onnx-microphone`` does not work for you.
