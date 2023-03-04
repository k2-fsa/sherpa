LSTM-transducer-based Models
============================

.. hint::

   Please refer to :ref:`install_sherpa_onnx` to install `sherpa-onnx`_
   before you read this section.

csukuangfj/sherpa-onnx-lstm-en-2023-02-17 (English)
---------------------------------------------------

This is a model trained using the `GigaSpeech`_ and the `LibriSpeech`_ dataset.

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

  # before running `git lfs pull`

  sherpa-onnx-lstm-en-2023-02-17 fangjun$ ls -lh *.onnx
  -rw-r--r--  1 fangjun  staff   132B Feb 17 15:55 decoder-epoch-99-avg-1.onnx
  -rw-r--r--  1 fangjun  staff   134B Feb 17 15:55 encoder-epoch-99-avg-1.onnx
  -rw-r--r--  1 fangjun  staff   132B Feb 17 15:55 joiner-epoch-99-avg-1.onnx

  # after running `git lfs pull`

  sherpa-onnx-lstm-en-2023-02-17 fangjun$ ls -lh *.onnx
  -rw-r--r--  1 fangjun  staff   2.0M Feb 17 15:55 decoder-epoch-99-avg-1.onnx
  -rw-r--r--  1 fangjun  staff   318M Feb 17 15:57 encoder-epoch-99-avg-1.onnx
  -rw-r--r--  1 fangjun  staff   1.0M Feb 17 15:55 joiner-epoch-99-avg-1.onnx

Decode a single wave file
~~~~~~~~~~~~~~~~~~~~~~~~~

.. hint::

   It supports decoding only wave files with a single channel and the sampling rate
   should be 16 kHz.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx \
    ./sherpa-onnx-lstm-en-2023-02-17/tokens.txt \
    ./sherpa-onnx-lstm-en-2023-02-17/encoder-epoch-99-avg-1.onnx \
    ./sherpa-onnx-lstm-en-2023-02-17/decoder-epoch-99-avg-1.onnx \
    ./sherpa-onnx-lstm-en-2023-02-17/joiner-epoch-99-avg-1.onnx \
    ./sherpa-onnx-lstm-en-2023-02-17/test_wavs/1089-134686-0001.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx.exe`` for Windows.

You should see the following output:

.. literalinclude:: ./code-lstm/sherpa-onnx-lstm-en-2023-02-17.txt

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

  # before running `git lfs pull`

  sherpa-onnx-lstm-zh-2023-02-20 fangjun$ ls -lh *.onnx
  -rw-r--r--  1 fangjun  staff   133B Feb 20 15:01 decoder-epoch-11-avg-1.onnx
  -rw-r--r--  1 fangjun  staff   134B Feb 20 15:01 encoder-epoch-11-avg-1.onnx
  -rw-r--r--  1 fangjun  staff   133B Feb 20 15:01 joiner-epoch-11-avg-1.onnx

  # after running `git lfs pull`

  sherpa-onnx-lstm-zh-2023-02-20 fangjun$ ls -lh *.onnx
  -rw-r--r--  1 fangjun  staff    12M Feb 20 15:02 decoder-epoch-11-avg-1.onnx
  -rw-r--r--  1 fangjun  staff   318M Feb 20 15:04 encoder-epoch-11-avg-1.onnx
  -rw-r--r--  1 fangjun  staff    11M Feb 20 15:02 joiner-epoch-11-avg-1.onnx

Decode a single wave file
~~~~~~~~~~~~~~~~~~~~~~~~~

.. hint::

   It supports decoding only wave files with a single channel and the sampling rate
   should be 16 kHz.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx \
    ./sherpa-onnx-lstm-zh-2023-02-20/tokens.txt \
    ./sherpa-onnx-lstm-zh-2023-02-20/encoder-epoch-11-avg-1.onnx \
    ./sherpa-onnx-lstm-zh-2023-02-20/decoder-epoch-11-avg-1.onnx \
    ./sherpa-onnx-lstm-zh-2023-02-20/joiner-epoch-11-avg-1.onnx \
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
