LSTM-transducer-based Models
=============================

.. hint::

   Please refer to :ref:`install_sherpa_ncnn` to install `sherpa-ncnn`_
   before you read this section.

.. _marcoyang_sherpa_ncnn_lstm_transducer_small_2023_02_13_bilingual:

marcoyang/sherpa-ncnn-lstm-transducer-small-2023-02-13 (Bilingual, Chinese + English)
--------------------------------------------------------------------------------------

This model is a small version of `lstm-transducer <https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/lstm_transducer_stateless3>`_
trained in `icefall`_.

It only has ``13.3 million parameters`` and can be deployed on ``embedded devices``
for real-time speech recognition. You can find the models in ``fp16`` format
at `<https://huggingface.co/marcoyang/sherpa-ncnn-lstm-transducer-small-2023-02-13>`_.

The model is trained on a bi-lingual dataset ``tal_csasr`` (Chinese + English), so it can be used
for both Chinese and English. 

In the following, we show you how to download it and
deploy it with `sherpa-ncnn`_.

Please use the following commands to download it.

.. code-block:: bash

   cd /path/to/sherpa-ncnn

   GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/marcoyang/sherpa-ncnn-lstm-transducer-small-2023-02-13

   cd sherpa-ncnn-lstm-transducer-small-2023-02-13
   git lfs pull --include "*.bin"

  
.. note::

  Please refer to :ref:`sherpa-ncnn-embedded-linux-arm-install` for how to
  compile `sherpa-ncnn`_ for a 32-bit ARM platform. 

Decode a single wave file with ./build/bin/sherpa-ncnn
::::::::::::::::::::::::::::::::::::::::::::::::::::::

.. hint::

   It supports decoding only wave files with a single channel and the sampling rate
   should be 16 kHz.

.. code-block:: bash

  cd /path/to/sherpa-ncnn

  ./build/bin/sherpa-ncnn \
    ./sherpa-ncnn-lstm-transducer-small-2023-02-13/tokens.txt \
    ./sherpa-ncnn-lstm-transducer-small-2023-02-13/encoder_jit_trace-pnnx.ncnn.param \
    ./sherpa-ncnn-lstm-transducer-small-2023-02-13/encoder_jit_trace-pnnx.ncnn.bin \
    ./sherpa-ncnn-lstm-transducer-small-2023-02-13/decoder_jit_trace-pnnx.ncnn.param \
    ./sherpa-ncnn-lstm-transducer-small-2023-02-13/decoder_jit_trace-pnnx.ncnn.bin \
    ./sherpa-ncnn-lstm-transducer-small-2023-02-13/joiner_jit_trace-pnnx.ncnn.param \
    ./sherpa-ncnn-lstm-transducer-small-2023-02-13/joiner_jit_trace-pnnx.ncnn.bin \
    ./sherpa-ncnn-lstm-transducer-small-2023-02-13/test_wavs/0.wav

.. note::

   The default option uses 4 threads and ``greedy_search`` for decoding.

.. note::

   Please use ``./build/bin/Release/sherpa-ncnn.exe`` for Windows.

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

csukuangfj/sherpa-ncnn-2022-09-05 (English)
-------------------------------------------

This is a model trained using the `GigaSpeech`_ and the `LibriSpeech`_ dataset.

Please see `<https://github.com/k2-fsa/icefall/pull/558>`_ for how the model
is trained.

You can find the training code at

`<https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/lstm_transducer_stateless2>`_

In the following, we describe how to download it and use it with `sherpa-ncnn`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

  cd /path/to/sherpa-ncnn

  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/sherpa-ncnn-2022-09-05
  cd sherpa-ncnn-2022-09-05
  git lfs pull --include "*.bin"

Please check that the file sizes of the pre-trained models are correct. See
the file sizes of ``*.bin`` files below.

.. code-block:: bash

  # before running `git lfs pull`

  sherpa-ncnn-2022-09-05 fangjun$ ls -lh *.bin
  -rw-r--r--  1 fangjun  staff   131B Feb 16 11:12 decoder_jit_trace-pnnx.ncnn.bin
  -rw-r--r--  1 fangjun  staff   134B Feb 16 11:12 encoder_jit_trace-pnnx.ncnn.bin
  -rw-r--r--  1 fangjun  staff   132B Feb 16 11:12 joiner_jit_trace-pnnx.ncnn.bin

  sherpa-ncnn-2022-09-05 fangjun$ git lfs pull --include "*.bin"

  # after running `git lfs pull`

  sherpa-ncnn-2022-09-05 fangjun$ ls -lh *.bin
  -rw-r--r--  1 fangjun  staff   502K Feb 16 11:12 decoder_jit_trace-pnnx.ncnn.bin
  -rw-r--r--  1 fangjun  staff   159M Feb 16 11:13 encoder_jit_trace-pnnx.ncnn.bin
  -rw-r--r--  1 fangjun  staff   1.5M Feb 16 11:12 joiner_jit_trace-pnnx.ncnn.bin


Decode a single wave file
~~~~~~~~~~~~~~~~~~~~~~~~~

.. hint::

   It supports decoding only wave files with a single channel and the sampling rate
   should be 16 kHz.

.. code-block:: bash

  cd /path/to/sherpa-ncnn

  for method in greedy_search modified_beam_search; do
    ./build/bin/sherpa-ncnn \
      ./sherpa-ncnn-2022-09-05/tokens.txt \
      ./sherpa-ncnn-2022-09-05/encoder_jit_trace-pnnx.ncnn.param \
      ./sherpa-ncnn-2022-09-05/encoder_jit_trace-pnnx.ncnn.bin \
      ./sherpa-ncnn-2022-09-05/decoder_jit_trace-pnnx.ncnn.param \
      ./sherpa-ncnn-2022-09-05/decoder_jit_trace-pnnx.ncnn.bin \
      ./sherpa-ncnn-2022-09-05/joiner_jit_trace-pnnx.ncnn.param \
      ./sherpa-ncnn-2022-09-05/joiner_jit_trace-pnnx.ncnn.bin \
      ./sherpa-ncnn-2022-09-05/test_wavs/1089-134686-0001.wav \
      2 \
      $method
  done

You should see the following output:

.. literalinclude:: ./code-lstm/2022-09-05.txt

.. note::

   Please use ``./build/bin/Release/sherpa-ncnn.exe`` for Windows.


Real-time speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-ncnn

  ./build/bin/sherpa-ncnn-microphone \
    ./sherpa-ncnn-2022-09-05/tokens.txt \
    ./sherpa-ncnn-2022-09-05/encoder_jit_trace-pnnx.ncnn.param \
    ./sherpa-ncnn-2022-09-05/encoder_jit_trace-pnnx.ncnn.bin \
    ./sherpa-ncnn-2022-09-05/decoder_jit_trace-pnnx.ncnn.param \
    ./sherpa-ncnn-2022-09-05/decoder_jit_trace-pnnx.ncnn.bin \
    ./sherpa-ncnn-2022-09-05/joiner_jit_trace-pnnx.ncnn.param \
    ./sherpa-ncnn-2022-09-05/joiner_jit_trace-pnnx.ncnn.bin \
    2 \
    greedy_search

.. note::

   Please use ``./build/bin/Release/sherpa-ncnn-microphone.exe`` for Windows.

It will print something like below:

.. code-block::

  Number of threads: 4
  num devices: 4
  Use default device: 2
    Name: MacBook Pro Microphone
    Max input channels: 1
  Started

Speak and it will show you the recognition result in real-time.

You can find a demo below:

..  youtube:: m6ynSxycpX0
   :width: 120%

csukuangfj/sherpa-ncnn-2022-09-30 (Chinese)
-------------------------------------------

This is a model trained using the `WenetSpeech`_ dataset.

Please see `<https://github.com/k2-fsa/icefall/pull/595>`_ for how the model
is trained.

In the following, we describe how to download it and use it with `sherpa-ncnn`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

  cd /path/to/sherpa-ncnn

  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/sherpa-ncnn-2022-09-30
  cd sherpa-ncnn-2022-09-30
  git lfs pull --include "*.bin"

Please check that the file sizes of the pre-trained models are correct. See
the file sizes of ``*.bin`` files below.

.. code-block:: bash

  # before running `git lfs pull`

  sherpa-ncnn-2022-09-30 fangjun$ ls -lh *.bin
  -rw-r--r--  1 fangjun  staff   132B Feb 16 11:30 decoder_jit_trace-pnnx.ncnn.bin
  -rw-r--r--  1 fangjun  staff   134B Feb 16 11:30 encoder_jit_trace-pnnx.ncnn.bin
  -rw-r--r--  1 fangjun  staff   132B Feb 16 11:30 joiner_jit_trace-pnnx.ncnn.bin

  sherpa-ncnn-2022-09-30 fangjun$ git lfs pull --include "*.bin"

  # after running `git lfs pull`

  sherpa-ncnn-2022-09-30 fangjun$ ls -lh *.bin
  -rw-r--r--  1 fangjun  staff   5.4M Feb 16 11:30 decoder_jit_trace-pnnx.ncnn.bin
  -rw-r--r--  1 fangjun  staff   159M Feb 16 11:31 encoder_jit_trace-pnnx.ncnn.bin
  -rw-r--r--  1 fangjun  staff   6.4M Feb 16 11:30 joiner_jit_trace-pnnx.ncnn.bin

Decode a single wave file
~~~~~~~~~~~~~~~~~~~~~~~~~

.. hint::

   It supports decoding only wave files with a single channel and the sampling rate
   should be 16 kHz.

.. code-block:: bash

  cd /path/to/sherpa-ncnn

  for method in greedy_search modified_beam_search; do
    ./build/bin/sherpa-ncnn \
      ./sherpa-ncnn-2022-09-30/tokens.txt \
      ./sherpa-ncnn-2022-09-30/encoder_jit_trace-pnnx.ncnn.param \
      ./sherpa-ncnn-2022-09-30/encoder_jit_trace-pnnx.ncnn.bin \
      ./sherpa-ncnn-2022-09-30/decoder_jit_trace-pnnx.ncnn.param \
      ./sherpa-ncnn-2022-09-30/decoder_jit_trace-pnnx.ncnn.bin \
      ./sherpa-ncnn-2022-09-30/joiner_jit_trace-pnnx.ncnn.param \
      ./sherpa-ncnn-2022-09-30/joiner_jit_trace-pnnx.ncnn.bin \
      ./sherpa-ncnn-2022-09-30/test_wavs/0.wav \
      2 \
      $method
  done

You should see the following output:

.. literalinclude:: ./code-lstm/2022-09-30.txt

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

Real-time speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-ncnn

  ./build/bin/sherpa-ncnn-microphone \
    ./sherpa-ncnn-2022-09-30/tokens.txt \
    ./sherpa-ncnn-2022-09-30/encoder_jit_trace-pnnx.ncnn.param \
    ./sherpa-ncnn-2022-09-30/encoder_jit_trace-pnnx.ncnn.bin \
    ./sherpa-ncnn-2022-09-30/decoder_jit_trace-pnnx.ncnn.param \
    ./sherpa-ncnn-2022-09-30/decoder_jit_trace-pnnx.ncnn.bin \
    ./sherpa-ncnn-2022-09-30/joiner_jit_trace-pnnx.ncnn.param \
    ./sherpa-ncnn-2022-09-30/joiner_jit_trace-pnnx.ncnn.bin \
    2 \
    greedy_search

.. note::

   Please use ``./build/bin/Release/sherpa-ncnn-microphone.exe`` for Windows.

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

You can find a demo below:

..  youtube:: bbQfoRT75oM
   :width: 120%
