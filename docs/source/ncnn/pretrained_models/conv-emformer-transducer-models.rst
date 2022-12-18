Conv-Emformer-transducer-based Models
=====================================

csukuangfj/sherpa-ncnn-conv-emformer-transducer-2022-12-06 (Chinese + English)
------------------------------------------------------------------------------

This model is converted from `<https://huggingface.co/ptrnull/icefall-asr-conv-emformer-transducer-stateless2-zh>`_,
which supports both Chinese and English.

.. hint::

  If you want to train your own model that is able to support both Chinese and
  English, please refer to our training code:

    `<https://github.com/k2-fsa/icefall/tree/master/egs/tal_csasr/ASR>`_

  You can also try the pre-trained models in your browser without installing anything
  by visiting:

    `<https://huggingface.co/spaces/k2-fsa/automatic-speech-recognition>`_

In the following, we describe how to download and use it with `sherpa-ncnn`_.

Please use the following commands to download it.

.. code-block:: bash

  cd /path/to/sherpa-ncnn

  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/sherpa-ncnn-conv-emformer-transducer-2022-12-06
  cd sherpa-ncnn-conv-emformer-transducer-2022-12-06
  git lfs pull --include "*.bin"

Please check that the file size of the pre-trained models is correct (see the
screen shot below):

.. figure:: ./pic/2022-12-06-filesize.png
   :alt: File size for sherpa-ncnn-2022-12-06
   :width: 800

Decode a single wave file with ./build/bin/sherpa-ncnn
::::::::::::::::::::::::::::::::::::::::::::::::::::::

.. hint::

   It supports decoding only wave files with a single channel and the sampling rate
   should be 16 kHz.

.. code-block:: bash

  cd /path/to/sherpa-ncnn

  ./build/bin/sherpa-ncnn \
    ./sherpa-ncnn-conv-emformer-transducer-2022-12-06/tokens.txt \
    ./sherpa-ncnn-conv-emformer-transducer-2022-12-06/encoder_jit_trace-pnnx.ncnn.param \
    ./sherpa-ncnn-conv-emformer-transducer-2022-12-06/encoder_jit_trace-pnnx.ncnn.bin \
    ./sherpa-ncnn-conv-emformer-transducer-2022-12-06/decoder_jit_trace-pnnx.ncnn.param \
    ./sherpa-ncnn-conv-emformer-transducer-2022-12-06/decoder_jit_trace-pnnx.ncnn.bin \
    ./sherpa-ncnn-conv-emformer-transducer-2022-12-06/joiner_jit_trace-pnnx.ncnn.param \
    ./sherpa-ncnn-conv-emformer-transducer-2022-12-06/joiner_jit_trace-pnnx.ncnn.bin \
    ./sherpa-ncnn-conv-emformer-transducer-2022-12-06/test_wavs/0.wav \

.. note::

   Please use ``./build/bin/Release/sherpa-ncnn.exe`` for Windows.

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

Real-time speech recognition from a microphone with build/bin/sherpa-ncnn-microphone
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

.. code-block:: bash

  cd /path/to/sherpa-ncnn
  ./build/bin/sherpa-ncnn-microphone \
    ./sherpa-ncnn-conv-emformer-transducer-2022-12-06/tokens.txt \
    ./sherpa-ncnn-conv-emformer-transducer-2022-12-06/encoder_jit_trace-pnnx.ncnn.param \
    ./sherpa-ncnn-conv-emformer-transducer-2022-12-06/encoder_jit_trace-pnnx.ncnn.bin \
    ./sherpa-ncnn-conv-emformer-transducer-2022-12-06/decoder_jit_trace-pnnx.ncnn.param \
    ./sherpa-ncnn-conv-emformer-transducer-2022-12-06/decoder_jit_trace-pnnx.ncnn.bin \
    ./sherpa-ncnn-conv-emformer-transducer-2022-12-06/joiner_jit_trace-pnnx.ncnn.param \
    ./sherpa-ncnn-conv-emformer-transducer-2022-12-06/joiner_jit_trace-pnnx.ncnn.bin

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

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.
