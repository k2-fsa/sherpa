.. _sherpa_onnx_t_one_ctc_models:

T-one CTC-based Models
======================

.. hint::

   Please refer to :ref:`install_sherpa_onnx` to install `sherpa-onnx`_
   before you read this section.

.. _sherpa-onnx-streaming-t-one-russian-2025-09-08:

sherpa-onnx-streaming-t-one-russian-2025-09-08 (Russian, 俄语)
----------------------------------------------------------------------

This model is converted from `<https://github.com/voicekit-team/T-one>`_
using scripts from `<https://github.com/k2-fsa/sherpa-onnx/tree/master/scripts/t-one>`_

It expects sample rate 8000 Hz.


Download the model
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-t-one-russian-2025-09-08.tar.bz2
  tar xvf sherpa-onnx-streaming-t-one-russian-2025-09-08.tar.bz2
  rm sherpa-onnx-streaming-t-one-russian-2025-09-08.tar.bz2

  ls -lh sherpa-onnx-streaming-t-one-russian-2025-09-08

The output is given below:

.. code-block:: bash

  -rw-r--r--  1 fangjun  staff    99K Sep  8 17:12 0.wav
  -rw-r--r--  1 fangjun  staff   553B Sep  8 17:12 LICENSE
  -rw-r--r--  1 fangjun  staff   126B Sep  8 17:12 README.md
  -rw-r--r--  1 fangjun  staff   138M Sep  8 17:12 model.onnx
  -rw-r--r--  1 fangjun  staff   202B Sep  8 17:12 tokens.txt

Decode a single wave file
~~~~~~~~~~~~~~~~~~~~~~~~~

.. hint::

   It supports decoding only wave files of a single channel with 16-bit
   encoded samples, while the sampling rate does not need to be 8 kHz.

The following code shows how to use the model to decode a wave file:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx \
    --t-one-ctc-model=./sherpa-onnx-streaming-t-one-russian-2025-09-08/model.onnx \
    --tokens=./sherpa-onnx-streaming-t-one-russian-2025-09-08/tokens.txt \
    ./sherpa-onnx-streaming-t-one-russian-2025-09-08/0.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx.exe`` for Windows.

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

You should see the following output:

.. literalinclude:: ./code-t-one/sherpa-onnx-streaming-russian-2025-09-08.txt

Real-time speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-microphone \
    --t-one-ctc-model=./sherpa-onnx-streaming-t-one-russian-2025-09-08/model.onnx \
    --tokens=./sherpa-onnx-streaming-t-one-russian-2025-09-08/tokens.txt

.. hint::

   If your system is Linux (including embedded Linux), you can also use
   :ref:`sherpa-onnx-alsa` to do real-time speech recognition with your
   microphone if ``sherpa-onnx-microphone`` does not work for you.

Huggingface space
~~~~~~~~~~~~~~~~~

You can try this model by visiting

  `<https://huggingface.co/spaces/k2-fsa/automatic-speech-recognition>`_
