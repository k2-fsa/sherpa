Japanese
========

.. hint::

   Please refer to :ref:`install_sherpa_onnx` to install `sherpa-onnx`_
   before you read this section.

This page lists offline CTC models from `NeMo`_ for Japanese.

sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8 (Japanese, 日语)
------------------------------------------------------------------------

This model is converted from `<https://huggingface.co/nvidia/parakeet-tdt_ctc-0.6b-ja>`_.

You can find the code for exporting the model from `NeMo`_ to `sherpa-onnx`_
at `<https://github.com/k2-fsa/sherpa-onnx/tree/master/scripts/nemo/parakeet-tdt_ctc-0.6b-ja>`_.

The model was trained on `ReazonSpeech`_ v2.0 speech corpus containing more than 35k
hours of natural Japanese speech.

In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8.tar.bz2
   tar xvf sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8.tar.bz2
   rm sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8.tar.bz2

You should see something like below after downloading::

  ls -lh sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8/
  total 1310808
  -rw-r--r--  1 fangjun  staff   625M Jul  9 11:11 model.int8.onnx
  drwxr-xr-x  5 fangjun  staff   160B Jul  9 14:22 test_wavs
  -rw-r--r--  1 fangjun  staff    28K Jul  9 14:21 tokens.txt

Decode wave files
~~~~~~~~~~~~~~~~~

.. hint::

   It supports decoding only wave files of a single channel with 16-bit
   encoded samples, while the sampling rate does not need to be 16 kHz.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-offline \
    --nemo-ctc-model=./sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8/model.int8.onnx \
    --tokens=./sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8/tokens.txt \
    ./sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8/test_wavs/test_ja_1.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

You should see the following output:

.. literalinclude:: ./code-japanese/tdt-ctc-0.6b-int8.txt

Speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-microphone-offline \
    --nemo-ctc-model=./sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8/model.int8.onnx \
    --tokens=./sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8/tokens.txt

Speech recognition from a microphone with VAD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

  ./build/bin/sherpa-onnx-vad-microphone-offline-asr \
    --silero-vad-model=./silero_vad.onnx \
    --nemo-ctc-model=./sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8/model.int8.onnx \
    --tokens=./sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8/tokens.txt
