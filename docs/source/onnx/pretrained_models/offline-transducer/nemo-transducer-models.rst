.. _sherpa_onnx_offline_nemo_transducer_models:

NeMo transducer-based Models
============================

.. hint::

   Please refer to :ref:`install_sherpa_onnx` to install `sherpa-onnx`
   before you read this section.

sherpa-onnx-nemo-transducer-giga-am-russian-2024-10-24 (Russian, 俄语)
----------------------------------------------------------------------

This model is converted from

  `<https://github.com/salute-developers/GigaAM>`_

You can find the conversion script at

  `<https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/nemo/GigaAM/run-rnnt.sh>`_

.. warning::

   The license of the model can be found at `<https://github.com/salute-developers/GigaAM/blob/main/GigaAM%20License_NC.pdf>`_.

   It is for non-commercial use only.

In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-transducer-giga-am-russian-2024-10-24.tar.bz2
  tar xvf sherpa-onnx-nemo-transducer-giga-am-russian-2024-10-24.tar.bz2
  rm sherpa-onnx-nemo-transducer-giga-am-russian-2024-10-24.tar.bz2

You should see something like below after downloading::

  ls -lh sherpa-onnx-nemo-transducer-giga-am-russian-2024-10-24/
  total 548472
  -rw-r--r--  1 fangjun  staff    89K Oct 25 13:36 GigaAM%20License_NC.pdf
  -rw-r--r--  1 fangjun  staff   318B Oct 25 13:37 README.md
  -rw-r--r--  1 fangjun  staff   3.8M Oct 25 13:36 decoder.onnx
  -rw-r--r--  1 fangjun  staff   262M Oct 25 13:37 encoder.int8.onnx
  -rw-r--r--  1 fangjun  staff   3.8K Oct 25 13:32 export-onnx-rnnt.py
  -rw-r--r--  1 fangjun  staff   2.0M Oct 25 13:36 joiner.onnx
  -rwxr-xr-x  1 fangjun  staff   2.0K Oct 25 13:32 run-rnnt.sh
  -rwxr-xr-x  1 fangjun  staff   8.7K Oct 25 13:32 test-onnx-rnnt.py
  drwxr-xr-x  4 fangjun  staff   128B Oct 25 13:37 test_wavs
  -rw-r--r--  1 fangjun  staff   5.8K Oct 25 13:36 tokens.txt

Decode wave files
~~~~~~~~~~~~~~~~~

.. hint::

   It supports decoding only wave files of a single channel with 16-bit
   encoded samples, while the sampling rate does not need to be 16 kHz.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-offline \
    --encoder=./sherpa-onnx-nemo-transducer-giga-am-russian-2024-10-24/encoder.int8.onnx \
    --decoder=./sherpa-onnx-nemo-transducer-giga-am-russian-2024-10-24/decoder.onnx \
    --joiner=./sherpa-onnx-nemo-transducer-giga-am-russian-2024-10-24/joiner.onnx \
    --tokens=./sherpa-onnx-nemo-transducer-giga-am-russian-2024-10-24/tokens.txt \
    --model-type=nemo_transducer \
    ./sherpa-onnx-nemo-transducer-giga-am-russian-2024-10-24/test_wavs/example.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

You should see the following output:

.. literalinclude:: ./code-nemo/sherpa-onnx-nemo-transducer-giga-am-russian-2024-10-24.int8.txt

Speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-microphone-offline \
    --encoder=./sherpa-onnx-nemo-transducer-giga-am-russian-2024-10-24/encoder.int8.onnx \
    --decoder=./sherpa-onnx-nemo-transducer-giga-am-russian-2024-10-24/decoder.onnx \
    --joiner=./sherpa-onnx-nemo-transducer-giga-am-russian-2024-10-24/joiner.onnx \
    --tokens=./sherpa-onnx-nemo-transducer-giga-am-russian-2024-10-24/tokens.txt \
    --model-type=nemo_transducer

Speech recognition from a microphone with VAD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

  ./build/bin/sherpa-onnx-vad-microphone-offline-asr \
    --silero-vad-model=./silero_vad.onnx \
    --encoder=./sherpa-onnx-nemo-transducer-giga-am-russian-2024-10-24/encoder.int8.onnx \
    --decoder=./sherpa-onnx-nemo-transducer-giga-am-russian-2024-10-24/decoder.onnx \
    --joiner=./sherpa-onnx-nemo-transducer-giga-am-russian-2024-10-24/joiner.onnx \
    --tokens=./sherpa-onnx-nemo-transducer-giga-am-russian-2024-10-24/tokens.txt \
    --model-type=nemo_transducer
