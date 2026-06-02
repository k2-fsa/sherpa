sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19 (Russian, 俄语)
-------------------------------------------------------------------------

This model is converted from

  `<https://github.com/salute-developers/GigaAM>`_

You can find the conversion script at

  `<https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/nemo/GigaAM/run-rnnt-v2.sh>`_

In the following, we describe how to download it and use it with `sherpa-onnx`_.

Android APK for real-time speech recognition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Please visit `<https://k2-fsa.github.io/sherpa/onnx/android/apk-simulate-streaming-asr.html>`_
and search for ``nemo_transducer_giga_am_v2_2025_04_19``.

.. hint::

   Please always use the latest version. For instance, you can use
   `sherpa-onnx-1.12.40-arm64-v8a-simulated_streaming_asr-ru-nemo_transducer_giga_am_v2_2025_04_19.apk <https://huggingface.co/csukuangfj2/sherpa-onnx-apk/resolve/main/vad-asr-simulated-streaming/1.12.40/sherpa-onnx-1.12.40-arm64-v8a-simulated_streaming_asr-ru-nemo_transducer_giga_am_v2_2025_04_19.apk>`_
   for ``arm64-v8a``.

Download the model
^^^^^^^^^^^^^^^^^^

Please use the following commands to download it.

.. code-block:: bash

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19.tar.bz2
  tar xvf sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19.tar.bz2
  rm sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19.tar.bz2

You should see something like below after downloading::

  ls -lh sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19

  total 231M
  -rw-r--r-- 1 501 staff 3.2M Apr 20 01:58 decoder.onnx
  -rw-r--r-- 1 501 staff 226M Apr 20 01:59 encoder.int8.onnx
  -rw-r--r-- 1 501 staff 1.4M Apr 20 01:58 joiner.onnx
  -rw-r--r-- 1 501 staff 219K Apr 20 01:59 LICENSE
  -rw-r--r-- 1 501 staff  302 Apr 20 01:59 README.md
  -rwxr-xr-x 1 501 staff  868 Apr 20 01:51 run-rnnt-v2.sh
  -rwxr-xr-x 1 501 staff 8.9K Apr 20 01:59 test-onnx-rnnt.py
  drwxr-xr-x 2 501 staff 4.0K Apr 21 09:35 test_wavs
  -rw-r--r-- 1 501 staff  196 Apr 20 01:58 tokens.txt

Decode wave files
^^^^^^^^^^^^^^^^^

.. hint::

   It supports decoding only wave files of a single channel with 16-bit
   encoded samples, while the sampling rate does not need to be 16 kHz.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-offline \
    --encoder=./sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19/encoder.int8.onnx \
    --decoder=./sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19/decoder.onnx \
    --joiner=./sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19/joiner.onnx \
    --tokens=./sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19/tokens.txt \
    --model-type=nemo_transducer \
    ./sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19/test_wavs/example.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

You should see the following output:

.. literalinclude:: ./code-nemo/sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19.txt

Real-time/Streaming Speech recognition from a microphone with VAD
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block::

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

  ./build/bin/sherpa-onnx-vad-microphone-simulated-streaming-asr \
    --silero-vad-model=./silero_vad.onnx \
    --encoder=./sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19/encoder.int8.onnx \
    --decoder=./sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19/decoder.onnx \
    --joiner=./sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19/joiner.onnx \
    --tokens=./sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19/tokens.txt \
    --model-type=nemo_transducer

Speech recognition from a microphone
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-microphone-offline \
    --encoder=./sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19/encoder.int8.onnx \
    --decoder=./sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19/decoder.onnx \
    --joiner=./sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19/joiner.onnx \
    --tokens=./sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19/tokens.txt \
    --model-type=nemo_transducer

Speech recognition from a microphone with VAD
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  cd /path/to/sherpa-onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

  ./build/bin/sherpa-onnx-vad-microphone-offline-asr \
    --silero-vad-model=./silero_vad.onnx \
    --encoder=./sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19/encoder.int8.onnx \
    --decoder=./sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19/decoder.onnx \
    --joiner=./sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19/joiner.onnx \
    --tokens=./sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19/tokens.txt \
    --model-type=nemo_transducer

