sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming (English)
----------------------------------------------------------------------

This model is converted from `<https://huggingface.co/nvidia/parakeet-unified-en-0.6b>`_.

Note that only the non-streaming mode is supported by this ONNX model.

You can find the conversion script at

  `<https://github.com/k2-fsa/sherpa-onnx/tree/master/scripts/nemo/parakeet-unified-en-0.6b>`_

In the following, we describe how to download it and use it with `sherpa-onnx`_.

.. hint::

   This model supports punctuations and cases.

Android APK for real-time speech recognition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Please visit `<https://k2-fsa.github.io/sherpa/onnx/android/apk-simulate-streaming-asr.html>`_
and search for ``parakeet_unified_en_non_streaming_0.6b_int8``.

.. hint::

   Please always use the latest version. For instance, you can use
   `sherpa-onnx-1.12.40-arm64-v8a-simulated_streaming_asr-en-parakeet_unified_en_non_streaming_0.6b_int8.apk <https://huggingface.co/csukuangfj2/sherpa-onnx-apk/resolve/main/vad-asr-simulated-streaming/1.12.40/sherpa-onnx-1.12.40-arm64-v8a-simulated_streaming_asr-en-parakeet_unified_en_non_streaming_0.6b_int8.apk>`_
   for ``arm64-v8a``.

Download the model
^^^^^^^^^^^^^^^^^^

Please use the following commands to download it.

.. code-block:: bash

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming.tar.bz2
   tar xvf sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming.tar.bz2
   rm sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming.tar.bz2

   ls -lh sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming.tar.bz2

You should see something like below after downloading::

  ls -lh  sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming

  total 1296016
  -rw-r--r--@ 1 fangjun  staff   1.0K 27 Apr 15:47 bias.md
  -rw-r--r--@ 1 fangjun  staff   6.9M 27 Apr 15:47 decoder.int8.onnx
  -rw-r--r--@ 1 fangjun  staff   624M 27 Apr 15:47 encoder.int8.onnx
  -rw-r--r--@ 1 fangjun  staff   2.4K 27 Apr 15:47 explainability.md
  -rw-r--r--@ 1 fangjun  staff   1.7M 27 Apr 15:47 joiner.int8.onnx
  -rw-r--r--@ 1 fangjun  staff   4.0K 27 Apr 15:47 privacy.md
  -rw-r--r--@ 1 fangjun  staff   782B 27 Apr 15:47 safety.md
  drwxr-xr-x@ 3 fangjun  staff    96B 27 Apr 15:47 test_wavs
  -rw-r--r--@ 1 fangjun  staff   8.7K 27 Apr 15:47 tokens.txt

.. hint::

   To use the ``fp32`` model, please visit `<https://huggingface.co/csukuangfj2/sherpa-onnx-nemo-parakeet-unified-en-0.6b-non-streaming>`_

Decode wave files
^^^^^^^^^^^^^^^^^

.. hint::

   It supports decoding only wave files of a single channel with 16-bit
   encoded samples, while the sampling rate does not need to be 16 kHz.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-offline \
    --encoder=./sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming/encoder.int8.onnx \
    --decoder=./sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming/decoder.int8.onnx \
    --joiner=./sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming/joiner.int8.onnx \
    --tokens=./sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming/tokens.txt \
    --model-type=nemo_transducer \
    ./sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming/test_wavs/0.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

You should see the following output:

.. literalinclude:: ./code-nemo/parakeet-unified-en-0.6b-0-wav.txt

Real-time/Streaming Speech recognition from a microphone with VAD
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block::

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

  ./build/bin/sherpa-onnx-vad-microphone-simulated-streaming-asr \
    --silero-vad-model=./silero_vad.onnx \
    --encoder=./sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming/encoder.int8.onnx \
    --decoder=./sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming/decoder.int8.onnx \
    --joiner=./sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming/joiner.int8.onnx \
    --tokens=./sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming/tokens.txt \
    --model-type=nemo_transducer

Speech recognition from a microphone
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-microphone-offline \
    --encoder=./sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming/encoder.int8.onnx \
    --decoder=./sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming/decoder.int8.onnx \
    --joiner=./sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming/joiner.int8.onnx \
    --tokens=./sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming/tokens.txt \
    --model-type=nemo_transducer

Speech recognition from a microphone with VAD
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  cd /path/to/sherpa-onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

  ./build/bin/sherpa-onnx-vad-microphone-offline-asr \
    --silero-vad-model=./silero_vad.onnx \
    --encoder=./sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming/encoder.int8.onnx \
    --decoder=./sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming/decoder.int8.onnx \
    --joiner=./sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming/joiner.int8.onnx \
    --tokens=./sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming/tokens.txt \
    --model-type=nemo_transducer

Decode a long audio file with VAD
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following examples show how to decode a very long audio file with the help
of VAD.

.. code-block:: bash

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/Obama.wav

  ./build/bin/sherpa-onnx-vad-with-offline-asr \
    --silero-vad-model=./silero_vad.onnx \
    --silero-vad-threshold=0.2 \
    --silero-vad-min-speech-duration=0.2 \
    --encoder=./sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming/encoder.int8.onnx \
    --decoder=./sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming/decoder.int8.onnx \
    --joiner=./sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming/joiner.int8.onnx \
    --tokens=./sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming/tokens.txt \
    --model-type=nemo_transducer \
    ./Obama.wav

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
    </tr>
    <tr>
      <td>Obama.wav</td>
      <td>
       <audio title="Obama.wav" controls="controls">
             <source src="/sherpa/_static/sense-voice/Obama.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
    </tr>
  </table>

You should see the following output:

.. literalinclude:: ./code-nemo/obama-parakeet-unified-en-0.6b.txt

.. hint::

   If you want to use a GUI version and want to export ``SRT`` format, please visit
   `<https://k2-fsa.github.io/sherpa/onnx/tauri/app/vad-asr-file.html>`_ and search for
   ``parakeet_unified_en_0.6b_int8``. Please always use the latest version.


