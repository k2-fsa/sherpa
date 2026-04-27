.. _sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8:

sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8 (25 European Languages)
----------------------------------------------------------------------

This model is converted from

  `<https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3>`_

You can find the conversion script at

  `<https://github.com/k2-fsa/sherpa-onnx/tree/master/scripts/nemo/parakeet-tdt-0.6b-v3>`_

It supports 25 European languages:

  - Bulgarian (bg), Croatian (hr), Czech (cs), Danish (da), Dutch (nl)
  - English (en), Estonian (et), Finnish (fi), French (fr), German (de)
  - Greek (el), Hungarian (hu), Italian (it), Latvian (lv), Lithuanian (lt)
  - Maltese (mt), Polish (pl), Portuguese (pt), Romanian (ro), Slovak (sk)
  - Slovenian (sl), Spanish (es), Swedish (sv), Russian (ru), Ukrainian (uk)

In the following, we describe how to download it and use it with `sherpa-onnx`_.

Colab
^^^^^

We provide two colab notebooks for this model:

  - `Colab with CPU <https://colab.research.google.com/drive/1ixBBirCv7vOcM0QNITwad9iSGFpG5an4?usp=sharing>`_
  - `Colab with NVIDIA GPU <https://colab.research.google.com/drive/1EUgBbM165YZLnef2iYf_ZIv6mBn9GhLG?usp=sharing>`_

Huggingface space
^^^^^^^^^^^^^^^^^

You can try it by visiting `<https://huggingface.co/spaces/k2-fsa/automatic-speech-recognition>`_

Android APK for real-time speech recognition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Please visit `<https://k2-fsa.github.io/sherpa/onnx/android/apk-simulate-streaming-asr.html>`_
and search for ``1parakeet_tdt_0.6b_v3``.

.. hint::

   Please always use the latest version. For instance, you can use
   `sherpa-onnx-1.12.40-arm64-v8a-simulated_streaming_asr-multi-parakeet_tdt_0.6b_v3.apk <https://huggingface.co/csukuangfj2/sherpa-onnx-apk/resolve/main/vad-asr-simulated-streaming/1.12.40/sherpa-onnx-1.12.40-arm64-v8a-simulated_streaming_asr-multi-parakeet_tdt_0.6b_v3.apk>`_
   for ``arm64-v8a``.


Download the model
^^^^^^^^^^^^^^^^^^

Please use the following commands to download it.

.. code-block:: bash

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2
   tar xvf sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2
   rm sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2

You should see something like below after downloading::

  ls -lh sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/

  total 640M
  -rw-r--r-- 1 501 staff  12M Aug 16 09:00 decoder.int8.onnx
  -rw-r--r-- 1 501 staff 622M Aug 16 09:00 encoder.int8.onnx
  -rw-r--r-- 1 501 staff 6.1M Aug 16 09:00 joiner.int8.onnx
  drwxr-xr-x 2 501 staff 4.0K Aug 16 09:00 test_wavs
  -rw-r--r-- 1 501 staff  92K Aug 16 09:00 tokens.txt

Decode wave files
^^^^^^^^^^^^^^^^^

.. hint::

   It supports decoding only wave files of a single channel with 16-bit
   encoded samples, while the sampling rate does not need to be 16 kHz.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  build/bin/sherpa-onnx-offline \
    --encoder=./sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/encoder.int8.onnx \
    --decoder=./sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/decoder.int8.onnx \
    --joiner=./sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/joiner.int8.onnx \
    --tokens=./sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/tokens.txt \
    --model-type=nemo_transducer \
    ./sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/test_wavs/en.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

You should see the following output:

.. literalinclude:: ./code-nemo/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.txt

Real-time/Streaming Speech recognition from a microphone with VAD
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block::

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

  ./build/bin/sherpa-onnx-vad-microphone-simulated-streaming-asr \
    --silero-vad-model=./silero_vad.onnx \
    --encoder=./sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/encoder.int8.onnx \
    --decoder=./sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/decoder.int8.onnx \
    --joiner=./sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/joiner.int8.onnx \
    --tokens=./sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/tokens.txt \
    --model-type=nemo_transducer

Speech recognition from a microphone
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-microphone-offline \
    --encoder=./sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/encoder.int8.onnx \
    --decoder=./sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/decoder.int8.onnx \
    --joiner=./sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/joiner.int8.onnx \
    --tokens=./sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/tokens.txt \
    --model-type=nemo_transducer

Speech recognition from a microphone with VAD
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  cd /path/to/sherpa-onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

  ./build/bin/sherpa-onnx-vad-microphone-offline-asr \
    --silero-vad-model=./silero_vad.onnx \
    --encoder=./sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/encoder.int8.onnx \
    --decoder=./sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/decoder.int8.onnx \
    --joiner=./sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/joiner.int8.onnx \
    --tokens=./sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/tokens.txt \
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
    --encoder=./sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/encoder.int8.onnx \
    --decoder=./sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/decoder.int8.onnx \
    --joiner=./sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/joiner.int8.onnx \
    --tokens=./sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/tokens.txt \
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

.. literalinclude:: ./code-nemo/obama-tdt-v3.txt

.. hint::

   If you want to use a GUI version and want to export ``SRT`` format, please visit
   `<https://k2-fsa.github.io/sherpa/onnx/tauri/app/vad-asr-file.html>`_ and search for
   ``en-parakeet_tdt_v3``. Please always use the latest version.

