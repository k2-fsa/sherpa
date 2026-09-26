.. _{{model_name}}:

{{model_name}} ({{lang}})
----------------------------------------------------------------------

This model supports only ``{{lang}}``. The sections below show how to use it with `sherpa-onnx`_.

Real-time/streaming speech recognition on Android
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Please visit `<https://k2-fsa.github.io/sherpa/onnx/android/apk-simulate-streaming-asr.html>`_
and select the file

  ``sherpa-onnx-<version>-arm64-v8a-simulated_streaming_asr-{{code}}-moonshine_{{apk_name}}_2026_02_27.apk``

.. note::

   For instance, if you choose version ``1.12.27``, you should use `sherpa-onnx-1.12.27-arm64-v8a-simulated_streaming_asr-{{code}}-moonshine_{{apk_name}}_2026_02_27.apk <https://huggingface.co/csukuangfj2/sherpa-onnx-apk/resolve/main/vad-asr-simulated-streaming/1.12.27/sherpa-onnx-1.12.27-arm64-v8a-simulated_streaming_asr-{{code}}-moonshine_{{apk_name}}_2026_02_27.apk>`_


The source code for the APK can be found at

  `<https://github.com/k2-fsa/sherpa-onnx/tree/master/android/SherpaOnnxSimulateStreamingAsr>`_

See :ref:`sherpa-onnx-install-android-studio` for how to build our Android demo.


Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download the model:

.. code-block::

   wget {{url}}
   tar xvf {{model_name}}.tar.bz2
   rm {{model_name}}.tar.bz2

   ls -lh {{model_name}}

You should get the following output:

.. code-block::

    ls -lh sherpa-onnx-moonshine-{{m}}-quantized-2026-02-27
   {% if m == 'base-es' %}
    total 63M
    -rw-r--r-- 1 501 staff  42M Feb 27 09:26 decoder_model_merged.ort
    -rw-r--r-- 1 501 staff  20M Feb 27 09:26 encoder_model.ort
    -rw-r--r-- 1 501 staff  14K Feb 27 09:27 LICENSE
    drwxr-xr-x 2 501 staff 4.0K Mar  3 07:24 test_wavs
    -rw-r--r-- 1 501 staff 520K Feb 27 09:27 tokens.txt
   {% elif m == 'base-ja' %}
    total 135M
    -rw-r--r-- 1 501 staff 105M Feb 27 09:27 decoder_model_merged.ort
    -rw-r--r-- 1 501 staff  30M Feb 27 09:27 encoder_model.ort
    -rw-r--r-- 1 501 staff  14K Feb 27 09:27 LICENSE
    drwxr-xr-x 2 501 staff 4.0K Mar  3 07:25 test_wavs
    -rw-r--r-- 1 501 staff 537K Feb 27 09:27 tokens.txt
   {% elif m == 'base-uk' %}
    total 135M
    -rw-r--r-- 1 501 staff 105M Feb 27 09:27 decoder_model_merged.ort
    -rw-r--r-- 1 501 staff  30M Feb 27 09:27 encoder_model.ort
    -rw-r--r-- 1 501 staff  14K Feb 27 09:28 LICENSE
    drwxr-xr-x 2 501 staff 4.0K Mar  3 07:25 test_wavs
    -rw-r--r-- 1 501 staff 537K Feb 27 09:28 tokens.txt
   {% elif m == 'base-vi' %}
    total 135M
    -rw-r--r-- 1 501 staff 105M Feb 27 09:27 decoder_model_merged.ort
    -rw-r--r-- 1 501 staff  30M Feb 27 09:27 encoder_model.ort
    -rw-r--r-- 1 501 staff  14K Feb 27 09:28 LICENSE
    drwxr-xr-x 2 501 staff 4.0K Mar  3 07:26 test_wavs
    -rw-r--r-- 1 501 staff 537K Feb 27 09:28 tokens.txt
   {% elif m == 'base-zh' %}
    total 135M
    -rw-r--r-- 1 501 staff 105M Feb 27 09:26 decoder_model_merged.ort
    -rw-r--r-- 1 501 staff  30M Feb 27 09:26 encoder_model.ort
    -rw-r--r-- 1 501 staff  14K Feb 27 09:28 LICENSE
    drwxr-xr-x 2 501 staff 4.0K Mar  3 07:26 test_wavs
    -rw-r--r-- 1 501 staff 537K Feb 27 09:28 tokens.txt
   {% elif m == 'tiny-en' %}
    total 43M
    -rw-r--r-- 1 501 staff  30M Feb 27 09:26 decoder_model_merged.ort
    -rw-r--r-- 1 501 staff  13M Feb 27 09:26 encoder_model.ort
    -rw-r--r-- 1 501 staff  14K Feb 27 09:28 LICENSE
    drwxr-xr-x 2 501 staff 4.0K Mar  3 07:26 test_wavs
    -rw-r--r-- 1 501 staff 537K Feb 27 09:28 tokens.txt
   {% elif m == 'tiny-ja' %}
    total 69M
    -rw-r--r-- 1 501 staff  56M Feb 27 09:27 decoder_model_merged.ort
    -rw-r--r-- 1 501 staff  13M Feb 27 09:27 encoder_model.ort
    -rw-r--r-- 1 501 staff  14K Feb 27 09:28 LICENSE
    drwxr-xr-x 2 501 staff 4.0K Mar  3 07:26 test_wavs
    -rw-r--r-- 1 501 staff 537K Feb 27 09:28 tokens.txt
   {% elif m == 'tiny-ko' %}
    total 69M
    -rw-r--r-- 1 501 staff  56M Feb 27 09:27 decoder_model_merged.ort
    -rw-r--r-- 1 501 staff  13M Feb 27 09:27 encoder_model.ort
    -rw-r--r-- 1 501 staff  14K Feb 27 09:28 LICENSE
    drwxr-xr-x 2 501 staff 4.0K Mar  3 07:27 test_wavs
    -rw-r--r-- 1 501 staff 537K Feb 27 09:28 tokens.txt
   {% elif m == 'base-ar' %}
    total 135M
    -rw-r--r-- 1 501 staff 105M Feb 27 09:26 decoder_model_merged.ort
    -rw-r--r-- 1 501 staff  30M Feb 27 09:26 encoder_model.ort
    -rw-r--r-- 1 501 staff  14K Feb 27 09:27 LICENSE
    drwxr-xr-x 2 501 staff 4.0K Mar  3 07:27 test_wavs
    -rw-r--r-- 1 501 staff 537K Feb 27 09:27 tokens.txt
   {% elif m == 'base-en' %}
    total 135M
    -rw-r--r-- 1 501 staff 105M Feb 27 09:27 decoder_model_merged.ort
    -rw-r--r-- 1 501 staff  30M Feb 27 09:26 encoder_model.ort
    -rw-r--r-- 1 501 staff  14K Feb 27 09:27 LICENSE
    drwxr-xr-x 2 501 staff 4.0K Mar  3 07:28 test_wavs
    -rw-r--r-- 1 501 staff 537K Feb 27 09:27 tokens.txt
   {% endif %}

Decode wave files
~~~~~~~~~~~~~~~~~

.. hint::

   It supports decoding only wave files of a single channel with 16-bit
   encoded samples, while the sampling rate does not need to be 16 kHz.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  build/bin/sherpa-onnx-offline \
    --moonshine-encoder=./{{model_name}}/encoder_model.ort \
    --moonshine-merged-decoder=./{{model_name}}/decoder_model_merged.ort \
    --tokens=./{{model_name}}/tokens.txt \
    ./{{model_name}}/test_wavs/0.wav

The output is given below:

.. literalinclude:: ./code/{{m}}.txt

Decode long files with a VAD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following example demonstrates how to use the model to decode a long wave file.

.. code-block::

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

  build/bin/sherpa-onnx-vad-with-offline-asr \
    --silero-vad-model=./silero_vad.onnx \
    --moonshine-encoder=./{{model_name}}/encoder_model.ort \
    --moonshine-merged-decoder=./{{model_name}}/decoder_model_merged.ort \
    --tokens=./{{model_name}}/tokens.txt \
    ./a-very-long-audio-file.wav

Real-time/Streaming Speech recognition from a microphone with VAD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block::

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

  ./build/bin/sherpa-onnx-vad-microphone-simulated-streaming-asr \
    --silero-vad-model=./silero_vad.onnx \
    --moonshine-encoder=./{{model_name}}/encoder_model.ort \
    --moonshine-merged-decoder=./{{model_name}}/decoder_model_merged.ort \
    --tokens=./{{model_name}}/tokens.txt

Speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-microphone-offline \
    --moonshine-encoder=./{{model_name}}/encoder_model.ort \
    --moonshine-merged-decoder=./{{model_name}}/decoder_model_merged.ort \
    --tokens=./{{model_name}}/tokens.txt


Speech recognition from a microphone with VAD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

  ./build/bin/sherpa-onnx-vad-microphone-offline-asr \
    --silero-vad-model=./silero_vad.onnx \
    --moonshine-encoder=./{{model_name}}/encoder_model.ort \
    --moonshine-merged-decoder=./{{model_name}}/decoder_model_merged.ort \
    --tokens=./{{model_name}}/tokens.txt
