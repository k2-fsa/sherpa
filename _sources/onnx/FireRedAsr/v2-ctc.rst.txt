.. _sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25:

sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25 (v2, CTC, Chinese + English, 普通话、粤语（香港和广东）、四川话、上海话、吴语、闽南话、安徽话、 福建话、甘肃话、贵州话、河北话、河南话、湖北话、湖南话、江西话、辽宁话、宁夏话、 陕西话、山西话、山东话、天津话、云南话等20多种方言)
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

This model is converted from `<https://www.modelscope.cn/models/FireRedTeam/FireRedASR2-AED>`_. Note that only the CTC branch
is converted. The attention decoder branch is excluded.

It supports both Chinese and English. Also, it supports more than 20 dialects/accents.

.. note::

   支持的中文方言/口音：粤语（香港和广东）、四川话、上海话、吴语、闽南话、安徽话、
   福建话、甘肃话、贵州话、河北话、河南话、湖北话、湖南话、江西话、辽宁话、宁夏话、
   陕西话、山西话、山东话、天津话、云南话等。

In the following, we describe how to use it.

Real-time/streaming speech recognition on Android
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pease visit `<https://k2-fsa.github.io/sherpa/onnx/android/apk.html>`_
and select the file

  ``sherpa-onnx-<version>-arm64-v8a-simulated_streaming_asr-zh_en-fire_red_asr2_ctc_int8_2026_02_25.apk``

.. note::

   For instance, if you choose version ``1.12.27``, you should use `sherpa-onnx-1.12.27-arm64-v8a-simulated_streaming_asr-zh_en-fire_red_asr2_ctc_int8_2026_02_25.apk <https://huggingface.co/csukuangfj2/sherpa-onnx-apk/resolve/main/vad-asr-simulated-streaming/1.12.27/sherpa-onnx-1.12.27-arm64-v8a-simulated_streaming_asr-zh_en-fire_red_asr2_ctc_int8_2026_02_25.apk>`_

   中国用户，请使用 `sherpa-onnx-1.12.27-arm64-v8a-simulated_streaming_asr-zh_en-fire_red_asr2_ctc_int8_2026_02_25.apk <https://hf-mirror.com/csukuangfj2/sherpa-onnx-apk/blob/main/vad-asr-simulated-streaming/1.12.27/sherpa-onnx-1.12.27-arm64-v8a-simulated_streaming_asr-zh_en-fire_red_asr2_ctc_int8_2026_02_25.apk>`_

The source code for the APK can be found at

  `<https://github.com/k2-fsa/sherpa-onnx/tree/master/android/SherpaOnnxSimulateStreamingAsr>`_

Please refer to :ref:`sherpa-onnx-install-android-studio` for how to build our Android demo.

Download
^^^^^^^^

Please use the following commands to download it::

  cd /path/to/sherpa-onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25.tar.bz2
  tar xvf sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25.tar.bz2
  rm sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25.tar.bz2

After downloading, you should find the following files::

  ls -lh sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25
  total 1515528
  -rw-r--r--@  1 fangjun  staff   740M 26 Feb 13:42 model.int8.onnx
  -rw-r--r--@  1 fangjun  staff   190B 26 Feb 13:35 README.md
  drwxr-xr-x@ 10 fangjun  staff   320B 26 Feb 13:42 test_wavs
  -rw-r--r--@  1 fangjun  staff    77K 26 Feb 13:42 tokens.txt

  ls -lh sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25/test_wavs/
  total 3848
  -rw-r--r--@ 1 fangjun  staff   314K 26 Feb 13:42 0.wav
  -rw-r--r--@ 1 fangjun  staff   159K 26 Feb 13:42 1.wav
  -rw-r--r--@ 1 fangjun  staff   147K 26 Feb 13:42 2.wav
  -rw-r--r--@ 1 fangjun  staff   245K 26 Feb 13:42 3-sichuan.wav
  -rw-r--r--@ 1 fangjun  staff   276K 26 Feb 13:42 3.wav
  -rw-r--r--@ 1 fangjun  staff   244K 26 Feb 13:42 4-tianjin.wav
  -rw-r--r--@ 1 fangjun  staff   250K 26 Feb 13:42 5-henan.wav
  -rw-r--r--@ 1 fangjun  staff   276K 26 Feb 13:42 8k.wav

Decode a file
^^^^^^^^^^^^^

Please use the following command to decode a wave file:

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline \
    --num-threads=1 \
    --fire-red-asr-ctc=./sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25/model.int8.onnx \
    --tokens=./sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25/tokens.txt \
    ./sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25/test_wavs/1.wav


You should see the following output:

.. literalinclude:: ./code/2026-02-25-ctc.txt

Decode long files with a VAD
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following example demonstrates how to use the model to decode a long wave file.

.. code-block::

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/lei-jun-test.wav

  build/bin/sherpa-onnx-vad-with-offline-asr \
    --num-threads=3 \
    --silero-vad-model=./silero_vad.onnx \
    --fire-red-asr-ctc=./sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25/model.int8.onnx \
    --tokens=./sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25/tokens.txt \
    ./lei-jun-test.wav

You should see the following output:

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
    </tr>
    <tr>
      <td>lei-jun-test.wav</td>
      <td>
       <audio title="lei-jun-test.wav" controls="controls">
             <source src="/sherpa/_static/sense-voice/lei-jun-test.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
    </tr>
  </table>

.. literalinclude:: ./code/2026-02-25-ctc-long.txt

Real-time/Streaming Speech recognition from a microphone with VAD
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block::

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

  ./build/bin/sherpa-onnx-vad-microphone-simulated-streaming-asr \
    --silero-vad-model=./silero_vad.onnx \
    --fire-red-asr-ctc=./sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25/model.int8.onnx \
    --tokens=./sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25/tokens.txt

Speech recognition from a microphone
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-microphone-offline \
    --fire-red-asr-ctc=./sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25/model.int8.onnx \
    --tokens=./sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25/tokens.txt

Speech recognition from a microphone with VAD
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  cd /path/to/sherpa-onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

  ./build/bin/sherpa-onnx-vad-microphone-offline-asr \
    --silero-vad-model=./silero_vad.onnx \
    --fire-red-asr-ctc=./sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25/model.int8.onnx \
    --tokens=./sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25/tokens.txt

