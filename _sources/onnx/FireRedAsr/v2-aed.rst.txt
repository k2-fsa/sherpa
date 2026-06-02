.. _sherpa-onnx-fire-red-asr2-zh_en-int8-2026-02-26:

sherpa-onnx-fire-red-asr2-zh_en-int8-2026-02-26 (v2, AED, Chinese + English, 普通话、粤语（香港和广东）、四川话、上海话、吴语、闽南话、安徽话、 福建话、甘肃话、贵州话、河北话、河南话、湖北话、湖南话、江西话、辽宁话、宁夏话、 陕西话、山西话、山东话、天津话、云南话等20多种方言)
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

This model is converted from `<https://www.modelscope.cn/models/FireRedTeam/FireRedASR2-AED>`_.

It supports both Chinese and English, as well as more than 20 dialects and accents.

.. note::

   支持的中文方言/口音：粤语（香港和广东）、四川话、上海话、吴语、闽南话、安徽话、
   福建话、甘肃话、贵州话、河北话、河南话、湖北话、湖南话、江西话、辽宁话、宁夏话、
   陕西话、山西话、山东话、天津话、云南话等。

The sections below show how to use it.

Download
^^^^^^^^

Please use the following commands to download it::

  cd /path/to/sherpa-onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-fire-red-asr2-zh_en-int8-2026-02-26.tar.bz2
  tar xvf sherpa-onnx-fire-red-asr2-zh_en-int8-2026-02-26.tar.bz2
  rm sherpa-onnx-fire-red-asr2-zh_en-int8-2026-02-26.tar.bz2

After downloading, you should find the following files::

  ls -lh sherpa-onnx-fire-red-asr2-zh_en-int8-2026-02-26
  total 2426960
  -rw-r--r--@  1 fangjun  staff   398M 26 Feb 13:57 decoder.int8.onnx
  -rw-r--r--@  1 fangjun  staff   779M 26 Feb 13:57 encoder.int8.onnx
  -rw-r--r--@  1 fangjun  staff   107B 26 Feb 13:57 README.md
  drwxr-xr-x@ 10 fangjun  staff   320B 26 Feb 13:57 test_wavs
  -rw-r--r--@  1 fangjun  staff    77K 26 Feb 13:57 tokens.txt

Decode a file
^^^^^^^^^^^^^

Please use the following command to decode a wave file:

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline \
    --fire-red-asr-encoder=./sherpa-onnx-fire-red-asr2-zh_en-int8-2026-02-26/encoder.int8.onnx \
    --fire-red-asr-decoder=./sherpa-onnx-fire-red-asr2-zh_en-int8-2026-02-26/decoder.int8.onnx \
    --tokens=./sherpa-onnx-fire-red-asr2-zh_en-int8-2026-02-26/tokens.txt \
    --num-threads=1 \
    ./sherpa-onnx-fire-red-asr2-zh_en-int8-2026-02-26/test_wavs/0.wav

You should see the following output:

.. literalinclude:: ./code/2026-02-26-aed.txt

Decode long files with a VAD
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following example demonstrates how to use the model to decode a long wave file.

.. code-block::

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/lei-jun-test.wav

  build/bin/sherpa-onnx-vad-with-offline-asr \
    --num-threads=3 \
    --silero-vad-model=./silero_vad.onnx \
    --fire-red-asr-encoder=./sherpa-onnx-fire-red-asr2-zh_en-int8-2026-02-26/encoder.int8.onnx \
    --fire-red-asr-decoder=./sherpa-onnx-fire-red-asr2-zh_en-int8-2026-02-26/decoder.int8.onnx \
    --tokens=./sherpa-onnx-fire-red-asr2-zh_en-int8-2026-02-26/tokens.txt \
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

.. literalinclude:: ./code/2026-02-26-aed-long.txt

Real-time/Streaming Speech recognition from a microphone with VAD
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block::

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

  ./build/bin/sherpa-onnx-vad-microphone-simulated-streaming-asr \
    --silero-vad-model=./silero_vad.onnx \
    --fire-red-asr-encoder=./sherpa-onnx-fire-red-asr2-zh_en-int8-2026-02-26/encoder.int8.onnx \
    --fire-red-asr-decoder=./sherpa-onnx-fire-red-asr2-zh_en-int8-2026-02-26/decoder.int8.onnx \
    --tokens=./sherpa-onnx-fire-red-asr2-zh_en-int8-2026-02-26/tokens.txt

Speech recognition from a microphone
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-microphone-offline \
    --fire-red-asr-encoder=./sherpa-onnx-fire-red-asr2-zh_en-int8-2026-02-26/encoder.int8.onnx \
    --fire-red-asr-decoder=./sherpa-onnx-fire-red-asr2-zh_en-int8-2026-02-26/decoder.int8.onnx \
    --tokens=./sherpa-onnx-fire-red-asr2-zh_en-int8-2026-02-26/tokens.txt

Speech recognition from a microphone with VAD
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  cd /path/to/sherpa-onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

  ./build/bin/sherpa-onnx-vad-microphone-offline-asr \
    --silero-vad-model=./silero_vad.onnx \
    --fire-red-asr-encoder=./sherpa-onnx-fire-red-asr2-zh_en-int8-2026-02-26/encoder.int8.onnx \
    --fire-red-asr-decoder=./sherpa-onnx-fire-red-asr2-zh_en-int8-2026-02-26/decoder.int8.onnx \
    --tokens=./sherpa-onnx-fire-red-asr2-zh_en-int8-2026-02-26/tokens.txt
