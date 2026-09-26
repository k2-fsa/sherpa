.. _sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10:

sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10 (Cantonese, 粤语)
------------------------------------------------------------------------------------------------

This model is converted from

  `<https://huggingface.co/ASLP-lab/WSYue-ASR/tree/main/u2pp_conformer_yue>`_

It uses 21.8k hours of training data.

.. hint::

   If you want a ``Cantonese`` ASR model, please choose this model
   or :ref:`sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09`

Huggingface space
~~~~~~~~~~~~~~~~~~

You can visit

  `<https://huggingface.co/spaces/k2-fsa/automatic-speech-recognition>`_

to try this model in your browser.

.. hint::

   You need to first select the language ``Cantonese``
   and then select the model  ``csukuangfj/sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10``.

Android APKs
~~~~~~~~~~~~~~~~~~

Real-time speech recognition Android APKs can be found at

  `<https://k2-fsa.github.io/sherpa/onnx/android/apk-simulate-streaming-asr.html>`_

.. hint::

   Please always download the latest version.

   Please search for ``wenetspeech_yue_u2pconformer_ctc_2025_09_10``.

Download
~~~~~~~~

Please use the following commands to download it::

  cd /path/to/sherpa-onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10.tar.bz2
  tar xf sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10.tar.bz2
  rm sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10.tar.bz2

After downloading, you should find the following files::

  ls -lh sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10/

  total 263264
  -rw-r--r--   1 fangjun  staff   129B Sep 10 14:18 README.md
  -rw-r--r--   1 fangjun  staff   128M Sep 10 14:18 model.int8.onnx
  drwxr-xr-x  22 fangjun  staff   704B Sep 10 14:18 test_wavs
  -rw-r--r--   1 fangjun  staff    83K Sep 10 14:18 tokens.txt

Real-time/Streaming Speech recognition from a microphone with VAD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block::

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

  ./build/bin/sherpa-onnx-vad-microphone-simulated-streaming-asr \
    --silero-vad-model=./silero_vad.onnx \
    --tokens=./{{model_path}}/tokens.txt \
    --wenet-ctc-model=./{{model_path}}/model.int8.onnx \
    --num-threads=1

Decode wave files
~~~~~~~~~~~~~~~~~~

{% for wav in wav_files %}
{{ wav.filename }}
{{ '"' * wav.filename|length }}

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Ground truth</th>
    </tr>
    <tr>
      <td>{{ wav.filename }}</td>
      <td>
       <audio title="{{ wav.filename }}" controls="controls">
             <source src="{{ wav.audio_src }}" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
      {{ wav.ground_truth }}
      </td>
    </tr>
  </table>

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline \
    --tokens=./{{model_path}}/tokens.txt \
    --wenet-ctc-model=./{{model_path}}/model.int8.onnx \
    --num-threads=1 \
    ./{{model_path}}/test_wavs/{{ wav.filename }}

.. literalinclude:: ./code/{{model_path}}/{{ wav.filename | replace(".wav", ".txt") }}

{% endfor %}

