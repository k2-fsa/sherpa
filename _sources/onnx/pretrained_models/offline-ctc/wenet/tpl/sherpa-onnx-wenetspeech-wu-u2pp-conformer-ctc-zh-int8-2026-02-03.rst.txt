.. _sherpa-onnx-wenetspeech-wu-u2pp-conformer-ctc-zh-int8-2026-02-03:

sherpa-onnx-wenetspeech-wu-u2pp-conformer-ctc-zh-int8-2026-02-03 (吴语)
------------------------------------------------------------------------------------------------

This model is converted from

  `<https://huggingface.co/ASLP-lab/WenetSpeech-Wu-Speech-Understanding/tree/main/u2++>`_

It uses 8k hours of training data.

It supports Shanghainese, Suzhounese, Shaoxingnese, Ningbonese, Hangzhounese, Jiaxingnese, Taizhounese, and Wenzhounese.

.. hint::

   该模型支持

    1. 普通话
    2. 上海话
    3. 苏州话
    4. 绍兴话
    5. 宁波话
    6. 杭州话
    7. 嘉兴话
    8. 台州话
    9. 温州话

Huggingface space
~~~~~~~~~~~~~~~~~~

You can visit

  `<https://huggingface.co/spaces/k2-fsa/automatic-speech-recognition>`_

to try this model in your browser.

.. hint::

   You need to first select the language ``吴语``
   and then select the model  ``csukuangfj2/sherpa-onnx-wenetspeech-wu-u2pp-conformer-ctc-zh-int8-2026-02-03``.

Android APKs
~~~~~~~~~~~~~~~~~~

Real-time speech recognition Android APKs can be found at

  `<https://k2-fsa.github.io/sherpa/onnx/android/apk-simulate-streaming-asr.html>`_

.. hint::

   Please always download the latest version.

   Please search for ``wu-wenetspeech_wu_u2pconformer_ctc_2026_02_03``.

Download
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it::

  cd /path/to/sherpa-onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-wenetspeech-wu-u2pp-conformer-ctc-zh-int8-2026-02-03.tar.bz2
  tar xf sherpa-onnx-wenetspeech-wu-u2pp-conformer-ctc-zh-int8-2026-02-03.tar.bz2
  rm sherpa-onnx-wenetspeech-wu-u2pp-conformer-ctc-zh-int8-2026-02-03.tar.bz2

After downloading, you should find the following files::

  ls -lh sherpa-onnx-wenetspeech-wu-u2pp-conformer-ctc-zh-int8-2026-02-03/
  total 264120
  -rw-r--r--@  1 fangjun  staff   127M  3 Feb 18:44 model.int8.onnx
  -rw-r--r--@  1 fangjun  staff   239B  3 Feb 18:44 README.md
  drwxr-xr-x@ 27 fangjun  staff   864B  3 Feb 18:44 test_wavs
  -rw-r--r--@  1 fangjun  staff    51K  3 Feb 18:44 tokens.txt

  ls -lh sherpa-onnx-wenetspeech-wu-u2pp-conformer-ctc-zh-int8-2026-02-03/test_wavs/
  total 10888
  -rw-r--r--@ 1 fangjun  staff   184K  3 Feb 18:44 1.wav
  -rw-r--r--@ 1 fangjun  staff   238K  3 Feb 18:44 10.wav
  -rw-r--r--@ 1 fangjun  staff   228K  3 Feb 18:44 11.wav
  -rw-r--r--@ 1 fangjun  staff   179K  3 Feb 18:44 12.wav
  -rw-r--r--@ 1 fangjun  staff   214K  3 Feb 18:44 13.wav
  -rw-r--r--@ 1 fangjun  staff   374K  3 Feb 18:44 14.wav
  -rw-r--r--@ 1 fangjun  staff   383K  3 Feb 18:44 15.wav
  -rw-r--r--@ 1 fangjun  staff   181K  3 Feb 18:44 16.wav
  -rw-r--r--@ 1 fangjun  staff   181K  3 Feb 18:44 17.wav
  -rw-r--r--@ 1 fangjun  staff   186K  3 Feb 18:44 18.wav
  -rw-r--r--@ 1 fangjun  staff   181K  3 Feb 18:44 19.wav
  -rw-r--r--@ 1 fangjun  staff   183K  3 Feb 18:44 2.wav
  -rw-r--r--@ 1 fangjun  staff   238K  3 Feb 18:44 20.wav
  -rw-r--r--@ 1 fangjun  staff   193K  3 Feb 18:44 21.wav
  -rw-r--r--@ 1 fangjun  staff   184K  3 Feb 18:44 22.wav
  -rw-r--r--@ 1 fangjun  staff   264K  3 Feb 18:44 23.wav
  -rw-r--r--@ 1 fangjun  staff   180K  3 Feb 18:44 24.wav
  -rw-r--r--@ 1 fangjun  staff   251K  3 Feb 18:44 3.wav
  -rw-r--r--@ 1 fangjun  staff   229K  3 Feb 18:44 4.wav
  -rw-r--r--@ 1 fangjun  staff   257K  3 Feb 18:44 5.wav
  -rw-r--r--@ 1 fangjun  staff   218K  3 Feb 18:44 6.wav
  -rw-r--r--@ 1 fangjun  staff   241K  3 Feb 18:44 7.wav
  -rw-r--r--@ 1 fangjun  staff   183K  3 Feb 18:44 8.wav
  -rw-r--r--@ 1 fangjun  staff   234K  3 Feb 18:44 9.wav
  -rw-r--r--@ 1 fangjun  staff   2.0K  3 Feb 18:44 transcript.txt

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

