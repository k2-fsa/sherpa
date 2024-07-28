Python API for SenseVoice
=========================

This page describes how to use the Python API for `SenseVoice`_.

Please refer to :ref:`install_sherpa_onnx_python` for how to install the Python package
of `sherpa-onnx`_.

The following is a quick way to do that::

  pip install sherpa-onnx

Decode a file
-------------

After installing the Python package, you can download the Python example code and run it with
the following commands::

  cd /tmp
  git clone http://github.com/k2-fsa/sherpa-onnx
  cd sherpa-onnx

  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
  tar xvf sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
  rm sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2

  python3 ./python-api-examples/offline-sense-voice-ctc-decode-files.py

You should see something like below::

  ./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/test_wavs/zh.wav
  {"text": "开饭时间早上9点至下午5点。", "timestamps": [0.72, 0.96, 1.26, 1.44, 1.92, 2.10, 2.58, 2.82, 3.30, 3.90, 4.20, 4.56, 4.74, 5.46], "tokens":["开", "饭", "时", "间", "早", "上", "9", "点", "至", "下", "午", "5", "点", "。"], "words": []}
  (py38) fangjuns-MacBook-Pro:sherpa-onnx fangjun$ #python3 ./python-api-examples/offline-sense-voice-ctc-decode-files.py

You can find ``offline-sense-voice-ctc-decode-files.py`` at the following address:

  `<https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/offline-sense-voice-ctc-decode-files.py>`_

Generate subtitles
------------------

This section describes how to use `SenseVoice`_ and  `silero-vad`_
to generate subtitles.

Chinese
^^^^^^^

Test with a wave file containing Chinese:

.. code-block:: bash

  cd /tmp/sherpa-onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/lei-jun-test.wav

  python3 ./python-api-examples/generate-subtitles.py \
    --silero-vad-model=./silero_vad.onnx \
    --sense-voice=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.onnx \
    --tokens=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt \
    --num-threads=2 \
    ./lei-jun-test.wav

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

It will generate a text file ``lei-jun-test.srt``, which is given below:


.. container:: toggle

    .. container:: header

      Click ▶ to see ``lei-jun-test.srt``.

    .. literalinclude:: ./code/lei-jun-test.srt

English
^^^^^^^

Test with a wave file containing English:

.. code-block:: bash

  cd /tmp/sherpa-onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/Obama.wav

  python3 ./python-api-examples/generate-subtitles.py \
    --silero-vad-model=./silero_vad.onnx \
    --sense-voice=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.onnx \
    --tokens=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt \
    --num-threads=2 \
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

It will generate a text file ``Obama.srt``, which is given below:

.. container:: toggle

    .. container:: header

      Click ▶ to see ``Obama.srt``.

    .. literalinclude:: ./code/Obama.srt
