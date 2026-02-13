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
  git clone https://github.com/k2-fsa/sherpa-onnx.git/
  cd sherpa-onnx

  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
  tar xvf sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
  rm sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2

  python3 ./python-api-examples/offline-sense-voice-ctc-decode-files.py



You should see something like below::

  ./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/test_wavs/zh.wav
  {"text": "开放时间早上9点至下午5点。", "timestamps": [0.72, 0.96, 1.26, 1.44, 1.92, 2.10, 2.58, 2.82, 3.30, 3.90, 4.20, 4.56, 4.74, 5.46], "tokens":["开", "放", "时", "间", "早", "上", "9", "点", "至", "下", "午", "5", "点", "。"], "words": []}

  (py38) fangjuns-MacBook-Pro:sherpa-onnx fangjun$ #python3 ./python-api-examples/offline-sense-voice-ctc-decode-files.py

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
    </tr>
    <tr>
      <td>zh.wav</td>
      <td>
       <audio title="zh.wav" controls="controls">
             <source src="/sherpa/_static/sense-voice/zh.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
    </tr>
  </table>

You can find ``offline-sense-voice-ctc-decode-files.py`` at the following address:

  `<https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/offline-sense-voice-ctc-decode-files.py>`_

Speech recognition from a microphone
------------------------------------

The following example shows how to use a microphone with `SenseVoice`_ and `silero-vad`_
for speech recognition::

  cd /tmp/sherpa-onnx

  # Assuem you have downloaded the SenseVoice model

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

  python3 ./python-api-examples/vad-with-non-streaming-asr.py  \
    --silero-vad-model=./silero_vad.onnx \
    --sense-voice=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.int8.onnx \
    --tokens=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt \
    --num-threads=2

You should see something like below::

    0 Background Music, Core Audio (2 in, 2 out)
    1 Background Music (UI Sounds), Core Audio (2 in, 2 out)
  > 2 MacBook Pro Microphone, Core Audio (1 in, 0 out)
  < 3 MacBook Pro Speakers, Core Audio (0 in, 2 out)
    4 WeMeet Audio Device, Core Audio (2 in, 2 out)
  Use default device: MacBook Pro Microphone
  Creating recognizer. Please wait...
  Started! Please speak

If you start speaking, you should see some output after you stop speaking.

.. hint::

   It starts speech recognition after `silero-vad`_ detects a pause.

Generate subtitles
------------------

This section describes how to use `SenseVoice`_ and  `silero-vad`_
to generate subtitles.

Chinese
^^^^^^^

Test with a wave file containing Chinese:

.. code-block:: bash

  cd /tmp/sherpa-onnx

  # Assuem you have downloaded the SenseVoice model

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

  # Assuem you have downloaded the SenseVoice model

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

WebSocket server and client example
-----------------------------------

This example shows how to use a WebSocket server with `SenseVoice`_ for speech recognition.

1. Start the server
^^^^^^^^^^^^^^^^^^^

Please run

.. code-block:: bash

   cd /tmp/sherpa-onnx

   # Assuem you have downloaded the SenseVoice model

   python3 ./python-api-examples/non_streaming_server.py \
     --sense-voice=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.int8.onnx \
     --tokens=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt

You should see the following output after starting the server::

  2024-07-28 20:22:38,389 INFO [non_streaming_server.py:1001] {'encoder': '', 'decoder': '', 'joiner': '', 'paraformer': '', 'sense_voice': './sherpa-o
  nnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.int8.onnx', 'nemo_ctc': '', 'wenet_ctc': '', 'tdnn_model': '', 'whisper_encoder': '', 'whisper_decod
  er': '', 'whisper_language': '', 'whisper_task': 'transcribe', 'whisper_tail_paddings': -1, 'tokens': './sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024
  -07-17/tokens.txt', 'num_threads': 2, 'provider': 'cpu', 'sample_rate': 16000, 'feat_dim': 80, 'decoding_method': 'greedy_search', 'max_active_paths'
  : 4, 'hotwords_file': '', 'hotwords_score': 1.5, 'blank_penalty': 0.0, 'port': 6006, 'max_batch_size': 3, 'max_wait_ms': 5, 'nn_pool_size': 1, 'max_m
  essage_size': 1048576, 'max_queue_size': 32, 'max_active_connections': 200, 'certificate': None, 'doc_root': './python-api-examples/web'}
  2024-07-28 20:22:41,861 INFO [non_streaming_server.py:647] started
  2024-07-28 20:22:41,861 INFO [non_streaming_server.py:659] No certificate provided
  2024-07-28 20:22:41,866 INFO [server.py:707] server listening on 0.0.0.0:6006
  2024-07-28 20:22:41,866 INFO [server.py:707] server listening on [::]:6006
  2024-07-28 20:22:41,866 INFO [non_streaming_server.py:679] Please visit one of the following addresses:

    http://localhost:6006

You can either visit the address `<http://localhost:6006>`_ or write code to interact with the server.

In the following, we describe possible approaches for interacting with the WebSocket server.

.. hint::

   The WebSocket server is able to handle multiple clients/connections at the same time.

2. Start the client (decode files sequentially)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following code sends the files in sequential one by one to the server for decoding.

.. code-block:: bash

   cd /tmp/sherpa-onnx

   python3 ./python-api-examples/offline-websocket-client-decode-files-sequential.py ./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/test_wavs/zh.wav  ./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/test_wavs/en.wav

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
    </tr>
    <tr>
      <td>zh.wav</td>
      <td>
       <audio title="zh.wav" controls="controls">
             <source src="/sherpa/_static/sense-voice/zh.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
    </tr>
    <tr>
      <td>en.wav</td>
      <td>
       <audio title="en.wav" controls="controls">
             <source src="/sherpa/_static/sense-voice/en.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
    </tr>
  </table>

You should see something like below on the server side::

  2024-07-28 20:28:15,749 INFO [server.py:642] connection open
  2024-07-28 20:28:15,749 INFO [non_streaming_server.py:835] Connected: ('::1', 53252, 0, 0). Number of connections: 1/200
  2024-07-28 20:28:15,933 INFO [non_streaming_server.py:851] result: 开放时间早上9点至下午5点。
  2024-07-28 20:28:16,194 INFO [non_streaming_server.py:851] result: The tribal chieftain called for the boy and presented him with 50 pieces of gold.
  2024-07-28 20:28:16,195 INFO [non_streaming_server.py:819] Disconnected: ('::1', 53252, 0, 0). Number of connections: 0/200
  2024-07-28 20:28:16,196 INFO [server.py:260] connection closed

You should see something like below on the client side::

  2024-07-28 20:28:15,750 INFO [offline-websocket-client-decode-files-sequential.py:114] Sending ./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/test_wavs/zh.wav
  开放时间早上9点至下午5点。
  2024-07-28 20:28:15,934 INFO [offline-websocket-client-decode-files-sequential.py:114] Sending ./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/test_wavs/en.wav
  The tribal chieftain called for the boy and presented him with 50 pieces of gold.

3. Start the client (decode files in parallel)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following code sends the files in parallel at the same time to the server for decoding.

.. code-block:: bash

   cd /tmp/sherpa-onnx

   python3 ./python-api-examples/offline-websocket-client-decode-files-paralell.py ./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/test_wavs/zh.wav  ./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/test_wavs/en.wav


.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
    </tr>
    <tr>
      <td>zh.wav</td>
      <td>
       <audio title="zh.wav" controls="controls">
             <source src="/sherpa/_static/sense-voice/zh.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
    </tr>
    <tr>
      <td>en.wav</td>
      <td>
       <audio title="en.wav" controls="controls">
             <source src="/sherpa/_static/sense-voice/en.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
    </tr>
  </table>
You should see something like below on the server side::

  2024-07-28 20:31:10,147 INFO [server.py:642] connection open
  2024-07-28 20:31:10,148 INFO [non_streaming_server.py:835] Connected: ('::1', 53436, 0, 0). Number of connections: 1/200
  2024-07-28 20:31:10,149 INFO [server.py:642] connection open
  2024-07-28 20:31:10,149 INFO [non_streaming_server.py:835] Connected: ('::1', 53437, 0, 0). Number of connections: 2/200
  2024-07-28 20:31:10,353 INFO [non_streaming_server.py:851] result: 开放时间早上9点至下午5点。
  2024-07-28 20:31:10,354 INFO [non_streaming_server.py:819] Disconnected: ('::1', 53436, 0, 0). Number of connections: 1/200
  2024-07-28 20:31:10,356 INFO [server.py:260] connection closed
  2024-07-28 20:31:10,541 INFO [non_streaming_server.py:851] result: The tribal chieftain called for the boy and presented him with 50 pieces of gold.
  2024-07-28 20:31:10,542 INFO [non_streaming_server.py:819] Disconnected: ('::1', 53437, 0, 0). Number of connections: 0/200
  2024-07-28 20:31:10,544 INFO [server.py:260] connection closed

You should see something like below on the client side::

  2024-07-28 20:31:10,112 INFO [offline-websocket-client-decode-files-paralell.py:139] {'server_addr': 'localhost', 'server_port': 6006, 'sound_files': ['./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/test_wavs/zh.wav', './sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/test_wavs/en.wav']}
  2024-07-28 20:31:10,148 INFO [offline-websocket-client-decode-files-paralell.py:113] Sending ./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/test_wavs/zh.wav
  2024-07-28 20:31:10,191 INFO [offline-websocket-client-decode-files-paralell.py:113] Sending ./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/test_wavs/en.wav
  2024-07-28 20:31:10,353 INFO [offline-websocket-client-decode-files-paralell.py:131] ./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/test_wavs/zh.wav
  开放时间早上9点至下午5点。
  2024-07-28 20:31:10,542 INFO [offline-websocket-client-decode-files-paralell.py:131] ./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/test_wavs/en.wav
  The tribal chieftain called for the boy and presented him with 50 pieces of gold.

4. Start the  Web browser client
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can also start a browser to interact with the WebSocket server.

Please visit `<http://localhost:6006>`_.

.. warning::

   We are not using a certificate to start the server, so the only
   ``correct`` URL is `<http://localhost:6006>`_.

   All of the following addresses are ``incorrect``:

    - Incorrect/Wrong address: `<https://localhost:6006>`_
    - Incorrect/Wrong address: `<http://127.0.0.1:6006>`_
    - Incorrect/Wrong address: `<https://127.0.0.1:6006>`_
    - Incorrect/Wrong address: `<http://a.b.c.d:6006>`_
    - Incorrect/Wrong address: `<https://a.b.c.d:6006>`_

After starting the browser, you should see the following page:

  .. image:: ./pic/python-websocket/client-1.jpg
     :align: center
     :width: 600

Upload a file for recognition
:::::::::::::::::::::::::::::

If we click ``Upload``, we will see the following page:

  .. image:: ./pic/python-websocket/client-2.jpg
     :align: center
     :width: 600

After clicking ``Click me to connect`` and ``Choose File``, you will
see the recognition result returned from the server:

  .. image:: ./pic/python-websocket/client-3.jpg
     :align: center
     :width: 600

Record your speech with a microphone for recognition
::::::::::::::::::::::::::::::::::::::::::::::::::::

If you click ``Offline-Record``, you should see the following page:

  .. image:: ./pic/python-websocket/client-4.jpg
     :align: center
     :width: 600

Please click the button ``Click me to connect``, and then click the button
``Offline-Record``, then speak, finally, click the button ``Offline-Stop``;

you should see the results from the server. A screenshot is given below:

  .. image:: ./pic/python-websocket/client-5.jpg
     :align: center
     :width: 600

Note that you can save the recorded audio into a wave file for debugging.

The recorded audio from the above screenshot is saved to ``test.wav`` and
is given below::

  Input File     : 'test.wav'
  Channels       : 1
  Sample Rate    : 16000
  Precision      : 16-bit
  Duration       : 00:00:07.00 = 112012 samples ~ 525.056 CDDA sectors
  File Size      : 224k
  Bit Rate       : 256k
  Sample Encoding: 16-bit Signed Integer PCM

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
    </tr>
    <tr>
      <td>test.wav</td>
      <td>
       <audio title="test.wav" controls="controls">
             <source src="/sherpa/_static/sense-voice/python-websocket/test.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
    </tr>
  </table>
