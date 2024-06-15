Non-Streaming WebSocket Server
==============================

This section describes how to use the Python non-streaming WebSocket server
of `sherpa-onnx`_ for speech recognition.

.. hint::

    The server supports multiple clients connecting at the same time.

The code for the non-streaming server can be found at

  `<https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/non_streaming_server.py>`_

Please refer to :ref:`sherpa-onnx-pre-trained-models` to download a non-streaming model
before you continue.

We use the following types of models for demonstration.

.. list-table::

 * - Description
   - URL
 * - Non-streaming transducer
   - :ref:`sherpa-onnx-zipformer-en-2023-06-26-english`
 * - Non-streaming paraformer
   - :ref:`sherpa_onnx_offline_paraformer_zh_2023_03_28_chinese`
 * - Non-streaming CTC model from NeMo
   - :ref:`stt-en-conformer-ctc-medium-nemo-sherpa-onnx`
 * - Non-streaming Whisper tiny.en
   - :ref:`whisper_tiny_en_sherpa_onnx`

Non-streaming transducer
------------------------

Start the server
^^^^^^^^^^^^^^^^^

.. code-block:: bash

  cd /path/to/sherpa-onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-en-2023-06-26.tar.bz2
  tar xvf sherpa-onnx-zipformer-en-2023-06-26.tar.bz2
  rm sherpa-onnx-zipformer-en-2023-06-26.tar.bz2

  python3 ./python-api-examples/non_streaming_server.py \
    --encoder ./sherpa-onnx-zipformer-en-2023-06-26/encoder-epoch-99-avg-1.onnx \
    --decoder ./sherpa-onnx-zipformer-en-2023-06-26/decoder-epoch-99-avg-1.onnx \
    --joiner ./sherpa-onnx-zipformer-en-2023-06-26/joiner-epoch-99-avg-1.onnx \
    --tokens ./sherpa-onnx-zipformer-en-2023-06-26/tokens.txt \
    --port 6006

Start the client
^^^^^^^^^^^^^^^^

**Decode multiple files in parallel**

.. code-block:: bash

  python3 ./python-api-examples/offline-websocket-client-decode-files-paralell.py \
    --server-addr localhost \
    --server-port 6006 \
    ./sherpa-onnx-zipformer-en-2023-06-26/test_wavs/0.wav \
    ./sherpa-onnx-zipformer-en-2023-06-26/test_wavs/1.wav \
    ./sherpa-onnx-zipformer-en-2023-06-26/test_wavs/8k.wav

You should see the following output:

.. code-block:: bash

  2023-08-11 18:19:26,000 INFO [offline-websocket-client-decode-files-paralell.py:139] {'server_addr': 'localhost', 'server_port': 6006, 'sound_files': ['./sherpa-onnx-zipformer-en-2023-06-26/test_wavs/0.wav', './sherpa-onnx-zipformer-en-2023-06-26/test_wavs/1.wav', './sherpa-onnx-zipformer-en-2023-06-26/test_wavs/8k.wav']}
  2023-08-11 18:19:26,034 INFO [offline-websocket-client-decode-files-paralell.py:113] Sending ./sherpa-onnx-zipformer-en-2023-06-26/test_wavs/8k.wav
  2023-08-11 18:19:26,058 INFO [offline-websocket-client-decode-files-paralell.py:113] Sending ./sherpa-onnx-zipformer-en-2023-06-26/test_wavs/1.wav
  2023-08-11 18:19:26,205 INFO [offline-websocket-client-decode-files-paralell.py:113] Sending ./sherpa-onnx-zipformer-en-2023-06-26/test_wavs/0.wav
  2023-08-11 18:19:26,262 INFO [offline-websocket-client-decode-files-paralell.py:131] ./sherpa-onnx-zipformer-en-2023-06-26/test_wavs/8k.wav
   YET THESE THOUGHTS AFFECTED HESTER PRYNNE LESS WITH HOPE THAN APPREHENSION
  2023-08-11 18:19:26,609 INFO [offline-websocket-client-decode-files-paralell.py:131] ./sherpa-onnx-zipformer-en-2023-06-26/test_wavs/1.wav
   GOD AS A DIRECT CONSEQUENCE OF THE SIN WHICH MAN THUS PUNISHED HAD GIVEN HER A LOVELY CHILD WHOSE PLACE WAS ON THAT SAME DISHONORED BOSOM TO CONNECT HER PARENT FOREVER WITH THE RACE AND DESCENT OF MORTALS AND TO BE FINALLY A BLESSED SOUL IN HEAVEN
  2023-08-11 18:19:26,773 INFO [offline-websocket-client-decode-files-paralell.py:131] ./sherpa-onnx-zipformer-en-2023-06-26/test_wavs/0.wav
   AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE THE SQUALID QUARTER OF THE BROTHELS

**Decode multiple files sequentially**

.. code-block:: bash

  python3 ./python-api-examples/offline-websocket-client-decode-files-sequential.py \
    --server-addr localhost \
    --server-port 6006 \
    ./sherpa-onnx-zipformer-en-2023-06-26/test_wavs/0.wav \
    ./sherpa-onnx-zipformer-en-2023-06-26/test_wavs/1.wav \
    ./sherpa-onnx-zipformer-en-2023-06-26/test_wavs/8k.wav

You should see the following output:

.. code-block:: bash

  2023-08-11 18:20:36,677 INFO [offline-websocket-client-decode-files-sequential.py:114] Sending ./sherpa-onnx-zipformer-en-2023-06-26/test_wavs/0.wav
   AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE THE SQUALID QUARTER OF THE BROTHELS
  2023-08-11 18:20:36,861 INFO [offline-websocket-client-decode-files-sequential.py:114] Sending ./sherpa-onnx-zipformer-en-2023-06-26/test_wavs/1.wav
   GOD AS A DIRECT CONSEQUENCE OF THE SIN WHICH MAN THUS PUNISHED HAD GIVEN HER A LOVELY CHILD WHOSE PLACE WAS ON THAT SAME DISHONORED BOSOM TO CONNECT HER PARENT FOREVER WITH THE RACE AND DESCENT OF MORTALS AND TO BE FINALLY A BLESSED SOUL IN HEAVEN
  2023-08-11 18:20:37,375 INFO [offline-websocket-client-decode-files-sequential.py:114] Sending ./sherpa-onnx-zipformer-en-2023-06-26/test_wavs/8k.wav
   YET THESE THOUGHTS AFFECTED HESTER PRYNNE LESS WITH HOPE THAN APPREHENSION

Non-streaming paraformer
------------------------

Start the server
^^^^^^^^^^^^^^^^

.. code-block:: bash

  cd /path/to/sherpa-onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2023-03-28.tar.bz2
  tar xvf sherpa-onnx-paraformer-zh-2023-03-28.tar.bz2
  rm sherpa-onnx-paraformer-zh-2023-03-28.tar.bz2

  python3 ./python-api-examples/non_streaming_server.py \
    --paraformer ./sherpa-onnx-paraformer-zh-2023-03-28/model.int8.onnx \
    --tokens ./sherpa-onnx-paraformer-zh-2023-03-28/tokens.txt \
    --port 6006

Start the client
^^^^^^^^^^^^^^^^

**Decode multiple files in parallel**

.. code-block:: bash

    python3 ./python-api-examples/offline-websocket-client-decode-files-paralell.py \
      --server-addr localhost \
      --server-port 6006 \
      ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/0.wav \
      ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/1.wav \
      ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/2.wav \
      ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/8k.wav


You should see the following output:

.. code-block:: bash

  2023-08-11 18:22:54,189 INFO [offline-websocket-client-decode-files-paralell.py:113] Sending ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/0.wav
  2023-08-11 18:22:54,233 INFO [offline-websocket-client-decode-files-paralell.py:113] Sending ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/1.wav
  2023-08-11 18:22:54,275 INFO [offline-websocket-client-decode-files-paralell.py:113] Sending ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/8k.wav
  2023-08-11 18:22:54,295 INFO [offline-websocket-client-decode-files-paralell.py:113] Sending ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/2.wav
  2023-08-11 18:22:54,380 INFO [offline-websocket-client-decode-files-paralell.py:131] ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/0.wav
  对我做了介绍啊那么我想说的是呢大家如果对我的研究感兴趣呢你
  2023-08-11 18:22:54,673 INFO [offline-websocket-client-decode-files-paralell.py:131] ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/8k.wav
  甚至出现交易几乎停滞的情况
  2023-08-11 18:22:54,673 INFO [offline-websocket-client-decode-files-paralell.py:131] ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/2.wav
  深入的分析这一次全球金融动荡背后的根源
  2023-08-11 18:22:54,674 INFO [offline-websocket-client-decode-files-paralell.py:131] ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/1.wav
  重点呢想谈三个问题首先呢就是这一轮全球金融动荡的表现

**Decode multiple files sequentially**

.. code-block:: bash

  python3 ./python-api-examples/offline-websocket-client-decode-files-sequential.py \
    --server-addr localhost \
    --server-port 6006 \
    ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/0.wav \
    ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/1.wav \
    ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/2.wav \
    ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/8k.wav

You should see the following output:

.. code-block:: bash

  2023-08-11 18:24:32,678 INFO [offline-websocket-client-decode-files-sequential.py:141] {'server_addr': 'localhost', 'server_port': 6006, 'sound_files': ['./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/0.wav', './sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/1.wav', './sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/2.wav', './sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/8k.wav']}
  2023-08-11 18:24:32,709 INFO [offline-websocket-client-decode-files-sequential.py:114] Sending ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/0.wav
  对我做了介绍啊那么我想说的是呢大家如果对我的研究感兴趣呢你
  2023-08-11 18:24:32,883 INFO [offline-websocket-client-decode-files-sequential.py:114] Sending ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/1.wav
  重点呢想谈三个问题首先呢就是这一轮全球金融动荡的表现
  2023-08-11 18:24:33,042 INFO [offline-websocket-client-decode-files-sequential.py:114] Sending ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/2.wav
  深入的分析这一次全球金融动荡背后的根源
  2023-08-11 18:24:33,175 INFO [offline-websocket-client-decode-files-sequential.py:114] Sending ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/8k.wav
  甚至出现交易几乎停滞的情况

Non-streaming CTC model from NeMo
---------------------------------

Start the server
^^^^^^^^^^^^^^^^

.. code-block:: bash

  cd /path/to/sherpa-onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-ctc-en-conformer-medium.tar.bz2
  tar xvf sherpa-onnx-nemo-ctc-en-conformer-medium.tar.bz2
  rm sherpa-onnx-nemo-ctc-en-conformer-medium.tar.bz2

  python3 ./python-api-examples/non_streaming_server.py \
    --nemo-ctc ./sherpa-onnx-nemo-ctc-en-conformer-medium/model.onnx \
    --tokens ./sherpa-onnx-nemo-ctc-en-conformer-medium/tokens.txt \
    --port 6006

Start the client
^^^^^^^^^^^^^^^^

**Decode multiple files in parallel**

.. code-block:: bash

  python3 ./python-api-examples/offline-websocket-client-decode-files-paralell.py \
    --server-addr localhost \
    --server-port 6006 \
    ./sherpa-onnx-nemo-ctc-en-conformer-medium/test_wavs/0.wav \
    ./sherpa-onnx-nemo-ctc-en-conformer-medium/test_wavs/1.wav \
    ./sherpa-onnx-nemo-ctc-en-conformer-medium/test_wavs/8k.wav

You should see the following output:

.. code-block:: bash

  2023-08-11 18:31:32,432 INFO [offline-websocket-client-decode-files-paralell.py:139] {'server_addr': 'localhost', 'server_port': 6006, 'sound_files': ['./sherpa-onnx-nemo-ctc-en-conformer-medium/test_wavs/0.wav', './sherpa-onnx-nemo-ctc-en-conformer-medium/test_wavs/1.wav', './sherpa-onnx-nemo-ctc-en-conformer-medium/test_wavs/8k.wav']}
  2023-08-11 18:31:32,462 INFO [offline-websocket-client-decode-files-paralell.py:113] Sending ./sherpa-onnx-nemo-ctc-en-conformer-medium/test_wavs/0.wav
  2023-08-11 18:31:32,513 INFO [offline-websocket-client-decode-files-paralell.py:113] Sending ./sherpa-onnx-nemo-ctc-en-conformer-medium/test_wavs/8k.wav
  2023-08-11 18:31:32,533 INFO [offline-websocket-client-decode-files-paralell.py:113] Sending ./sherpa-onnx-nemo-ctc-en-conformer-medium/test_wavs/1.wav
  2023-08-11 18:31:32,670 INFO [offline-websocket-client-decode-files-paralell.py:131] ./sherpa-onnx-nemo-ctc-en-conformer-medium/test_wavs/0.wav
   after early nightfall the yellow lamps would light up here and there the squalid quarter of the brothels
  2023-08-11 18:31:32,741 INFO [offline-websocket-client-decode-files-paralell.py:131] ./sherpa-onnx-nemo-ctc-en-conformer-medium/test_wavs/8k.wav
   yet these thoughts affected hester pryne less with hope than apprehension
  2023-08-11 18:31:33,117 INFO [offline-websocket-client-decode-files-paralell.py:131] ./sherpa-onnx-nemo-ctc-en-conformer-medium/test_wavs/1.wav
   god as a direct consequence of the sin which man thus punished had given her a lovely child whose place was on that same dishonored bosom to connect her parent for ever with the race and descent of mortals and to be finally a blessed soul in heaven

**Decode multiple files sequentially**

.. code-block:: bash

  python3 ./python-api-examples/offline-websocket-client-decode-files-sequential.py \
    --server-addr localhost \
    --server-port 6006 \
    ./sherpa-onnx-nemo-ctc-en-conformer-medium/test_wavs/0.wav \
    ./sherpa-onnx-nemo-ctc-en-conformer-medium/test_wavs/1.wav \
    ./sherpa-onnx-nemo-ctc-en-conformer-medium/test_wavs/8k.wav

You should see the following output:

.. code-block:: bash

  2023-08-11 18:33:14,520 INFO [offline-websocket-client-decode-files-sequential.py:141] {'server_addr': 'localhost', 'server_port': 6006, 'sound_files': ['./sherpa-onnx-nemo-ctc-en-conformer-medium/test_wavs/0.wav', './sherpa-onnx-nemo-ctc-en-conformer-medium/test_wavs/1.wav', './sherpa-onnx-nemo-ctc-en-conformer-medium/test_wavs/8k.wav']}
  2023-08-11 18:33:14,547 INFO [offline-websocket-client-decode-files-sequential.py:114] Sending ./sherpa-onnx-nemo-ctc-en-conformer-medium/test_wavs/0.wav
   after early nightfall the yellow lamps would light up here and there the squalid quarter of the brothels
  2023-08-11 18:33:14,716 INFO [offline-websocket-client-decode-files-sequential.py:114] Sending ./sherpa-onnx-nemo-ctc-en-conformer-medium/test_wavs/1.wav
   god as a direct consequence of the sin which man thus punished had given her a lovely child whose place was on that same dishonored bosom to connect her parent for ever with the race and descent of mortals and to be finally a blessed soul in heaven
  2023-08-11 18:33:15,218 INFO [offline-websocket-client-decode-files-sequential.py:114] Sending ./sherpa-onnx-nemo-ctc-en-conformer-medium/test_wavs/8k.wav
   yet these thoughts affected hester pryne less with hope than apprehension

Non-streaming Whisper tiny.en
-----------------------------

Start the server
^^^^^^^^^^^^^^^^^

.. code-block:: bash

  cd /path/to/sherpa-onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.en.tar.bz2
  tar xvf sherpa-onnx-whisper-tiny.en.tar.bz2
  rm sherpa-onnx-whisper-tiny.en.tar.bz2

  python3 ./python-api-examples/non_streaming_server.py \
    --whisper-encoder=./sherpa-onnx-whisper-tiny.en/tiny.en-encoder.onnx \
    --whisper-decoder=./sherpa-onnx-whisper-tiny.en/tiny.en-decoder.onnx \
    --tokens=./sherpa-onnx-whisper-tiny.en/tiny.en-tokens.txt \
    --port 6006

Start the client
^^^^^^^^^^^^^^^^

**Decode multiple files in parallel**

.. code-block:: bash

    python3 ./python-api-examples/offline-websocket-client-decode-files-paralell.py \
      --server-addr localhost \
      --server-port 6006 \
      ./sherpa-onnx-whisper-tiny.en/test_wavs/0.wav \
      ./sherpa-onnx-whisper-tiny.en/test_wavs/1.wav \
      ./sherpa-onnx-whisper-tiny.en/test_wavs/8k.wav

You should see the following output:

.. code-block:: bash

  2023-08-11 18:35:28,866 INFO [offline-websocket-client-decode-files-paralell.py:139] {'server_addr': 'localhost', 'server_port': 6006, 'sound_files': ['./sherpa-onnx-whisper-tiny.en/test_wavs/0.wav', './sherpa-onnx-whisper-tiny.en/test_wavs/1.wav', './sherpa-onnx-whisper-tiny.en/test_wavs/8k.wav']}
  2023-08-11 18:35:28,894 INFO [offline-websocket-client-decode-files-paralell.py:113] Sending ./sherpa-onnx-whisper-tiny.en/test_wavs/0.wav
  2023-08-11 18:35:28,947 INFO [offline-websocket-client-decode-files-paralell.py:113] Sending ./sherpa-onnx-whisper-tiny.en/test_wavs/1.wav
  2023-08-11 18:35:29,082 INFO [offline-websocket-client-decode-files-paralell.py:113] Sending ./sherpa-onnx-whisper-tiny.en/test_wavs/8k.wav
  2023-08-11 18:35:29,754 INFO [offline-websocket-client-decode-files-paralell.py:131] ./sherpa-onnx-whisper-tiny.en/test_wavs/0.wav
   After early nightfall, the yellow lamps would light up here and there, the squalid quarter of the brothels.
  2023-08-11 18:35:30,276 INFO [offline-websocket-client-decode-files-paralell.py:131] ./sherpa-onnx-whisper-tiny.en/test_wavs/8k.wav
   Yet these thoughts affected Hester Prin less with hope than apprehension.
  2023-08-11 18:35:31,592 INFO [offline-websocket-client-decode-files-paralell.py:131] ./sherpa-onnx-whisper-tiny.en/test_wavs/1.wav
   God, as a direct consequence of the sin which man thus punished, had given her a lovely child, whose place was on that same dishonored bosom to connect her parent forever with the race and descent of mortals, and to be finally a blessed soul in heaven.

**Decode multiple files sequentially**

.. code-block:: bash

  python3 ./python-api-examples/offline-websocket-client-decode-files-sequential.py \
    --server-addr localhost \
    --server-port 6006 \
    ./sherpa-onnx-whisper-tiny.en/test_wavs/0.wav \
    ./sherpa-onnx-whisper-tiny.en/test_wavs/1.wav \
    ./sherpa-onnx-whisper-tiny.en/test_wavs/8k.wav

You should see the following output:

.. code-block:: bash

  2023-08-11 18:36:42,148 INFO [offline-websocket-client-decode-files-sequential.py:141] {'server_addr': 'localhost', 'server_port': 6006, 'sound_files': ['./sherpa-onnx-whisper-tiny.en/test_wavs/0.wav', './sherpa-onnx-whisper-tiny.en/test_wavs/1.wav', './sherpa-onnx-whisper-tiny.en/test_wavs/8k.wav']}
  2023-08-11 18:36:42,176 INFO [offline-websocket-client-decode-files-sequential.py:114] Sending ./sherpa-onnx-whisper-tiny.en/test_wavs/0.wav
   After early nightfall, the yellow lamps would light up here and there, the squalid quarter of the brothels.
  2023-08-11 18:36:42,926 INFO [offline-websocket-client-decode-files-sequential.py:114] Sending ./sherpa-onnx-whisper-tiny.en/test_wavs/1.wav
   God, as a direct consequence of the sin which man thus punished, had given her a lovely child, whose place was on that same dishonored bosom to connect her parent forever with the race and descent of mortals, and to be finally a blessed soul in heaven.
  2023-08-11 18:36:44,314 INFO [offline-websocket-client-decode-files-sequential.py:114] Sending ./sherpa-onnx-whisper-tiny.en/test_wavs/8k.wav
   Yet these thoughts affected Hester Prin less with hope than apprehension.

colab
-----

We provide a colab notebook
|Sherpa-onnx python non-streaming websocket example colab notebook|
for you to try the Python non-streaming websocket server example of `sherpa-onnx`_.

.. |Sherpa-onnx python non-streaming websocket example colab notebook| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://github.com/k2-fsa/colab/blob/master/sherpa-onnx/sherpa_onnx_python_non_streaming_websocket_server.ipynb
