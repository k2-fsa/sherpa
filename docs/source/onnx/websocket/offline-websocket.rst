.. _onnx_non_streaming_websocket_server_and_client:

Non-streaming WebSocket server and client
=========================================

.. hint::

   Please refer to :ref:`install_sherpa_onnx` to install `sherpa-onnx`_
   before you read this section.

Build `sherpa-onnx` with WebSocket support
------------------------------------------

By default, it will generate the following binaries after :ref:`install_sherpa_onnx`:

.. code-block:: bash

  sherpa-onnx fangjun$ ls -lh build/bin/*websocket*
  -rwxr-xr-x  1 fangjun  staff   1.1M Mar 31 22:09 build/bin/sherpa-onnx-offline-websocket-server
  -rwxr-xr-x  1 fangjun  staff   1.0M Mar 31 22:09 build/bin/sherpa-onnx-online-websocket-client
  -rwxr-xr-x  1 fangjun  staff   1.2M Mar 31 22:09 build/bin/sherpa-onnx-online-websocket-server

Please refer to :ref:`onnx_streaming_websocket_server_and_client`
for the usage of ``sherpa-onnx-online-websocket-server``
and ``sherpa-onnx-online-websocket-client``.

View the server usage
---------------------

Before starting the server, let us view the help message of ``sherpa-onnx-offline-websocket-server``:

.. code-block:: bash

  build/bin/sherpa-onnx-offline-websocket-server

The above command will print the following help information:

.. literalinclude:: ./code/sherpa-onnx-offline-websocket-server-help.txt

Start the server
----------------

.. hint::

  Please refer to :ref:`sherpa-onnx-pre-trained-models`
  for a list of pre-trained models.

Start the server with a transducer model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline-websocket-server \
    --port=6006 \
    --num-work-threads=5 \
    --tokens=./sherpa-onnx-zipformer-en-2023-03-30/tokens.txt \
    --encoder=./sherpa-onnx-zipformer-en-2023-03-30/encoder-epoch-99-avg-1.onnx \
    --decoder=./sherpa-onnx-zipformer-en-2023-03-30/decoder-epoch-99-avg-1.onnx \
    --joiner=./sherpa-onnx-zipformer-en-2023-03-30/joiner-epoch-99-avg-1.onnx \
    --log-file=./log.txt \
    --max-batch-size=5

.. caution::

   The arguments are in the form ``--key=value``.

   It does not support ``--key value``.

   It does not support ``--key value``.

   It does not support ``--key value``.

.. hint::

   In the above demo, the model files are
   from :ref:`sherpa_onnx_zipformer_en_2023_03_30`.

.. note::

  Note that the server supports processing multiple clients in a batch in parallel.
  You can use ``--max-batch-size`` to limit the batch size.

Start the server with a paraformer model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline-websocket-server \
    --port=6006 \
    --num-work-threads=5 \
    --tokens=./sherpa-onnx-paraformer-zh-2023-03-28/tokens.txt \
    --paraformer=./sherpa-onnx-paraformer-zh-2023-03-28/model.onnx \
    --log-file=./log.txt \
    --max-batch-size=5

.. hint::

   In the above demo, the model files are
   from :ref:`sherpa_onnx_offline_paraformer_zh_2023_03_28_chinese`.

Start the client (Python)
-------------------------

We provide two clients written in Python:

  - `offline-websocket-client-decode-files-paralell.py <https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/offline-websocket-client-decode-files-paralell.py>`_: It decodes multiple files in parallel
    by creating a separate connection for each file
  - `offline-websocket-client-decode-files-sequential.py <https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/offline-websocket-client-decode-files-sequential.py>`_: It decodes multiple files sequentially
    by creating only a single connection

offline-websocket-client-decode-files-paralell.py
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  python3 ./python-api-examples/offline-websocket-client-decode-files-paralell.py \
    --server-addr localhost \
    --server-port 6006 \
    ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/0.wav \
    ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/1.wav \
    ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/2.wav \
    ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/8k.wav

offline-websocket-client-decode-files-sequential.py
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  python3 ./python-api-examples/offline-websocket-client-decode-files-sequential.py \
    --server-addr localhost \
    --server-port 6006 \
    ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/0.wav \
    ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/1.wav \
    ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/2.wav \
    ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/8k.wav
