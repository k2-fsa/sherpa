.. _onnx_streaming_websocket_server_and_client:

Streaming WebSocket server and client
=====================================

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

Please refer to :ref:`onnx_non_streaming_websocket_server_and_client`
for the usage of ``sherpa-onnx-offline-websocket-server``.

View the server usage
---------------------

Before starting the server, let us view the help message of ``sherpa-onnx-online-websocket-server``:

.. code-block:: bash

  build/bin/sherpa-onnx-online-websocket-server

The above command will print the following help information:

.. literalinclude:: ./code/sherpa-onnx-online-websocket-server-help.txt

Start the server
----------------

.. hint::

  Please refer to :ref:`sherpa-onnx-pre-trained-models`
  for a list of pre-trained models.

.. code-block:: bash

  ./build/bin/sherpa-onnx-online-websocket-server \
    --port=6006 \
    --num-work-threads=3 \
    --num-io-threads=2 \
    --tokens=./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt \
    --encoder=./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.onnx \
    --decoder=./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx \
    --joiner=./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.onnx \
    --log-file=./log.txt \
    --max-batch-size=5 \
    --loop-interval-ms=20

.. caution::

   The arguments are in the form ``--key=value``.

   It does not support ``--key value``.

   It does not support ``--key value``.

   It does not support ``--key value``.

.. hint::

   In the above demo, the model files are
   from :ref:`sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20`.

.. note::

  Note that the server supports processing multiple clients in a batch in parallel.
  You can use ``--max-batch-size`` to limit the batch size.

View the usage of the client (C++)
----------------------------------

Let us view the usage of the C++ `WebSocket`_ client:

.. code-block:: bash

   ./build/bin/sherpa-onnx-online-websocket-client

The above command will print the following help information:

.. literalinclude:: ./code/sherpa-onnx-online-websocket-client-help-info.txt

.. caution::

   We only support using IP address for ``--server-ip``.

   For instance, please don't use ``--server-ip=localhost``, use ``--server-ip=127.0.0.1`` instead.

   For instance, please don't use ``--server-ip=localhost``, use ``--server-ip=127.0.0.1`` instead.

   For instance, please don't use ``--server-ip=localhost``, use ``--server-ip=127.0.0.1`` instead.

Start the client (C++)
----------------------

To start the C++ `WebSocket`_ client, use:

.. code-block:: bash

   build/bin/sherpa-onnx-online-websocket-client \
     --seconds-per-message=0.1 \
     --server-port=6006 \
     ./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/test_wavs/0.wav

Since the server is able to process multiple clients at the same time, you can
use the following command to start multiple clients:

.. code-block:: bash

  for i in $(seq 0 10); do
    k=$(expr $i % 5)
    build/bin/sherpa-onnx-online-websocket-client \
      --seconds-per-message=0.1 \
      --server-port=6006 \
      ./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/test_wavs/${k}.wav &
  done

  wait

  echo "done"

View the usage of the client (Python)
-------------------------------------

Use the following command to view the usage:

.. code-block:: bash

   python3 ./python-api-examples/online-websocket-client-decode-file.py  --help

.. hint::

   ``online-websocket-client-decode-file.py`` is from
   `<https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/online-websocket-client-decode-file.py>`_

It will print:

.. literalinclude:: ./code/python-online-websocket-client-decode-a-file.txt

.. hint::

   For the Python client, you can use either a domain name or an IP address
   for ``--server-addr``. For instance, you can use either
   ``--server-addr localhost`` or ``--server-addr 127.0.0.1``.

   For the input argument, you can either use ``--key=value`` or ``--key value``.


Start the client (Python)
-------------------------

.. code-block:: bash

  python3 ./python-api-examples/online-websocket-client-decode-file.py \
    --server-addr localhost \
    --server-port 6006 \
    --seconds-per-message 0.1 \
    ./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/test_wavs/4.wav

Start the client (Python, with microphone)
------------------------------------------

.. code-block:: bash

  python3 ./python-api-examples/online-websocket-client-microphone.py \
    --server-addr localhost \
    --server-port 6006

   ``online-websocket-client-microphone.py `` is from
   `<https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/online-websocket-client-microphone.py>`_
