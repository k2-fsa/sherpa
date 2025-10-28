Secure connections
==================

In this section, we describe how to use ``https`` and
secure websocket with ``sherpa``.

.. hint::

   If you don't use ``https``, you have to use ``http://localhost:port``.
   Otherwise, you won't be able to use the microphone within your browser.


Generate a certificate
----------------------

First, you need to have a `X.509 certificate <https://en.wikipedia.org/wiki/X.509>`_.
If you don't have one, we provide the following command to generate a
``self-signed`` certificate for you:

.. code-block::

   cd sherpa/bin/web
   ./generate-certificate.py

It will generate a file ``cert.pem``, which is the self-signed certificate.

.. caution::

   You have to make your browser trust the self-signed certificate later
   for both the https server and the websocket server. The reason to trust
   both servers is that they are running on different ports.

Start the https server
----------------------


.. code-block:: bash

   cd sherpa/bin/web
   ./start-https-server.py \
     --server-address 0.0.0.0 \
     --server-port 6007 \
     --certificate cert.pem

``0.0.0.0`` means the https server will listen on all IP addresses of the
current machine.

If you are using Firefox, you can visit `<https://0.0.0.0:6007>`_, which
will show you the following page:

.. figure:: ./images/secure-connections/1.png
    :alt: Your-connection-is-not-secure
    :align: center
    :figwidth: 600px

    Click the ``Advanced`` button.

.. hint::

   You get the above message because you are using a self-signed certificate.
   Also, you can use one of the public IP addresses of your machine to
   replace ``0.0.0.0`` in `<https://0.0.0.0:6007>`_.

After clicking the button ``Advanced``, you will see the following page:

.. figure:: ./images/secure-connections/2.png
    :alt: After-clicking-the-advanced-button
    :align: center
    :figwidth: 600px

    After clicking the ``Advanced`` button.

Now click ``Add exception`` and then click "Confirm security exception" below:

.. figure:: ./images/secure-connections/3.png
    :alt: Click-confirm-security-exception
    :align: center
    :figwidth: 600px

    Click ``Confirm security exception``.
    After clicking the advanced button

At this point, your browser should trust your self-signed certificate
for the host ``0.0.0.0:6007``.

One thing left is that you should also make your browser trust the
certificate of the websocket server. Otherwise, you won't be able
to make a connection to the websocket server.

Assume your websocket server runs on the same machine as your https
server but uses the port ``6006``. You can start a https server on
port ``6006`` and repeat the above steps to make your browser
trust the certificate for the host ``0.0.0.0:6006`` and then kill
the https server running on port ``6006``.

Start the websocket server
--------------------------

.. note::

   We use ``sherpa/bin/conv_emformer_transducer_stateless2/streaming_server.py``
   as a demo below. The steps should be similar for starting other
   streaming servers.

.. code-block:: bash

    cd /path/to/sherpa

    git lfs install
    git clone https://huggingface.co/Zengwei/icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05

    ./sherpa/bin/conv_emformer_transducer_stateless2/streaming_server.py \
      --endpoint.rule3.min-utterance-length 1000.0 \
      --port 6006 \
      --max-batch-size 50 \
      --max-wait-ms 5 \
      --nn-pool-size 1 \
      --nn-model-filename ./icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05/exp/cpu-jit-epoch-30-avg-10-torch-1.10.0.pt \
      --bpe-model-filename ./icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05/data/lang_bpe_500/bpe.model \
      --certificate ./sherpa/bin/web/cert.pem

Now visit `<https://0.0.0.0:6007>`_ and you should be able to make a secure
connection to the websocket server ``wss://0.0.0.0:6006``.
