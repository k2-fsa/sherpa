.. _conformer_rnnt_client_english:

Client
======

With the client you can record your voice in real-time, send it to the
server, and get the recognition results back from the server.

We provide a web client for this purpose.

.. caution::

   Please first start the :ref:`conformer_rnnt_server_english` before you start the client.

   Also, we have hard coded the server port to 6006. Please either pass
   ``--port 6006`` when starting the server or change the client
   `<https://github.com/k2-fsa/sherpa/blob/master/sherpa/bin/web/js/streaming_record.js#L7>`_
   to use whaterver the port the server is using.

Usage
-----

.. code-block:: bash

   cd /path/to/sherpa
   cd ./sherpa/bin/web
   python3 -m http.server 6008

Then open your browser, and visit `<http://localhost:6008/streaming_record.html>`_.

You will see a UI like the following screenshot. Click the ``Streaming-Record`` button
and speak! You should see the recognition results from the server.


.. image:: /_static/conformer-rnnt-streaming-asr-web-client.jpg
  :alt: Screen shot of the web client user interface

.. note::

   If you are unable to click the ``Streaming-Record`` button, please make sure
   the server port is 6006.

.. caution::

   You have to visit `<http://localhost:6008/streaming_record.html>`_, not
   `<http://0.0.0.0:6008/streaming_record.html>`_. Otherwise, you will not be able
   to use the microphone in the browser. One way to avoid this is to use ``https``,
   but that needs a certificate.

.. hint::

   If you are using Chrome, you can right click the page, and then click
   ``inspect`` in the popup menu, and then click ``console``. You will see
   some diagnostic message. This helps you to debug if you are unable to click
   the ``Streaming-Record`` button.
