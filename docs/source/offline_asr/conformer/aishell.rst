aishell demo
============

.. hint::

   Please first refer to :ref:`installation` to install `sherpa`_
   before proceeding.

In this section, we demonstrate how to use `sherpa`_ for offline ASR
using a `Conformer`_ `transducer`_ model trained on the `aishell`_ dataset.

Download the pre-trained model
------------------------------

The pre-trained model is in a git repository hosted on
`huggingface <https://huggingface.co/>`_.

Since the pre-trained model is over 10 MB and is managed by
`git LFS <https://git-lfs.github.com/>`_, you have
to first install ``git-lfs`` before you continue.

On Ubuntu, you can install ``git-lfs`` using

.. code-block:: bash

   sudo apt-get install git-lfs

.. hint::

   If you are using other operating systems, please refer to
   `<https://git-lfs.github.com/>`_ for how to install ``git-lfs`` on your
   systems.

After installing ``git-lfs``, we are ready to download the pre-trained model:

.. code-block:: bash

   git lfs install
   git clone https://huggingface.co/csukuangfj/icefall-aishell-pruned-transducer-stateless3-2022-06-20

.. caution::

   It is important that you did not forget to run ``git lfs install``.
   Otherwise, you will be SAD later.

The ``3`` most important files you just downloaded are:

.. code-block:: bash

    $ cd icefall-models/icefall-aishell-pruned-transducer-stateless3-2022-06-20/

    $ ls -lh exp/*jit*
    -rw-r--r-- 1 kuangfangjun root 390M Jun 20 11:48 exp/cpu_jit-epoch-29-avg-5-torch-1.10.0.pt
    -rw-r--r-- 1 kuangfangjun root 390M Jun 20 12:28 exp/cpu_jit-epoch-29-avg-5-torch-1.6.0.pt

    $ ls -lh data/lang_char/tokens.txt
    -rw-r--r-- 1 kuangfangjun root 38K Jun 20 10:32 data/lang_char/tokens.txt

``exp/cpu_jit-epoch-29-avg-5-torch-1.10.0.pt`` is a torchscript model
exported using torch 1.10, while ``exp/cpu_jit-epoch-29-avg-5-torch-1.6.0.pt``
is exported using torch 1.6.0.

If you are using a version of PyTorch that is older than 1.10, please select
``exp/cpu_jit-epoch-29-avg-5-torch-1.6.0.pt``. Otherwise, please use
``exp/cpu_jit-epoch-29-avg-5-torch-1.10.0.pt``.

``data/lang_char/tokens.txt`` is a token table (i.e., word table),
containing mappings between words and word IDs.

.. note::

   At present, we only implement ``greedy_search`` and ``modified beam_search``
   for decoding, so you only need a torchscript model file and a ``tokens.txt``
   to start the server.

   After we implement ``fast_beam_search``, you can also use an FST-based
   n-gram LM during decoding.

Start the server
----------------

The entry point of the server is
`sherpa/bin/pruned_transducer_statelessX/offline_server.py <https://github.com/k2-fsa/sherpa/blob/master/sherpa/bin/pruned_transducer_statelessX/offline_server.py>`_.

One thing worth mentioning is that the entry point is a Python script.
In `sherpa`_, the server is implemented using `asyncio`_, where **IO-bound**
tasks, such as communicating with clients, are implemented in Python,
while **CPU-bound** tasks, such as neural network computation, are implemented
in C++ and are invoked by a pool of threads created and managed by Python.

.. note::

  When a thread calls into C++ from Python, it releases the
  `global interpreter lock (GIL) <https://wiki.python.org/moin/GlobalInterpreterLock>`_
  and regains the ``GIL`` just before it returns.

  In this way, we can maximize the utilization of multi CPU cores.

To view the usage information of the server, you can use:

.. code-block:: bash

   $ ./sherpa/bin/pruned_transducer_statelessX/offline_server.py --help

which gives the following output:

.. literalinclude:: ./code/offline-server-help.txt
   :caption: Output of ``./sherpa/bin/pruned_transducer_statelessX/offline_server.py --help``

The following shows an example about how to use the above pre-trained model
to start the server:

.. literalinclude:: ./code/start-the-server.sh
   :language: bash
   :caption: Command to start the server using the above pre-trained model

When the server is started, you should see something like below:

.. code-block::
  :caption: Output after starting the server

  2022-06-21 17:33:10,000 INFO [offline_server.py:371] started
  2022-06-21 17:33:10,002 INFO [server.py:707] server listening on 0.0.0.0:6010
  2022-06-21 17:33:10,002 INFO [server.py:707] server listening on [::]:6010


Start the client
----------------

We also provide a Python script
`sherpa/bin/pruned_transducer_statelessX/offline_client.py <https://github.com/k2-fsa/sherpa/blob/master/sherpa/bin/pruned_transducer_statelessX/offline_client.py>`_ for the client.

.. code-block:: bash

   ./sherpa/bin/pruned_transducer_statelessX/offline_client.py --help

shows the following help information:

.. literalinclude:: ./code/offline-client-help.txt
   :caption: Output of ``./sherpa/bin/pruned_transducer_statelessX/offline_client.py --help``

We provide some test waves in the git repo you just cloned. The following command
shows you how to start the client:

.. literalinclude:: ./code/start-the-client.sh
   :caption: Start the client and send multiple sound files for recognition

You will see the following output from the client side:


.. literalinclude:: ./code/client-results.txt
   :caption: Recogntion results received by the client

while the server side log is:

.. code-block::

  2022-06-21 17:33:10,000 INFO [offline_server.py:371] started
  2022-06-21 17:33:10,002 INFO [server.py:707] server listening on 0.0.0.0:6010
  2022-06-21 17:33:10,002 INFO [server.py:707] server listening on [::]:6010
  2022-06-21 17:39:30,148 INFO [server.py:642] connection open
  2022-06-21 17:39:30,148 INFO [offline_server.py:552] Connected: ('127.0.0.1', 59558). Number of connections: 1/10
  2022-06-21 17:39:33,757 INFO [offline_server.py:573] Disconnected: ('127.0.0.1', 59558)
  2022-06-21 17:39:33,758 INFO [server.py:260] connection closed

Congratulations! You have succeeded in starting the server and client using
a pre-trained model with `sherpa`_.

We provide a colab notebook
|offline asr with aishell colab notebook|
for you to try this tutorial step by step.

It describes not only how to setup the environment, but it also
shows you how to compute the ``WER`` and ``RTF`` of the `aishell`_ **test** dataset.

.. |offline asr with aishell colab notebook| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/drive/1eeJ7WcWZdy1SI93jXlp0lYAccW29F_NO?usp=sharing
