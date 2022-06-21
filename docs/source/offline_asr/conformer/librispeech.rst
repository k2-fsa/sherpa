LibriSpeech demo
================

.. hint::

   Please first refer to :ref:`installation` to install `sherpa`_
   before proceeding.

In this section, we demonstrate how to use `sherpa`_ for offline ASR
using a `Conformer`_ `transducer`_ model trained on the `LibriSpeech`_ dataset.

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
   git clone https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13

.. caution::

   It is important that you did not forget to run ``git lfs install``.
   Otherwise, you will be SAD later.

The ``3`` most important files you just downloaded are:

.. code-block:: bash

    $ cd icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13/

    $ ls -lh exp/*jit*
    lrwxrwxrwx 1 kuangfangjun root   10 Jun 17 21:52 exp/cpu_jit-torch-1.10.0.pt -> cpu_jit.pt
    -rw-r--r-- 1 kuangfangjun root 326M Jun 18 08:58 exp/cpu_jit-torch-1.6.0.pt
    -rw-r--r-- 1 kuangfangjun root 326M May 23 00:05 exp/cpu_jit.pt


    $ ls -lh data/lang_bpe_500/bpe.mode
    -rw-r--r-- 1 kuangfangjun root 240K Mar 12 14:43 data/lang_bpe_500/bpe.model

``exp/cpu_jit-torch-1.10.0.pt`` is a torchscript model
exported using torch 1.10, while ``exp/cpu_jit-torch-1.6.0.pt``
is exported using torch 1.6.0.

If you are using a version of PyTorch that is older than 1.10, please select
``exp/cpu_jit-torch-1.6.0.pt``. Otherwise, please use
``exp/cpu_jit-torch-1.10.0.pt``.

``data/lang_bpe_500/bpe.model`` is the BPE model that we used during training.

.. note::

   At present, we only implement ``greedy_search`` and ``modified beam_search``
   for decoding, so you only need a torchscript model file and a ``bpe.model``
   to start the server.

   After we implement ``fast_beam_search``, you can also use an FST-based
   n-gram LM during decoding.

Start the server
----------------

The entry point of the server is
`sherpa/bin/conformer_rnnt/offline_server.py <https://github.com/k2-fsa/sherpa/blob/master/sherpa/bin/conformer_rnnt/offline_server.py>`_.

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

   $ ./sherpa/bin/conformer_rnnt/offline_server.py --help

which gives the following output:

.. literalinclude:: ./code/offline-server-help.txt
   :caption: Output of ``./sherpa/bin/conformer_rnnt/offline_server.py --help``

The following shows an example about how to use the above pre-trained model
to start the server:

.. literalinclude:: ./code/start-the-server-librispeech.sh
   :language: bash
   :caption: Command to start the server using the above pre-trained model

When the server is started, you should see something like below:

.. code-block::
  :caption: Output after starting the server

  2022-06-21 18:51:58,424 INFO [offline_server.py:371] started
  2022-06-21 18:51:58,426 INFO [server.py:707] server listening on 0.0.0.0:6010
  2022-06-21 18:51:58,426 INFO [server.py:707] server listening on [::]:6010


Start the client
----------------

We also provide a Python script
`sherpa/bin/conformer_rnnt/offline_client.py <https://github.com/k2-fsa/sherpa/blob/master/sherpa/bin/conformer_rnnt/offline_client.py>`_ for the client.

.. code-block:: bash

   ./sherpa/bin/conformer_rnnt/offline_client.py --help

shows the following help information:

.. literalinclude:: ./code/offline-client-help.txt
   :caption: Output of ``./sherpa/bin/conformer_rnnt/offline_client.py --help``

We provide some test waves in the git repo you just cloned. The following command
shows you how to start the client:

.. literalinclude:: ./code/start-the-client-librispeech.sh
   :caption: Start the client and send multiple sound files for recognition

You will see the following output from the client side:


.. literalinclude:: ./code/client-results-librispeech.txt
   :caption: Recogntion results received by the client

while the server side log is:

.. code-block::

  2022-06-21 18:51:58,424 INFO [offline_server.py:371] started
  2022-06-21 18:51:58,426 INFO [server.py:707] server listening on 0.0.0.0:6010
  2022-06-21 18:51:58,426 INFO [server.py:707] server listening on [::]:6010
  2022-06-21 18:54:05,655 INFO [server.py:642] connection open
  2022-06-21 18:54:05,655 INFO [offline_server.py:552] Connected: ('127.0.0.1', 33228). Number of connections: 1/10
  2022-06-21 18:54:09,391 INFO [offline_server.py:573] Disconnected: ('127.0.0.1', 33228)
  2022-06-21 18:54:09,392 INFO [server.py:260] connection closed

Congratulations! You have succeeded in starting the server and client using
a pre-trained model with `sherpa`_.

We provide a colab notebook
|offline asr with librispeech colab notebook|
for you to try this tutorial step by step.

It describes not only how to setup the environment, but it also
shows you how to compute the ``WER`` and ``RTF`` of the `LibriSpeech`_
**test-clean** dataset.

.. |offline asr with librispeech colab notebook| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/drive/1JX5Ph2onYm1ZjNP_94eGqZ-DIRMLlIca?usp=sharing

