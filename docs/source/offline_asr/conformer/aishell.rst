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

If you are using a version of PyTorch older than 1.10, please select
``exp/cpu_jit-epoch-29-5-torch-1.6.0.pt``. Otherwise, please use
``exp/cpu_jit-epoch-29-5-torch-1.10.0.pt``.

``data/lang_char/tokens.txt`` is a token table (i.e., word table),
containing mappings between words and word IDs.

.. note::

   At present, we only implement ``greedy_search`` and ``modified beam_search``
   for decoding, so you only need a torchscript model file and a ``tokens.txt``
   to start the server.

   After we implement ``fast_beam_search``, you can also use an FST based
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
