.. _onnx-x-asr:

X-ASR
=====

This section describes how to use models from
`X-ASR <https://github.com/Gilgamesh-J/X-ASR>`_.

`X-ASR <https://github.com/Gilgamesh-J/X-ASR>`_ is a zipformer transducer
model supporting both Chinese and English. We have exported it to
`sherpa-onnx`_ with the scripts in
`scripts/zipformer-transducer/x-asr
<https://github.com/k2-fsa/sherpa-onnx/tree/master/scripts/zipformer-transducer/x-asr>`_
and uploaded the models to the `asr-models
<https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models>`_ release.

We provide both non-streaming (offline) and streaming models. Every model
comes in four flavors:

  - ``punct`` or ``no-punct``: the ``punct`` models also add punctuations to
    the recognition result
  - ``int8`` or ``fp32``: the ``int8`` models are much smaller and faster;
    the ``fp32`` models are a bit more accurate

Non-streaming models
--------------------

The model names in the table below are shortened; for instance,
``punct-int8`` is short for
``sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03``.
Please click a name to download the model archive.

.. list-table::
   :header-rows: 1

   * - Model
     - Size
   * - `punct-int8 <https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03.tar.bz2>`_
     - 136 MB
   * - `punct-fp32 <https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-2026-06-03.tar.bz2>`_
     - 587 MB
   * - `no-punct-int8 <https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-zipformer-transducer-zh-en-int8-2026-06-03.tar.bz2>`_
     - 137 MB
   * - `no-punct-fp32 <https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-zipformer-transducer-zh-en-2026-06-03.tar.bz2>`_
     - 583 MB

Download
^^^^^^^^

Please use the following commands to download the ``punct``, ``int8`` model::

  cd /path/to/sherpa-onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03.tar.bz2
  tar xvf sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03.tar.bz2
  rm sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03.tar.bz2

After downloading, you should find the following files::

  ls -lh sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03

  -rw-r--r--@ 1 fangjun  staff   116K 29 Jun 16:31 bpe.model
  -rw-r--r--@ 1 fangjun  staff    11M 29 Jun 16:31 decoder-epoch-99-avg-1.onnx
  -rw-r--r--@ 1 fangjun  staff   154M 29 Jun 16:31 encoder-epoch-99-avg-1.int8.onnx
  -rw-r--r--@ 1 fangjun  staff   2.5M 29 Jun 16:31 joiner-epoch-99-avg-1.int8.onnx
  -rw-r--r--@ 1 fangjun  staff    96B 29 Jun 16:31 README.md
  -rwxr-xr-x@ 1 fangjun  staff   6.7K 29 Jun 16:31 test_onnx.py
  drwxr-xr-x@ 6 fangjun  staff   192B 29 Jun 16:27 test_wavs
  -rw-r--r--@ 1 fangjun  staff    57K 29 Jun 16:31 tokens.txt

.. hint::

   The ``fp32`` models use ``encoder-epoch-99-avg-1.onnx`` and
   ``joiner-epoch-99-avg-1.onnx`` instead; the ``no-punct`` models contain
   the same files with a different ``tokens.txt``. Every model archive
   contains ``test_wavs`` for testing.

Decode files
^^^^^^^^^^^^

Please use the following command to decode wave files:

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03/tokens.txt \
    --encoder=./sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03/encoder-epoch-99-avg-1.int8.onnx \
    --decoder=./sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03/decoder-epoch-99-avg-1.onnx \
    --joiner=./sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03/joiner-epoch-99-avg-1.int8.onnx \
    --num-threads=1 \
    --debug=0 \
    ./sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03/test_wavs/0.wav \
    ./sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03/test_wavs/1.wav

You should see the following output:

.. literalinclude:: ./code/offline-punct-int8.txt

.. hint::

   The results contain punctuations since this is a ``punct`` model.

Streaming models
----------------

The streaming models support four chunk sizes: 160 ms, 480 ms, 960 ms, and
1920 ms. A smaller chunk size means lower latency.

The model names in the table below are shortened; for instance,
``480ms-punct-int8`` is short for
``sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05``.
Please click a name to download the model archive.

.. list-table::
   :header-rows: 1

   * - Model
     - Size
   * - `160ms-punct-int8 <https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-160ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05.tar.bz2>`_
     - 134 MB
   * - `160ms-punct-fp32 <https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-160ms-streaming-zipformer-transducer-zh-en-punct-2026-06-05.tar.bz2>`_
     - 584 MB
   * - `160ms-no-punct-int8 <https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-160ms-streaming-zipformer-transducer-zh-en-int8-2026-06-05.tar.bz2>`_
     - 134 MB
   * - `160ms-no-punct-fp32 <https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-160ms-streaming-zipformer-transducer-zh-en-2026-06-05.tar.bz2>`_
     - 580 MB
   * - `480ms-punct-int8 <https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05.tar.bz2>`_
     - 134 MB
   * - `480ms-punct-fp32 <https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-2026-06-05.tar.bz2>`_
     - 584 MB
   * - `480ms-no-punct-int8 <https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-int8-2026-06-05.tar.bz2>`_
     - 134 MB
   * - `480ms-no-punct-fp32 <https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-2026-06-05.tar.bz2>`_
     - 580 MB
   * - `960ms-punct-int8 <https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-960ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05.tar.bz2>`_
     - 134 MB
   * - `960ms-punct-fp32 <https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-960ms-streaming-zipformer-transducer-zh-en-punct-2026-06-05.tar.bz2>`_
     - 584 MB
   * - `960ms-no-punct-int8 <https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-960ms-streaming-zipformer-transducer-zh-en-int8-2026-06-05.tar.bz2>`_
     - 134 MB
   * - `960ms-no-punct-fp32 <https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-960ms-streaming-zipformer-transducer-zh-en-2026-06-05.tar.bz2>`_
     - 580 MB
   * - `1920ms-punct-int8 <https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-1920ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05.tar.bz2>`_
     - 134 MB
   * - `1920ms-punct-fp32 <https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-1920ms-streaming-zipformer-transducer-zh-en-punct-2026-06-05.tar.bz2>`_
     - 584 MB
   * - `1920ms-no-punct-int8 <https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-1920ms-streaming-zipformer-transducer-zh-en-int8-2026-06-05.tar.bz2>`_
     - 134 MB
   * - `1920ms-no-punct-fp32 <https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-1920ms-streaming-zipformer-transducer-zh-en-2026-06-05.tar.bz2>`_
     - 580 MB

Download
^^^^^^^^

Please use the following commands to download the ``480 ms``, ``punct``,
``int8`` model::

  cd /path/to/sherpa-onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05.tar.bz2
  tar xvf sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05.tar.bz2
  rm sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05.tar.bz2

After downloading, you should find the following files::

  ls -lh sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05

  -rw-r--r--@ 1 fangjun  staff   116K 29 Jun 16:30 bpe.model
  -rw-r--r--@ 1 fangjun  staff    11M 29 Jun 16:30 decoder.onnx
  -rw-r--r--@ 1 fangjun  staff   148M 29 Jun 16:30 encoder.int8.onnx
  -rw-r--r--@ 1 fangjun  staff   2.5M 29 Jun 16:30 joiner.int8.onnx
  -rw-r--r--@ 1 fangjun  staff    96B 29 Jun 16:30 README.md
  -rwxr-xr-x@ 1 fangjun  staff   7.1K 29 Jun 16:30 test_onnx.py
  drwxr-xr-x@ 6 fangjun  staff   192B 29 Jun 16:27 test_wavs
  -rw-r--r--@ 1 fangjun  staff    57K 29 Jun 16:30 tokens.txt

.. hint::

   The ``fp32`` models use ``encoder.onnx`` and ``joiner.onnx`` instead; the
   ``no-punct`` models contain the same files with a different ``tokens.txt``.

Decode a file
^^^^^^^^^^^^^

Please use the following command to decode a wave file:

.. code-block:: bash

  ./build/bin/sherpa-onnx \
    --tokens=./sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05/tokens.txt \
    --encoder=./sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05/encoder.int8.onnx \
    --decoder=./sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05/decoder.onnx \
    --joiner=./sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05/joiner.int8.onnx \
    --num-threads=1 \
    --debug=0 \
    ./sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05/test_wavs/0.wav

You should see the following output:

.. literalinclude:: ./code/streaming-480ms-punct-int8.txt

Use X-ASR with our APIs
-----------------------

You can use the `X-ASR <https://github.com/Gilgamesh-J/X-ASR>`_ models with all APIs of `sherpa-onnx`_. In the
following pages, we describe step by step how to use the models above in
each programming language; every demo was tested and the expected output is
shown.

.. toctree::
   :maxdepth: 1

   ./python-api.rst
   ./c-api.rst
   ./cxx-api.rst
   ./java-api.rst
   ./kotlin-api.rst
   ./go-api.rst
   ./rust-api.rst
   ./csharp-api.rst
   ./javascript-api.rst
   ./javascript-wasm-api.rst
   ./dart-api.rst
   ./swift-api.rst
   ./pascal-api.rst
