Decode files
============

In this section, we demonstrate how to use the Python API of `sherpa-onnx`_
to decode files.

.. hint::

   We only support WAVE files of single channel and each sample should have
   16-bit, while the sample rate of the file can be arbitrary and it does
   not need to be 16 kHz


Streaming zipformer
--------------------

We use :ref:`sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20` as
an example below.

.. code-block:: bash

   cd /path/to/sherpa-onnx

   python3 ./python-api-examples/online-decode-files.py \
     --tokens=./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt \
     --encoder=./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.onnx \
     --decoder=./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx \
     --joiner=./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.onnx \
     ./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/test_wavs/0.wav \
     ./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/test_wavs/1.wav \
     ./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/test_wavs/2.wav \
     ./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/test_wavs/3.wav \
     ./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/test_wavs/8k.wav

.. hint::

   ``online-decode-files.py`` is from `<https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/online-decode-files.py>`_

.. note::

   You can replace ``encoder-epoch-99-avg-1.onnx`` with ``encoder-epoch-99-avg-1.int8.onnx``
   to use ``int8`` models for decoding.

The output is given below:

.. literalinclude:: ./code/decode-files/streaming-transducer-bilingual-zh-en-2023-02-20.txt

Non-streaming zipformer
-----------------------

We use :ref:`sherpa_onnx_zipformer_en_2023_04_01` as
an example below.

.. code-block:: bash

   cd /path/to/sherpa-onnx

   python3 ./python-api-examples/offline-decode-files.py \
      --tokens=./sherpa-onnx-zipformer-en-2023-04-01/tokens.txt \
      --encoder=./sherpa-onnx-zipformer-en-2023-04-01/encoder-epoch-99-avg-1.onnx \
      --decoder=./sherpa-onnx-zipformer-en-2023-04-01/decoder-epoch-99-avg-1.onnx \
      --joiner=./sherpa-onnx-zipformer-en-2023-04-01/joiner-epoch-99-avg-1.onnx \
      ./sherpa-onnx-zipformer-en-2023-04-01/test_wavs/0.wav \
      ./sherpa-onnx-zipformer-en-2023-04-01/test_wavs/1.wav \
      ./sherpa-onnx-zipformer-en-2023-04-01/test_wavs/8k.wav

.. hint::

   ``offline-decode-files.py`` is from `<https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/offline-decode-files.py>`_

.. note::

   You can replace ``encoder-epoch-99-avg-1.onnx`` with ``encoder-epoch-99-avg-1.int8.onnx``
   to use ``int8`` models for decoding.

The output is given below:

.. literalinclude:: ./code/decode-files/non-streaming-transducer-zipformer-2023-04-01.txt

Non-streaming paraformer
------------------------

We use :ref:`sherpa_onnx_offline_paraformer_zh_2023_03_28_chinese` as
an example below.

.. code-block:: bash

   cd /path/to/sherpa-onnx

   python3 ./python-api-examples/offline-decode-files.py \
    --tokens=./sherpa-onnx-paraformer-zh-2023-03-28/tokens.txt \
    --paraformer=./sherpa-onnx-paraformer-zh-2023-03-28/model.onnx \
    ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/0.wav \
    ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/1.wav \
    ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/2.wav \
    ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/8k.wav

.. note::

   You can replace ``model.onnx`` with ``model.int8.onnx``
   to use ``int8`` models for decoding.

The output is given below:

.. literalinclude:: ./code/decode-files/non-streaming-paraformer-2023-03-28.txt
