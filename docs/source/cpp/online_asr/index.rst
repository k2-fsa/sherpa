.. _cpp_streaming_asr:

Streaming ASR
=============

This page describes how to use the C++ API of `sherpa`_ for
streaming/online ASR.

.. warning::

   It supports only models from
   `<https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/conv_emformer_transducer_stateless2>`_
   at present.

Please refer to :ref:`cpp_installation` for installation.


After running ``make -j``, you should find the following files:

  - ``lib/libsherpa_online_recognizer.so``
  - ``include/sherpa/cpp_api/online_recognizer.h``
  - ``include/sherpa/cpp_api/online_stream.h``

You can include the above two header files in your application and link
``libsherpa_online_recognizer.so`` with you executable to use the C++ APIs.


`<https://github.com/k2-fsa/sherpa/blob/master/sherpa/cpp_api/test_online_recognizer_microphone.cc>`_
shows how to use the C++ API for real-time speech recognition with a microphone.
After running ``make -j``, you can also find an executable ``bin/test_online_recognizer_microphone``.
The following shows how to use it:

.. code-block:: bash

   cd /path/to/sherpa/build

   git lfs install
   git clone https://huggingface.co/Zengwei/icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05

   ./bin/test_online_recognizer_microphone \
     ./icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05/exp/cpu-jit-epoch-30-avg-10-torch-1.10.0.pt \
     ./icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05/data/lang_bpe_500/tokens.txt

It will print something like below:

.. code-block::

  num devices: 4
  Use default device: 2
    Name: MacBook Pro Microphone
    Max input channels: 1
  Started

Say something and you will see the recognition result printed to the console in real-time.

You can find a demo below:

..  youtube:: 86-YLg3u-WY
   :width: 120%
