.. _asr_websocket_server:

ASR Server
=============

This page describes how to use the C++ API of `sherpa`_ to deploy ASR Websocket Server

.. warning::

   It supports only models from
   `<https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/conv_emformer_transducer_stateless2>`_
   at present.

The following show how to build

.. code-block:: bash

   mkdir /path/to/sherpa/build
   cd /path/to/sherpa/build
   cmake -DSHERPA_ENABLE_WEBSOCKET_SERVER=ON ..
   make -j


After above installation, you should find the following bin:

  - ``bin/websocket-server``
  - ``bin/websocket-client``


The following shows how to use the C++ API for real-time speech recognition with a Websocket Server.

.. code-block:: bash

   cd /path/to/sherpa/build

   git lfs install
   git clone https://huggingface.co/Zengwei/icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05

   ./bin/websocket-server \
     --decoding-method=modified_beam_search \
     --nn-model=./icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05/exp/cpu-jit-epoch-30-avg-10-torch-1.10.0.pt \
     --tokens=./icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05/data/lang_bpe_500/tokens.txt \
     --server-port=6006

   ./bin/websocket-client \
     --server-ip=127.0.0.1 --server-port=6006 \
     --wav-path=./icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05/test_wavs/1089-134686-0001.wav

