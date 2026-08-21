Streaming ASR
=============

This page describes how to use `sherpa`_ for streaming ASR.

The following types of transducer models are supported:

  - `Emformer`_
  - `ConvEmformer`_
  - ``LSTM``
  - `Zipformer`_


We support standalone speech recognition as well as server/client based
speech recognition using `WebSocket`_.


.. toctree::
   :maxdepth: 3

   standalone/index

   endpointing
   secure-connections
   emformer/index
   conv_emformer/index
   conformer/index
   lstm/index
