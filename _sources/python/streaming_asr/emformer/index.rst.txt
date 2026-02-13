Emformer transducer based streaming ASR
=======================================

This page describes how to use `sherpa`_ for streaming
ASR with `Emformer`_ transducer models
trained with `pruned stateless transdcuer <https://github.com/k2-fsa/icefall>`_.

.. hint::

   To be specific, the pre-trained model is trained on the `LibriSpeech`_
   dataset using the code from
   `<https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/pruned_stateless_emformer_rnnt2>`_.

   The pre-trained model can be downloaded from
   `<https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-stateless-emformer-rnnt2-2022-06-01>`_

There are no **recurrent** modules in the transducer model:

  - The encoder network (i.e., the transcription network) is an Emformer model
  - The decoder network (i.e., the prediction network) is a
    `stateless network <https://ieeexplore.ieee.org/document/9054419>`_,
    consisting of an ``nn.Embedding()`` and a ``nn.Conv1d()``.
  - The joiner network (i.e., the joint network) contains an adder,
    a ``tanh`` activation, and a ``nn.Linear()``.


Streaming ASR in this section consists of two components:

.. toctree::
   :maxdepth: 2

   server
   client

The following is a `YouTube video <https://www.youtube.com/watch?v=z7HgaZv5W0U>`_,
demonstrating how to use the server and the client.

.. note::

   If you have no access to YouTube, please visit the following link from bilibili
   `<https://www.bilibili.com/video/BV1BU4y197bs>`_

..  youtube:: z7HgaZv5W0U
   :width: 120%
