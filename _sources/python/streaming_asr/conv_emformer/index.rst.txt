ConvEmformer transducer based streaming ASR
===========================================

This page describes how to use `sherpa`_ for streaming
ASR with `ConvEmformer`_ transducer models
trained with `pruned stateless transdcuer <https://github.com/k2-fsa/icefall>`_.

.. hint::

   To be specific, the pre-trained model is trained on the `LibriSpeech`_
   dataset using the code from
   `<https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/conv_emformer_transducer_stateless2>`_.

   The pre-trained model can be downloaded from
   `<https://huggingface.co/Zengwei/icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05>`_

There are no **recurrent** modules in the transducer model:

  - The encoder network (i.e., the transcription network) is a ConvEmformer model
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
