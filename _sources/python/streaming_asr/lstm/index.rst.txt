LSTM transducer based streaming ASR
===========================================

This page describes how to use `sherpa`_ for streaming
ASR with `LSTM` transducer models
trained with `pruned stateless transdcuer <https://github.com/k2-fsa/icefall>`_.

.. hint::

   To be specific, the pre-trained model for English is trained on the `LibriSpeech`_
   dataset using the code from
   `<https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/lstm_transducer_stateless>`_.

   The pre-trained model for English can be downloaded from
   `<https://huggingface.co/Zengwei/icefall-asr-librispeech-lstm-transducer-stateless-2022-08-18>`_

   While the pretrained model for Chinese is trained on the `WenetSpeech`_
   dataset. The model can be downloaded from
   `<https://huggingface.co/csukuangfj/icefall-asr-wenetspeech-lstm-transducer-stateless-2022-09-19>`_

There are no **recurrent** modules in the transducer model:

  - The encoder network (i.e., the transcription network) is a LSTM model
  - The decoder network (i.e., the prediction network) is a
    `stateless network <https://ieeexplore.ieee.org/document/9054419>`_,
    consisting of an ``nn.Embedding()`` and a ``nn.Conv1d()``.
  - The joiner network (i.e., the joint network) contains an adder,
    a ``tanh`` activation, and a ``nn.Linear()``.


We provide examples for following two languages:

English
-------

.. toctree::
   :maxdepth: 2

   english/server
   english/client

Chinese
-------

.. toctree::
   :maxdepth: 2

   chinese/server
   chinese/client
