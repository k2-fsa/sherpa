Streaming Chinese ASR
=====================

This page describes how to use `sherpa`_ for streaming
ASR with `Conformer`_ transducer models
trained with `pruned stateless transdcuer <https://github.com/k2-fsa/icefall>`_.

.. hint::

   To be specific, the pre-trained model is trained on the `WenetSpeech`_
   dataset using the code from
   `<https://github.com/k2-fsa/icefall/tree/master/egs/wenetspeech/ASR/pruned_transducer_stateless5>`_.

   The pre-trained model can be downloaded from
   `<https://huggingface.co/luomingshuang/icefall_asr_wenetspeech_pruned_transducer_stateless5_streaming>`_

There are no **recurrent** modules in the transducer model:

  - The encoder network (i.e., the transcription network) is a Conformer model
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
