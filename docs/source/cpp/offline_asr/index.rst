.. _cpp_non_streaming_asr:

Non-streaming ASR
=================

This page describes how to use the C++ frontend of `sherpa`_ for
non-streaming ASR.

We use pretrained models from the following two datasets for demonstration:
`GigaSpeech`_ and `WenetSpeech`_.

The pretrained model from `GigaSpeech`_ uses `BPE <https://github.com/google/sentencepiece>`_
as modeling units, while the model from `WenetSpeech`_ uses Chinese characters.


.. toctree::
   :maxdepth: 2
   :caption: Demo with pretrained models

   api
   gigaspeech
   wenetspeech
