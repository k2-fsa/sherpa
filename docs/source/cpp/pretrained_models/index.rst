.. _pretrained_models:

Pre-trained models
==================

Two kinds of end-to-end (E2E) models are supported by `sherpa`_:

- CTC
- Transducer

.. hint::

   For transducer-based models, we only support stateless transducers.
   To the best of our knowledge, only `icefall`_ supports that. In other words,
   only transducer models from `icefall`_ are currently supported.

   For CTC-based models, we support any type of models trained using CTC loss
   as long as you can export the model via torchscript. Models from the following
   frameworks are currently supported: `icefall`_, `wenet`_, and `torchaudio`_ (Wav2Vec 2.0).
   If you have a CTC model and want it to be supported in `sherpa`, please
   create an issue at `<https://github.com/k2-fsa/sherpa/issues>`_.

This page lists all available pre-trained models that you can download.


.. toctree::
   :maxdepth: 2
   :caption: Pretrained models

   offline_ctc
   offline_transducer
   online_transducer
