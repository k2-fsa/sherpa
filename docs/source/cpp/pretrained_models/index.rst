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

.. hint::

   You can try the pre-trained models in your browser without installing
   anything. See `<https://huggingface.co/spaces/k2-fsa/automatic-speech-recognition>`_.


This page lists all available pre-trained models that you can download.

.. hint::

   We provide pre-trained models for the following languages:

    - Arabic
    - Chinese
    - English
    - German
    - Tibetan


.. hint::

   We provide a colab notebook
   |Sherpa offline recognition python api colab notebook|
   for you to try offline recognition step by step.

   It shows how to install sherpa and use it as offline recognizer,
   which supports the models from icefall, the wenet framework and torchaudio.

.. |Sherpa offline recognition python api colab notebook| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/drive/1RdU06GcytTpI-r8vkQ7NkI0ugytnwJVB?usp=sharing

.. toctree::
   :maxdepth: 5
   :caption: Pretrained models

   offline_ctc/index
   offline_transducer
   online_transducer
