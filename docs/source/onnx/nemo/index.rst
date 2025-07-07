.. _onnx-nemo:

NeMo
====

This page lists models from `NeMo`_.

Speech Recognition models
-------------------------

.. toctree::
   :maxdepth: 5

   ../pretrained_models/offline-ctc/nemo/index.rst
   ../pretrained_models/offline-transducer/nemo-transducer-models.rst
   ./canary.rst

Speaker Embedding models
------------------------

Please see :ref:`onnx-speaker-identification`.

Models from `NeMo`_ are prefixed with ``nemo``, e.g.,

  - `nemo_en_titanet_small.onnx <https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_titanet_small.onnx>`_
  - `nemo_en_titanet_large.onnx <https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_titanet_large.onnx>`_
  - `nemo_en_speakerverification_speakernet.onnx <https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_speakerverification_speakernet.onnx>`_

You can find the export scripts at `<https://github.com/k2-fsa/sherpa-onnx/tree/master/scripts/nemo/speaker-verification>`_
