How to export models to RKNN for sherpa-onnx
============================================

Please don't use RKNN models from `<https://github.com/airockchip/rknn_model_zoo/tree/main/examples/zipformer>`_

.. warning::

   Many users use models from rknn model zoo in `sherpa-onnx`_ and get errors. That's expected.

   Please never do that.

If you want to export models from `icefall`_, please see

  - ``Entrypoint``: `<https://github.com/k2-fsa/icefall/blob/master/.github/workflows/rknn.yml>`_
  - `<https://github.com/k2-fsa/icefall/blob/master/.github/scripts/librispeech/ASR/run_rknn.sh>`_
  - `<https://github.com/k2-fsa/icefall/blob/master/.github/scripts/wenetspeech/ASR/run_rknn.sh>`_
  - `<https://github.com/k2-fsa/icefall/blob/master/.github/scripts/multi_zh-hans/ASR/run_rknn.sh>`_

.. caution::

   You have to start from the PyTorch checkpoints.

   Please don't try to convert the onnx model files from `sherpa-onnx`_ to RKNN.

   You would get errors.

