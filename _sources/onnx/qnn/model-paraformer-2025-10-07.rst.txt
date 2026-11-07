sherpa-onnx-qnn-SM8850-binary-5-seconds-paraformer-zh-2025-10-07-int8 (Sichuan dialect / Chuanyu dialect)
-----------------------------------------------------------------------------------------------------------------

This model is converted from :ref:`sherpa-onnx-paraformer-zh-int8-2025-10-07`.

The original PyTorch checkpoint is available at

  `<https://huggingface.co/ASLP-lab/WSChuan-ASR/tree/main/Paraformer-large-Chuan>`_

This model accepts input audio up to ``5`` seconds. Audio shorter than ``5`` seconds
is padded internally, and audio longer than ``5`` seconds is truncated.

We provide the same model for other durations from ``8`` to ``30`` seconds. We also
provide packages for other Qualcomm SoCs. See
`asr-models-qnn-binary <https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models-qnn-binary>`_
for the full list.

.. code-block::

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models-qnn-binary/sherpa-onnx-qnn-SM8850-binary-5-seconds-paraformer-zh-2025-10-07-int8.tar.bz2
  tar xvf sherpa-onnx-qnn-SM8850-binary-5-seconds-paraformer-zh-2025-10-07-int8.tar.bz2
  rm sherpa-onnx-qnn-SM8850-binary-5-seconds-paraformer-zh-2025-10-07-int8.tar.bz2

Use it the same way as the 2023-03-28 Paraformer model:

.. code-block::

   ./sherpa-onnx-offline \
     --provider=qnn \
     --tokens=./sherpa-onnx-qnn-SM8850-binary-5-seconds-paraformer-zh-2025-10-07-int8/tokens.txt \
     --paraformer.qnn-backend-lib=./libQnnHtp.so \
     --paraformer.qnn-system-lib=./libQnnSystem.so \
     --paraformer.qnn-context-binary=./sherpa-onnx-qnn-SM8850-binary-5-seconds-paraformer-zh-2025-10-07-int8/encoder.bin,./sherpa-onnx-qnn-SM8850-binary-5-seconds-paraformer-zh-2025-10-07-int8/predictor.bin,./sherpa-onnx-qnn-SM8850-binary-5-seconds-paraformer-zh-2025-10-07-int8/decoder.bin \
     ./sherpa-onnx-qnn-SM8850-binary-5-seconds-paraformer-zh-2025-10-07-int8/test_wavs/1.wav

