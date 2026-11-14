sherpa-onnx-qnn-SM8850-binary-5-seconds-paraformer-zh-2023-03-28-int8 (Chinese + English)
------------------------------------------------------------------------------------------------

This model is converted from :ref:`sherpa_onnx_offline_paraformer_zh_2023_03_28_chinese`.

This model accepts input audio up to ``5`` seconds. Audio shorter than ``5`` seconds
is padded internally, and audio longer than ``5`` seconds is truncated.

We provide the same model for other durations from ``8`` to ``30`` seconds. We also
provide packages for other Qualcomm SoCs. See
`asr-models-qnn-binary <https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models-qnn-binary>`_
for the full list.

.. code-block::

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models-qnn-binary/sherpa-onnx-qnn-SM8850-binary-5-seconds-paraformer-zh-2023-03-28-int8.tar.bz2
  tar xvf sherpa-onnx-qnn-SM8850-binary-5-seconds-paraformer-zh-2023-03-28-int8.tar.bz2
  rm sherpa-onnx-qnn-SM8850-binary-5-seconds-paraformer-zh-2023-03-28-int8.tar.bz2

Now copy them to your Qualcomm device. Make sure you have read
:ref:`run-exe-on-your-phone-with-qnn-binary` to copy the required ``*.so`` files
from QNN SDK ``2.40.0.251030`` and set ``ADSP_LIBRARY_PATH`` correctly.

Run it on your device:

.. code-block::

   ./sherpa-onnx-offline \
     --provider=qnn \
     --tokens=./sherpa-onnx-qnn-SM8850-binary-5-seconds-paraformer-zh-2023-03-28-int8/tokens.txt \
     --paraformer.qnn-backend-lib=./libQnnHtp.so \
     --paraformer.qnn-system-lib=./libQnnSystem.so \
     --paraformer.qnn-context-binary=./sherpa-onnx-qnn-SM8850-binary-5-seconds-paraformer-zh-2023-03-28-int8/encoder.bin,./sherpa-onnx-qnn-SM8850-binary-5-seconds-paraformer-zh-2023-03-28-int8/predictor.bin,./sherpa-onnx-qnn-SM8850-binary-5-seconds-paraformer-zh-2023-03-28-int8/decoder.bin \
     ./sherpa-onnx-qnn-SM8850-binary-5-seconds-paraformer-zh-2023-03-28-int8/test_wavs/0.wav

or write it in a single line:

.. code-block::

   ./sherpa-onnx-offline --provider=qnn --tokens=./sherpa-onnx-qnn-SM8850-binary-5-seconds-paraformer-zh-2023-03-28-int8/tokens.txt --paraformer.qnn-backend-lib=./libQnnHtp.so --paraformer.qnn-system-lib=./libQnnSystem.so --paraformer.qnn-context-binary=./sherpa-onnx-qnn-SM8850-binary-5-seconds-paraformer-zh-2023-03-28-int8/encoder.bin,./sherpa-onnx-qnn-SM8850-binary-5-seconds-paraformer-zh-2023-03-28-int8/predictor.bin,./sherpa-onnx-qnn-SM8850-binary-5-seconds-paraformer-zh-2023-03-28-int8/decoder.bin ./sherpa-onnx-qnn-SM8850-binary-5-seconds-paraformer-zh-2023-03-28-int8/test_wavs/0.wav

