Real-time speech recognition from a microphone
==============================================

In this section, we demonstrate how to use the Python API of `sherpa-onnx`_
for real-time speech recognition with a microphone.

With endpoint detection
-----------------------

.. code-block:: bash

   cd /path/to/sherpa-onnx

   python3 ./python-api-examples/speech-recognition-from-microphone-with-endpoint-detection.py \
     --tokens=./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt \
     --encoder=./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.onnx \
     --decoder=./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx \
     --joiner=./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.onnx

.. hint::

   ``speech-recognition-from-microphone-with-endpoint-detection.py`` is from `<https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/speech-recognition-from-microphone-with-endpoint-detection.py>`_

   In the above demo, the model files are
   from :ref:`sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20`.

Without endpoint detection
--------------------------

.. code-block:: bash

   cd /path/to/sherpa-onnx

   python3 ./python-api-examples/speech-recognition-from-microphone.py \
     --tokens=./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt \
     --encoder=./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.onnx \
     --decoder=./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx \
     --joiner=./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.onnx

.. hint::

   ``speech-recognition-from-microphone.py`` is from `<https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/speech-recognition-from-microphone.py>`_

   In the above demo, the model files are
   from :ref:`sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20`.
