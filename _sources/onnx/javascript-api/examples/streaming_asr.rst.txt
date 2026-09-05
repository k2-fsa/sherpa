Streaming ASR
=============

Recognize speech in a WAV file using a streaming (online) Zipformer
transducer model. The recognizer processes audio incrementally, producing
partial results as more audio arrives.

Source file
-----------

`nodejs-addon-examples/test_asr_streaming_transducer.js <https://github.com/k2-fsa/sherpa-onnx/blob/master/nodejs-addon-examples/test_asr_streaming_transducer.js>`_

Code
----

.. literalinclude:: ../code/streaming_asr.js
   :language: javascript
   :linenos:

How to run
----------

1. Install the package::

     npm install sherpa-onnx-node

2. Download the model and test files::

     curl -LS -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
     tar xvf sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
     rm sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2

3. Set the library path and run:

   .. code-block:: bash

      # macOS
      export DYLD_LIBRARY_PATH=$(npm root)/sherpa-onnx-node/lib:$DYLD_LIBRARY_PATH

      # Linux
      export LD_LIBRARY_PATH=$(npm root)/sherpa-onnx-node/lib:$LD_LIBRARY_PATH

      node streaming_asr.js

Expected output
^^^^^^^^^^^^^^^

.. code-block:: text

   Wave duration 6.625 seconds
   Elapsed 0.234 seconds
   RTF = 0.234/6.625 = 0.035
   ./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/test_wavs/0.wav
   result
    { text: ' 吃饭了吗', tokens: [...], timestamps: [...] }

Notes
-----

- ``OnlineRecognizer`` is the streaming recognizer. Call ``createStream()``
  to create a stream, then feed audio with ``acceptWaveform()``.
- Append 0.4 seconds of tail padding (zeros) after the main audio so the
  model can process the last chunk.
- Call ``isReady()`` and ``decode()`` in a loop until no more frames are
  available, then call ``getResult()`` for the final transcription.
- This model supports both Chinese and English.
