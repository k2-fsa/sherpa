Non-Streaming ASR
==================

Recognize speech in a WAV file using a non-streaming (offline) Zipformer
transducer model. The recognizer processes the entire audio file at once.

Source file
-----------

`nodejs-addon-examples/test_asr_non_streaming_transducer.js <https://github.com/k2-fsa/sherpa-onnx/blob/master/nodejs-addon-examples/test_asr_non_streaming_transducer.js>`_

Code
----

.. literalinclude:: ../code/non_streaming_asr.js
   :language: javascript
   :linenos:

How to run
----------

1. Install the package::

     npm install sherpa-onnx-node

2. Download the model and test files::

     curl -LS -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-en-2023-04-01.tar.bz2
     tar xvf sherpa-onnx-zipformer-en-2023-04-01.tar.bz2
     rm sherpa-onnx-zipformer-en-2023-04-01.tar.bz2

3. Set the library path and run:

   .. code-block:: bash

      # macOS
      export DYLD_LIBRARY_PATH=$(npm root)/sherpa-onnx-node/lib:$DYLD_LIBRARY_PATH

      # Linux
      export LD_LIBRARY_PATH=$(npm root)/sherpa-onnx-node/lib:$LD_LIBRARY_PATH

      node non_streaming_asr.js

Expected output
^^^^^^^^^^^^^^^

.. code-block:: text

   Wave duration 5.280 seconds
   Elapsed 0.156 seconds
   RTF = 0.156/5.280 = 0.030
   ./sherpa-onnx-zipformer-en-2023-04-01/test_wavs/1.wav
   result
    { text: ' ...', tokens: [...], timestamps: [...] }

Notes
-----

- ``OfflineRecognizer`` is the non-streaming recognizer. Unlike the
  streaming version, it processes the entire audio in one call.
- The API is simpler: ``createStream()`` -> ``acceptWaveform()`` ->
  ``decode()`` -> ``getResult()``. No loop needed.
- Non-streaming models generally produce more accurate results than
  streaming models, but cannot be used for real-time transcription.
