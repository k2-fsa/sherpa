TTS: ZipVoice (Voice Cloning)
==============================

Generate speech with the ZipVoice model using voice cloning. ZipVoice uses
a reference audio clip and its transcript to clone the speaker's voice.

Source files
------------

- Sync: `test_tts_non_streaming_zipvoice_zh_en.js <https://github.com/k2-fsa/sherpa-onnx/blob/master/nodejs-addon-examples/test_tts_non_streaming_zipvoice_zh_en.js>`_
- Async: `test_tts_non_streaming_zipvoice_zh_en_async.js <https://github.com/k2-fsa/sherpa-onnx/blob/master/nodejs-addon-examples/test_tts_non_streaming_zipvoice_zh_en_async.js>`_
- Play async: `test_tts_non_streaming_zipvoice_zh_en_play_async.js <https://github.com/k2-fsa/sherpa-onnx/blob/master/nodejs-addon-examples/test_tts_non_streaming_zipvoice_zh_en_play_async.js>`_

Synchronous generation
----------------------

.. literalinclude:: ../code/tts_zipvoice_sync.js
   :language: javascript
   :linenos:

How to run
----------

1. Install the packages::

     npm install sherpa-onnx-node
     npm install speaker  # only needed for play_async

2. Download the model and vocoder::

     curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-zipvoice-distill-int8-zh-en-emilia.tar.bz2
     tar xf sherpa-onnx-zipvoice-distill-int8-zh-en-emilia.tar.bz2
     rm sherpa-onnx-zipvoice-distill-int8-zh-en-emilia.tar.bz2

     curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos_24khz.onnx

3. Set the library path and run:

   .. code-block:: bash

      # macOS
      export DYLD_LIBRARY_PATH=$(npm root)/sherpa-onnx-node/lib:$DYLD_LIBRARY_PATH

      # Linux
      export LD_LIBRARY_PATH=$(npm root)/sherpa-onnx-node/lib:$LD_LIBRARY_PATH

      node tts_zipvoice_sync.js

Notes
-----

- ZipVoice requires a reference audio AND its transcript for voice cloning.
  The ``GenerationConfig`` must include:
  - ``referenceAudio``: Float32Array of the reference audio samples.
  - ``referenceSampleRate``: Sample rate of the reference audio.
  - ``referenceText``: Transcript of the reference audio.
  - ``numSteps``: Number of diffusion steps (e.g., 4).
  - ``extra.min_char_in_sentence``: Minimum characters per sentence.
- The config key is ``zipvoice`` with fields: ``tokens``, ``encoder``,
  ``decoder``, ``vocoder``, ``dataDir``, ``lexicon``.
- ZipVoice also supports async generation. See the
  `async example <https://github.com/k2-fsa/sherpa-onnx/blob/master/nodejs-addon-examples/test_tts_non_streaming_zipvoice_zh_en_async.js>`_
  and
  `play async example <https://github.com/k2-fsa/sherpa-onnx/blob/master/nodejs-addon-examples/test_tts_non_streaming_zipvoice_zh_en_play_async.js>`_.
