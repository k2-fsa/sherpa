TTS: Pocket (Voice Cloning)
============================

Generate speech with the Pocket TTS model using voice cloning. Pocket uses
a reference audio clip to clone the speaker's voice for the generated speech.

Source file
-----------

`nodejs-addon-examples/test_tts_non_streaming_pocket_en.js <https://github.com/k2-fsa/sherpa-onnx/blob/master/nodejs-addon-examples/test_tts_non_streaming_pocket_en.js>`_

Code
----

.. literalinclude:: ../code/tts_pocket_sync.js
   :language: javascript
   :linenos:

How to run
----------

1. Install the package::

     npm install sherpa-onnx-node

2. Download the model::

     curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2
     tar xf sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2
     rm sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2

3. Set the library path and run:

   .. code-block:: bash

      # macOS
      export DYLD_LIBRARY_PATH=$(npm root)/sherpa-onnx-node/lib:$DYLD_LIBRARY_PATH

      # Linux
      export LD_LIBRARY_PATH=$(npm root)/sherpa-onnx-node/lib:$LD_LIBRARY_PATH

      node tts_pocket_sync.js

Notes
-----

- Pocket TTS uses voice cloning via ``referenceAudio`` in the
  ``GenerationConfig``. Provide a WAV file of the target speaker.
- The config key is ``pocket`` with fields: ``lmFlow``, ``lmMain``,
  ``encoder``, ``decoder``, ``textConditioner``, ``vocabJson``,
  ``tokenScoresJson``, ``voiceEmbeddingCacheCapacity``.
- ``GenerationConfig`` fields for Pocket:
  - ``referenceAudio``: Float32Array of the reference audio samples.
  - ``referenceSampleRate``: Sample rate of the reference audio.
  - ``numSteps``: Number of diffusion steps (e.g., 5).
  - ``extra.max_reference_audio_len``: Max reference audio length in seconds.
  - ``extra.seed``: Random seed for reproducibility.
- Pocket also supports async generation with ``createAsync()`` and
  ``generateAsync()``. See the
  `async example <https://github.com/k2-fsa/sherpa-onnx/blob/master/nodejs-addon-examples/test_tts_non_streaming_pocket_en_async.js>`_
  and
  `play async example <https://github.com/k2-fsa/sherpa-onnx/blob/master/nodejs-addon-examples/test_tts_non_streaming_pocket_en_play_async.js>`_.
