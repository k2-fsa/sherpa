TTS: Supertonic
================

Generate speech with the Supertonic 3 model. Supertonic supports 31 languages
and provides sync, async, and real-time playback modes.

Source files
------------

- Sync: `test_tts_non_streaming_supertonic_en.js <https://github.com/k2-fsa/sherpa-onnx/blob/master/nodejs-addon-examples/test_tts_non_streaming_supertonic_en.js>`_
- Async: `test_tts_non_streaming_supertonic_en_async.js <https://github.com/k2-fsa/sherpa-onnx/blob/master/nodejs-addon-examples/test_tts_non_streaming_supertonic_en_async.js>`_
- Play async: `test_tts_non_streaming_supertonic_en_play_async.js <https://github.com/k2-fsa/sherpa-onnx/blob/master/nodejs-addon-examples/test_tts_non_streaming_supertonic_en_play_async.js>`_

Synchronous generation
----------------------

.. literalinclude:: ../code/offline_tts_sync.js
   :language: javascript
   :linenos:

Asynchronous generation
-----------------------

.. literalinclude:: ../code/offline_tts_async.js
   :language: javascript
   :linenos:

Asynchronous generation with real-time playback
-----------------------------------------------

.. literalinclude:: ../code/offline_tts_play_async.js
   :language: javascript
   :linenos:

How to run
----------

1. Install the packages::

     npm install sherpa-onnx-node
     npm install speaker  # only needed for play_async

2. Download the model::

     curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-supertonic-3-tts-int8-2026-05-11.tar.bz2
     tar xf sherpa-onnx-supertonic-3-tts-int8-2026-05-11.tar.bz2
     rm sherpa-onnx-supertonic-3-tts-int8-2026-05-11.tar.bz2

3. Set the library path and run:

   .. code-block:: bash

      # macOS
      export DYLD_LIBRARY_PATH=$(npm root)/sherpa-onnx-node/lib:$DYLD_LIBRARY_PATH

      # Linux
      export LD_LIBRARY_PATH=$(npm root)/sherpa-onnx-node/lib:$LD_LIBRARY_PATH

      # Choose one:
      node offline_tts_sync.js
      node offline_tts_async.js
      node offline_tts_play_async.js

Notes
-----

- The config key is ``supertonic`` with 7 model files: ``durationPredictor``,
  ``textEncoder``, ``vectorEstimator``, ``vocoder``, ``ttsJson``,
  ``unicodeIndexer``, ``voiceStyle``.
- ``GenerationConfig`` fields for Supertonic:
  - ``sid``: Speaker ID (range 0-9).
  - ``speed``: Speech speed (1.0 = normal).
  - ``numSteps``: Number of diffusion steps (e.g., 8).
  - ``extra.lang``: ISO 639-1 language code. Supported: ``ar``, ``bg``,
    ``cs``, ``da``, ``de``, ``el``, ``en``, ``es``, ``et``, ``fi``, ``fr``,
    ``hi``, ``hr``, ``hu``, ``id``, ``it``, ``ja``, ``ko``, ``lt``, ``lv``,
    ``nl``, ``pl``, ``pt``, ``ro``, ``ru``, ``sk``, ``sl``, ``sv``, ``tr``,
    ``uk``, ``vi``.
- The async API uses ``OfflineTts.createAsync()`` and ``tts.generateAsync()``
  with an ``onProgress`` callback.
- The play_async mode pipes audio chunks to the ``speaker`` npm package for
  immediate playback during generation.
