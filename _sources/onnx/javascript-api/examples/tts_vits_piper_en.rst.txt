TTS: VITS Piper (English)
=========================

Generate speech with the VITS Piper English (GB, cori-medium) model.
VITS is a popular end-to-end TTS architecture. It supports both synchronous
and asynchronous generation.

For model documentation, see
`VITS Piper <https://k2-fsa.github.io/sherpa/onnx/tts/all/>`_.

Source files
------------

- Sync: `test_tts_non_streaming_vits_piper_en.js <https://github.com/k2-fsa/sherpa-onnx/blob/master/nodejs-addon-examples/test_tts_non_streaming_vits_piper_en.js>`_
- Async: `test_tts_non_streaming_vits_piper_en_async.js <https://github.com/k2-fsa/sherpa-onnx/blob/master/nodejs-addon-examples/test_tts_non_streaming_vits_piper_en_async.js>`_

Synchronous generation
----------------------

.. literalinclude:: ../code/tts_vits_piper_en.js
   :language: javascript
   :linenos:

Asynchronous generation
-----------------------

.. literalinclude:: ../code/tts_vits_piper_en_async.js
   :language: javascript
   :linenos:

How to run
----------

1. Install the package::

     npm install sherpa-onnx-node

2. Download the model::

     curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-low.tar.bz2
     tar xf vits-piper-en_US-amy-low.tar.bz2

3. Set the library path and run:

   .. code-block:: bash

      # macOS
      export DYLD_LIBRARY_PATH=$(npm root)/sherpa-onnx-node/lib:$DYLD_LIBRARY_PATH

      # Linux
      export LD_LIBRARY_PATH=$(npm root)/sherpa-onnx-node/lib:$LD_LIBRARY_PATH

      # Choose one:
      node tts_vits_piper_en.js
      node tts_vits_piper_en_async.js

Notes
-----

- The config key is ``vits`` with fields: ``model``, ``tokens``, ``dataDir``.
- VITS models from Piper are self-contained (no separate vocoder needed).
- ``dataDir`` points to the espeak-ng data directory for phoneme conversion.
- The sync API uses ``new sherpa_onnx.OfflineTts(config)`` and
  ``tts.generate({text, generationConfig})``.
- The async API uses ``OfflineTts.createAsync()`` and ``tts.generateAsync()``
  with an ``onProgress`` callback.
