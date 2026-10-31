TTS: Kokoro (English)
=====================

Generate speech with the Kokoro English (v0.19) model. Kokoro is a
high-quality TTS model with multiple speaker voices. It supports both
synchronous and asynchronous generation.

For model documentation, see
`Kokoro English <https://k2-fsa.github.io/sherpa/onnx/tts/all/>`_.

Source files
------------

- Sync: `test_tts_non_streaming_kokoro_en.js <https://github.com/k2-fsa/sherpa-onnx/blob/master/nodejs-addon-examples/test_tts_non_streaming_kokoro_en.js>`_
- Async: `test_tts_non_streaming_kokoro_en_async.js <https://github.com/k2-fsa/sherpa-onnx/blob/master/nodejs-addon-examples/test_tts_non_streaming_kokoro_en_async.js>`_

Synchronous generation
----------------------

.. literalinclude:: ../code/tts_kokoro_en.js
   :language: javascript
   :linenos:

Asynchronous generation
-----------------------

.. literalinclude:: ../code/tts_kokoro_en_async.js
   :language: javascript
   :linenos:

How to run
----------

1. Install the package::

     npm install sherpa-onnx-node

2. Download the model::

     curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-en-v0_19.tar.bz2
     tar xf kokoro-en-v0_19.tar.bz2
     rm kokoro-en-v0_19.tar.bz2

3. Set the library path and run:

   .. code-block:: bash

      # macOS
      export DYLD_LIBRARY_PATH=$(npm root)/sherpa-onnx-node/lib:$DYLD_LIBRARY_PATH

      # Linux
      export LD_LIBRARY_PATH=$(npm root)/sherpa-onnx-node/lib:$LD_LIBRARY_PATH

      # Choose one:
      node tts_kokoro_en.js
      node tts_kokoro_en_async.js

Notes
-----

- The config key is ``kokoro`` with fields: ``model``, ``voices``, ``tokens``,
  ``dataDir``.
- ``sid`` selects the speaker voice (valid range depends on the model).
- The sync API uses ``new sherpa_onnx.OfflineTts(config)`` and
  ``tts.generate({text, generationConfig})``.
- The async API uses ``OfflineTts.createAsync()`` and ``tts.generateAsync()``
  with an ``onProgress`` callback.
- For Chinese+English, see :doc:`./tts_kokoro_zh_en`.
