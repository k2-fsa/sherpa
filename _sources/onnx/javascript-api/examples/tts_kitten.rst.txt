TTS: Kitten
============

Generate speech with the Kitten Nano English model. Kitten is a lightweight
TTS model that supports both synchronous and asynchronous generation.

For model documentation, see
`Kitten TTS <https://k2-fsa.github.io/sherpa/onnx/tts/all/>`_.

Source files
------------

- Sync: `test_tts_non_streaming_kitten_en_sync.js <https://github.com/k2-fsa/sherpa-onnx/blob/master/nodejs-addon-examples/test_tts_non_streaming_kitten_en_sync.js>`_
- Async: `test_tts_non_streaming_kitten_en.js <https://github.com/k2-fsa/sherpa-onnx/blob/master/nodejs-addon-examples/test_tts_non_streaming_kitten_en.js>`_

Synchronous generation
----------------------

.. literalinclude:: ../code/tts_kitten_sync.js
   :language: javascript
   :linenos:

Asynchronous generation
-----------------------

.. literalinclude:: ../code/tts_kitten_async.js
   :language: javascript
   :linenos:

How to run
----------

1. Install the package::

     npm install sherpa-onnx-node

2. Download the model::

     curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kitten-nano-en-v0_1-fp16.tar.bz2
     tar xf kitten-nano-en-v0_1-fp16.tar.bz2
     rm kitten-nano-en-v0_1-fp16.tar.bz2

3. Set the library path and run:

   .. code-block:: bash

      # macOS
      export DYLD_LIBRARY_PATH=$(npm root)/sherpa-onnx-node/lib:$DYLD_LIBRARY_PATH

      # Linux
      export LD_LIBRARY_PATH=$(npm root)/sherpa-onnx-node/lib:$LD_LIBRARY_PATH

      # Choose one:
      node tts_kitten_sync.js
      node tts_kitten_async.js

Notes
-----

- ``GenerationConfig`` fields: ``sid`` (speaker ID), ``speed`` (1.0 = normal),
  ``silenceScale`` (controls pause length).
- The sync API uses ``new sherpa_onnx.OfflineTts(config)`` and
  ``tts.generate({text, generationConfig})``.
- The async API uses ``OfflineTts.createAsync()`` and ``tts.generateAsync()``
  with an ``onProgress`` callback that receives audio chunks as they are
  generated. Return ``1`` to continue, ``0`` to cancel.
