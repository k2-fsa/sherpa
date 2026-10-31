TTS: Matcha (English)
=====================

Generate speech with the Matcha English (ljspeech) model. Matcha uses
a separate vocoder model (Vocos) for waveform synthesis and supports both
synchronous and asynchronous generation.

For model documentation, see
`Matcha English <https://k2-fsa.github.io/sherpa/onnx/tts/all/>`_.

Source files
------------

- Sync: `test_tts_non_streaming_matcha_icefall_en.js <https://github.com/k2-fsa/sherpa-onnx/blob/master/nodejs-addon-examples/test_tts_non_streaming_matcha_icefall_en.js>`_
- Async: `test_tts_non_streaming_matcha_icefall_en_async.js <https://github.com/k2-fsa/sherpa-onnx/blob/master/nodejs-addon-examples/test_tts_non_streaming_matcha_icefall_en_async.js>`_

Synchronous generation
----------------------

.. literalinclude:: ../code/tts_matcha_en.js
   :language: javascript
   :linenos:

Asynchronous generation
-----------------------

.. literalinclude:: ../code/tts_matcha_en_async.js
   :language: javascript
   :linenos:

How to run
----------

1. Install the package::

     npm install sherpa-onnx-node

2. Download the model and vocoder::

     curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-en_US-ljspeech.tar.bz2
     tar xvf matcha-icefall-en_US-ljspeech.tar.bz2
     rm matcha-icefall-en_US-ljspeech.tar.bz2

     curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos-22khz-univ.onnx

3. Set the library path and run:

   .. code-block:: bash

      # macOS
      export DYLD_LIBRARY_PATH=$(npm root)/sherpa-onnx-node/lib:$DYLD_LIBRARY_PATH

      # Linux
      export LD_LIBRARY_PATH=$(npm root)/sherpa-onnx-node/lib:$LD_LIBRARY_PATH

      # Choose one:
      node tts_matcha_en.js
      node tts_matcha_en_async.js

Notes
-----

- The config key is ``matcha`` with fields: ``acousticModel``, ``vocoder``,
  ``tokens``, ``dataDir``.
- Matcha requires a separate vocoder model. Download ``vocos-22khz-univ.onnx``
  and place it in the working directory.
- The sync API uses ``new sherpa_onnx.OfflineTts(config)`` and
  ``tts.generate({text, generationConfig})``.
- The async API uses ``OfflineTts.createAsync()`` and ``tts.generateAsync()``
  with an ``onProgress`` callback.
- For Chinese, see :doc:`./tts_matcha_zh`.
