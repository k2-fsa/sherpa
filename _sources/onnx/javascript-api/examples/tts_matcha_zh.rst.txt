TTS: Matcha (Chinese)
=====================

Generate speech with the Matcha Chinese (baker) model. This model requires
a vocoder and rule FSTs for Chinese text normalization. It supports both
synchronous and asynchronous generation.

For model documentation, see
`Matcha Chinese <https://k2-fsa.github.io/sherpa/onnx/tts/all/>`_.

Source files
------------

- Sync: `test_tts_non_streaming_matcha_icefall_zh.js <https://github.com/k2-fsa/sherpa-onnx/blob/master/nodejs-addon-examples/test_tts_non_streaming_matcha_icefall_zh.js>`_
- Async: `test_tts_non_streaming_matcha_icefall_zh_async.js <https://github.com/k2-fsa/sherpa-onnx/blob/master/nodejs-addon-examples/test_tts_non_streaming_matcha_icefall_zh_async.js>`_

Synchronous generation
----------------------

.. literalinclude:: ../code/tts_matcha_zh.js
   :language: javascript
   :linenos:

Asynchronous generation
-----------------------

.. literalinclude:: ../code/tts_matcha_zh_async.js
   :language: javascript
   :linenos:

How to run
----------

1. Install the package::

     npm install sherpa-onnx-node

2. Download the model, vocoder, and rule FSTs::

     curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-zh-baker.tar.bz2
     tar xvf matcha-icefall-zh-baker.tar.bz2
     rm matcha-icefall-zh-baker.tar.bz2

     curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos-22khz-univ.onnx

3. Set the library path and run:

   .. code-block:: bash

      # macOS
      export DYLD_LIBRARY_PATH=$(npm root)/sherpa-onnx-node/lib:$DYLD_LIBRARY_PATH

      # Linux
      export LD_LIBRARY_PATH=$(npm root)/sherpa-onnx-node/lib:$LD_LIBRARY_PATH

      # Choose one:
      node tts_matcha_zh.js
      node tts_matcha_zh_async.js

Notes
-----

- In addition to the ``matcha`` config fields, the Chinese model uses:
  - ``lexicon``: Maps Chinese characters to phonemes.
  - ``ruleFsts``: Comma-separated FST files for phone, date, and number
    normalization (e.g., ``phone.fst,date.fst,number.fst``).
- The text contains dates, phone numbers, and monetary amounts that are
  normalized by the rule FSTs before synthesis.
- The sync API uses ``new sherpa_onnx.OfflineTts(config)`` and
  ``tts.generate({text, generationConfig})``.
- The async API uses ``OfflineTts.createAsync()`` and ``tts.generateAsync()``
  with an ``onProgress`` callback.
- For English, see :doc:`./tts_matcha_en`.
