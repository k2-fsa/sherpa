TTS: Kokoro (Chinese + English)
=================================

Generate speech with the Kokoro multi-language (Chinese+English, v1.0) model.
This model supports mixed Chinese-English text with multiple speaker voices
and both synchronous and asynchronous generation.

For model documentation, see
`Kokoro Chinese+English <https://k2-fsa.github.io/sherpa/onnx/tts/all/>`_.

Source files
------------

- Sync: `test_tts_non_streaming_kokoro_zh_en.js <https://github.com/k2-fsa/sherpa-onnx/blob/master/nodejs-addon-examples/test_tts_non_streaming_kokoro_zh_en.js>`_
- Async: `test_tts_non_streaming_kokoro_zh_en_async.js <https://github.com/k2-fsa/sherpa-onnx/blob/master/nodejs-addon-examples/test_tts_non_streaming_kokoro_zh_en_async.js>`_

Synchronous generation
----------------------

.. literalinclude:: ../code/tts_kokoro_zh_en.js
   :language: javascript
   :linenos:

Asynchronous generation
-----------------------

.. literalinclude:: ../code/tts_kokoro_zh_en_async.js
   :language: javascript
   :linenos:

How to run
----------

1. Install the package::

     npm install sherpa-onnx-node

2. Download the model::

     curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-multi-lang-v1_0.tar.bz2
     tar xf kokoro-multi-lang-v1_0.tar.bz2
     rm kokoro-multi-lang-v1_0.tar.bz2

3. Set the library path and run:

   .. code-block:: bash

      # macOS
      export DYLD_LIBRARY_PATH=$(npm root)/sherpa-onnx-node/lib:$DYLD_LIBRARY_PATH

      # Linux
      export LD_LIBRARY_PATH=$(npm root)/sherpa-onnx-node/lib:$LD_LIBRARY_PATH

      # Choose one:
      node tts_kokoro_zh_en.js
      node tts_kokoro_zh_en_async.js

Notes
-----

- This model uses the same ``kokoro`` config key as the English model, but
  adds a ``lexicon`` field with comma-separated lexicon files for each
  language (e.g., ``lexicon-us-en.txt,lexicon-zh.txt``).
- ``sid: 48`` selects a specific speaker voice. The multi-lang model has
  more speakers than the English-only model.
- The sync API uses ``new sherpa_onnx.OfflineTts(config)`` and
  ``tts.generate({text, generationConfig})``.
- The async API uses ``OfflineTts.createAsync()`` and ``tts.generateAsync()``
  with an ``onProgress`` callback.
- For English-only, see :doc:`./tts_kokoro_en`.
