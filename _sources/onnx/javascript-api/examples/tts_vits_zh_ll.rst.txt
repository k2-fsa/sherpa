TTS: VITS (Chinese, zh-ll)
===========================

Generate speech with the VITS Chinese (sherpa-onnx-vits-zh-ll) model.
This model uses a lexicon and rule FSTs for Chinese text normalization.
It supports both synchronous and asynchronous generation.

For model documentation, see
`VITS zh-ll <https://k2-fsa.github.io/sherpa/onnx/tts/all/>`_.

Source files
------------

- Sync: `test_tts_non_streaming_vits_zh_ll.js <https://github.com/k2-fsa/sherpa-onnx/blob/master/nodejs-addon-examples/test_tts_non_streaming_vits_zh_ll.js>`_
- Async: `test_tts_non_streaming_vits_zh_ll_async.js <https://github.com/k2-fsa/sherpa-onnx/blob/master/nodejs-addon-examples/test_tts_non_streaming_vits_zh_ll_async.js>`_

Synchronous generation
----------------------

.. literalinclude:: ../code/tts_vits_zh_ll.js
   :language: javascript
   :linenos:

Asynchronous generation
-----------------------

.. literalinclude:: ../code/tts_vits_zh_ll_async.js
   :language: javascript
   :linenos:

How to run
----------

1. Install the package::

     npm install sherpa-onnx-node

2. Download the model::

     curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-vits-zh-ll.tar.bz2
     tar xvf sherpa-onnx-vits-zh-ll.tar.bz2

3. Set the library path and run:

   .. code-block:: bash

      # macOS
      export DYLD_LIBRARY_PATH=$(npm root)/sherpa-onnx-node/lib:$DYLD_LIBRARY_PATH

      # Linux
      export LD_LIBRARY_PATH=$(npm root)/sherpa-onnx-node/lib:$LD_LIBRARY_PATH

      # Choose one:
      node tts_vits_zh_ll.js
      node tts_vits_zh_ll_async.js

Notes
-----

- The config key is ``vits`` with fields: ``model``, ``tokens``, ``lexicon``.
- ``ruleFsts`` contains comma-separated FST files: ``date.fst``, ``phone.fst``,
  ``number.fst``.
- The example text contains dates, phone numbers, and monetary amounts
  that are normalized by the rule FSTs.
- The sync API uses ``new sherpa_onnx.OfflineTts(config)`` and
  ``tts.generate({text, generationConfig})``.
- The async API uses ``OfflineTts.createAsync()`` and ``tts.generateAsync()``
  with an ``onProgress`` callback.
