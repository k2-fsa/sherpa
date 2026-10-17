TTS: VITS Coqui (German)
=========================

Generate speech with the VITS Coqui German (CSS10) model. It supports both
synchronous and asynchronous generation.

For model documentation, see
`VITS Coqui <https://k2-fsa.github.io/sherpa/onnx/tts/all/>`_.

Source files
------------

- Sync: `test_tts_non_streaming_vits_coqui_de.js <https://github.com/k2-fsa/sherpa-onnx/blob/master/nodejs-addon-examples/test_tts_non_streaming_vits_coqui_de.js>`_
- Async: `test_tts_non_streaming_vits_coqui_de_async.js <https://github.com/k2-fsa/sherpa-onnx/blob/master/nodejs-addon-examples/test_tts_non_streaming_vits_coqui_de_async.js>`_

Synchronous generation
----------------------

.. literalinclude:: ../code/tts_vits_coqui_de.js
   :language: javascript
   :linenos:

Asynchronous generation
-----------------------

.. literalinclude:: ../code/tts_vits_coqui_de_async.js
   :language: javascript
   :linenos:

How to run
----------

1. Install the package::

     npm install sherpa-onnx-node

2. Download the model::

     curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-coqui-de-css10.tar.bz2
     tar xf vits-coqui-de-css10.tar.bz2

3. Set the library path and run:

   .. code-block:: bash

      # macOS
      export DYLD_LIBRARY_PATH=$(npm root)/sherpa-onnx-node/lib:$DYLD_LIBRARY_PATH

      # Linux
      export LD_LIBRARY_PATH=$(npm root)/sherpa-onnx-node/lib:$LD_LIBRARY_PATH

      # Choose one:
      node tts_vits_coqui_de.js
      node tts_vits_coqui_de_async.js

Notes
-----

- The config key is ``vits`` with fields: ``model``, ``tokens``.
- This model does not use ``dataDir`` (no espeak-ng dependency).
- The sync API uses ``new sherpa_onnx.OfflineTts(config)`` and
  ``tts.generate({text, generationConfig})``.
- The async API uses ``OfflineTts.createAsync()`` and ``tts.generateAsync()``
  with an ``onProgress`` callback.
