Keyword Spotting
=================

Detect predefined keywords in audio using a streaming Zipformer transducer
model. The model listens for specific words or phrases and reports when
they are detected.

Source file
-----------

`nodejs-addon-examples/test_keyword_spotter_transducer.js <https://github.com/k2-fsa/sherpa-onnx/blob/master/nodejs-addon-examples/test_keyword_spotter_transducer.js>`_

Code
----

.. literalinclude:: ../code/keyword_spotter.js
   :language: javascript
   :linenos:

How to run
----------

1. Install the package::

     npm install sherpa-onnx-node

2. Download the model and test files::

     curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2
     tar xvf sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2
     rm sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2

3. Set the library path and run:

   .. code-block:: bash

      # macOS
      export DYLD_LIBRARY_PATH=$(npm root)/sherpa-onnx-node/lib:$DYLD_LIBRARY_PATH

      # Linux
      export LD_LIBRARY_PATH=$(npm root)/sherpa-onnx-node/lib:$LD_LIBRARY_PATH

      node keyword_spotter.js

Expected output
^^^^^^^^^^^^^^^

.. code-block:: text

   Wave duration 3.456 seconds
   Elapsed 0.123 seconds
   RTF = 0.123/3.456 = 0.036
   ./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/test_wavs/3.wav
   Detected keywords: [ '你好', '小爱同学' ]

Notes
-----

- The ``keywordsFile`` in the config specifies a text file containing the
  keywords to detect, one per line. You can edit this file to change the
  keywords.
- Like streaming ASR, the keyword spotter uses a loop of ``isReady()`` and
  ``decode()`` calls, and checks ``getResult().keyword`` after each decode.
- Append 0.4 seconds of tail padding after the main audio.
- An empty keyword string means no keyword was detected in that decode step.
