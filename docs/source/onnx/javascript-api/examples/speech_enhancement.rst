Speech Enhancement
===================

Remove background noise from audio using a GTCRN (Global Token Channel
Attention Recurrent Network) model. This is useful for cleaning up noisy
recordings before transcription.

Source file
-----------

`nodejs-addon-examples/test_offline_speech_enhancement_gtcrn.js <https://github.com/k2-fsa/sherpa-onnx/blob/master/nodejs-addon-examples/test_offline_speech_enhancement_gtcrn.js>`_

Code
----

.. literalinclude:: ../code/speech_enhancement.js
   :language: javascript
   :linenos:

How to run
----------

1. Install the package::

     npm install sherpa-onnx-node

2. Download the model and test file::

     curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/gtcrn_simple.onnx
     curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/inp_16k.wav

3. Set the library path and run:

   .. code-block:: bash

      # macOS
      export DYLD_LIBRARY_PATH=$(npm root)/sherpa-onnx-node/lib:$DYLD_LIBRARY_PATH

      # Linux
      export LD_LIBRARY_PATH=$(npm root)/sherpa-onnx-node/lib:$LD_LIBRARY_PATH

      node speech_enhancement.js

Expected output
^^^^^^^^^^^^^^^

.. code-block:: text

   Saved to ./enhanced-16k.wav

Notes
-----

- ``OfflineSpeechDenoiser`` processes the entire audio file at once.
- ``run()`` accepts ``{samples, sampleRate, enableExternalBuffer}`` and
  returns ``{samples, sampleRate}``.
- ``enableExternalBuffer: true`` enables zero-copy buffer sharing.
- The output sample rate matches the input sample rate (16kHz in this example).
- You can also use ``dpdfnet_baseline.onnx`` as an alternative model.
