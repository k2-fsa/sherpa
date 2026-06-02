Audio Tagging
=============

Classify audio events in WAV files using a CED (Conditional Event Detection)
model. The model identifies sounds like speech, music, animal calls, etc.

Source file
-----------

`nodejs-addon-examples/test_audio_tagging_ced.js <https://github.com/k2-fsa/sherpa-onnx/blob/master/nodejs-addon-examples/test_audio_tagging_ced.js>`_

Code
----

.. literalinclude:: ../code/audio_tagging.js
   :language: javascript
   :linenos:

How to run
----------

1. Install the package::

     npm install sherpa-onnx-node

2. Download the model and test files::

     curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/audio-tagging-models/sherpa-onnx-ced-mini-audio-tagging-2024-04-19.tar.bz2
     tar xvf sherpa-onnx-ced-mini-audio-tagging-2024-04-19.tar.bz2
     rm sherpa-onnx-ced-mini-audio-tagging-2024-04-19.tar.bz2

3. Set the library path and run:

   .. code-block:: bash

      # macOS
      export DYLD_LIBRARY_PATH=$(npm root)/sherpa-onnx-node/lib:$DYLD_LIBRARY_PATH

      # Linux
      export LD_LIBRARY_PATH=$(npm root)/sherpa-onnx-node/lib:$LD_LIBRARY_PATH

      node audio_tagging.js

Expected output
^^^^^^^^^^^^^^^

.. code-block:: text

   ------
   input file: ./sherpa-onnx-ced-mini-audio-tagging-2024-04-19/test_wavs/1.wav
   Probability		Name
   0.987			Speech
   0.654			Narration, monologue
   0.321			Conversation
   0.123			Inside, small room
   0.045			Telephone
   Wave duration 3.200 seconds
   Elapsed 0.045 seconds
   RTF = 0.045/3.200 = 0.014
   ------

Notes
-----

- ``AudioTagging`` supports two model types: ``ced`` and ``zipformer``.
- ``topK`` controls how many events are returned (default 5).
- Each event has ``prob`` (probability) and ``name`` (event label).
- The ``labels`` file (``class_labels_indices.csv``) maps model output
  indices to human-readable event names.
