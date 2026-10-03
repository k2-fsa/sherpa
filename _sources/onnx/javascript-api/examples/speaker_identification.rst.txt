Speaker Identification
=======================

Identify and verify speakers using speaker embeddings. This example enrolls
two speakers, then identifies test utterances and verifies a specific speaker.

Source file
-----------

`nodejs-addon-examples/test_speaker_identification.js <https://github.com/k2-fsa/sherpa-onnx/blob/master/nodejs-addon-examples/test_speaker_identification.js>`_

Code
----

.. literalinclude:: ../code/speaker_identification.js
   :language: javascript
   :linenos:

How to run
----------

1. Install the package::

     npm install sherpa-onnx-node

2. Download the embedding model::

     curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx

3. Download test audio files::

     git clone https://github.com/csukuangfj/sr-data

4. Set the library path and run:

   .. code-block:: bash

      # macOS
      export DYLD_LIBRARY_PATH=$(npm root)/sherpa-onnx-node/lib:$DYLD_LIBRARY_PATH

      # Linux
      export LD_LIBRARY_PATH=$(npm root)/sherpa-onnx-node/lib:$LD_LIBRARY_PATH

      node speaker_identification.js

Expected output
^^^^^^^^^^^^^^^

.. code-block:: text

   --- All speakers ---
   [ 'fangjun', 'leijun' ]
   --------------------
   ./sr-data/test/fangjun-test-sr-1.wav: fangjun
   ./sr-data/test/leijun-test-sr-1.wav: leijun
   ./sr-data/test/liudehua-test-sr-1.wav: <Unknown>

Notes
-----

- ``SpeakerEmbeddingExtractor`` computes a fixed-dimensional embedding vector
  from a WAV file.
- ``SpeakerEmbeddingManager`` stores embeddings and provides three operations:

  - ``search({v, threshold})``: Find the best matching speaker, or return
    ``''`` if none match above the threshold.
  - ``verify({name, v, threshold})``: Check if the embedding matches a
    specific enrolled speaker.
  - ``addMulti({name, v})``: Enroll a speaker with multiple utterances for
    better accuracy.

- The ``threshold`` controls the trade-off between false accepts and false
  rejects. Tune it for your use case.
