Speaker Diarization
====================

Determine who speaks when in an audio file. This example uses Pyanote
segmentation, speaker embeddings, and clustering to identify and separate
speakers.

Source file
-----------

`nodejs-addon-examples/test_offline_speaker_diarization.js <https://github.com/k2-fsa/sherpa-onnx/blob/master/nodejs-addon-examples/test_offline_speaker_diarization.js>`_

Code
----

.. literalinclude:: ../code/speaker_diarization.js
   :language: javascript
   :linenos:

How to run
----------

1. Install the package::

     npm install sherpa-onnx-node

2. Download the models and test file::

     curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
     tar xvf sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
     rm sherpa-onnx-pyannote-segmentation-3-0.tar.bz2

     curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx

     curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/0-four-speakers-zh.wav

3. Set the library path and run:

   .. code-block:: bash

      # macOS
      export DYLD_LIBRARY_PATH=$(npm root)/sherpa-onnx-node/lib:$DYLD_LIBRARY_PATH

      # Linux
      export LD_LIBRARY_PATH=$(npm root)/sherpa-onnx-node/lib:$LD_LIBRARY_PATH

      node speaker_diarization.js

Expected output
^^^^^^^^^^^^^^^

.. code-block:: text

   Segments:
     Speaker 0: 0.20s - 5.40s
     Speaker 1: 5.80s - 12.30s
     Speaker 2: 12.60s - 18.90s
     Speaker 0: 19.20s - 25.10s
     Speaker 3: 25.50s - 31.20s

Notes
-----

- The config has three parts: ``segmentation`` (detects speech/non-speech
  boundaries), ``embedding`` (computes speaker vectors), and ``clustering``
  (groups segments by speaker).
- Set ``clustering.numClusters`` to the expected number of speakers if known,
  or ``-1`` to let the algorithm decide automatically using the threshold.
- ``minDurationOn`` discards segments shorter than the given seconds.
- ``minDurationOff`` merges two segments if the gap is less than the given
  seconds.
- Each returned segment has ``start``, ``end``, and ``speaker`` fields.
