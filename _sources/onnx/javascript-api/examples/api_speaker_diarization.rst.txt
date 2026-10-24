Speaker Diarization API
=======================

Offline speaker diarization API reference for ``sherpa-onnx-node``.

Source file
-----------

`scripts/node-addon-api/lib/non-streaming-speaker-diarization.js <https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/node-addon-api/lib/non-streaming-speaker-diarization.js>`_

API
---

OfflineSpeakerDiarization
^^^^^^^^^^^^^^^^^^^^^^^^^

Identifies "who spoke when" in an audio recording.

Constructor
"""""""""""

.. code-block:: javascript

   const diarizer = new sherpa_onnx.OfflineSpeakerDiarization(config);

:param config: Configuration object with:

- ``segmentation`` (object, optional) — Segmentation model config:

  - ``pyannote`` — ``{ model: string }`` path to the segmentation ONNX model.
  - ``numThreads`` (number, optional).
  - ``debug`` (boolean, optional).
  - ``provider`` (string, optional).

- ``embedding`` (object, optional) — Speaker embedding model config:

  - ``model`` (string) — Path to the embedding ONNX model.
  - ``numThreads`` (number, optional).
  - ``debug`` (boolean, optional).
  - ``provider`` (string, optional).

- ``clustering`` (object, optional) — Clustering config:

  - ``numClusters`` (number, optional) — Number of speakers (0 = auto).
  - ``threshold`` (number, optional) — Clustering threshold.

- ``minDurationOn`` (number, optional) — Min speaker segment duration.
- ``minDurationOff`` (number, optional) — Min non-speech duration.

Methods
"""""""

``diarizer.process(samples)``
..............................

Run diarization on the input audio.

:param samples: Audio samples in ``[-1, 1]`` (``Float32Array``).
:returns: An array of ``SpeakerDiarizationSegment`` objects, each with:

  - ``start`` (number) — Start time in seconds.
  - ``end`` (number) — End time in seconds.
  - ``speaker`` (number) — Speaker ID (integer).

``diarizer.setConfig(config)``
...............................

Update clustering configuration at runtime.

:param config: ``{ clustering: { numClusters?, threshold? } }``.

Properties
""""""""""

- ``diarizer.config`` — The configuration object.
- ``diarizer.sampleRate`` — Expected sample rate in Hz (number).

Example
^^^^^^^

.. code-block:: javascript

   const sherpa_onnx = require('sherpa-onnx-node');

   const diarizer = new sherpa_onnx.OfflineSpeakerDiarization({
     segmentation: { pyannote: { model: './segmentation-3-0.onnx' } },
     embedding: { model: './3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx' },
     clustering: { numClusters: 0, threshold: 0.5 },
   });

   const wave = sherpa_onnx.readWave('./audio.wav');
   const segments = diarizer.process(wave.samples);

   for (const seg of segments) {
     console.log(`Speaker ${seg.speaker}: ${seg.start.toFixed(2)}s - ${seg.end.toFixed(2)}s`);
   }

Notes
-----

- The input audio should be mono, 16kHz, float32 in ``[-1, 1]``.
- Set ``numClusters: 0`` to auto-detect the number of speakers.
- Use ``setConfig()`` to adjust clustering parameters without re-creating the diarizer.
