Audio Tagging API
=================

Audio tagging API reference for ``sherpa-onnx-node``.

Source file
-----------

`scripts/node-addon-api/lib/audio-tagg.js <https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/node-addon-api/lib/audio-tagg.js>`_

API
---

AudioTagging
^^^^^^^^^^^^

Classifies audio into predefined event categories.

Constructor
"""""""""""

.. code-block:: javascript

   const tagger = new sherpa_onnx.AudioTagging(config);

:param config: Configuration object with:

- ``model`` (object, optional) — Model configuration with one of:

  - ``ced`` (string) — Path to the CED ONNX model.
  - ``zipformer`` — ``{ model: string }`` path to the ZipFormer ONNX model.

  Plus common fields: ``numThreads``, ``debug``, ``provider``.

- ``labels`` (string, optional) — Path to the labels file.
- ``topK`` (number, optional) — Number of top results to return.

Methods
"""""""

``tagger.createStream()``
..........................

:returns: A new ``OfflineStream`` for feeding audio.

``tagger.compute(stream, topK?)``
...................................

Compute audio tags for the given stream.

:param stream: An ``OfflineStream``.
:param topK: Number of top results (``number``, default ``-1`` for all).
:returns: An array of ``AudioEvent`` objects, each with:

  - ``name`` (string) — Event category name.
  - ``prob`` (number) — Probability score.
  - ``index`` (number) — Category index.

Properties
""""""""""

- ``tagger.config`` — The configuration object.

Example
^^^^^^^

.. code-block:: javascript

   const sherpa_onnx = require('sherpa-onnx-node');

   const tagger = new sherpa_onnx.AudioTagging({
     model: { ced: './ced.onnx' },
     labels: './labels.txt',
     topK: 5,
   });

   const stream = tagger.createStream();
   const wave = sherpa_onnx.readWave('./audio.wav');
   stream.acceptWaveform({ samples: wave.samples, sampleRate: wave.sampleRate });

   const events = tagger.compute(stream);
   for (const event of events) {
     console.log(`${event.name}: ${event.prob.toFixed(3)}`);
   }

Notes
-----

- The input audio should be mono, 16kHz, float32 in ``[-1, 1]``.
- Use ``topK`` to limit the number of returned results.
