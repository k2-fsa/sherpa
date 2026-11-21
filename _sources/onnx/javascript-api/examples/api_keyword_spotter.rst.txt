Keyword Spotter API
===================

Streaming keyword spotting API reference for ``sherpa-onnx-node``.

Source file
-----------

`scripts/node-addon-api/lib/keyword-spotter.js <https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/node-addon-api/lib/keyword-spotter.js>`_

API
---

KeywordSpotter
^^^^^^^^^^^^^^

Streaming keyword spotter. Detects keywords in audio as it arrives.

Constructor
"""""""""""

.. code-block:: javascript

   const spotter = new sherpa_onnx.KeywordSpotter(config);

:param config: Configuration object with:

- ``featConfig`` (object, optional) — ``{ sampleRate: number, featureDim: number }``
- ``modelConfig`` (object, optional) — Model configuration with one of:

  - ``transducer`` — ``{ encoder: string, decoder: string, joiner: string }``
  - ``paraformer`` — ``{ encoder: string, decoder: string }``
  - ``zipformerCtc`` — ``{ model: string }``
  - ``nemoCtc`` — ``{ model: string }``

  Plus common fields: ``tokens``, ``numThreads``, ``debug``, ``provider``.

- ``maxActivePaths`` (number, optional) — Max active paths for beam search.
- ``numTrailingBlanks`` (number, optional) — Number of trailing blanks.
- ``keywordsScore`` (number, optional) — Score boost for keywords.
- ``keywordsThreshold`` (number, optional) — Threshold for keyword detection.
- ``keywordsFile`` (string, optional) — Path to keywords file.

Methods
"""""""

``spotter.createStream()``
...........................

:returns: A new ``OnlineStream``.

``spotter.isReady(stream)``
.............................

Check if the stream has enough data for decoding.

:param stream: An ``OnlineStream``.
:returns: ``true`` if ready (``boolean``).

``spotter.decode(stream)``
............................

Trigger one decoding step on the stream.

:param stream: An ``OnlineStream``.

``spotter.reset(stream)``
...........................

Reset the stream for a new utterance.

:param stream: An ``OnlineStream``.

``spotter.getResult(stream)``
...............................

Get the current keyword detection result.

:param stream: An ``OnlineStream``.
:returns: A ``KeywordResult`` object with:

  - ``keyword`` (string) — Detected keyword text.
  - ``start_time`` (number) — Start time in seconds.
  - ``timestamps`` (number[]) — Per-token timestamps.
  - ``tokens`` (string[]) — Token strings.

Properties
""""""""""

- ``spotter.config`` — The configuration object.

Example
^^^^^^^

.. code-block:: javascript

   const sherpa_onnx = require('sherpa-onnx-node');

   const spotter = new sherpa_onnx.KeywordSpotter({
     modelConfig: {
       transducer: {
         encoder: './encoder.onnx',
         decoder: './decoder.onnx',
         joiner: './joiner.onnx',
       },
       tokens: './tokens.txt',
     },
     keywordsFile: './keywords.txt',
   });

   const stream = spotter.createStream();

   // Feed audio incrementally
   stream.acceptWaveform({ samples: audioChunk, sampleRate: 16000 });

   while (spotter.isReady(stream)) {
     spotter.decode(stream);
   }

   const result = spotter.getResult(stream);
   if (result.keyword) {
     console.log(`Detected: ${result.keyword}`);
   }

   spotter.reset(stream);

Notes
-----

- Feed audio continuously; call ``decode()`` whenever ``isReady()`` returns
  ``true``.
- Use ``keywordsFile`` to define custom keywords.
- Call ``reset()`` between utterances.
