Streaming ASR API
=================

Streaming (online) speech recognition API reference for ``sherpa-onnx-node``.

Source file
-----------

`scripts/node-addon-api/lib/streaming-asr.js <https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/node-addon-api/lib/streaming-asr.js>`_

API
---

OnlineRecognizer
^^^^^^^^^^^^^^^^

Streaming speech recognizer. Processes audio incrementally and returns
partial results as audio arrives.

Constructor
"""""""""""

.. code-block:: javascript

   const recognizer = new sherpa_onnx.OnlineRecognizer(config);

:param config: Recognizer configuration object (``OnlineRecognizerConfig``).

The ``config`` object supports:

- ``featConfig`` (object, optional) — ``{ sampleRate: number, featureDim: number }``
- ``modelConfig`` (object, optional) — Model configuration with one of:

  - ``transducer`` — ``{ encoder: string, decoder: string, joiner: string }``
  - ``paraformer`` — ``{ encoder: string, decoder: string }``
  - ``zipformer2Ctc`` — ``{ model: string }``
  - ``nemoCtc`` — ``{ model: string }``

  Plus common fields: ``tokens``, ``numThreads``, ``debug``, ``provider``.

- ``decodingMethod`` (string, optional) — e.g. ``'greedy_search'``.
- ``maxActivePaths`` (number, optional) — For beam search.
- ``enableEndpoint`` (boolean, optional) — Enable endpoint detection.
- ``rule1MinTrailingSilence`` (number, optional) — Endpoint rule 1.
- ``rule2MinTrailingSilence`` (number, optional) — Endpoint rule 2.
- ``rule3MinUtteranceLength`` (number, optional) — Endpoint rule 3.
- ``blankPenalty`` (number, optional) — Blank penalty for CTC models.

Methods
"""""""

``recognizer.createStream()``
..............................

:returns: A new ``OnlineStream``.

``recognizer.isReady(stream)``
..............................

Check if the stream has enough frames for decoding.

:param stream: An ``OnlineStream``.
:returns: ``true`` if ready (``boolean``).

``recognizer.decode(stream)``
..............................

Trigger one decoding step on the stream.

:param stream: An ``OnlineStream``.

``recognizer.isEndpoint(stream)``
.................................

Check if an endpoint has been detected.

:param stream: An ``OnlineStream``.
:returns: ``true`` if endpoint detected (``boolean``).

``recognizer.reset(stream)``
.............................

Reset the stream for a new utterance.

:param stream: An ``OnlineStream``.

``recognizer.getResult(stream)``
.................................

Get the current recognition result.

:param stream: An ``OnlineStream``.
:returns: An ``OnlineRecognizerResult`` object with:

  - ``text`` (string) — Recognized text.
  - ``tokens`` (string[]) — Token strings.
  - ``timestamps`` (number[]) — Per-token timestamps.
  - ``is_final`` (boolean) — Whether this is a final result.

Properties
""""""""""

- ``recognizer.config`` — The configuration object.

OnlineStream
^^^^^^^^^^^^

An active streaming recognition stream.

Methods
"""""""

``stream.acceptWaveform(obj)``
...............................

Feed audio to the stream.

:param obj: ``{ samples: Float32Array, sampleRate: number }``.

``stream.inputFinished()``
...........................

Signal that no more audio will be fed to this stream.

Display
^^^^^^^

Helper for printing recognized words to the console.

Constructor
"""""""""""

.. code-block:: javascript

   const display = new sherpa_onnx.Display(maxWordPerLine);

:param maxWordPerLine: Max words per line (number).

Methods
"""""""

``display.print(idx, text)``
.............................

Print recognized text.

:param idx: Segment index (number).
:param text: Text to display (string).

Example
^^^^^^^

.. code-block:: javascript

   const sherpa_onnx = require('sherpa-onnx-node');

   const recognizer = new sherpa_onnx.OnlineRecognizer({ /* config */ });
   const stream = recognizer.createStream();

   stream.acceptWaveform({ samples: audioSamples, sampleRate: 16000 });

   while (recognizer.isReady(stream)) {
     recognizer.decode(stream);
   }

   const result = recognizer.getResult(stream);
   console.log(result.text);

   if (recognizer.isEndpoint(stream)) {
     recognizer.reset(stream);  // ready for next utterance
   }

Notes
-----

- Feed audio continuously; call ``decode()`` whenever ``isReady()`` returns
  ``true``.
- Use ``isEndpoint()`` to detect utterance boundaries.
- Call ``inputFinished()`` when no more audio will arrive.
