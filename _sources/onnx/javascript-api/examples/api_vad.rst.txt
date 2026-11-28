VAD API
=======

Voice Activity Detection (VAD) API reference for ``sherpa-onnx-node``.

Source file
-----------

`scripts/node-addon-api/lib/vad.js <https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/node-addon-api/lib/vad.js>`_

API
---

Vad
^^^

Voice Activity Detector. Detects speech segments in an audio stream.

Constructor
"""""""""""

.. code-block:: javascript

   const vad = new sherpa_onnx.Vad(config, bufferSizeInSeconds);

:param config: VAD configuration object.
:param bufferSizeInSeconds: Buffer size in seconds (number).

The ``config`` object has the following properties:

- ``sileroVad`` (object, optional) — Silero VAD model config:

  - ``model`` (string) — Path to ``silero_vad.onnx``.
  - ``threshold`` (number) — Speech threshold, e.g. ``0.5``.
  - ``minSpeechDuration`` (number) — Min speech duration in seconds.
  - ``minSilenceDuration`` (number) — Min silence duration in seconds.
  - ``windowSize`` (number) — Window size in samples, e.g. ``512``.
  - ``maxSpeechDuration`` (number, optional) — Max speech duration in seconds.

- ``tenVad`` (object, optional) — Ten VAD model config (same fields as ``sileroVad``).
- ``sampleRate`` (number) — Sample rate in Hz, e.g. ``16000``.
- ``numThreads`` (number, optional) — Number of threads.
- ``debug`` (boolean, optional) — Enable debug output.

Methods
"""""""

``vad.acceptWaveform(samples)``
.................................

Feed audio samples to the VAD.

:param samples: Input audio samples (``Float32Array``).

``vad.isDetected()``
.....................

:returns: ``true`` if speech is currently being detected (``boolean``).

``vad.isEmpty()``
.................

:returns: ``true`` if no completed speech segments are available (``boolean``).

``vad.front(enableExternalBuffer?)``
.....................................

Get the earliest completed speech segment without removing it.

:param enableExternalBuffer: Whether to use an external buffer (``boolean``, default ``true``).
:returns: A ``SpeechSegment`` object with ``start`` (number) and ``samples`` (``Float32Array``).

``vad.pop()``
..............

Remove the earliest completed speech segment from the queue.

``vad.flush()``
...............

Flush the internal buffer to emit any pending speech segments.

``vad.clear()``
...............

Clear all internal state.

``vad.reset()``
...............

Reset the detector to its initial state.

Properties
""""""""""

- ``vad.config`` — The configuration object passed to the constructor.

CircularBuffer
^^^^^^^^^^^^^^

A circular buffer that stores ``Float32`` audio samples.

Constructor
"""""""""""

.. code-block:: javascript

   const buffer = new sherpa_onnx.CircularBuffer(capacity);

:param capacity: Buffer capacity in samples (number).

Methods
"""""""

``buffer.push(samples)``
..........................

Push samples into the buffer.

:param samples: Audio samples (``Float32Array``).

``buffer.get(startIndex, n, enableExternalBuffer?)``
.....................................................

Get a slice of samples from the buffer.

:param startIndex: Start index (number).
:param n: Number of samples to read (number).
:param enableExternalBuffer: Whether to use an external buffer (``boolean``, default ``true``).
:returns: Audio samples (``Float32Array``).

``buffer.pop(n)``
..................

Remove ``n`` samples from the front of the buffer.

:param n: Number of samples to remove (number).

``buffer.size()``
.................

:returns: Current number of samples in the buffer (``number``).

``buffer.head()``
.................

:returns: The head index of the buffer (``number``).

``buffer.reset()``
...................

Reset the buffer to empty state.

Example
^^^^^^^

.. code-block:: javascript

   const sherpa_onnx = require('sherpa-onnx-node');

   const vad = new sherpa_onnx.Vad({
     sileroVad: { model: './silero_vad.onnx', threshold: 0.5,
       minSpeechDuration: 0.25, minSilenceDuration: 0.5, windowSize: 512 },
     sampleRate: 16000, debug: false, numThreads: 1,
   }, 60);

   // Feed audio in chunks of windowSize samples
   vad.acceptWaveform(samples);

   if (vad.isDetected()) {
     console.log('Speech detected');
   }

   while (!vad.isEmpty()) {
     const segment = vad.front();
     vad.pop();
     console.log(`Segment: ${segment.samples.length} samples`);
   }

Notes
-----

- Feed audio in chunks matching ``windowSize`` (512 for Silero VAD at 16kHz).
- ``isDetected()`` returns ``true`` while speech is ongoing.
- ``isEmpty()`` / ``front()`` / ``pop()`` are used to extract completed speech
  segments (speech followed by enough silence).
- Use ``flush()`` at the end of a stream to emit any remaining buffered speech.
- You can use ``ten-vad.onnx`` instead of ``silero_vad.onnx`` by setting the
  ``tenVad`` config field and leaving ``sileroVad.model`` empty.
