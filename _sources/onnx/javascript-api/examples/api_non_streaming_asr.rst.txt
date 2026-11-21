Non-Streaming ASR API
=====================

Non-streaming (offline) speech recognition API reference for ``sherpa-onnx-node``.

Source file
-----------

`scripts/node-addon-api/lib/non-streaming-asr.js <https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/node-addon-api/lib/non-streaming-asr.js>`_

API
---

OfflineRecognizer
^^^^^^^^^^^^^^^^^

Non-streaming speech recognizer. Processes a complete audio file at once.

Constructor
"""""""""""

.. code-block:: javascript

   const recognizer = new sherpa_onnx.OfflineRecognizer(config);

:param config: Recognizer configuration object (``OfflineRecognizerConfig``).

The ``config`` object supports:

- ``featConfig`` (object, optional) — ``{ sampleRate: number, featureDim: number }``
- ``modelConfig`` (object, optional) — Model configuration with one of:

  - ``transducer`` — ``{ encoder: string, decoder: string, joiner: string }``
  - ``paraformer`` — ``{ encoder: string, decoder: string }``
  - ``zipformerCtc`` — ``{ model: string }``
  - ``nemoCtc`` — ``{ model: string }``
  - ``senseVoice`` — ``{ model: string }``
  - ``whisper`` — ``{ encoder: string, decoder: string, language: string, task: string }``
  - ``moonshine`` — ``{ encoder: string, decoder: string, uncachedDecoder: string }``
  - ``fireRedAsr`` — ``{ encoder: string, decoder: string }``
  - ``cohereTranscribe`` — ``{ model: string }``

  Plus common fields: ``tokens``, ``numThreads``, ``debug``, ``provider``.

Static Methods
""""""""""""""

``OfflineRecognizer.createAsync(config)``
..........................................

Create a recognizer asynchronously (non-blocking).

:param config: Recognizer configuration (``OfflineRecognizerConfig``).
:returns: A ``Promise<OfflineRecognizer>``.

Methods
"""""""

``recognizer.createStream()``
..............................

:returns: A new ``OfflineStream``.

``recognizer.decode(stream)``
..............................

Decode the stream synchronously.

:param stream: An ``OfflineStream``.

``recognizer.decodeAsync(stream)``
...................................

Decode the stream asynchronously (non-blocking).

:param stream: An ``OfflineStream``.
:returns: A ``Promise<OfflineRecognizerResult>``.

``recognizer.getResult(stream)``
.................................

Get the recognition result.

:param stream: An ``OfflineStream``.
:returns: An ``OfflineRecognizerResult`` object with:

  - ``text`` (string) — Recognized text.
  - ``tokens`` (string[]) — Token strings.
  - ``timestamps`` (number[]) — Per-token timestamps in seconds.
  - ``durations`` (number[]) — Per-token durations in seconds.
  - ``lang`` (string) — Detected language (SenseVoice).
  - ``emotion`` (string) — Detected emotion (SenseVoice).
  - ``event`` (string) — Detected event (SenseVoice).

``recognizer.setConfig(config)``
.................................

Update the recognizer configuration at runtime.

:param config: New configuration (``OfflineRecognizerConfig``).

Properties
""""""""""

- ``recognizer.config`` — The configuration object.

OfflineStream
^^^^^^^^^^^^^

A non-streaming recognition stream.

Methods
"""""""

``stream.acceptWaveform(obj)``
...............................

Feed audio to the stream.

:param obj: ``{ samples: Float32Array, sampleRate: number }``.

``stream.setOption(key, value)``
.................................

Set a string option on the stream.

:param key: Option name (string).
:param value: Option value (string).

Example
^^^^^^^

.. code-block:: javascript

   const sherpa_onnx = require('sherpa-onnx-node');

   const recognizer = new sherpa_onnx.OfflineRecognizer({ /* config */ });
   const stream = recognizer.createStream();

   const wave = sherpa_onnx.readWave('./audio.wav');
   stream.acceptWaveform({ samples: wave.samples, sampleRate: wave.sampleRate });

   recognizer.decode(stream);
   const result = recognizer.getResult(stream);
   console.log(result.text);

Notes
-----

- The entire audio must be available before calling ``decode()``.
- Use ``createAsync()`` for non-blocking construction in async contexts.
- Use ``decodeAsync()`` for non-blocking decoding.
