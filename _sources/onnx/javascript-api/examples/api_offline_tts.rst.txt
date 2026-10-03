Offline TTS API
===============

Text-to-Speech (TTS) API reference for ``sherpa-onnx-node``.

Source file
-----------

`scripts/node-addon-api/lib/non-streaming-tts.js <https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/node-addon-api/lib/non-streaming-tts.js>`_

API
---

OfflineTts
^^^^^^^^^^

Text-to-Speech engine. Converts text to audio.

Constructor
"""""""""""

.. code-block:: javascript

   const tts = new sherpa_onnx.OfflineTts(config);

:param config: TTS configuration object (``OfflineTtsConfig``).

The ``config`` object supports:

- ``model`` (object, optional) — Model configuration with one of:

  - ``vits`` — VITS model config.
  - ``matcha`` — Matcha model config (requires a vocoder).
  - ``kokoro`` — Kokoro model config.
  - ``kitten`` — Kitten model config.
  - ``zipvoice`` — ZipVoice model config (requires a vocoder).
  - ``pocket`` — Pocket model config.

- ``maxNumSentences`` (number, optional) — Max sentences to process.
- ``silenceScale`` (number, optional) — Silence scaling factor.
- ``numThreads`` (number, optional) — Number of threads.
- ``provider`` (string, optional) — e.g. ``'cpu'``.

Static Methods
""""""""""""""

``OfflineTts.createAsync(config)``
...................................

Create a TTS engine asynchronously (non-blocking).

:param config: TTS configuration (``OfflineTtsConfig``).
:returns: A ``Promise<OfflineTts>``.

Methods
"""""""

``tts.generate(obj)``
......................

Generate audio synchronously.

:param obj: Generation request object with:

  - ``text`` (string) — Input text to synthesize.
  - ``sid`` (number) — Speaker ID.
  - ``speed`` (number) — Playback speed (e.g. ``1.0``).
  - ``generationConfig`` (GenerationConfig, optional) — Advanced generation parameters.

:returns: A ``GeneratedAudio`` object with ``samples`` (``Float32Array``) and ``sampleRate`` (number).

``tts.generateAsync(obj)``
............................

Generate audio asynchronously (non-blocking).

:param obj: Same as ``generate()`` plus:

  - ``onProgress`` (function, optional) — Callback receiving ``{ samples, progress }``.
    Return truthy to continue, falsy to cancel.

:returns: A ``Promise<GeneratedAudio>``.

Properties
""""""""""

- ``tts.config`` — The configuration object.
- ``tts.numSpeakers`` — Number of available speakers (number).
- ``tts.sampleRate`` — Output sample rate in Hz (number).

GenerationConfig
^^^^^^^^^^^^^^^^

Advanced generation parameters for TTS.

Constructor
"""""""""""

.. code-block:: javascript

   const genConfig = new sherpa_onnx.GenerationConfig({
     speed: 1.0, sid: 0, numSteps: 5
   });

All properties are optional:

- ``speed`` (number) — Playback speed.
- ``sid`` (number) — Speaker ID.
- ``numSteps`` (number) — Number of steps (for flow-matching models).
- ``silenceScale`` (number) — Silence scaling factor.
- ``referenceAudio`` (Float32Array) — Reference audio for voice cloning.
- ``referenceSampleRate`` (number) — Sample rate of the reference audio.
- ``referenceText`` (string) — Transcript of the reference audio.
- ``extra`` (object) — Extra key-value pairs (e.g. ``{ lang: 'en' }``).

Example
^^^^^^^

.. code-block:: javascript

   const sherpa_onnx = require('sherpa-onnx-node');

   const tts = new sherpa_onnx.OfflineTts({ /* config */ });
   console.log(`Sample rate: ${tts.sampleRate}`);

   // Synchronous
   const audio = tts.generate({ text: 'Hello world', sid: 0, speed: 1.0 });
   sherpa_onnx.writeWave('output.wav', { samples: audio.samples, sampleRate: audio.sampleRate });

   // Async with progress
   const audio2 = await tts.generateAsync({
     text: 'Hello world', sid: 0, speed: 1.0,
     onProgress: ({ progress }) => { console.log(`${(progress * 100).toFixed(1)}%`); return 1; }
   });

Notes
-----

- Use ``createAsync()`` for non-blocking construction in async contexts.
- The ``onProgress`` callback receives streaming audio chunks during generation.
  Return ``0`` or ``false`` to cancel generation.
- Matcha and ZipVoice models require a vocoder model (e.g. ``vocos-22khz-univ.onnx``).
