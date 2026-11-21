Spoken Language Identification API
===================================

Spoken language identification API reference for ``sherpa-onnx-node``.

Source file
-----------

`scripts/node-addon-api/lib/spoken-language-identification.js <https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/node-addon-api/lib/spoken-language-identification.js>`_

API
---

SpokenLanguageIdentification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Identifies the spoken language in an audio recording.

Constructor
"""""""""""

.. code-block:: javascript

   const sli = new sherpa_onnx.SpokenLanguageIdentification(config);

:param config: Configuration object with:

- ``whisper`` (object, optional) — Whisper model configuration:

  - ``encoder`` (string) — Path to the Whisper encoder ONNX model.
  - ``decoder`` (string) — Path to the Whisper decoder ONNX model.
  - ``tailPaddings`` (number, optional) — Number of tail padding samples.

- ``numThreads`` (number, optional).
- ``debug`` (boolean, optional).
- ``provider`` (string, optional).

Methods
"""""""

``sli.createStream()``
........................

:returns: A new ``OfflineStream`` for feeding audio.

``sli.compute(stream)``
.........................

Identify the spoken language.

:param stream: An ``OfflineStream``.
:returns: A two-letter language code (``string``), e.g. ``'en'``, ``'de'``,
  ``'fr'``, ``'es'``, ``'zh'``, ``'ja'``, ``'ko'``.

Properties
""""""""""

- ``sli.config`` — The configuration object.

Example
^^^^^^^

.. code-block:: javascript

   const sherpa_onnx = require('sherpa-onnx-node');

   const sli = new sherpa_onnx.SpokenLanguageIdentification({
     whisper: {
       encoder: './whisper-encoder.onnx',
       decoder: './whisper-decoder.onnx',
     },
   });

   const stream = sli.createStream();
   const wave = sherpa_onnx.readWave('./audio.wav');
   stream.acceptWaveform({ samples: wave.samples, sampleRate: wave.sampleRate });

   const lang = sli.compute(stream);
   console.log(`Detected language: ${lang}`);

Notes
-----

- Uses a Whisper-based model for language identification.
- The input audio should be mono, 16kHz, float32 in ``[-1, 1]``.
- Supported languages depend on the Whisper model variant used.
