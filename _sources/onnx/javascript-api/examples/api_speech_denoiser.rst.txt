Speech Denoiser API
===================

Speech denoising API reference for ``sherpa-onnx-node``.

Source file
-----------

- `scripts/node-addon-api/lib/non-streaming-speech-denoiser.js <https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/node-addon-api/lib/non-streaming-speech-denoiser.js>`_
- `scripts/node-addon-api/lib/online-speech-denoiser.js <https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/node-addon-api/lib/online-speech-denoiser.js>`_

API
---

OfflineSpeechDenoiser
^^^^^^^^^^^^^^^^^^^^^

Removes background noise from a complete audio recording.

Constructor
"""""""""""

.. code-block:: javascript

   const denoiser = new sherpa_onnx.OfflineSpeechDenoiser(config);

:param config: Configuration object with:

- ``model`` (object, optional) — Model configuration with one of:

  - ``gtcrn`` — ``{ model: string }`` path to the GTCRN ONNX model.
  - ``dpdfnet`` — ``{ model: string }`` path to the DPDFNet ONNX model.

  Plus common fields: ``numThreads``, ``debug``, ``provider``.

Methods
"""""""

``denoiser.run(obj)``
.......................

Run denoising on the input audio.

:param obj: Audio request object with:

  - ``samples`` (Float32Array) — Audio samples in ``[-1, 1]``.
  - ``sampleRate`` (number) — Sample rate in Hz.
  - ``enableExternalBuffer`` (boolean, optional, default ``true``).

:returns: A ``GeneratedAudio`` object with ``samples`` (``Float32Array``) and
  ``sampleRate`` (number).

Properties
""""""""""

- ``denoiser.config`` — The configuration object.
- ``denoiser.sampleRate`` — Expected input sample rate in Hz (number).

OnlineSpeechDenoiser
^^^^^^^^^^^^^^^^^^^^

Removes background noise from audio in a streaming fashion.

Constructor
"""""""""""

.. code-block:: javascript

   const denoiser = new sherpa_onnx.OnlineSpeechDenoiser(config);

:param config: Configuration object with:

- ``model`` (object, optional) — Model configuration with one of:

  - ``gtcrn`` — ``{ model: string }`` path to the GTCRN ONNX model.
  - ``dpdfnet`` — ``{ model: string }`` path to the DPDFNet ONNX model.

  Plus common fields: ``numThreads``, ``debug``, ``provider``.

Methods
"""""""

``denoiser.run(obj)``
.......................

Process a chunk of audio.

:param obj: Audio request object with:

  - ``samples`` (Float32Array) — Audio samples in ``[-1, 1]``.
  - ``sampleRate`` (number) — Sample rate in Hz.
  - ``enableExternalBuffer`` (boolean, optional, default ``true``).

:returns: A ``GeneratedAudio`` object with ``samples`` (``Float32Array``) and
  ``sampleRate`` (number).

``denoiser.flush(enableExternalBuffer?)``
..........................................

Flush remaining buffered audio and return denoised output.

:param enableExternalBuffer: Whether to use an external buffer (``boolean``,
  default ``true``).
:returns: A ``GeneratedAudio`` object with ``samples`` (``Float32Array``) and
  ``sampleRate`` (number).

``denoiser.reset()``
......................

Reset the internal state for reuse.

Properties
""""""""""

- ``denoiser.config`` — The configuration object.
- ``denoiser.sampleRate`` — Expected input sample rate in Hz (number).
- ``denoiser.frameShiftInSamples`` — Frame shift in samples (number).

Example
^^^^^^^

.. code-block:: javascript

   const sherpa_onnx = require('sherpa-onnx-node');

   // Offline denoising
   const denoiser = new sherpa_onnx.OfflineSpeechDenoiser({
     model: { gtcrn: './gtcrn.onnx' },
   });

   const wave = sherpa_onnx.readWave('./noisy-audio.wav');
   const denoised = denoiser.run({
     samples: wave.samples,
     sampleRate: wave.sampleRate,
   });
   sherpa_onnx.writeWave('clean-audio.wav', {
     samples: denoised.samples,
     sampleRate: denoised.sampleRate,
   });

   // Online denoising
   const onlineDenoiser = new sherpa_onnx.OnlineSpeechDenoiser({
     model: { gtcrn: './gtcrn.onnx' },
   });

   // Process chunks as they arrive
   const chunk1 = onlineDenoiser.run({ samples: audioChunk1, sampleRate: 16000 });
   const chunk2 = onlineDenoiser.run({ samples: audioChunk2, sampleRate: 16000 });
   const remaining = onlineDenoiser.flush();

   onlineDenoiser.reset();  // ready for next audio

Notes
-----

- ``OfflineSpeechDenoiser`` processes the entire audio at once.
- ``OnlineSpeechDenoiser`` processes audio incrementally; call ``flush()`` after
  the last chunk to get any remaining buffered audio.
- Call ``reset()`` on the online denoiser to reuse it for new audio.
