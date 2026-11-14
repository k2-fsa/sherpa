LinearResampler
===============

Resample audio from one sample rate to another using a linear resampler.

Source file
-----------

`scripts/node-addon-api/lib/resampler.js <https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/node-addon-api/lib/resampler.js>`_

API
---

Constructor
^^^^^^^^^^^

.. code-block:: javascript

   const resampler = new sherpa_onnx.LinearResampler(inputSampleRate, outputSampleRate);

Creates a resampler that converts audio from ``inputSampleRate`` Hz to
``outputSampleRate`` Hz.

Methods
^^^^^^^

``resampler.resample(samples)``
"""""""""""""""""""""""""""""""

Resample a chunk of audio samples. Call this for each chunk of input audio.

:param samples: Input audio samples (``Float32Array``).
:returns: Resampled audio samples (``Float32Array``).

``resampler.flush(samples)``
""""""""""""""""""""""""""""

Resample the final chunk of audio and flush internal buffers. This is the
same as ``resample()`` but signals the resampler to emit any remaining
buffered samples. Call this once after the last chunk of input audio.

:param samples: The final chunk of input audio samples (``Float32Array``).
:returns: Resampled audio samples including buffered tail (``Float32Array``).

``resampler.reset()``
"""""""""""""""""""""

Reset the resampler to its initial state, discarding any internally buffered
samples.

``resampler.getInputSampleRate()``
""""""""""""""""""""""""""""""""""

:returns: The input sample rate in Hz.

``resampler.getOutputSampleRate()``
"""""""""""""""""""""""""""""""""""

:returns: The output sample rate in Hz.

Properties
^^^^^^^^^^

- ``resampler.inputSampleRate`` - Input sample rate in Hz (number).
- ``resampler.outputSampleRate`` - Output sample rate in Hz (number).

Example
^^^^^^^

.. code-block:: javascript

   const sherpa_onnx = require('sherpa-onnx-node');

   // Resample from 48000 Hz to 16000 Hz
   const resampler = new sherpa_onnx.LinearResampler(48000, 16000);

   // Process audio chunks
   const resampledChunk = resampler.resample(inputChunk);

   // For the final chunk, use flush() to get all remaining samples
   const finalChunk = resampler.flush(lastInputChunk);

Notes
-----

- The resampler maintains internal state across calls to ``resample()`` to
  handle chunk boundaries correctly.
- Always use ``flush()`` for the last chunk; otherwise the resampler may
  hold back a few trailing samples.
- Call ``reset()`` to discard internal state if you want to reuse the
  resampler for a new audio stream.
- The low-pass filter parameters follow the same convention used throughout
  the sherpa-onnx codebase (see ``alsa-play.cc``).
