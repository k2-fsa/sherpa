VAD with Microphone
====================

Detect speech from a microphone in real time using Silero VAD (Voice Activity
Detection). Each detected speech segment is saved as a separate WAV file.

Source file
-----------

`nodejs-addon-examples/test_vad_microphone.js <https://github.com/k2-fsa/sherpa-onnx/blob/master/nodejs-addon-examples/test_vad_microphone.js>`_

Code
----

.. literalinclude:: ../code/vad_microphone.js
   :language: javascript
   :linenos:

How to run
----------

1. Install the packages::

     npm install sherpa-onnx-node
     npm install node-cpal

2. Download the VAD model::

     curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

3. Set the library path and run:

   .. code-block:: bash

      # macOS
      export DYLD_LIBRARY_PATH=$(npm root)/sherpa-onnx-node/lib:$DYLD_LIBRARY_PATH

      # Linux
      export LD_LIBRARY_PATH=$(npm root)/sherpa-onnx-node/lib:$LD_LIBRARY_PATH

      node vad_microphone.js

4. Speak into the microphone. Detected speech segments will be printed and
   saved as WAV files. Press ``Ctrl+C`` to stop.

Expected output
^^^^^^^^^^^^^^^

.. code-block:: text

   Started! Please speak
   0: Detected speech
   0 End of speech. Duration: 2.345 seconds
   Saved to 0-14-30-25.wav

Notes
-----

- `node-cpal <https://github.com/saeta-eth/node-cpal>`_ provides
  cross-platform microphone access via Rust's CPAL library. Install it with
  ``npm install node-cpal``.
- ``cpal.getDefaultInputDevice()`` returns the default microphone device
  object. Use ``device.deviceId`` to get the device ID string.
- ``cpal.getDefaultInputConfig(deviceId)`` returns the device's native
  sample rate and format.
- ``cpal.createStream(deviceId, true, config, callback)`` opens an input
  stream. The callback receives a ``Float32Array`` of audio samples.
- ``new sherpa_onnx.LinearResampler(inputRate, outputRate)`` creates a
  resampler to convert audio from the device's native sample rate to the
  rate required by the model (e.g., 16kHz).
- ``resampler.resample(samples)`` resamples a ``Float32Array`` chunk and
  returns the resampled ``Float32Array``.
- ``CircularBuffer`` stores incoming microphone samples. The VAD processes
  audio in fixed-size windows (512 samples for Silero VAD at 16kHz).
- ``isDetected()`` returns ``true`` when speech is currently being detected.
- ``isEmpty()`` / ``front()`` / ``pop()`` are used to extract completed
  speech segments (speech followed by enough silence).
- You can also use ``ten-vad.onnx`` instead of ``silero_vad.onnx``.
