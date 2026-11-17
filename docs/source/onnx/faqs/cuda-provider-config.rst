.. _sherpa-onnx-cuda-provider-config:

How do I configure CUDA for Whisper and FireRedASR?
============================================================

Use ``provider="cuda:/path/to/cuda.config"`` to pass CUDA execution-provider
options to `sherpa-onnx`_. For Whisper and FireRedASR AED, this also keeps the
encoder and decoder attention caches on the selected GPU, just as
``provider="cuda"`` does. No additional cache option is needed.

This behavior requires a CUDA-enabled build containing
`sherpa-onnx PR #4160 <https://github.com/k2-fsa/sherpa-onnx/pull/4160>`_.
Before that fix, using a CUDA configuration file disabled these models'
CUDA I/O binding, causing extra CPU/GPU cache copies during decoding.
FireRedASR CTC is not affected by this cache-binding change.

See :doc:`../python/install` for CUDA-enabled Python packages. The CUDA and
cuDNN libraries must match the ONNX Runtime included in your build.
Forwarding CUDA options from a file requires ONNX Runtime 1.12 or newer.

Create a configuration file
---------------------------

Save the following as ``cuda.config``:

.. code-block:: text

   # Select the first GPU visible to this process.
   device_id=0

The file uses one ``key=value`` pair per line. Empty lines and lines starting
with ``#`` are ignored. The path must point to a readable file; a relative
path is resolved from the application's working directory.

You can add other CUDA execution-provider options, for example:

.. code-block:: text

   device_id=0
   arena_extend_strategy=kSameAsRequested

See the `ONNX Runtime CUDA execution-provider options
<https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#configuration-options>`_
for supported keys and values. Available options depend on your ONNX Runtime
version; the second setting above is optional.

Use it from Python or the command line
--------------------------------------

For example, download :ref:`whisper_tiny_en_sherpa_onnx`, then create a
recognizer with the FP32 model files:

.. code-block:: python

   import sherpa_onnx

   recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
       encoder="./sherpa-onnx-whisper-tiny.en/tiny.en-encoder.onnx",
       decoder="./sherpa-onnx-whisper-tiny.en/tiny.en-decoder.onnx",
       tokens="./sherpa-onnx-whisper-tiny.en/tiny.en-tokens.txt",
       provider="cuda:./cuda.config",
   )

Accept audio and decode as usual. For FireRedASR AED, pass the same
``provider="cuda:./cuda.config"`` argument to
``sherpa_onnx.OfflineRecognizer.from_fire_red_asr()`` with your FireRedASR
model files.

The equivalent Whisper command-line example is:

.. code-block:: bash

   ./sherpa-onnx-offline \
     --provider="cuda:./cuda.config" \
     --whisper-encoder=./sherpa-onnx-whisper-tiny.en/tiny.en-encoder.onnx \
     --whisper-decoder=./sherpa-onnx-whisper-tiny.en/tiny.en-decoder.onnx \
     --tokens=./sherpa-onnx-whisper-tiny.en/tiny.en-tokens.txt \
     ./sherpa-onnx-whisper-tiny.en/test_wavs/0.wav

Select another GPU
------------------

Set ``device_id=1`` to select the second GPU visible to the process. The ID
must be a non-negative integer identifying an available CUDA device.
With the offline recognizers above, omitting ``device_id`` defaults to 0.

``CUDA_VISIBLE_DEVICES`` changes which devices are visible and renumbers them.
For example, with ``CUDA_VISIBLE_DEVICES=1``, only one GPU is visible to the
process, so use ``device_id=0`` in the file, not ``device_id=1``. Set the
environment variable before starting the application.

If CUDA is unavailable or its provider options are rejected during provider
registration, check the log: recognition can fall back to CPU, without CUDA
cache binding. A successful transcript alone does not confirm GPU execution.
