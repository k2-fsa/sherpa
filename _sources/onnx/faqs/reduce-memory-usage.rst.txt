Memory usage stays high after decoding long audio files
=========================================================

The ONNX Runtime CPU memory arena keeps the working set of each run resident
after decoding returns. If you decode several long recordings in one process,
the RSS stays at several GB even when the recognizer is idle.

To reduce it, create a provider configuration file, e.g. ``cpu.config``, with
the following content::

  EnableCpuMemArena=0
  EnableMemPattern=0

and pass it via the ``provider`` field, which every language binding accepts::

  recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
      ...,
      provider="cpu:/path/to/cpu.config",
  )

For example, with the FireRedASR2 CTC int8 model and 180 s inputs, RSS stays
at ~8 GB after decoding with the default settings, and drops to ~1.2 GB with
the above config file.

The config file uses one ``key=value`` pair per line (``#`` starts a comment)
and also supports:

- ``GraphOptimizationLevel``
- ``LogSeverityLevel``
- ``ProfilingFilePrefix``
- keys prefixed with ``SessionConfig.``, which are forwarded to ONNX Runtime
  session configuration entries, e.g. ``SessionConfig.mlas.disable_kleidiai=1``
- execution-provider options, e.g. ``gpu_mem_limit`` for cuda

See the comments in ``sherpa-onnx/csrc/session.cc`` for details. Note that
ONNX Runtime only exposes ``Disable*`` for the memory arena and the memory
pattern, so a config file can only disable them, never re-enable them.

Disabling the CPU memory arena may slightly slow down decoding of long
segments. If you decode long recordings, we also recommend using VAD to
segment them into chunks of 20-30 s, e.g. with the approach in
`vad-with-non-streaming-asr.py
<https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/vad-with-non-streaming-asr.py>`_.
