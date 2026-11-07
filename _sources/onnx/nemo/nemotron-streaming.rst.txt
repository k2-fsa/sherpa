Nemotron ASR Streaming
======================

This page describes how to use the `nemotron-speech-streaming-en-0.6b <https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b>`_
in `sherpa-onnx`_.

The model supports 4 different chunk sizes: 80ms, 160ms, 560ms, and 1120ms.
For each chunk size, there is a corresponding ONNX model. The following table
lists the model for each chunk size:

.. list-table::

 * - Model
   - Chunk size
   - URL
 * - ``sherpa-onnx-nemotron-speech-streaming-en-0.6b-80ms-int8-2026-04-25.tar.bz2``
   - 80 ms
   - `Download address <https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemotron-speech-streaming-en-0.6b-80ms-int8-2026-04-25.tar.bz2>`_
 * - ``sherpa-onnx-nemotron-speech-streaming-en-0.6b-160ms-int8-2026-04-25.tar.bz2``
   - 160 ms
   - `Download address <https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemotron-speech-streaming-en-0.6b-160ms-int8-2026-04-25.tar.bz2>`_
 * - ``sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25.tar.bz2``
   - 560 ms
   - `Download address <https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25.tar.bz2>`_
 * - ``sherpa-onnx-nemotron-speech-streaming-en-0.6b-1120ms-int8-2026-04-25.tar.bz2``
   - 1120 ms
   - `Download address <https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemotron-speech-streaming-en-0.6b-1120ms-int8-2026-04-25.tar.bz2>`_

.. hint::

   The larger the chunk size, the higher the accuracy.

   The following figure is from `<https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b>`_.

   .. image:: ./pic/results_wer_and_scaling_nemotron_streaming_0.6b.png
      :align: center
      :alt: WER vs chunk size
      :width: 600

The following shows how to use the model with chunk size ``560ms``

sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25 (English)
--------------------------------------------------------------------------------

Export to ONNX
^^^^^^^^^^^^^^

In case you want to export the model by yourself, please see the export script at

  `<https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/nemo/nemotron-speech-streaming-en-0.6b/export_onnx.py>`_

For normal users, you don't need to care about the export step. Just use the pre-exported model from us.

Pre-built Android APK for real-time speech recognition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Please visit `<https://k2-fsa.github.io/sherpa/onnx/android/apk.html>`_
and search for `nemotron-speech-streaming-en-0.6b`.

For instance, you can select `sherpa-onnx-1.12.40-arm64-v8a-asr-en-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25.apk <https://huggingface.co/csukuangfj2/sherpa-onnx-apk/resolve/main/asr/1.12.40/sherpa-onnx-1.12.40-arm64-v8a-asr-en-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25.apk>`_
for Android ABI ``arm64-v8a``.

.. hint::

  Please always use the latest version.

Download the model
^^^^^^^^^^^^^^^^^^

Please use the following command to download the model:

.. code-block::

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25.tar.bz2
   tar xvf sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25.tar.bz2
   rm sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25.tar.bz2

   ls -lh sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25

You should see the following output:

.. code-block::

  ls -lh sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25

  total 1296904
  -rw-r--r--@ 1 fangjun  staff   6.9M 25 Apr 18:33 decoder.int8.onnx
  -rw-r--r--@ 1 fangjun  staff   623M 25 Apr 18:33 encoder.int8.onnx
  -rw-r--r--@ 1 fangjun  staff   1.7M 25 Apr 18:33 joiner.int8.onnx
  -rw-r--r--@ 1 fangjun  staff   159B 25 Apr 18:34 README.md
  drwxr-xr-x@ 6 fangjun  staff   192B 25 Apr 18:34 test_wavs
  -rw-r--r--@ 1 fangjun  staff   8.7K 25 Apr 18:25 tokens.txt

Decode a wave file
^^^^^^^^^^^^^^^^^^

Please use the following command to decode a wave file:

.. code-block::

   build/bin/sherpa-onnx \
     --encoder=./sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25/encoder.int8.onnx \
     --decoder=./sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25/decoder.int8.onnx \
     --joiner=./sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25/joiner.int8.onnx \
     --tokens=./sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25/tokens.txt \
     ./sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25/test_wavs/0.wav

The output of the above command is given below:

.. literalinclude:: ./code-nemotron-streaming/560ms.txt

Real-time speech recognition from a microphone
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Please use the following command for real-time speech recognition with a microphone:

.. code-block::

   build/bin/sherpa-onnx-microphone \
     --encoder=./sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25/encoder.int8.onnx \
     --decoder=./sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25/decoder.int8.onnx \
     --joiner=./sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25/joiner.int8.onnx \
     --tokens=./sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25/tokens.txt

sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11 (Multilingual)
--------------------------------------------------------------------------------

This section describes how to use the multilingual model
`nemotron-3.5-asr-streaming-0.6b <https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b>`_
in `sherpa-onnx`_.

Unlike the English-only model above, a single model supports 40 language-locales
from one checkpoint. The language is selected per-stream as a string
(e.g. ``en``, ``ja``, ``hi``); an empty string or ``auto`` lets the model
detect the language automatically.

The 19 transcription-ready locales work out of the box:

  English (en-US, en-GB), Spanish (es-US, es-ES), French (fr-FR, fr-CA)

  Italian (it-IT), Portuguese (pt-BR, pt-PT), Dutch (nl-NL), German (de-DE)

  Turkish (tr-TR), Russian (ru-RU), Arabic (ar-AR), Hindi (hi-IN)

  Japanese (ja-JP), Korean (ko-KR), Vietnamese (vi-VN), Ukrainian (uk-UA)

It also covers 13 broad-coverage locales (usable, but with higher error rates):

  Polish (pl-PL), Swedish (sv-SE), Czech (cs-CZ), Norwegian Bokmål (nb-NO)

  Danish (da-DK), Bulgarian (bg-BG), Finnish (fi-FI), Croatian (hr-HR)

  Slovak (sk-SK), Mandarin (zh-CN), Hungarian (hu-HU), Romanian (ro-RO)

  Estonian (et-EE)

A further 8 adaptation-ready locales require fine-tuning before they can transcribe:

  Greek (el-GR), Lithuanian (lt-LT), Latvian (lv-LV), Maltese (mt-MT)

  Slovenian (sl-SI), Hebrew (he-IL), Thai (th-TH), Norwegian Nynorsk (nn-NO)

.. hint::

   In ``auto`` mode the model detects the language and emits a tag such as
   ``<en-US>`` in the output. `sherpa-onnx`_ strips these language tags so the
   returned transcript stays clean.

It also supports the same 4 chunk sizes: 80ms, 160ms, 560ms, and 1120ms.
The following table lists the model for each chunk size:

.. list-table::

 * - Model
   - Chunk size
   - URL
 * - ``sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-80ms-int8-2026-06-11.tar.bz2``
   - 80 ms
   - `Download address <https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-80ms-int8-2026-06-11.tar.bz2>`_
 * - ``sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-160ms-int8-2026-06-11.tar.bz2``
   - 160 ms
   - `Download address <https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-160ms-int8-2026-06-11.tar.bz2>`_
 * - ``sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11.tar.bz2``
   - 560 ms
   - `Download address <https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11.tar.bz2>`_
 * - ``sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-1120ms-int8-2026-06-11.tar.bz2``
   - 1120 ms
   - `Download address <https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-1120ms-int8-2026-06-11.tar.bz2>`_

.. hint::

   The larger the chunk size, the higher the accuracy.

The following shows how to use the model with chunk size ``560ms``.

Export to ONNX
^^^^^^^^^^^^^^

In case you want to export the model by yourself, please see the export script at

  `<https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/nemo/nemotron-3.5-asr-streaming-0.6b/export_onnx.py>`_

For normal users, you don't need to care about the export step. Just use the pre-exported model from us.

Download the model
^^^^^^^^^^^^^^^^^^

Please use the following command to download the model:

.. code-block::

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11.tar.bz2
   tar xvf sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11.tar.bz2
   rm sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11.tar.bz2

   ls -lh sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11

You should see the following output:

.. code-block::

  ls -lh sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11

  total 1359632
  -rw-r--r--@ 1 rpwr021  wheel    14M Jun 11 19:41 decoder.int8.onnx
  -rw-r--r--@ 1 rpwr021  wheel   627M Jun 11 19:41 encoder.int8.onnx
  -rw-r--r--@ 1 rpwr021  wheel   9.1M Jun 11 19:41 joiner.int8.onnx
  -rw-r--r--@ 1 rpwr021  wheel   222B Jun 11 19:43 README.md
  drwxr-xr-x@ 4 rpwr021  wheel   128B Jun 11 19:44 test_wavs
  -rw-r--r--@ 1 rpwr021  wheel   128K Jun 11 19:33 tokens.txt

Decode a wave file
^^^^^^^^^^^^^^^^^^

To force a specific language, pass ``--language`` with a language string,
for example ``--language=ja`` for Japanese:

.. code-block::

   build/bin/sherpa-onnx \
     --encoder=./sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11/encoder.int8.onnx \
     --decoder=./sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11/decoder.int8.onnx \
     --joiner=./sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11/joiner.int8.onnx \
     --tokens=./sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11/tokens.txt \
     --language=ja \
     ./sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11/test_wavs/ja.wav

To let the model detect the language automatically, use ``--language=auto``
(this is also the behavior when ``--language`` is omitted):

.. code-block::

   build/bin/sherpa-onnx \
     --encoder=./sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11/encoder.int8.onnx \
     --decoder=./sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11/decoder.int8.onnx \
     --joiner=./sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11/joiner.int8.onnx \
     --tokens=./sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11/tokens.txt \
     --language=auto \
     ./sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11/test_wavs/en.wav

.. hint::

   The ``--language`` value is set per-stream. When using the C/C++/Python/Swift
   APIs, set it on the stream with ``SetOption("language", "<lang>")`` before
   feeding waveform samples.

Real-time speech recognition from a microphone
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Please use the following command for real-time speech recognition with a microphone:

.. code-block::

   build/bin/sherpa-onnx-microphone \
     --encoder=./sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11/encoder.int8.onnx \
     --decoder=./sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11/decoder.int8.onnx \
     --joiner=./sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11/joiner.int8.onnx \
     --tokens=./sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11/tokens.txt \
     --language=auto
