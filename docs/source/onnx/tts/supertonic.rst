.. _onnx-tts-supertonic:

SupertonicTTS
=============

This page shows how to use `sherpa-onnx`_ with SupertonicTTS.

SupertonicTTS is an offline multi-speaker, multi-language TTS model. Typical
usage selects a speaker with ``--sid`` and a language with ``--lang``.

Download a pre-trained model
----------------------------

Download the released SupertonicTTS archive from
`<https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models>`_:

.. code-block:: bash

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-supertonic-tts-int8-2026-03-06.tar.bz2
   tar xf sherpa-onnx-supertonic-tts-int8-2026-03-06.tar.bz2
   rm sherpa-onnx-supertonic-tts-int8-2026-03-06.tar.bz2

Run it from the command line
----------------------------

The following command matches the model configuration used by
`rust-api-examples/examples/supertonic_tts.rs <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/examples/supertonic_tts.rs>`_:

.. code-block:: bash

   ./build/bin/sherpa-onnx-offline-tts \
     --supertonic-duration-predictor=./sherpa-onnx-supertonic-tts-int8-2026-03-06/duration_predictor.int8.onnx \
     --supertonic-text-encoder=./sherpa-onnx-supertonic-tts-int8-2026-03-06/text_encoder.int8.onnx \
     --supertonic-vector-estimator=./sherpa-onnx-supertonic-tts-int8-2026-03-06/vector_estimator.int8.onnx \
     --supertonic-vocoder=./sherpa-onnx-supertonic-tts-int8-2026-03-06/vocoder.int8.onnx \
     --supertonic-tts-json=./sherpa-onnx-supertonic-tts-int8-2026-03-06/tts.json \
     --supertonic-unicode-indexer=./sherpa-onnx-supertonic-tts-int8-2026-03-06/unicode_indexer.bin \
     --supertonic-voice-style=./sherpa-onnx-supertonic-tts-int8-2026-03-06/voice.bin \
     --sid=0 \
     --lang=en \
     --output-filename=./supertonic.wav \
     "Today as always, men fall into two groups: slaves and free men."

The tracked helper script in the repo is:

- `rust-api-examples/run-supertonic-tts.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/run-supertonic-tts.sh>`_

API examples
------------

Please see the following tracked example code:

- Rust

  - `rust-api-examples/examples/supertonic_tts.rs <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/examples/supertonic_tts.rs>`_
  - `rust-api-examples/run-supertonic-tts.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/run-supertonic-tts.sh>`_

- C++ and C

  - `cxx-api-examples/supertonic-tts-en-cxx-api.cc <https://github.com/k2-fsa/sherpa-onnx/blob/master/cxx-api-examples/supertonic-tts-en-cxx-api.cc>`_
  - `c-api-examples/supertonic-tts-en-c-api.c <https://github.com/k2-fsa/sherpa-onnx/blob/master/c-api-examples/supertonic-tts-en-c-api.c>`_

- Python

  - `python-api-examples/supertonic-tts.py <https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/supertonic-tts.py>`_

- Go

  - `go-api-examples/supertonic-tts/main.go <https://github.com/k2-fsa/sherpa-onnx/blob/master/go-api-examples/supertonic-tts/main.go>`_
  - `go-api-examples/supertonic-tts/run.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/go-api-examples/supertonic-tts/run.sh>`_

- Java and Kotlin

  - `java-api-examples/SupertonicTts.java <https://github.com/k2-fsa/sherpa-onnx/blob/master/java-api-examples/SupertonicTts.java>`_
  - `java-api-examples/run-supertonic-tts.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/java-api-examples/run-supertonic-tts.sh>`_
  - `kotlin-api-examples/test_supertonic_tts.kt <https://github.com/k2-fsa/sherpa-onnx/blob/master/kotlin-api-examples/test_supertonic_tts.kt>`_

- Dart and Swift

  - `dart-api-examples/tts/bin/supertonic-en.dart <https://github.com/k2-fsa/sherpa-onnx/blob/master/dart-api-examples/tts/bin/supertonic-en.dart>`_
  - `dart-api-examples/tts/run-supertonic-en.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/dart-api-examples/tts/run-supertonic-en.sh>`_
  - `swift-api-examples/tts-supertonic-en.swift <https://github.com/k2-fsa/sherpa-onnx/blob/master/swift-api-examples/tts-supertonic-en.swift>`_
  - `swift-api-examples/run-tts-supertonic-en.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/swift-api-examples/run-tts-supertonic-en.sh>`_

- .NET

  - `dotnet-examples/supertonic-tts/Program.cs <https://github.com/k2-fsa/sherpa-onnx/blob/master/dotnet-examples/supertonic-tts/Program.cs>`_
  - `dotnet-examples/supertonic-tts/run.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/dotnet-examples/supertonic-tts/run.sh>`_

- JavaScript

  - `nodejs-addon-examples/test_tts_non_streaming_supertonic_en.js <https://github.com/k2-fsa/sherpa-onnx/blob/master/nodejs-addon-examples/test_tts_non_streaming_supertonic_en.js>`_
  - `nodejs-addon-examples/test_tts_non_streaming_supertonic_en_async.js <https://github.com/k2-fsa/sherpa-onnx/blob/master/nodejs-addon-examples/test_tts_non_streaming_supertonic_en_async.js>`_
  - `nodejs-addon-examples/test_tts_non_streaming_supertonic_en_play_async.js <https://github.com/k2-fsa/sherpa-onnx/blob/master/nodejs-addon-examples/test_tts_non_streaming_supertonic_en_play_async.js>`_

- Pascal

  - `pascal-api-examples/tts/supertonic-en.pas <https://github.com/k2-fsa/sherpa-onnx/blob/master/pascal-api-examples/tts/supertonic-en.pas>`_
  - `pascal-api-examples/tts/run-supertonic-en.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/pascal-api-examples/tts/run-supertonic-en.sh>`_

Notes
-----

- Use ``--sid`` to choose a speaker.
- Use ``--lang`` to select the synthesis language.
- The model files include ``tts.json`` and ``unicode_indexer.bin`` in addition to ONNX files.
