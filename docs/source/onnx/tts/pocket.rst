.. _onnx-tts-pocket:

PocketTTS
=========

This page explains how to use `sherpa-onnx`_ with PocketTTS.

PocketTTS is an offline zero-shot text-to-speech model. It uses a short
reference audio clip to clone the target voice.

Unlike :ref:`onnx-tts-zipvoice`, PocketTTS does **not** require a reference
transcript. You only need ``--reference-audio``.

Download a pre-trained model
----------------------------

Download the released PocketTTS archive from
`<https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models>`_:

.. code-block:: bash

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2
   tar xf sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2
   rm sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2

Run a command-line example
--------------------------

The following command uses the same model files as
`rust-api-examples/examples/pocket_tts.rs <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/examples/pocket_tts.rs>`_:

.. code-block:: bash

   ./build/bin/sherpa-onnx-offline-tts \
     --pocket-lm-flow=./sherpa-onnx-pocket-tts-int8-2026-01-26/lm_flow.int8.onnx \
     --pocket-lm-main=./sherpa-onnx-pocket-tts-int8-2026-01-26/lm_main.int8.onnx \
     --pocket-encoder=./sherpa-onnx-pocket-tts-int8-2026-01-26/encoder.onnx \
     --pocket-decoder=./sherpa-onnx-pocket-tts-int8-2026-01-26/decoder.int8.onnx \
     --pocket-text-conditioner=./sherpa-onnx-pocket-tts-int8-2026-01-26/text_conditioner.onnx \
     --pocket-vocab-json=./sherpa-onnx-pocket-tts-int8-2026-01-26/vocab.json \
     --pocket-token-scores-json=./sherpa-onnx-pocket-tts-int8-2026-01-26/token_scores.json \
     --reference-audio=./sherpa-onnx-pocket-tts-int8-2026-01-26/test_wavs/bria.wav \
     --num-steps=2 \
     --output-filename=./pocket.wav \
     "Today as always, men fall into two groups: slaves and free men."

You can also use this tracked helper script:

- `rust-api-examples/run-pocket-tts.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/run-pocket-tts.sh>`_

API examples
------------

Additional example code is available here:

- Rust

  - `rust-api-examples/examples/pocket_tts.rs <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/examples/pocket_tts.rs>`_
  - `rust-api-examples/run-pocket-tts.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/run-pocket-tts.sh>`_

- C++ and C

  - `cxx-api-examples/pocket-tts-en-cxx-api.cc <https://github.com/k2-fsa/sherpa-onnx/blob/master/cxx-api-examples/pocket-tts-en-cxx-api.cc>`_
  - `c-api-examples/pocket-tts-en-c-api.c <https://github.com/k2-fsa/sherpa-onnx/blob/master/c-api-examples/pocket-tts-en-c-api.c>`_

- Python

  - `python-api-examples/pocket-tts.py <https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/pocket-tts.py>`_
  - `python-api-examples/pocket-tts-play.py <https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/pocket-tts-play.py>`_

- Go

  - `go-api-examples/zero-shot-pocket-tts/main.go <https://github.com/k2-fsa/sherpa-onnx/blob/master/go-api-examples/zero-shot-pocket-tts/main.go>`_
  - `go-api-examples/zero-shot-pocket-tts/run.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/go-api-examples/zero-shot-pocket-tts/run.sh>`_
  - `go-api-examples/zero-shot-pocket-tts-play/main.go <https://github.com/k2-fsa/sherpa-onnx/blob/master/go-api-examples/zero-shot-pocket-tts-play/main.go>`_
  - `go-api-examples/zero-shot-pocket-tts-play/run.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/go-api-examples/zero-shot-pocket-tts-play/run.sh>`_

- Java and Kotlin

  - `java-api-examples/PocketTts.java <https://github.com/k2-fsa/sherpa-onnx/blob/master/java-api-examples/PocketTts.java>`_
  - `java-api-examples/run-pocket-tts.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/java-api-examples/run-pocket-tts.sh>`_
  - `kotlin-api-examples/test_pocket_tts.kt <https://github.com/k2-fsa/sherpa-onnx/blob/master/kotlin-api-examples/test_pocket_tts.kt>`_

- Dart and Swift

  - `dart-api-examples/tts/bin/pocket-en.dart <https://github.com/k2-fsa/sherpa-onnx/blob/master/dart-api-examples/tts/bin/pocket-en.dart>`_
  - `dart-api-examples/tts/run-pocket-en.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/dart-api-examples/tts/run-pocket-en.sh>`_
  - `swift-api-examples/tts-pocket-en.swift <https://github.com/k2-fsa/sherpa-onnx/blob/master/swift-api-examples/tts-pocket-en.swift>`_
  - `swift-api-examples/run-tts-pocket-en.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/swift-api-examples/run-tts-pocket-en.sh>`_

- .NET

  - `dotnet-examples/pocket-tts-zero-shot/Program.cs <https://github.com/k2-fsa/sherpa-onnx/blob/master/dotnet-examples/pocket-tts-zero-shot/Program.cs>`_
  - `dotnet-examples/pocket-tts-zero-shot/run.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/dotnet-examples/pocket-tts-zero-shot/run.sh>`_
  - `dotnet-examples/pocket-tts-zero-shot-play/Program.cs <https://github.com/k2-fsa/sherpa-onnx/blob/master/dotnet-examples/pocket-tts-zero-shot-play/Program.cs>`_
  - `dotnet-examples/pocket-tts-zero-shot-play/run.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/dotnet-examples/pocket-tts-zero-shot-play/run.sh>`_

- JavaScript

  - `nodejs-examples/test-offline-tts-pocket-en.js <https://github.com/k2-fsa/sherpa-onnx/blob/master/nodejs-examples/test-offline-tts-pocket-en.js>`_
  - `nodejs-addon-examples/test_tts_non_streaming_pocket_en.js <https://github.com/k2-fsa/sherpa-onnx/blob/master/nodejs-addon-examples/test_tts_non_streaming_pocket_en.js>`_
  - `nodejs-addon-examples/test_tts_non_streaming_pocket_en_async.js <https://github.com/k2-fsa/sherpa-onnx/blob/master/nodejs-addon-examples/test_tts_non_streaming_pocket_en_async.js>`_
  - `nodejs-addon-examples/test_tts_non_streaming_pocket_en_play_async.js <https://github.com/k2-fsa/sherpa-onnx/blob/master/nodejs-addon-examples/test_tts_non_streaming_pocket_en_play_async.js>`_

- Pascal

  - `pascal-api-examples/tts/pocket-en.pas <https://github.com/k2-fsa/sherpa-onnx/blob/master/pascal-api-examples/tts/pocket-en.pas>`_
  - `pascal-api-examples/tts/run-pocket-en.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/pascal-api-examples/tts/run-pocket-en.sh>`_

Notes
-----

- PocketTTS needs a reference audio clip.
- PocketTTS does **not** require reference text. This is different from :ref:`onnx-tts-zipvoice`.
- The reference audio should contain the voice that you want to clone.
- ``--num-steps`` controls the generation quality/speed tradeoff.
