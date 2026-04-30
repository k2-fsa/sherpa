.. _onnx-tts-kitten:

KittenTTS
=========

This page explains how to use `sherpa-onnx`_ with
`KittenTTS <https://github.com/KittenML/KittenTTS>`_.

KittenTTS is a compact English text-to-speech model. It does not require a
reference audio prompt. You select a speaker with ``--sid`` and synthesize
audio directly.

Download a pre-trained model
----------------------------

The quickest way is to download one of the pre-built model archives from
`<https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models>`_.

For example:

.. code-block:: bash

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kitten-nano-en-v0_1-fp16.tar.bz2
   tar xf kitten-nano-en-v0_1-fp16.tar.bz2
   rm kitten-nano-en-v0_1-fp16.tar.bz2

Other released KittenTTS models are listed in :ref:`kitten-nano-v01`.

Run a command-line example
--------------------------

The following command uses the same model files as
`rust-api-examples/examples/kitten_tts_en.rs <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/examples/kitten_tts_en.rs>`_:

.. code-block:: bash

   ./build/bin/sherpa-onnx-offline-tts \
     --kitten-model=./kitten-nano-en-v0_1-fp16/model.fp16.onnx \
     --kitten-voices=./kitten-nano-en-v0_1-fp16/voices.bin \
     --kitten-tokens=./kitten-nano-en-v0_1-fp16/tokens.txt \
     --kitten-data-dir=./kitten-nano-en-v0_1-fp16/espeak-ng-data \
     --sid=0 \
     --output-filename=./kitten-en.wav \
     "Today as always, men fall into two groups: slaves and free men."

You can also use this tracked helper script:

- `rust-api-examples/run-kitten-tts-en.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/run-kitten-tts-en.sh>`_

API examples
------------

Additional example code is available in
`k2-fsa/sherpa-onnx <https://github.com/k2-fsa/sherpa-onnx>`_:

- Rust

  - `rust-api-examples/examples/kitten_tts_en.rs <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/examples/kitten_tts_en.rs>`_
  - `rust-api-examples/run-kitten-tts-en.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/run-kitten-tts-en.sh>`_

- C++ and C

  - `cxx-api-examples/kitten-tts-en-cxx-api.cc <https://github.com/k2-fsa/sherpa-onnx/blob/master/cxx-api-examples/kitten-tts-en-cxx-api.cc>`_
  - `c-api-examples/kitten-tts-en-c-api.c <https://github.com/k2-fsa/sherpa-onnx/blob/master/c-api-examples/kitten-tts-en-c-api.c>`_

- Python

  - `python-api-examples/offline-tts.py <https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/offline-tts.py>`_
  - `python-api-examples/offline-tts-play.py <https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/offline-tts-play.py>`_

- Go

  - `go-api-examples/non-streaming-tts/main.go <https://github.com/k2-fsa/sherpa-onnx/blob/master/go-api-examples/non-streaming-tts/main.go>`_
  - `go-api-examples/non-streaming-tts/run-kitten-en.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/go-api-examples/non-streaming-tts/run-kitten-en.sh>`_
  - `go-api-examples/offline-tts-play/main.go <https://github.com/k2-fsa/sherpa-onnx/blob/master/go-api-examples/offline-tts-play/main.go>`_
  - `go-api-examples/offline-tts-play/run-kitten-en.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/go-api-examples/offline-tts-play/run-kitten-en.sh>`_

- Java and Kotlin

  - `java-api-examples/NonStreamingTtsKittenEn.java <https://github.com/k2-fsa/sherpa-onnx/blob/master/java-api-examples/NonStreamingTtsKittenEn.java>`_
  - `java-api-examples/run-non-streaming-tts-kitten-en.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/java-api-examples/run-non-streaming-tts-kitten-en.sh>`_
  - `kotlin-api-examples/test_tts.kt <https://github.com/k2-fsa/sherpa-onnx/blob/master/kotlin-api-examples/test_tts.kt>`_

- Dart and Swift

  - `dart-api-examples/tts/bin/kitten-en.dart <https://github.com/k2-fsa/sherpa-onnx/blob/master/dart-api-examples/tts/bin/kitten-en.dart>`_
  - `dart-api-examples/tts/run-kitten-en.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/dart-api-examples/tts/run-kitten-en.sh>`_
  - `swift-api-examples/tts-kitten-en.swift <https://github.com/k2-fsa/sherpa-onnx/blob/master/swift-api-examples/tts-kitten-en.swift>`_
  - `swift-api-examples/run-tts-kitten-en.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/swift-api-examples/run-tts-kitten-en.sh>`_

- .NET

  - `dotnet-examples/kitten-tts/Program.cs <https://github.com/k2-fsa/sherpa-onnx/blob/master/dotnet-examples/kitten-tts/Program.cs>`_
  - `dotnet-examples/kitten-tts/run-kitten.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/dotnet-examples/kitten-tts/run-kitten.sh>`_
  - `dotnet-examples/kitten-tts-play/Program.cs <https://github.com/k2-fsa/sherpa-onnx/blob/master/dotnet-examples/kitten-tts-play/Program.cs>`_
  - `dotnet-examples/kitten-tts-play/run-kitten.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/dotnet-examples/kitten-tts-play/run-kitten.sh>`_

- JavaScript

  - `nodejs-examples/test-offline-tts-kitten-en.js <https://github.com/k2-fsa/sherpa-onnx/blob/master/nodejs-examples/test-offline-tts-kitten-en.js>`_
  - `nodejs-addon-examples/test_tts_non_streaming_kitten_en.js <https://github.com/k2-fsa/sherpa-onnx/blob/master/nodejs-addon-examples/test_tts_non_streaming_kitten_en.js>`_

- Pascal

  - `pascal-api-examples/tts/kitten-en.pas <https://github.com/k2-fsa/sherpa-onnx/blob/master/pascal-api-examples/tts/kitten-en.pas>`_
  - `pascal-api-examples/tts/run-kitten-en.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/pascal-api-examples/tts/run-kitten-en.sh>`_
  - `pascal-api-examples/tts/kitten-en-playback.pas <https://github.com/k2-fsa/sherpa-onnx/blob/master/pascal-api-examples/tts/kitten-en-playback.pas>`_
  - `pascal-api-examples/tts/run-kitten-en-playback.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/pascal-api-examples/tts/run-kitten-en-playback.sh>`_

See also
--------

- :ref:`onnx-tts-pretrained-models`
- :ref:`kitten-nano-v01`
