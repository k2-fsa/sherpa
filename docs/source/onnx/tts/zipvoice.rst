.. _onnx-tts-zipvoice:

ZipVoice
========

This page shows how to use `sherpa-onnx`_ with ZipVoice.

ZipVoice is an offline zero-shot voice cloning model. It uses both a reference
audio clip and the matching reference text.

Unlike :ref:`onnx-tts-pocket`, ZipVoice requires both ``--reference-audio`` and
``--reference-text``.

Download a pre-trained model
----------------------------

Download the released ZipVoice archive from
`<https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models>`_:

.. code-block:: bash

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-zipvoice-distill-int8-zh-en-emilia.tar.bz2
   tar xf sherpa-onnx-zipvoice-distill-int8-zh-en-emilia.tar.bz2
   rm sherpa-onnx-zipvoice-distill-int8-zh-en-emilia.tar.bz2

You also need to download the vocoder model ``vocos_24khz.onnx``:

.. code-block:: bash

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos_24khz.onnx

Run it from the command line
----------------------------

The following command matches the model configuration used by
`rust-api-examples/examples/zipvoice_tts.rs <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/examples/zipvoice_tts.rs>`_:

.. code-block:: bash

   ./build/bin/sherpa-onnx-offline-tts \
     --zipvoice-encoder=./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/encoder.int8.onnx \
     --zipvoice-decoder=./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/decoder.int8.onnx \
     --zipvoice-data-dir=./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/espeak-ng-data \
     --zipvoice-lexicon=./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/lexicon.txt \
     --zipvoice-tokens=./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/tokens.txt \
     --zipvoice-vocoder=./vocos_24khz.onnx \
     --reference-audio=./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/test_wavs/leijun-1.wav \
     --reference-text="那还是三十六年前, 一九八七年. 我呢考上了武汉大学的计算机系." \
     --num-steps=4 \
     --output-filename=./zipvoice.wav \
     "小米的价值观是真诚, 热爱, 真诚，就是不欺人也不自欺."

The tracked helper script in the repo is:

- `rust-api-examples/run-zipvoice-tts.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/run-zipvoice-tts.sh>`_

.. important::

   ``--reference-text`` should be the exact transcript of ``--reference-audio``.
   If they do not match, the synthesized voice quality can degrade noticeably.

API examples
------------

Please see the following tracked example code:

- Rust

  - `rust-api-examples/examples/zipvoice_tts.rs <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/examples/zipvoice_tts.rs>`_
  - `rust-api-examples/run-zipvoice-tts.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/run-zipvoice-tts.sh>`_

- C++ and C

  - `cxx-api-examples/zipvoice-tts-zh-en-cxx-api.cc <https://github.com/k2-fsa/sherpa-onnx/blob/master/cxx-api-examples/zipvoice-tts-zh-en-cxx-api.cc>`_
  - `c-api-examples/zipvoice-tts-zh-en-c-api.c <https://github.com/k2-fsa/sherpa-onnx/blob/master/c-api-examples/zipvoice-tts-zh-en-c-api.c>`_

- Python

  - `python-api-examples/zipvoice-tts.py <https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/zipvoice-tts.py>`_
  - `python-api-examples/zipvoice-tts-play.py <https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/zipvoice-tts-play.py>`_

- Go

  - `go-api-examples/zero-shot-zipvoice-tts/main.go <https://github.com/k2-fsa/sherpa-onnx/blob/master/go-api-examples/zero-shot-zipvoice-tts/main.go>`_
  - `go-api-examples/zero-shot-zipvoice-tts/run.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/go-api-examples/zero-shot-zipvoice-tts/run.sh>`_
  - `go-api-examples/zero-shot-zipvoice-tts-play/main.go <https://github.com/k2-fsa/sherpa-onnx/blob/master/go-api-examples/zero-shot-zipvoice-tts-play/main.go>`_
  - `go-api-examples/zero-shot-zipvoice-tts-play/run.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/go-api-examples/zero-shot-zipvoice-tts-play/run.sh>`_

- Java and Kotlin

  - `java-api-examples/ZipVoiceTts.java <https://github.com/k2-fsa/sherpa-onnx/blob/master/java-api-examples/ZipVoiceTts.java>`_
  - `java-api-examples/run-zipvoice-tts.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/java-api-examples/run-zipvoice-tts.sh>`_
  - `kotlin-api-examples/test_zipvoice_tts.kt <https://github.com/k2-fsa/sherpa-onnx/blob/master/kotlin-api-examples/test_zipvoice_tts.kt>`_

- Dart and Swift

  - `dart-api-examples/tts/bin/zipvoice-zh-en.dart <https://github.com/k2-fsa/sherpa-onnx/blob/master/dart-api-examples/tts/bin/zipvoice-zh-en.dart>`_
  - `dart-api-examples/tts/run-zipvoice-zh-en.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/dart-api-examples/tts/run-zipvoice-zh-en.sh>`_
  - `swift-api-examples/tts-zipvoice.swift <https://github.com/k2-fsa/sherpa-onnx/blob/master/swift-api-examples/tts-zipvoice.swift>`_
  - `swift-api-examples/run-tts-zipvoice.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/swift-api-examples/run-tts-zipvoice.sh>`_

- .NET

  - `dotnet-examples/zipvoice-tts/Program.cs <https://github.com/k2-fsa/sherpa-onnx/blob/master/dotnet-examples/zipvoice-tts/Program.cs>`_
  - `dotnet-examples/zipvoice-tts/run.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/dotnet-examples/zipvoice-tts/run.sh>`_
  - `dotnet-examples/zipvoice-tts-play/Program.cs <https://github.com/k2-fsa/sherpa-onnx/blob/master/dotnet-examples/zipvoice-tts-play/Program.cs>`_
  - `dotnet-examples/zipvoice-tts-play/run.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/dotnet-examples/zipvoice-tts-play/run.sh>`_

- JavaScript

  - `nodejs-examples/test-offline-tts-zipvoice-zh-en.js <https://github.com/k2-fsa/sherpa-onnx/blob/master/nodejs-examples/test-offline-tts-zipvoice-zh-en.js>`_
  - `nodejs-addon-examples/test_tts_non_streaming_zipvoice_zh_en.js <https://github.com/k2-fsa/sherpa-onnx/blob/master/nodejs-addon-examples/test_tts_non_streaming_zipvoice_zh_en.js>`_
  - `nodejs-addon-examples/test_tts_non_streaming_zipvoice_zh_en_async.js <https://github.com/k2-fsa/sherpa-onnx/blob/master/nodejs-addon-examples/test_tts_non_streaming_zipvoice_zh_en_async.js>`_
  - `nodejs-addon-examples/test_tts_non_streaming_zipvoice_zh_en_play_async.js <https://github.com/k2-fsa/sherpa-onnx/blob/master/nodejs-addon-examples/test_tts_non_streaming_zipvoice_zh_en_play_async.js>`_

- Pascal

  - `pascal-api-examples/tts/zipvoice-zh-en.pas <https://github.com/k2-fsa/sherpa-onnx/blob/master/pascal-api-examples/tts/zipvoice-zh-en.pas>`_
  - `pascal-api-examples/tts/run-zipvoice-zh-en.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/pascal-api-examples/tts/run-zipvoice-zh-en.sh>`_

Notes
-----

- ZipVoice needs both ``--reference-audio`` and ``--reference-text``.
- ``--reference-text`` should exactly match what is spoken in ``--reference-audio``.
- The released example model is bilingual Chinese/English.
- ``--num-steps`` controls the generation quality/speed tradeoff.
