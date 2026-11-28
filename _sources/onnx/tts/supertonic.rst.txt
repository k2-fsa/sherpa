.. _onnx-tts-supertonic:

SupertonicTTS
=============

This page explains how to use `sherpa-onnx`_ with `SupertonicTTS`_.

.. hint::

   Support of this model in `sherpa-onnx`_ is contributed by `<https://github.com/Wasser1462>`_
   in the PR `<https://github.com/k2-fsa/sherpa-onnx/pull/3605>`_.

`SupertonicTTS`_ 3 is an offline multi-speaker, multi-language TTS model supporting
**31 languages**. In a typical setup, you select a speaker with ``--sid`` and
a language with ``--lang``.

You can try it online at `HuggingFace Spaces <https://huggingface.co/spaces/k2-fsa/text-to-speech>`_.

The following table lists the supported languages and links to their
documentation, download instructions, and code examples.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Language
     - Documentation
   * - Arabic
     - `supertonic-3-ar <https://k2-fsa.github.io/sherpa/onnx/tts/all/Arabic/supertonic-3-ar.html>`_
   * - Bulgarian
     - `supertonic-3-bg <https://k2-fsa.github.io/sherpa/onnx/tts/all/Bulgarian/supertonic-3-bg.html>`_
   * - Croatian
     - `supertonic-3-hr <https://k2-fsa.github.io/sherpa/onnx/tts/all/Croatian/supertonic-3-hr.html>`_
   * - Czech
     - `supertonic-3-cs <https://k2-fsa.github.io/sherpa/onnx/tts/all/Czech/supertonic-3-cs.html>`_
   * - Danish
     - `supertonic-3-da <https://k2-fsa.github.io/sherpa/onnx/tts/all/Danish/supertonic-3-da.html>`_
   * - Dutch
     - `supertonic-3-nl <https://k2-fsa.github.io/sherpa/onnx/tts/all/Dutch/supertonic-3-nl.html>`_
   * - English
     - `supertonic-3-en <https://k2-fsa.github.io/sherpa/onnx/tts/all/English/supertonic-3-en.html>`_
   * - Estonian
     - `supertonic-3-et <https://k2-fsa.github.io/sherpa/onnx/tts/all/Estonian/supertonic-3-et.html>`_
   * - Finnish
     - `supertonic-3-fi <https://k2-fsa.github.io/sherpa/onnx/tts/all/Finnish/supertonic-3-fi.html>`_
   * - French
     - `supertonic-3-fr <https://k2-fsa.github.io/sherpa/onnx/tts/all/French/supertonic-3-fr.html>`_
   * - German
     - `supertonic-3-de <https://k2-fsa.github.io/sherpa/onnx/tts/all/German/supertonic-3-de.html>`_
   * - Greek
     - `supertonic-3-el <https://k2-fsa.github.io/sherpa/onnx/tts/all/Greek/supertonic-3-el.html>`_
   * - Hindi
     - `supertonic-3-hi <https://k2-fsa.github.io/sherpa/onnx/tts/all/Hindi/supertonic-3-hi.html>`_
   * - Hungarian
     - `supertonic-3-hu <https://k2-fsa.github.io/sherpa/onnx/tts/all/Hungarian/supertonic-3-hu.html>`_
   * - Indonesian
     - `supertonic-3-id <https://k2-fsa.github.io/sherpa/onnx/tts/all/Indonesian/supertonic-3-id.html>`_
   * - Italian
     - `supertonic-3-it <https://k2-fsa.github.io/sherpa/onnx/tts/all/Italian/supertonic-3-it.html>`_
   * - Japanese
     - `supertonic-3-ja <https://k2-fsa.github.io/sherpa/onnx/tts/all/Japanese/supertonic-3-ja.html>`_
   * - Korean
     - `supertonic-3-ko <https://k2-fsa.github.io/sherpa/onnx/tts/all/Korean/supertonic-3-ko.html>`_
   * - Latvian
     - `supertonic-3-lv <https://k2-fsa.github.io/sherpa/onnx/tts/all/Latvian/supertonic-3-lv.html>`_
   * - Lithuanian
     - `supertonic-3-lt <https://k2-fsa.github.io/sherpa/onnx/tts/all/Lithuanian/supertonic-3-lt.html>`_
   * - Polish
     - `supertonic-3-pl <https://k2-fsa.github.io/sherpa/onnx/tts/all/Polish/supertonic-3-pl.html>`_
   * - Portuguese
     - `supertonic-3-pt <https://k2-fsa.github.io/sherpa/onnx/tts/all/Portuguese/supertonic-3-pt.html>`_
   * - Romanian
     - `supertonic-3-ro <https://k2-fsa.github.io/sherpa/onnx/tts/all/Romanian/supertonic-3-ro.html>`_
   * - Russian
     - `supertonic-3-ru <https://k2-fsa.github.io/sherpa/onnx/tts/all/Russian/supertonic-3-ru.html>`_
   * - Slovak
     - `supertonic-3-sk <https://k2-fsa.github.io/sherpa/onnx/tts/all/Slovak/supertonic-3-sk.html>`_
   * - Slovenian
     - `supertonic-3-sl <https://k2-fsa.github.io/sherpa/onnx/tts/all/Slovenian/supertonic-3-sl.html>`_
   * - Spanish
     - `supertonic-3-es <https://k2-fsa.github.io/sherpa/onnx/tts/all/Spanish/supertonic-3-es.html>`_
   * - Swedish
     - `supertonic-3-sv <https://k2-fsa.github.io/sherpa/onnx/tts/all/Swedish/supertonic-3-sv.html>`_
   * - Turkish
     - `supertonic-3-tr <https://k2-fsa.github.io/sherpa/onnx/tts/all/Turkish/supertonic-3-tr.html>`_
   * - Ukrainian
     - `supertonic-3-uk <https://k2-fsa.github.io/sherpa/onnx/tts/all/Ukrainian/supertonic-3-uk.html>`_
   * - Vietnamese
     - `supertonic-3-vi <https://k2-fsa.github.io/sherpa/onnx/tts/all/Vietnamese/supertonic-3-vi.html>`_

Download a pre-trained model
----------------------------

Download the released SupertonicTTS archive from
`<https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models>`_:

.. code-block:: bash

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-supertonic-3-tts-int8-2026-05-11.tar.bz2
   tar xf sherpa-onnx-supertonic-3-tts-int8-2026-05-11.tar.bz2
   rm sherpa-onnx-supertonic-3-tts-int8-2026-05-11.tar.bz2

Run a command-line example
--------------------------

The following command uses the same model files as
`rust-api-examples/examples/supertonic_tts.rs <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/examples/supertonic_tts.rs>`_:

.. code-block:: bash

   ./build/bin/sherpa-onnx-offline-tts \
     --supertonic-duration-predictor=./sherpa-onnx-supertonic-3-tts-int8-2026-05-11/duration_predictor.int8.onnx \
     --supertonic-text-encoder=./sherpa-onnx-supertonic-3-tts-int8-2026-05-11/text_encoder.int8.onnx \
     --supertonic-vector-estimator=./sherpa-onnx-supertonic-3-tts-int8-2026-05-11/vector_estimator.int8.onnx \
     --supertonic-vocoder=./sherpa-onnx-supertonic-3-tts-int8-2026-05-11/vocoder.int8.onnx \
     --supertonic-tts-json=./sherpa-onnx-supertonic-3-tts-int8-2026-05-11/tts.json \
     --supertonic-unicode-indexer=./sherpa-onnx-supertonic-3-tts-int8-2026-05-11/unicode_indexer.bin \
     --supertonic-voice-style=./sherpa-onnx-supertonic-3-tts-int8-2026-05-11/voice.bin \
     --sid=0 \
     --lang=en \
     --output-filename=./supertonic.wav \
     "Today as always, men fall into two groups: slaves and free men."

You can change ``--lang`` to any of the 31 supported language codes
(e.g., ``ja`` for Japanese, ``ko`` for Korean, ``fr`` for French).

You can also use this tracked helper script:

- `rust-api-examples/run-supertonic-tts.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/run-supertonic-tts.sh>`_

API examples
------------

Additional example code is available here:

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
- Use ``--lang`` to select the synthesis language (e.g., ``en``, ``ja``, ``ko``, ``fr``, etc.).
- The model files include ``tts.json`` and ``unicode_indexer.bin`` in addition to ONNX files.

See also
--------

- :ref:`onnx-tts-pretrained-models`
- `<https://k2-fsa.github.io/sherpa/onnx/tts/all/>`_
