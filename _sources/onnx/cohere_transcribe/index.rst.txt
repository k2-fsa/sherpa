.. _onnx-cohere-transcribe:

Cohere Transcribe
=================

This section describes how to use models from `<https://huggingface.co/CohereLabs/cohere-transcribe-03-2026>`_.

A single model from `Cohere Transcribe`_ supports the following 14 languages:

  - European: English, French, German, Italian, Spanish, Portuguese, Greek, Dutch, Polish
  - AIPAC: Chinese (Mandarin), Japanese, Korean, Vietnamese
  - MENA: Arabic

The converted `sherpa-onnx`_ model is an offline non-streaming ASR model. When
decoding, you need to specify the input language, for example ``en`` for
English, ``zh`` for Chinese, and ``de`` for German.

We have provided Cohere Transcribe examples for the following 12 programming
languages:

.. list-table::
   :widths: 25 25 25 25
   :header-rows: 0

   * - C++
     - C
     - Python
     - C#
   * - Go
     - Kotlin
     - Java
     - JavaScript (Node.js and node-addon)
   * - Swift
     - `Dart`_ (Support `Flutter`_)
     - Rust
     - Object Pascal

In the following, we describe how to download a pre-trained model and where to
find the corresponding examples in `k2-fsa/sherpa-onnx`_.

.. toctree::
   :maxdepth: 5

   ./pretrained.rst
   ./huggingface-space.rst
   ./examples.rst
