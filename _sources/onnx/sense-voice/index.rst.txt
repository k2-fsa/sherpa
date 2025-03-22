SenseVoice
==========

This section describes how to use models from `<https://github.com/FunAudioLLM/SenseVoice>`_.

A single model from `SenseVoice`_ supports the following languages

  - Chinese (Mandarin, 普通话)
  - Cantonese (粤语, 广东话)
  - English
  - Japanese
  - Korean

which is similar to what multilingual `Whisper`_ is doing.

We have converted `SenseVoice`_ to onnx and provided APIs for the following programming languages

  - 1. C++
  - 2. C
  - 3. Python
  - 4. C#
  - 5. Go
  - 6. Kotlin
  - 7. Java
  - 8. JavaScript (Support `WebAssembly`_ and `Node`_)
  - 9. Swift
  - 10. `Dart`_ (Support `Flutter`_)

Note that you can use `SenseVoice`_ with `sherpa-onnx`_ on the following platforms:

  - Linux (x64, aarch64, arm, riscv64)
  - macOS (x64, arm64)
  - Windows (x64, x86, arm64)
  - Android (arm64-v8a, armv7-eabi, x86, x86_64)
  - iOS (arm64)

In the following, we describe how to download pre-trained `SenseVoice`_ models
and use them in `sherpa-onnx`_.


.. toctree::
   :maxdepth: 5

   ./huggingface-space.rst
   ./export.rst
   ./pretrained.rst
   ./c-api.rst
   ./dart-api.rst
   ./python-api.rst
