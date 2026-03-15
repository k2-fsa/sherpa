.. _onnx-funasr-nano:

FunASR Nano
===========

This section describes how to use models from `<https://github.com/FunAudioLLM/Fun-ASR>`_.

Fun-ASR-Nano-2512
-----------------

A single model from `Fun-ASR-Nano-2512`_ supports the following languages

  - Chinese
  - English
  - Japanese

.. hint::

   中文包括 7 种方言（吴语、粤语、闽语、客家话、赣语、湘语、晋语）和
   26 种地方口音（河南、山西、湖北、四川、重庆、云南、贵州、广东、广西
   及其他 20 多个地区）。

   英文和日文涵盖多种地方口音。

   此外还支持歌词识别和说唱语音识别。

We have converted `Fun-ASR-Nano-2512`_ to onnx and provided APIs for the following programming languages

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

You can find the onnx export script at `<https://github.com/Wasser1462/FunASR-nano-onnx>`_

Note that you can use `Fun-ASR-Nano-2512`_ with `sherpa-onnx`_ on the following platforms:

  - Linux (x64, aarch64, arm, riscv64)
  - macOS (x64, arm64)
  - Windows (x64, x86, arm64)
  - Android (arm64-v8a, armv7-eabi, x86, x86_64)
  - iOS (arm64)

In the following, we describe how to download pre-trained `Fun-ASR-Nano-2512`_ models
and use them in `sherpa-onnx`_.


.. toctree::
   :maxdepth: 6

   ./huggingface-space.rst
   ./export.rst
   ./pretrained.rst
