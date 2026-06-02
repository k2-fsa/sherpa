Dolphin
=======

This section describes how to use models from `<https://github.com/DataoceanAI/Dolphin>`_.

Note that this model supports 40 Eastern languages across East Asia,
South Asia, Southeast Asia, and the Middle East;
it also supports 22 Chinese dialects

Please see its GitHub description for more information.

We have converted `Dolphin`_ to onnx and provided APIs for the following programming languages

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
  - 11. Object Pascal

Note that you can use `Dolphin`_ with `sherpa-onnx`_ on the following platforms:

  - Linux (x64, aarch64, arm, riscv64)
  - macOS (x64, arm64)
  - Windows (x64, x86, arm64)
  - Android (arm64-v8a, armv7-eabi, x86, x86_64)
  - iOS (arm64)

In the following, we describe how to download pre-trained `Dolphin`_ models
and use them in `sherpa-onnx`_.

.. toctree::
   :maxdepth: 5

   ./huggingface-space.rst
   ./pretrained.rst
