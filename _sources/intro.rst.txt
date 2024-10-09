Introduction
============

`sherpa`_ is the deployment framework of the ``Next-gen Kaldi`` project.

`sherpa`_ does only one thing, using a pre-trained model to transcribe speech.
If you are interested in how to train your own model or fine tune a pre-trained
model, please refer to `icefall`_.

At present, `sherpa`_ has the following sub-projects:

  - `k2-fsa/sherpa`_
  - `k2-fsa/sherpa-onnx`_
  - `k2-fsa/sherpa-ncnn`_


The differences are compared below:

.. list-table::

 * - ****
   - `k2-fsa/sherpa`_
   - `k2-fsa/sherpa-onnx`_
   - `k2-fsa/sherpa-ncnn`_
 * - Installation difficulty
   - **hard**
   - ``easy``
   - ``easy``
 * - NN lib
   - `PyTorch`_
   - `onnxruntime`_
   - `ncnn`_
 * - CPU Support
   - x86, x86_64
   - | x86, x86_64,
     | ``arm32``, ``arm64``
   - | x86, x86_64,
     | ``arm32``, ``arm64``,
     | ``**RISC-V**``
 * - GPU Support
   - | Yes
     | (with ``CUDA`` for NVIDIA GPUs)
   - Yes
   - | Yes
     | (with ``Vulkan`` for ARM GPUs)
 * - OS Support
   - | Linux, Windows,
     | macOS
   - | Linux, Windows,
     | macOS, ``iOS``,
     | ``Android``
   - | Linux, Windows,
     | macOS, ``iOS``,
     | ``Android``
 * - Support batch_size > 1
   - Yes
   - Yes
   - ``No``
 * - Provided APIs
   - C++, Python
   - | C, C++, Python,
     | C#, Java, Kotlin,
     | Swift, Go,
     | JavaScript, Dart
   - | C, C++, Python,
     | C#, Kotlin,
     | Swift, Go
 * - Supported functions
   - | streaming speech recognition,
     | non-streaming speech recognition
   - | streaming speech recognition,
     | non-streaming speech recognition,
     | text-to-speech,
     | speaker identification,
     | speaker verification,
     | spoken language identification,
     | audio tagging,
     | VAD,
     | keyword spotting,
   - streaming speech recognition


We also support `Triton`_. Please see :ref:`triton_overview`.
