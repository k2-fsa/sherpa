sherpa-onnx
===========

.. hint::

  During speech recognition, it does not need to access the Internet.
  Everyting is processed locally on your device.

We support using `onnx`_ with `onnxruntime`_ to replace `PyTorch`_ for neural
network computation. The code is put in a separate repository `sherpa-onnx`_.

`sherpa-onnx`_ is self-contained and everything can be compiled from source.

Please refer to
`<https://k2-fsa.github.io/icefall/model-export/export-onnx.html>`_
for how to export models to `onnx`_ format.

In the following, we describe how to build `sherpa-onnx`_ for Linux, macOS,
Windows, embedded systems, Android, and iOS.

Also, we show how to use it for speech recognition with pre-trained models.

.. toctree::
   :maxdepth: 5

   ./tutorials/index
   ./install/index
   ./faqs/index
   ./python/index
   ./c-api/index
   ./java-api/index
   ./javascript-api/index
   ./kotlin-api/index
   ./swift-api/index
   ./go-api/index
   ./csharp-api/index
   ./wasm/index
   ./android/index
   ./ios/index
   ./websocket/index
   ./hotwords/index
   ./kws/index
   ./punctuation/index
   ./audio-tagging/index
   ./spoken-language-identification/index
   ./vad/index
   ./pretrained_models/index

.. toctree::
   :maxdepth: 5
   :caption: Speaker Identification

   ./speaker-identification/index

.. toctree::
   :maxdepth: 5
   :caption: tts

   ./tts/index
