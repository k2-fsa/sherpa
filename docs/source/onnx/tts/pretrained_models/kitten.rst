KittenTTS
=========

This page lists pre-trained models from `<https://github.com/KittenML/KittenTTS>`_.

kitten-nano-en-v0_1-fp16
------------------------

This model provides 8 voices in total: 4 male and 4 female.

Please see

`<https://github.com/k2-fsa/sherpa-onnx/pull/2460>`_

for details. We have listed the voices of each speaker in the above pull request.

This model is converted from `<https://huggingface.co/KittenML/kitten-tts-nano-0.1>`_

You can find the conversion script from `<https://github.com/k2-fsa/sherpa-onnx/tree/master/scripts/kitten-tts/nano_v0_1>`_

Download the model
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kitten-nano-en-v0_1-fp16.tar.bz2
   tar xf kitten-nano-en-v0_1-fp16.tar.bz2
   rm kitten-nano-en-v0_1-fp16.tar.bz2

Huggingface space
~~~~~~~~~~~~~~~~~~~~

You can try this model by visiting

  `<https://huggingface.co/spaces/k2-fsa/text-to-speech>`_

First select English and then select ``kitten-nano-en-v0_1-fp16`` from the available models.

You don't need to install anything to try it.

Android TTS Engine APK
~~~~~~~~~~~~~~~~~~~~~~~~

You can use it to replace your Android system text to speech engine so that it can be called from 3rd party
ebook-reader APPs.

Please download the pre-built APK from

  `<https://k2-fsa.github.io/sherpa/onnx/tts/apk-engine.html>`_

Search for ``kitten`` in the above page to find ``KittenTTS``.

The source code is available at

  `<https://github.com/k2-fsa/sherpa-onnx/tree/master/android/SherpaOnnxTtsEngine>`_

1. C++ API example
~~~~~~~~~~~~~~~~~~

Please see `<https://github.com/k2-fsa/sherpa-onnx/blob/master/cxx-api-examples/kitten-tts-en-cxx-api.cc>`_

2. Python API example
~~~~~~~~~~~~~~~~~~~~~

Please see

  - `<https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/offline-tts.py>`_
  - `<https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/offline-tts-play.py>`_

3. C API example
~~~~~~~~~~~~~~~~~~

Please see `<https://github.com/k2-fsa/sherpa-onnx/blob/master/c-api-examples/kitten-tts-en-c-api.c>`_

4. Go API example
~~~~~~~~~~~~~~~~~

Please see

  - `<https://github.com/k2-fsa/sherpa-onnx/blob/master/go-api-examples/offline-tts-play/run-kitten-en.sh>`_
  - `<https://github.com/k2-fsa/sherpa-onnx/blob/master/go-api-examples/offline-tts-play/main.go>`_

5. C# API example
~~~~~~~~~~~~~~~~~~~~~~

Please see

  - `<https://github.com/k2-fsa/sherpa-onnx/tree/master/dotnet-examples/kitten-tts>`_
  - `<https://github.com/k2-fsa/sherpa-onnx/tree/master/dotnet-examples/kitten-tts-play>`_

6. Dart API example
~~~~~~~~~~~~~~~~~~~~

Please see

  - `<https://github.com/k2-fsa/sherpa-onnx/blob/master/dart-api-examples/tts/run-kitten-en.sh>`_
  - `<https://github.com/k2-fsa/sherpa-onnx/blob/master/dart-api-examples/tts/bin/kitten-en.dart>`_

7. Swift API example
~~~~~~~~~~~~~~~~~~~~

Please see

  - `<https://github.com/k2-fsa/sherpa-onnx/blob/master/swift-api-examples/run-tts-kitten-en.sh>`_
  - `<https://github.com/k2-fsa/sherpa-onnx/blob/master/swift-api-examples/tts-kitten-en.swift>`_

8. Kotlin API example
~~~~~~~~~~~~~~~~~~~~~~

Please see `<https://github.com/k2-fsa/sherpa-onnx/blob/master/kotlin-api-examples/test_tts.kt#L100>`_

9. Java API example
~~~~~~~~~~~~~~~~~~~~

Please see

  - `<https://github.com/k2-fsa/sherpa-onnx/blob/master/java-api-examples/run-non-streaming-tts-kitten-en.sh>`_
  - `<https://github.com/k2-fsa/sherpa-onnx/blob/master/java-api-examples/NonStreamingTtsKittenEn.java>`_

10. JavaScript API example (nodejs + WebAssembly)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Please see `<https://github.com/k2-fsa/sherpa-onnx/blob/master/nodejs-examples/test-offline-tts-kitten-en.js>`_

11. JavaScript API example (nodejs + node-addon)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Please see `<https://github.com/k2-fsa/sherpa-onnx/blob/master/nodejs-addon-examples/test_tts_non_streaming_kitten_en.js>`_

12. Pascal API example
~~~~~~~~~~~~~~~~~~~~~~~

Please see

  - `<https://github.com/k2-fsa/sherpa-onnx/blob/master/pascal-api-examples/tts/kitten-en.pas>`_
  - `<https://github.com/k2-fsa/sherpa-onnx/blob/master/pascal-api-examples/tts/kitten-en-playback.pas>`_
  - `<https://github.com/k2-fsa/sherpa-onnx/blob/master/pascal-api-examples/tts/run-kitten-en.sh>`_
  - `<https://github.com/k2-fsa/sherpa-onnx/blob/master/pascal-api-examples/tts/run-kitten-en-playback.sh>`_
