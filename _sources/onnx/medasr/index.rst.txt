.. _onnx-medasr:

Google MedASR
=============

This section describes how to use Google MedASR with `sherpa-onnx`_.

At present, `sherpa-onnx`_ provides an offline English CTC model:

  - ``sherpa-onnx-medasr-ctc-en-int8-2025-12-25``

The model implementation in `sherpa-onnx`_ is in

  - `<https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/offline-medasr-ctc-model.h>`_

You can find related export and test scripts in

  - `<https://github.com/k2-fsa/sherpa-onnx/tree/master/scripts/medasr>`_

Quick start
-----------

Download the model:

.. code-block:: bash

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-medasr-ctc-en-int8-2025-12-25.tar.bz2
  tar xvf sherpa-onnx-medasr-ctc-en-int8-2025-12-25.tar.bz2
  rm sherpa-onnx-medasr-ctc-en-int8-2025-12-25.tar.bz2

Then run:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-medasr-ctc-en-int8-2025-12-25/tokens.txt \
    --medasr=./sherpa-onnx-medasr-ctc-en-int8-2025-12-25/model.int8.onnx \
    ./sherpa-onnx-medasr-ctc-en-int8-2025-12-25/test_wavs/0.wav \
    ./sherpa-onnx-medasr-ctc-en-int8-2025-12-25/test_wavs/1.wav \
    ./sherpa-onnx-medasr-ctc-en-int8-2025-12-25/test_wavs/2.wav

Examples
--------

We provide MedASR examples for several APIs:

  - C: `<https://github.com/k2-fsa/sherpa-onnx/blob/master/c-api-examples/medasr-ctc-c-api.c>`_
  - C++: `<https://github.com/k2-fsa/sherpa-onnx/blob/master/cxx-api-examples/medasr-ctc-cxx-api.cc>`_
  - Python: `<https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/offline-medasr-ctc-decode-files.py>`_
  - JavaScript (Node.js): `<https://github.com/k2-fsa/sherpa-onnx/blob/master/nodejs-examples/test-offline-medasr-ctc.js>`_
  - Node.js addon: `<https://github.com/k2-fsa/sherpa-onnx/blob/master/nodejs-addon-examples/test_asr_non_streaming_medasr_ctc.js>`_
  - Kotlin: `<https://github.com/k2-fsa/sherpa-onnx/blob/master/kotlin-api-examples/test_offline_medasr_ctc.kt>`_
  - Swift: `<https://github.com/k2-fsa/sherpa-onnx/blob/master/swift-api-examples/medasr-ctc.swift>`_
  - Dart: `<https://github.com/k2-fsa/sherpa-onnx/blob/master/dart-api-examples/non-streaming-asr/bin/medasr-ctc.dart>`_
  - C#: `<https://github.com/k2-fsa/sherpa-onnx/blob/master/dotnet-examples/offline-decode-files/run-medasr-ctc.sh>`_
  - Pascal: `<https://github.com/k2-fsa/sherpa-onnx/blob/master/pascal-api-examples/non-streaming-asr/medasr_ctc.pas>`_

See also
--------

  - `Google MedASR <https://huggingface.co/google/medasr>`_
  - `<https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/medasr/README.md>`_
