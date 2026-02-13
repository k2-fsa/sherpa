Pre-trained Models
==================

This page describes how to download pre-trained `FireRedAsr`_ models.

sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16 (Chinese + English, 普通话、四川话、河南话等)
------------------------------------------------------------------------------------------------

This model is converted from `<https://huggingface.co/FireRedTeam/FireRedASR-AED-L>`_

It supports the following 2 languages:

  - Chinese (普通话, 四川话、天津话、河南话等方言)
  - English

In the following, we describe how to download it.

Download
^^^^^^^^

Please use the following commands to download it::

  cd /path/to/sherpa-onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16.tar.bz2
  tar xvf sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16.tar.bz2
  rm sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16.tar.bz2

After downloading, you should find the following files::

  ls -lh sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16/
  total 1.7G
  -rw-r--r--  1 kuangfangjun root  188 Feb 16 16:22 README.md
  -rw-r--r--  1 kuangfangjun root 425M Feb 16 16:21 decoder.int8.onnx
  -rw-r--r--  1 kuangfangjun root 1.3G Feb 16 16:21 encoder.int8.onnx
  drwxr-xr-x 10 kuangfangjun root    0 Feb 16 16:26 test_wavs
  -rw-r--r--  1 kuangfangjun root  70K Feb 16 16:21 tokens.txt

  ls -lh sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16/test_wavs/
  total 1.9M
  -rw-r--r-- 1 kuangfangjun root 315K Feb 16 16:24 0.wav
  -rw-r--r-- 1 kuangfangjun root 160K Feb 16 16:24 1.wav
  -rw-r--r-- 1 kuangfangjun root 147K Feb 16 16:24 2.wav
  -rw-r--r-- 1 kuangfangjun root 245K Feb 16 16:25 3-sichuan.wav
  -rw-r--r-- 1 kuangfangjun root 276K Feb 16 16:24 3.wav
  -rw-r--r-- 1 kuangfangjun root 245K Feb 16 16:25 4-tianjin.wav
  -rw-r--r-- 1 kuangfangjun root 250K Feb 16 16:26 5-henan.wav
  -rw-r--r-- 1 kuangfangjun root 276K Feb 16 16:24 8k.wav

Decode a file
^^^^^^^^^^^^^

Please use the following command to decode a wave file:

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16/tokens.txt \
    --fire-red-asr-encoder=./sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16/encoder.int8.onnx \
    --fire-red-asr-decoder=./sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16/decoder.int8.onnx \
    --num-threads=1 \
    ./sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16/test_wavs/0.wav

You should see the following output:

.. literalinclude:: ./code/2025-02-16.txt
