Export FunASR Nano to sherpa-onnx
==================================

This page describes how to export `Fun-ASR-Nano-2512`_ to onnx so that you can use
it with `sherpa-onnx`_.


The code
--------

Please refer to `<https://github.com/Wasser1462/FunASR-nano-onnx>`_

Where to find exported models
------------------------------

You can find the exported `Fun-ASR-Nano-2512`_ models at

  `<https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models>`_

The following is an example about how to download an exported `Fun-ASR-Nano-2512`_ model::

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2

  tar xvf sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2
  rm sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2

To view the downloaded files, please use::

  ls -lh sherpa-onnx-funasr-nano-int8-2025-12-30/

  total 948M
  drwxr-xr-x  5 kuangfangjun root    0 Jan  7 19:28 Qwen3-0.6B
  -rw-r--r--  1 kuangfangjun root  253 Jan  7 19:33 README.md
  -rw-r--r--  1 kuangfangjun root 149M Jan  7 19:33 embedding.int8.onnx
  -rw-r--r--  1 kuangfangjun root 227M Jan  7 19:34 encoder_adaptor.int8.onnx
  -rw-r--r--  1 kuangfangjun root 573M Jan  7 19:34 llm.int8.onnx
  drwxr-xr-x 27 kuangfangjun root    0 Jan  7 19:28 test_wavs

.. code-block::

  ls -lh sherpa-onnx-funasr-nano-int8-2025-12-30/Qwen3-0.6B/
  total 16M
  -rw-r--r-- 1 kuangfangjun root 1.6M Jan  7 19:34 merges.txt
  -rw-r--r-- 1 kuangfangjun root  11M Jan  7 19:34 tokenizer.json
  -rw-r--r-- 1 kuangfangjun root 2.7M Jan  7 19:34 vocab.json

.. code-block::

  ls -lh sherpa-onnx-funasr-nano-int8-2025-12-30/test_wavs/
  total 9.7M
  -rw-r--r-- 1 kuangfangjun root 6.9K Jan  7 19:33 README.md
  -rw-r--r-- 1 kuangfangjun root 220K Jan  7 19:33 dia_hunan.wav
  -rw-r--r-- 1 kuangfangjun root 253K Jan  7 19:33 dia_minnan.wav
  -rw-r--r-- 1 kuangfangjun root 229K Jan  7 19:33 dia_sh.wav
  -rw-r--r-- 1 kuangfangjun root 297K Jan  7 19:33 dia_yue.wav
  -rw-r--r-- 1 kuangfangjun root 215K Jan  7 19:33 far_2.wav
  -rw-r--r-- 1 kuangfangjun root 682K Jan  7 19:33 far_3.wav
  -rw-r--r-- 1 kuangfangjun root 284K Jan  7 19:33 far_4.wav
  -rw-r--r-- 1 kuangfangjun root 279K Jan  7 19:33 far_5.wav
  -rw-r--r-- 1 kuangfangjun root 254K Jan  7 19:33 ja.wav
  -rw-r--r-- 1 kuangfangjun root 255K Jan  7 19:33 ja_en_codeswitch.wav
  -rw-r--r-- 1 kuangfangjun root 259K Jan  7 19:33 lyrics.wav
  -rw-r--r-- 1 kuangfangjun root 431K Jan  7 19:33 lyrics_2.wav
  -rw-r--r-- 1 kuangfangjun root 546K Jan  7 19:33 lyrics_3.wav
  -rw-r--r-- 1 kuangfangjun root 1.3M Jan  7 19:33 lyrics_en_1.wav
  -rw-r--r-- 1 kuangfangjun root 679K Jan  7 19:33 lyrics_en_2.wav
  -rw-r--r-- 1 kuangfangjun root 1.7M Jan  7 19:33 lyrics_en_3.wav
  -rw-r--r-- 1 kuangfangjun root 331K Jan  7 19:33 noise_en.wav
  -rw-r--r-- 1 kuangfangjun root 267K Jan  7 19:33 rag_biochemistry.wav
  -rw-r--r-- 1 kuangfangjun root 214K Jan  7 19:33 rag_chemistry.wav
  -rw-r--r-- 1 kuangfangjun root 248K Jan  7 19:33 rag_history.wav
  -rw-r--r-- 1 kuangfangjun root 173K Jan  7 19:33 rag_math.wav
  -rw-r--r-- 1 kuangfangjun root 192K Jan  7 19:33 rag_medical.wav
  -rw-r--r-- 1 kuangfangjun root 379K Jan  7 19:33 rag_physics.wav
  -rw-r--r-- 1 kuangfangjun root 224K Jan  7 19:33 vietnamese.wav
