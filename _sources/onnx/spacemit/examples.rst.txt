.. _spacemit-provider-examples:

Examples
========

This page collects starter examples for running `sherpa-onnx`_ with the
SpacemiT execution provider.

.. note::

   The first inference run is usually used for initialization, so its timing
   is not representative. In real usage, you can run a first inference with
   empty data and ignore that timing result.

Prepare models
-----------

You can use `xslim` to quantize your ONNX models to `dynamic int8` format or `static int8` format, which is optimized for SpacemiT. For example, you can quantize a VITS TTS model with:

.. code-block:: bash

   # You can install xslim with pip, or build it from source. For building from source, please refer to
   # https://github.com/spacemit-com/xslim
   # pip install xslim

   python3 -m xslim -i ./model.onnx -o ./model.dynq.onnx --dynq


Offline TTS
-----------

.. code-block:: bash

   ${SHERPA_ONNX_INSTALL_DIR}/bin/sherpa-onnx-offline-tts \
      --provider=spacemit \
      --vits-model=./en_US-lessac-medium.dynq.onnx \
      --vits-data-dir=./espeak-ng-data \
      --vits-tokens=./tokens.txt \
      --output-filename=./liliana-piper-en_US-lessac-medium.wav \
      'liliana, the most beautiful and lovely assistant of our team!'


Offline ASR
-----------

The local SpacemiT test directory also contains an offline ASR example based on
SenseVoice:

.. code-block:: bash

   ${SHERPA_ONNX_INSTALL_DIR}/bin/sherpa-onnx-offline \
     --provider=spacemit \
     --tokens=./tokens.txt \
     --sense-voice-model=./model.dynq.onnx \
     --num-threads=4 \
     ./test_wavs/zh.wav \
     ./test_wavs/en.wav \
     ./test_wavs/ja.wav \
     ./test_wavs/ko.wav

You should see the following output::

   Creating recognizer ...
   ........./sherpa-onnx/sherpa-onnx/csrc/session.cc:SpiltProviderAndConfig:63 Provider string: spacemit
   ........./sherpa-onnx/sherpa-onnx/csrc/session.cc:GetSessionOptionsImpl:337 Use SpacemiT Execution Provider
   ........./sherpa-onnx/sherpa-onnx/csrc/session.cc:GetSessionOptionsImpl:347 Set IntraOpNumThreads to 1
   ........./sherpa-onnx/sherpa-onnx/csrc/session.cc:GetSessionOptionsImpl:349 Set InterOpNumThreads to 1
   ........./sherpa-onnx/sherpa-onnx/csrc/session.cc:GetSessionOptionsImpl:354 Set SPACEMIT_EP_INTRA_THREAD_NUM to 4
   recognizer created in 4.014 s
   Started
   Done!

   ./test_wavs/zh.wav
   {"lang": "<|zh|>", "emotion": "<|NEUTRAL|>", "event": "<|Speech|>", "text": "开饭时间早上九点至下午五点", "timestamps": [0.72, 0.96, 1.26, 1.44, 1.92, 2.10, 2.58, 2.82, 3.30, 3.90, 4.20, 4.56, 4.74], "durations": [], "tokens":["开", "饭", "时", "间", "早", "上", "九", "点", "至", "下", "午", "五", "点"], "ys_log_probs": [], "words": []}
   ----
   ./test_wavs/en.wav
   {"lang": "<|en|>", "emotion": "<|NEUTRAL|>", "event": "<|Speech|>", "text": "the tribal chieftain called for the boy and presented him with fifty pieces of gold", "timestamps": [0.90, 1.26, 1.56, 1.80, 2.16, 2.46, 2.76, 2.94, 3.12, 3.60, 3.96, 4.50, 4.74, 5.10, 5.46, 5.88, 6.18], "durations": [], "tokens":["the", " tri", "bal", " chief", "tain", " called", " for", " the", " boy", " and", " presented", " him", " with", " fifty", " pieces", " of", " gold"], "ys_log_probs": [], "words": []}
   ----
   ./test_wavs/ja.wav
   {"lang": "<|ja|>", "emotion": "<|NEUTRAL|>", "event": "<|Speech|>", "text": "うちの中学は弁当制で持っていけない場合は50円の学校販売のパンを買う", "timestamps": [0.42, 0.60, 0.72, 0.90, 1.08, 1.26, 1.44, 1.62, 1.80, 2.04, 2.46, 2.52, 2.64, 2.76, 2.88, 3.00, 3.12, 3.24, 3.36, 3.48, 3.78, 3.96, 4.20, 4.38, 4.56, 4.68, 4.92, 5.10, 5.28, 5.40, 5.52, 5.70, 5.82, 6.00], "durations": [], "tokens":["う", "ち", "の", "中", "学", "は", "弁", "当", "制", "で", "持", "っ", "て", "い", "け", "な", "い", "場", "合", "は", "5", "0", "円", "の", "学", "校", "販", "売", "の", "パ", "ン", "を", "買", "う"], "ys_log_probs": [], "words": []}
   ----
   ./test_wavs/ko.wav
   {"lang": "<|ko|>", "emotion": "<|NEUTRAL|>", "event": "<|Speech|>", "text": "조 금만 생각 을 하 면서 살 면 훨씬 편할 거야", "timestamps": [0.78, 0.96, 1.14, 1.32, 1.56, 1.62, 1.80, 1.86, 1.98, 2.22, 2.40, 2.76, 3.06, 3.30, 3.42, 3.54], "durations": [], "tokens":["조", " 금", "만", " 생각", " ", "을", " 하", " ", "면서", " 살", " 면", " 훨씬", " 편", "할", " 거", "야"], "ys_log_probs": [], "words": []}
   ----
   num threads: 4
   decoding method: greedy_search
   Elapsed seconds: 6.831 s
   Real time factor (RTF): 6.831 / 24.552 = 0.278
