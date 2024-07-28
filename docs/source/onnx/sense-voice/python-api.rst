Python API for SenseVoice
=========================

This page describes how to use the Python API for `SenseVoice`_.

Please refer to :ref:`install_sherpa_onnx_python` for how to install the Python package
of `sherpa-onnx`_.

The following is a quick way to do that::

  pip install sherpa-onnx

After installing the Python package, you can download the Python example code and run it with
the following commands::

  cd /tmp
  git clone http://github.com/k2-fsa/sherpa-onnx
  cd sherpa-onnx

  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
  tar xvf sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
  rm sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2

  python3 ./python-api-examples/offline-sense-voice-ctc-decode-files.py

You should see something like below::

  ./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/test_wavs/zh.wav
  {"text": "开饭时间早上9点至下午5点。", "timestamps": [0.72, 0.96, 1.26, 1.44, 1.92, 2.10, 2.58, 2.82, 3.30, 3.90, 4.20, 4.56, 4.74, 5.46], "tokens":["开", "饭", "时", "间", "早", "上", "9", "点", "至", "下", "午", "5", "点", "。"], "words": []}
  (py38) fangjuns-MacBook-Pro:sherpa-onnx fangjun$ #python3 ./python-api-examples/offline-sense-voice-ctc-decode-files.py

You can find ``offline-sense-voice-ctc-decode-files.py`` at the following address:

  `<https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/offline-sense-voice-ctc-decode-files.py>`_
