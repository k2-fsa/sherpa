Export SenseVoice to sherpa-onnx
================================

This page describes how to export `SenseVoice`_ to onnx so that you can use
it with `sherpa-onnx`_.


The code
--------

Please refer to `export-onnx.py <https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/sense-voice/export-onnx.py>`_

The entry point is `run.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/sense-voice/run.sh>`_

After executing `run.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/sense-voice/run.sh>`_, you should get
the following files

  - ``model.onnx``, the float32 onnx model
  - ``model.int8.onnx``, the 8-bit quantized model
  - ``tokens.txt``, for converting integer token IDs to strings
  - ``test_wavs/zh.wav``, test wave for Chinese
  - ``test_wavs/en.wav``, test wave for English
  - ``test_wavs/ko.wav``, test wave for Korean
  - ``test_wavs/ja.wav``, test wave for Japanese
  - ``test_wavs/yue.wav``, test wave for Cantonese

Test the exported model
-----------------------

You can use `test.py <https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/sense-voice/test.py>`_
to test the exported model.

Note that `test.py <https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/sense-voice/test.py>`_
does not depend on `sherpa-onnx`_. It uses onnxruntime Python API.

Where to find exported models
------------------------------

You can find the exported `SenseVoice`_ models at

  `<https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models>`_

The following is an example about how to download an exported `SenseVoice`_ model::

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2

  # For Chinese users, you can use the following mirror
  # wget https://hub.nuaa.cf/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2

  tar xvf sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
  rm sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2


To view the downloaded files, please use::

  ls -lh sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17

  total 1.1G
  -rw-r--r-- 1 runner docker   71 Jul 18 13:06 LICENSE
  -rw-r--r-- 1 runner docker  104 Jul 18 13:06 README.md
  -rwxr-xr-x 1 runner docker 5.8K Jul 18 13:06 export-onnx.py
  -rw-r--r-- 1 runner docker 229M Jul 18 13:06 model.int8.onnx
  -rw-r--r-- 1 runner docker 895M Jul 18 13:06 model.onnx
  drwxr-xr-x 2 runner docker 4.0K Jul 18 13:06 test_wavs
  -rw-r--r-- 1 runner docker 309K Jul 18 13:06 tokens.txt

  ls -lh sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/test_wavs

  total 940K
  -rw-r--r-- 1 runner docker 224K Jul 18 13:06 en.wav
  -rw-r--r-- 1 runner docker 226K Jul 18 13:06 ja.wav
  -rw-r--r-- 1 runner docker 145K Jul 18 13:06 ko.wav
  -rw-r--r-- 1 runner docker 161K Jul 18 13:06 yue.wav
  -rw-r--r-- 1 runner docker 175K Jul 18 13:06 zh.wav
