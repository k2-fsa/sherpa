Pre-trained Models
==================

This page describes how to download pre-trained `SenseVoice`_ models.


sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17
--------------------------------------------------

This model is converted from `<https://www.modelscope.cn/models/iic/SenseVoiceSmall>`_
using the script `export-onnx.py <https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/sense-voice/export-onnx.py>`_.

It supports the following 5 languages:

  - Chinese (Mandarin, 普通话)
  - Cantonese (粤语, 广东话)
  - English
  - Japanese
  - Korean

In the following, we describe how to download it.

Download
^^^^^^^^

Please use the following commands to download it::

  cd /path/to/sherpa-onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2

  # For Chinese users, you can use the following mirror
  # wget https://hub.nuaa.cf/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2

  tar xvf sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
  rm sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2

After downloading, you should find the following files::

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

Decode a file with model.onnx
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Without inverse text normalization
::::::::::::::::::::::::::::::::::

To decode a file without inverse text normalization, please use:

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt \
    --sense-voice-model=/Users/fangjun/open-source/sherpa-onnx/scripts/sense-voice/model.onnx \
    --num-threads=1 \
    --debug=0 \
    ./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/test_wavs/zh.wav \
    ./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/test_wavs/en.wav

You should see the following output:

.. literalinclude:: ./code/2024-07-17.txt

With inverse text normalization
:::::::::::::::::::::::::::::::

To decode a file with inverse text normalization, please use:

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt \
    --sense-voice-model=/Users/fangjun/open-source/sherpa-onnx/scripts/sense-voice/model.onnx \
    --num-threads=1 \
    --sense-voice-use-itn=1 \
    --debug=0 \
    ./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/test_wavs/zh.wav \
    ./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/test_wavs/en.wav

You should see the following output:

.. literalinclude:: ./code/2024-07-17-itn.txt

.. hint::

   When inverse text normalziation is enabled, the results also
   punctuations.

Specify a language
::::::::::::::::::

If you don't provide a language when decoding, it uses ``auto``.

To specify the language when decoding, please use:

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt \
    --sense-voice-model=/Users/fangjun/open-source/sherpa-onnx/scripts/sense-voice/model.onnx \
    --num-threads=1 \
    --sense-voice-language=zh \
    --debug=0 \
    ./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/test_wavs/zh.wav

You should see the following output:

.. literalinclude:: ./code/2024-07-17-lang.txt

.. hint::

   Valid values for ``--sense-voice-language`` are ``auto``, ``zh``, ``en``, ``ko``, ``ja``, and ``yue``.
   where ``zh`` is for Chinese, ``en`` for English, ``ko`` for Korean, ``ja`` for Japanese, and
   ``yue`` for ``Cantonese``.
