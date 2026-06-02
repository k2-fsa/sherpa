Pre-trained Models
==================

This page describes how to download pre-trained `Dolphin`_ models.

sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02
-------------------------------------------------------

This model is converted from `<https://huggingface.co/DataoceanAI/dolphin-base>`_

In the following, we describe how to download it.

Download
^^^^^^^^

Please use the following commands to download it::

  cd /path/to/sherpa-onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02.tar.bz2
  tar xvf sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02.tar.bz2
  rm sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02.tar.bz2

After downloading, you should find the following files::

  ls -lh sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02

  total 100M
  -rw-r--r-- 1 501 staff  99M Apr  2 10:19 model.int8.onnx
  -rw-r--r-- 1 501 staff  141 Apr  2 10:19 README.md
  drwxr-xr-x 2 501 staff 4.0K Apr  2 10:19 test_wavs
  -rw-r--r-- 1 501 staff 493K Apr  2 10:19 tokens.txt

Decode a file
^^^^^^^^^^^^^

Please use the following command to decode a wave file:

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02/tokens.txt \
    --dolphin-model=./sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02/model.int8.onnx \
    --num-threads=1 \
    ./sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02/test_wavs/0.wav

You should see the following output:

.. literalinclude:: ./code/base-int8-2025-04-02.txt

sherpa-onnx-dolphin-base-ctc-multi-lang-2025-04-02
--------------------------------------------------

This model is converted from `<https://huggingface.co/DataoceanAI/dolphin-base>`_

In the following, we describe how to download it.

Download
^^^^^^^^

Please use the following commands to download it::

  cd /path/to/sherpa-onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-dolphin-base-ctc-multi-lang-2025-04-02.tar.bz2
  tar xvf sherpa-onnx-dolphin-base-ctc-multi-lang-2025-04-02.tar.bz2
  rm sherpa-onnx-dolphin-base-ctc-multi-lang-2025-04-02.tar.bz2

After downloading, you should find the following files::

  ls -lh sherpa-onnx-dolphin-base-ctc-multi-lang-2025-04-02

  total 303M
  -rw-r--r-- 1 501 staff 303M Apr  2 10:19 model.onnx
  -rw-r--r-- 1 501 staff  142 Apr  2 10:19 README.md
  drwxr-xr-x 2 501 staff 4.0K Apr  2 10:19 test_wavs
  -rw-r--r-- 1 501 staff 493K Apr  2 10:19 tokens.txt

Decode a file
^^^^^^^^^^^^^

Please use the following command to decode a wave file:

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-dolphin-base-ctc-multi-lang-2025-04-02/tokens.txt \
    --dolphin-model=./sherpa-onnx-dolphin-base-ctc-multi-lang-2025-04-02/model.onnx \
    --num-threads=1 \
    ./sherpa-onnx-dolphin-base-ctc-multi-lang-2025-04-02/test_wavs/0.wav

You should see the following output:

.. literalinclude:: ./code/base-2025-04-02.txt

sherpa-onnx-dolphin-small-ctc-multi-lang-int8-2025-04-02
--------------------------------------------------------

This model is converted from `<https://huggingface.co/DataoceanAI/dolphin-small>`_

In the following, we describe how to download it.

Download
^^^^^^^^

Please use the following commands to download it::

  cd /path/to/sherpa-onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-dolphin-small-ctc-multi-lang-int8-2025-04-02.tar.bz2
  tar xvf sherpa-onnx-dolphin-small-ctc-multi-lang-int8-2025-04-02.tar.bz2
  rm sherpa-onnx-dolphin-small-ctc-multi-lang-int8-2025-04-02.tar.bz2

After downloading, you should find the following files::

  ls -lh sherpa-onnx-dolphin-small-ctc-multi-lang-int8-2025-04-02

  total 239M
  -rw-r--r-- 1 501 staff 239M Apr  2 10:20 model.int8.onnx
  -rw-r--r-- 1 501 staff  141 Apr  2 10:19 README.md
  drwxr-xr-x 2 501 staff 4.0K Apr  2 10:19 test_wavs
  -rw-r--r-- 1 501 staff 493K Apr  2 10:19 tokens.txt

Decode a file
^^^^^^^^^^^^^

Please use the following command to decode a wave file:

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-dolphin-small-ctc-multi-lang-int8-2025-04-02/tokens.txt \
    --dolphin-model=./sherpa-onnx-dolphin-small-ctc-multi-lang-int8-2025-04-02/model.int8.onnx \
    --num-threads=1 \
    ./sherpa-onnx-dolphin-small-ctc-multi-lang-int8-2025-04-02/test_wavs/0.wav

You should see the following output:

.. literalinclude:: ./code/small-int8-2025-04-02.txt

sherpa-onnx-dolphin-small-ctc-multi-lang-2025-04-02
---------------------------------------------------

This model is converted from `<https://huggingface.co/DataoceanAI/dolphin-small>`_

In the following, we describe how to download it.

Download
^^^^^^^^

Please use the following commands to download it::

  cd /path/to/sherpa-onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-dolphin-small-ctc-multi-lang-2025-04-02.tar.bz2
  tar xvf sherpa-onnx-dolphin-small-ctc-multi-lang-2025-04-02.tar.bz2
  rm sherpa-onnx-dolphin-small-ctc-multi-lang-2025-04-02.tar.bz2

After downloading, you should find the following files::

  ls -lh sherpa-onnx-dolphin-small-ctc-multi-lang-2025-04-02

  total 784M
  -rw-r--r-- 1 501 staff 783M Apr  2 10:20 model.onnx
  -rw-r--r-- 1 501 staff  141 Apr  2 10:20 README.md
  drwxr-xr-x 2 501 staff 4.0K Apr  2 10:20 test_wavs
  -rw-r--r-- 1 501 staff 493K Apr  2 10:20 tokens.txt

Decode a file
^^^^^^^^^^^^^

Please use the following command to decode a wave file:

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-dolphin-small-ctc-multi-lang-2025-04-02/tokens.txt \
    --dolphin-model=./sherpa-onnx-dolphin-small-ctc-multi-lang-2025-04-02/model.onnx \
    --num-threads=1 \
    ./sherpa-onnx-dolphin-small-ctc-multi-lang-2025-04-02/test_wavs/0.wav

You should see the following output:

.. literalinclude:: ./code/small-2025-04-02.txt
