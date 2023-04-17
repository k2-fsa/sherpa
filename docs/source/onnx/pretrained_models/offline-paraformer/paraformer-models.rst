Paraformer models
=================

.. hint::

   Please refer to :ref:`install_sherpa_onnx` to install `sherpa-onnx`_
   before you read this section.

.. _sherpa_onnx_offline_paraformer_zh_2023_03_28_chinese:

csukuangfj/sherpa-onnx-paraformer-zh-2023-03-28 (Chinese)
---------------------------------------------------------


This model is converted from

`<https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch>`_

The code for converting can be found at

`<https://huggingface.co/csukuangfj/paraformer-onnxruntime-python-example/tree/main>`_


In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/sherpa-onnx-paraformer-zh-2023-03-28
  cd sherpa-onnx-paraformer-zh-2023-03-28
  git lfs pull --include "*.onnx"

Please check that the file sizes of the pre-trained models are correct. See
the file sizes of ``*.onnx`` files below.

.. code-block:: bash

  sherpa-onnx-paraformer-zh-2023-03-28$ ls -lh *.onnx
  -rw-r--r-- 1 kuangfangjun root 214M Apr  1 07:28 model.int8.onnx
  -rw-r--r-- 1 kuangfangjun root 824M Apr  1 07:28 model.onnx

Decode wave files
~~~~~~~~~~~~~~~~~

.. hint::

   It supports decoding only wave files of a single channel with 16-bit
   encoded samples, while the sampling rate does not need to be 16 kHz.

fp32
^^^^

The following code shows how to use ``fp32`` models to decode wave files:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-paraformer-zh-2023-03-28/tokens.txt \
    --paraformer=./sherpa-onnx-paraformer-zh-2023-03-28/model.onnx \
    ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/0.wav \
    ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/1.wav \
    ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/2.wav \
    ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/8k.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

You should see the following output:

.. literalinclude:: ./code-paraformer/sherpa-onnx-paraformer-zh-2023-03-28.txt

int8
^^^^

The following code shows how to use ``int8`` models to decode wave files:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-paraformer-zh-2023-03-28/tokens.txt \
    --paraformer=./sherpa-onnx-paraformer-zh-2023-03-28/model.int8.onnx \
    ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/0.wav \
    ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/1.wav \
    ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/2.wav \
    ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/8k.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

You should see the following output:

.. literalinclude:: ./code-paraformer/sherpa-onnx-paraformer-zh-2023-03-28-int8.txt

Speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-microphone-offline \
    --tokens=./sherpa-onnx-paraformer-zh-2023-03-28/tokens.txt \
    --paraformer=./sherpa-onnx-paraformer-zh-2023-03-28/model.int8.onnx
