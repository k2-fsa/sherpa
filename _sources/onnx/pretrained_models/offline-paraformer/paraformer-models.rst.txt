Paraformer models
=================

.. hint::

   Please refer to :ref:`install_sherpa_onnx` to install `sherpa-onnx`_
   before you read this section.

.. _sherpa_onnx_offline_paraformer_trilingual_zh_cantonese_en:

csukuangfj/sherpa-onnx-paraformer-trilingual-zh-cantonese-en (Chinese + English + Cantonese 粤语)
-------------------------------------------------------------------------------------------------

.. note::

   This model does not support timestamps. It is a trilingual model, supporting
   both Chinese and English. (支持普通话、``粤语``、河南话、天津话、四川话等方言)

This model is converted from

`<https://www.modelscope.cn/models/dengcunqin/speech_seaco_paraformer_large_asr_nat-zh-cantonese-en-16k-common-vocab11666-pytorch/summary>`_

In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

  cd /path/to/sherpa-onnx
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-trilingual-zh-cantonese-en.tar.bz2

  # For Chinese users
  # wget https://hub.nuaa.cf/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-trilingual-zh-cantonese-en.tar.bz2

  tar xvf sherpa-onnx-paraformer-trilingual-zh-cantonese-en.tar.bz2

Please check that the file sizes of the pre-trained models are correct. See
the file sizes of ``*.onnx`` files below.

.. code-block:: bash

  sherpa-onnx-paraformer-trilingual-zh-cantonese-en$ ls -lh *.onnx

  -rw-r--r-- 1 1001 127 234M Mar 10 02:12 model.int8.onnx
  -rw-r--r-- 1 1001 127 831M Mar 10 02:12 model.onnx

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
    --tokens=./sherpa-onnx-paraformer-trilingual-zh-cantonese-en/tokens.txt \
    --paraformer=./sherpa-onnx-paraformer-trilingual-zh-cantonese-en/model.onnx \
    ./sherpa-onnx-paraformer-trilingual-zh-cantonese-en/test_wavs/1.wav \
    ./sherpa-onnx-paraformer-trilingual-zh-cantonese-en/test_wavs/2.wav \
    ./sherpa-onnx-paraformer-trilingual-zh-cantonese-en/test_wavs/3-sichuan.wav \
    ./sherpa-onnx-paraformer-trilingual-zh-cantonese-en/test_wavs/4-tianjin.wav \
    ./sherpa-onnx-paraformer-trilingual-zh-cantonese-en/test_wavs/5-henan.wav \
    ./sherpa-onnx-paraformer-trilingual-zh-cantonese-en/test_wavs/6-zh-en.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

You should see the following output:

.. literalinclude:: ./code-paraformer/sherpa-onnx-paraformer-trilingual-zh-cantonese-en.txt

int8
^^^^

The following code shows how to use ``int8`` models to decode wave files:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-paraformer-trilingual-zh-cantonese-en/tokens.txt \
    --paraformer=./sherpa-onnx-paraformer-trilingual-zh-cantonese-en/model.int8.onnx \
    ./sherpa-onnx-paraformer-trilingual-zh-cantonese-en/test_wavs/1.wav \
    ./sherpa-onnx-paraformer-trilingual-zh-cantonese-en/test_wavs/2.wav \
    ./sherpa-onnx-paraformer-trilingual-zh-cantonese-en/test_wavs/3-sichuan.wav \
    ./sherpa-onnx-paraformer-trilingual-zh-cantonese-en/test_wavs/4-tianjin.wav \
    ./sherpa-onnx-paraformer-trilingual-zh-cantonese-en/test_wavs/5-henan.wav \
    ./sherpa-onnx-paraformer-trilingual-zh-cantonese-en/test_wavs/6-zh-en.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

You should see the following output:

.. literalinclude:: ./code-paraformer/sherpa-onnx-paraformer-trilingual-zh-cantonese-en-int8.txt

Speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-microphone-offline \
    --tokens=./sherpa-onnx-paraformer-trilingual-zh-cantonese-en/tokens.txt \
    --paraformer=./sherpa-onnx-paraformer-trilingual-zh-cantonese-en/model.int8.onnx

.. _sherpa_onnx_offline_paraformer_en_2024_03_09_english:

csukuangfj/sherpa-onnx-paraformer-en-2024-03-09 (English)
---------------------------------------------------------

.. note::

   This model does not support timestamps. It supports only English.

This model is converted from

`<https://www.modelscope.cn/models/iic/speech_paraformer_asr-en-16k-vocab4199-pytorch/summary>`_

In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

  cd /path/to/sherpa-onnx
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-en-2024-03-09.tar.bz2

  # For Chinese users
  # wget https://hub.nuaa.cf/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-en-2024-03-09.tar.bz2

  tar xvf sherpa-onnx-paraformer-en-2024-03-09.tar.bz2

Please check that the file sizes of the pre-trained models are correct. See
the file sizes of ``*.onnx`` files below.

.. code-block:: bash

  sherpa-onnx-paraformer-en-2024-03-09$ ls -lh *.onnx

  -rw-r--r-- 1 1001 127 220M Mar 10 02:12 model.int8.onnx
  -rw-r--r-- 1 1001 127 817M Mar 10 02:12 model.onnx

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
    --tokens=./sherpa-onnx-paraformer-en-2024-03-09/tokens.txt \
    --paraformer=./sherpa-onnx-paraformer-en-2024-03-09/model.onnx \
    ./sherpa-onnx-paraformer-en-2024-03-09/test_wavs/0.wav \
    ./sherpa-onnx-paraformer-en-2024-03-09/test_wavs/1.wav \
    ./sherpa-onnx-paraformer-en-2024-03-09/test_wavs/8k.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

You should see the following output:

.. literalinclude:: ./code-paraformer/sherpa-onnx-paraformer-en-2024-03-09.txt

int8
^^^^

The following code shows how to use ``int8`` models to decode wave files:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-paraformer-en-2024-03-09/tokens.txt \
    --paraformer=./sherpa-onnx-paraformer-en-2024-03-09/model.int8.onnx \
    ./sherpa-onnx-paraformer-en-2024-03-09/test_wavs/0.wav \
    ./sherpa-onnx-paraformer-en-2024-03-09/test_wavs/1.wav \
    ./sherpa-onnx-paraformer-en-2024-03-09/test_wavs/8k.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

You should see the following output:

.. literalinclude:: ./code-paraformer/sherpa-onnx-paraformer-en-2024-03-09-int8.txt

Speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-microphone-offline \
    --tokens=./sherpa-onnx-paraformer-en-2024-03-09/tokens.txt \
    --paraformer=./sherpa-onnx-paraformer-en-2024-03-09/model.int8.onnx

.. _sherpa_onnx_offline_paraformer_zh_small_2024_03_09_chinese_english:

csukuangfj/sherpa-onnx-paraformer-zh-small-2024-03-09 (Chinese + English)
-------------------------------------------------------------------------

.. note::

   This model does not support timestamps. It is a bilingual model, supporting
   both Chinese and English. (支持普通话、河南话、天津话、四川话等方言)

This model is converted from

`<https://www.modelscope.cn/models/crazyant/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-onnx/summary>`_

In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

  cd /path/to/sherpa-onnx
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-small-2024-03-09.tar.bz2

  # For Chinese users
  wget https://hub.nuaa.cf/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-small-2024-03-09.tar.bz2

  tar xvf sherpa-onnx-paraformer-zh-small-2024-03-09.tar.bz2

Please check that the file sizes of the pre-trained models are correct. See
the file sizes of ``*.onnx`` files below.

.. code-block:: bash

  sherpa-onnx-paraformer-zh-small-2024-03-09$ ls -lh *.onnx

  -rw-r--r-- 1 1001 127 79M Mar 10 00:48 model.int8.onnx

Decode wave files
~~~~~~~~~~~~~~~~~

.. hint::

   It supports decoding only wave files of a single channel with 16-bit
   encoded samples, while the sampling rate does not need to be 16 kHz.

int8
^^^^

The following code shows how to use ``int8`` models to decode wave files:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-paraformer-zh-small-2024-03-09/tokens.txt \
    --paraformer=./sherpa-onnx-paraformer-zh-small-2024-03-09/model.int8.onnx \
    ./sherpa-onnx-paraformer-zh-small-2024-03-09/test_wavs/0.wav \
    ./sherpa-onnx-paraformer-zh-small-2024-03-09/test_wavs/1.wav \
    ./sherpa-onnx-paraformer-zh-small-2024-03-09/test_wavs/8k.wav \
    ./sherpa-onnx-paraformer-zh-small-2024-03-09/test_wavs/2-zh-en.wav \
    ./sherpa-onnx-paraformer-zh-small-2024-03-09/test_wavs/3-sichuan.wav \
    ./sherpa-onnx-paraformer-zh-small-2024-03-09/test_wavs/4-tianjin.wav \
    ./sherpa-onnx-paraformer-zh-small-2024-03-09/test_wavs/5-henan.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

You should see the following output:

.. literalinclude:: ./code-paraformer/sherpa-onnx-paraformer-zh-small-2024-03-09-int8.txt

Speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-microphone-offline \
    --tokens=./sherpa-onnx-paraformer-zh-small-2024-03-09/tokens.txt \
    --paraformer=./sherpa-onnx-paraformer-zh-small-2024-03-09/model.int8.onnx

.. _sherpa_onnx_offline_paraformer_zh_2024_03_09_chinese_english:

csukuangfj/sherpa-onnx-paraformer-zh-2024-03-09 (Chinese + English)
-------------------------------------------------------------------------

.. note::

   This model does not support timestamps. It is a bilingual model, supporting
   both Chinese and English. (支持普通话、河南话、天津话、四川话等方言)

This model is converted from

`<https://www.modelscope.cn/models/crazyant/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-onnx/summary>`_

In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

  cd /path/to/sherpa-onnx
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2024-03-09.tar.bz2

  # For Chinese users
  # wget https://hub.nuaa.cf/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2024-03-09.tar.bz2

  tar xvf sherpa-onnx-paraformer-zh-2024-03-09.tar.bz2

Please check that the file sizes of the pre-trained models are correct. See
the file sizes of ``*.onnx`` files below.

.. code-block:: bash

  sherpa-onnx-paraformer-zh-2024-03-09$ ls -lh *.onnx

  -rw-r--r-- 1 1001 127 217M Mar 10 02:22 model.int8.onnx
  -rw-r--r-- 1 1001 127 785M Mar 10 02:22 model.onnx

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
    --tokens=./sherpa-onnx-paraformer-zh-2024-03-09/tokens.txt \
    --paraformer=./sherpa-onnx-paraformer-zh-2024-03-09/model.onnx \
    ./sherpa-onnx-paraformer-zh-2024-03-09/test_wavs/0.wav \
    ./sherpa-onnx-paraformer-zh-2024-03-09/test_wavs/1.wav \
    ./sherpa-onnx-paraformer-zh-2024-03-09/test_wavs/8k.wav \
    ./sherpa-onnx-paraformer-zh-2024-03-09/test_wavs/2-zh-en.wav \
    ./sherpa-onnx-paraformer-zh-2024-03-09/test_wavs/3-sichuan.wav \
    ./sherpa-onnx-paraformer-zh-2024-03-09/test_wavs/4-tianjin.wav \
    ./sherpa-onnx-paraformer-zh-2024-03-09/test_wavs/5-henan.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

You should see the following output:

.. literalinclude:: ./code-paraformer/sherpa-onnx-paraformer-zh-2024-03-09.txt

int8
^^^^

The following code shows how to use ``int8`` models to decode wave files:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-paraformer-zh-2024-03-09/tokens.txt \
    --paraformer=./sherpa-onnx-paraformer-zh-2024-03-09/model.int8.onnx \
    ./sherpa-onnx-paraformer-zh-2024-03-09/test_wavs/0.wav \
    ./sherpa-onnx-paraformer-zh-2024-03-09/test_wavs/1.wav \
    ./sherpa-onnx-paraformer-zh-2024-03-09/test_wavs/8k.wav \
    ./sherpa-onnx-paraformer-zh-2024-03-09/test_wavs/2-zh-en.wav \
    ./sherpa-onnx-paraformer-zh-2024-03-09/test_wavs/3-sichuan.wav \
    ./sherpa-onnx-paraformer-zh-2024-03-09/test_wavs/4-tianjin.wav \
    ./sherpa-onnx-paraformer-zh-2024-03-09/test_wavs/5-henan.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

You should see the following output:

.. literalinclude:: ./code-paraformer/sherpa-onnx-paraformer-zh-2024-03-09.txt

Speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-microphone-offline \
    --tokens=./sherpa-onnx-paraformer-zh-2024-03-09/tokens.txt \
    --paraformer=./sherpa-onnx-paraformer-zh-2024-03-09/model.int8.onnx


.. _sherpa_onnx_offline_paraformer_zh_2023_03_28_chinese:

csukuangfj/sherpa-onnx-paraformer-zh-2023-03-28 (Chinese + English)
-------------------------------------------------------------------

.. note::

   This model does not support timestamps. It is a bilingual model, supporting
   both Chinese and English. (支持普通话、河南话、天津话、四川话等方言)


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
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2023-03-28.tar.bz2

  # For Chinese users
  # wget https://hub.nuaa.cf/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2023-03-28.tar.bz2

  tar xvf sherpa-onnx-paraformer-zh-2023-03-28.tar.bz2

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
    ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/3-sichuan.wav \
    ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/4-tianjin.wav \
    ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/5-henan.wav \
    ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/6-zh-en.wav \
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
    ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/3-sichuan.wav \
    ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/4-tianjin.wav \
    ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/5-henan.wav \
    ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/6-zh-en.wav \
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

.. _sherpa_onnx_offline_paraformer_zh_2023_09_14_chinese:

csukuangfj/sherpa-onnx-paraformer-zh-2023-09-14 (Chinese + English))
---------------------------------------------------------------------

.. note::

   This model supports timestamps. It is a bilingual model, supporting
   both Chinese and English. (支持普通话、河南话、天津话、四川话等方言)


This model is converted from

`<https://www.modelscope.cn/models/iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx/summary>`_

In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2

  # For Chinese users
  # wget https://hub.nuaa.cf/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2

  tar xvf sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2

Please check that the file sizes of the pre-trained models are correct. See
the file sizes of ``*.onnx`` files below.

.. code-block:: bash

  sherpa-onnx-paraformer-zh-2023-09-14$ ls -lh *.onnx
  -rw-r--r--  1 fangjun  staff   232M Sep 14 13:46 model.int8.onnx

Decode wave files
~~~~~~~~~~~~~~~~~

.. hint::

   It supports decoding only wave files of a single channel with 16-bit
   encoded samples, while the sampling rate does not need to be 16 kHz.

int8
^^^^

The following code shows how to use ``int8`` models to decode wave files:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-paraformer-zh-2023-09-14/tokens.txt \
    --paraformer=./sherpa-onnx-paraformer-zh-2023-09-14/model.int8.onnx \
    --model-type=paraformer \
    ./sherpa-onnx-paraformer-zh-2023-09-14/test_wavs/0.wav \
    ./sherpa-onnx-paraformer-zh-2023-09-14/test_wavs/1.wav \
    ./sherpa-onnx-paraformer-zh-2023-09-14/test_wavs/2.wav \
    ./sherpa-onnx-paraformer-zh-2023-09-14/test_wavs/3-sichuan.wav \
    ./sherpa-onnx-paraformer-zh-2023-09-14/test_wavs/4-tianjin.wav \
    ./sherpa-onnx-paraformer-zh-2023-09-14/test_wavs/5-henan.wav \
    ./sherpa-onnx-paraformer-zh-2023-09-14/test_wavs/6-zh-en.wav \
    ./sherpa-onnx-paraformer-zh-2023-09-14/test_wavs/8k.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

You should see the following output:

.. literalinclude:: ./code-paraformer/sherpa-onnx-paraformer-zh-2023-09-14-int8.txt

Speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-microphone-offline \
    --tokens=./sherpa-onnx-paraformer-zh-2023-09-14/tokens.txt \
    --paraformer=./sherpa-onnx-paraformer-zh-2023-09-14/model.int8.onnx \
    --model-type=paraformer
