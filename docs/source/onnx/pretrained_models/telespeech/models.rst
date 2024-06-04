Models
======

sherpa-onnx-telespeech-ctc-int8-zh-2024-06-04 (支持非常多种方言)
-------------------------------------------------------------------------------------

.. hint::

   这个模型支持很多种方言。

Please use the following commands to download it.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-telespeech-ctc-int8-zh-2024-06-04.tar.bz2

  # For Chinese users, please use the following mirror
  # wget https://hub.nuaa.cf/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-telespeech-ctc-int8-zh-2024-06-04.tar.bz2

  tar xvf sherpa-onnx-telespeech-ctc-int8-zh-2024-06-04.tar.bz2
  rm sherpa-onnx-telespeech-ctc-int8-zh-2024-06-04.tar.bz2

Please check that the file sizes of the pre-trained models are correct. See
the file sizes of ``*.onnx`` files below.

.. code-block:: bash

  $ ls -lh *.onnx
  -rw-r--r--  1 fangjun  staff   325M Jun  4 11:56 model.int8.onnx

Decode wave files
~~~~~~~~~~~~~~~~~

.. hint::

   It supports decoding only wave files of a single channel with 16-bit
   encoded samples, while the sampling rate does not need to be 16 kHz.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-offline \
    --telespeech-ctc=./sherpa-onnx-telespeech-ctc-int8-zh-2024-06-04/model.int8.onnx \
    --tokens=./sherpa-onnx-telespeech-ctc-int8-zh-2024-06-04/tokens.txt \
    --model-type=telespeech_ctc \
    --num-threads=1 \
    ./sherpa-onnx-telespeech-ctc-int8-zh-2024-06-04/test_wavs/3-sichuan.wav \
    ./sherpa-onnx-telespeech-ctc-int8-zh-2024-06-04/test_wavs/4-tianjin.wav \
    ./sherpa-onnx-telespeech-ctc-int8-zh-2024-06-04/test_wavs/5-henan.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

You should see the following output:

.. literalinclude:: ./code/sherpa-onnx-telespeech-ctc-int8-zh-2024-06-04.txt

.. note::

   The ``feature_dim=80`` is incorrect in the above logs. The actual value is 40.

.. hint::

   There is also a float32 model. Please see `<https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-telespeech-ctc-zh-2024-06-04.tar.bz2>`_
