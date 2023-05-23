Conformer-transducer-based Models
=================================

.. hint::

   Please refer to :ref:`install_sherpa_onnx` to install `sherpa-onnx`_
   before you read this section.

csukuangfj/sherpa-onnx-conformer-zh-stateless2-2023-05-23 (Chinese)
-------------------------------------------------------------------

This model is converted from

`<https://huggingface.co/luomingshuang/icefall_asr_wenetspeech_pruned_transducer_stateless2>`_

which supports only Chinese as it is trained on the `WenetSpeech`_ corpus.

You can find the training code at

`<https://github.com/k2-fsa/icefall/tree/master/egs/wenetspeech/ASR/pruned_transducer_stateless2>`_

In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

  cd /path/to/sherpa-onnx
  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/sherpa-onnx-conformer-zh-stateless2-2023-05-23
  cd sherpa-onnx-conformer-zh-stateless2-2023-05-23
  git lfs pull --include "*.onnx"

Please check that the file sizes of the pre-trained models are correct. See
the file sizes of ``*.onnx`` files below.

.. code-block:: bash

  sherpa-onnx-conformer-zh-stateless2-2023-05-23 fangjun$ ls -lh *.onnx
  -rw-r--r--  1 fangjun  staff    11M May 23 15:29 decoder-epoch-99-avg-1.int8.onnx
  -rw-r--r--  1 fangjun  staff    12M May 23 15:29 decoder-epoch-99-avg-1.onnx
  -rw-r--r--  1 fangjun  staff   122M May 23 15:30 encoder-epoch-99-avg-1.int8.onnx
  -rw-r--r--  1 fangjun  staff   315M May 23 15:31 encoder-epoch-99-avg-1.onnx
  -rw-r--r--  1 fangjun  staff   2.7M May 23 15:29 joiner-epoch-99-avg-1.int8.onnx
  -rw-r--r--  1 fangjun  staff    11M May 23 15:29 joiner-epoch-99-avg-1.onnx

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
    --tokens=./sherpa-onnx-conformer-zh-stateless2-2023-05-23/tokens.txt \
    --encoder=./sherpa-onnx-conformer-zh-stateless2-2023-05-23/encoder-epoch-99-avg-1.onnx \
    --decoder=./sherpa-onnx-conformer-zh-stateless2-2023-05-23/decoder-epoch-99-avg-1.onnx \
    --joiner=./sherpa-onnx-conformer-zh-stateless2-2023-05-23/joiner-epoch-99-avg-1.onnx \
    ./sherpa-onnx-conformer-zh-stateless2-2023-05-23/test_wavs/0.wav \
    ./sherpa-onnx-conformer-zh-stateless2-2023-05-23/test_wavs/1.wav \
    ./sherpa-onnx-conformer-zh-stateless2-2023-05-23/test_wavs/2.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

You should see the following output:

.. literalinclude:: ./code-conformer/sherpa-onnx-conformer-zh-stateless2-2023-05-23.txt

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

int8
^^^^

The following code shows how to use ``int8`` models to decode wave files:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-conformer-zh-stateless2-2023-05-23/tokens.txt \
    --encoder=./sherpa-onnx-conformer-zh-stateless2-2023-05-23/encoder-epoch-99-avg-1.int8.onnx \
    --decoder=./sherpa-onnx-conformer-zh-stateless2-2023-05-23/decoder-epoch-99-avg-1.onnx \
    --joiner=./sherpa-onnx-conformer-zh-stateless2-2023-05-23/joiner-epoch-99-avg-1.int8.onnx \
    ./sherpa-onnx-conformer-zh-stateless2-2023-05-23/test_wavs/0.wav \
    ./sherpa-onnx-conformer-zh-stateless2-2023-05-23/test_wavs/1.wav \
    ./sherpa-onnx-conformer-zh-stateless2-2023-05-23/test_wavs/2.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

.. caution::

   We did not use ``int8`` for the decoder model above.

You should see the following output:

.. literalinclude:: ./code-conformer/sherpa-onnx-conformer-zh-stateless2-2023-05-23.int8.txt

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

Speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-microphone-offline \
    --tokens=./sherpa-onnx-conformer-zh-stateless2-2023-05-23/tokens.txt \
    --encoder=./sherpa-onnx-conformer-zh-stateless2-2023-05-23/encoder-epoch-99-avg-1.onnx \
    --decoder=./sherpa-onnx-conformer-zh-stateless2-2023-05-23/decoder-epoch-99-avg-1.onnx \
    --joiner=./sherpa-onnx-conformer-zh-stateless2-2023-05-23/joiner-epoch-99-avg-1.onnx

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.


csukuangfj/sherpa-onnx-conformer-zh-2023-05-23 (Chinese)
--------------------------------------------------------

This model is converted from

`<https://huggingface.co/luomingshuang/icefall_asr_wenetspeech_pruned_transducer_stateless5_offline>`_

which supports only Chinese as it is trained on the `WenetSpeech`_ corpus.

You can find the training code at

`<https://github.com/k2-fsa/icefall/tree/master/egs/wenetspeech/ASR/pruned_transducer_stateless5>`_

In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

  cd /path/to/sherpa-onnx
  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/sherpa-onnx-conformer-zh-2023-05-23
  cd sherpa-onnx-conformer-zh-2023-05-23
  git lfs pull --include "*.onnx"

Please check that the file sizes of the pre-trained models are correct. See
the file sizes of ``*.onnx`` files below.

.. code-block:: bash

  sherpa-onnx-conformer-zh-2023-05-23 fangjun$ ls -lh *.onnx
  -rw-r--r--  1 fangjun  staff    11M May 23 13:45 decoder-epoch-99-avg-1.int8.onnx
  -rw-r--r--  1 fangjun  staff    12M May 23 13:45 decoder-epoch-99-avg-1.onnx
  -rw-r--r--  1 fangjun  staff   129M May 23 13:47 encoder-epoch-99-avg-1.int8.onnx
  -rw-r--r--  1 fangjun  staff   345M May 23 13:48 encoder-epoch-99-avg-1.onnx
  -rw-r--r--  1 fangjun  staff   2.7M May 23 13:45 joiner-epoch-99-avg-1.int8.onnx
  -rw-r--r--  1 fangjun  staff    11M May 23 13:45 joiner-epoch-99-avg-1.onnx

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
    --tokens=./sherpa-onnx-conformer-zh-2023-05-23/tokens.txt \
    --encoder=./sherpa-onnx-conformer-zh-2023-05-23/encoder-epoch-99-avg-1.onnx \
    --decoder=./sherpa-onnx-conformer-zh-2023-05-23/decoder-epoch-99-avg-1.onnx \
    --joiner=./sherpa-onnx-conformer-zh-2023-05-23/joiner-epoch-99-avg-1.onnx \
    ./sherpa-onnx-conformer-zh-2023-05-23/test_wavs/0.wav \
    ./sherpa-onnx-conformer-zh-2023-05-23/test_wavs/1.wav \
    ./sherpa-onnx-conformer-zh-2023-05-23/test_wavs/2.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

You should see the following output:

.. literalinclude:: ./code-conformer/sherpa-onnx-conformer-zh-2023-05-23.txt

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

int8
^^^^

The following code shows how to use ``int8`` models to decode wave files:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-conformer-zh-2023-05-23/tokens.txt \
    --encoder=./sherpa-onnx-conformer-zh-2023-05-23/encoder-epoch-99-avg-1.int8.onnx \
    --decoder=./sherpa-onnx-conformer-zh-2023-05-23/decoder-epoch-99-avg-1.onnx \
    --joiner=./sherpa-onnx-conformer-zh-2023-05-23/joiner-epoch-99-avg-1.int8.onnx \
    ./sherpa-onnx-conformer-zh-2023-05-23/test_wavs/0.wav \
    ./sherpa-onnx-conformer-zh-2023-05-23/test_wavs/1.wav \
    ./sherpa-onnx-conformer-zh-2023-05-23/test_wavs/2.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

.. caution::

   We did not use ``int8`` for the decoder model above.

You should see the following output:

.. literalinclude:: ./code-conformer/sherpa-onnx-conformer-zh-2023-05-23.int8.txt

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

Speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-microphone-offline \
    --tokens=./sherpa-onnx-conformer-zh-2023-05-23/tokens.txt \
    --encoder=./sherpa-onnx-conformer-zh-2023-05-23/encoder-epoch-99-avg-1.onnx \
    --decoder=./sherpa-onnx-conformer-zh-2023-05-23/decoder-epoch-99-avg-1.onnx \
    --joiner=./sherpa-onnx-conformer-zh-2023-05-23/joiner-epoch-99-avg-1.onnx

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

csukuangfj/sherpa-onnx-conformer-en-2023-03-18 (English)
--------------------------------------------------------

This model is converted from

`<https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13>`_

which supports only English as it is trained on the `LibriSpeech`_ corpus.

You can find the training code at

`<https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/pruned_transducer_stateless3>`_

In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/sherpa-onnx-conformer-en-2023-03-18
  cd sherpa-onnx-conformer-en-2023-03-18
  git lfs pull --include "*.onnx"

Please check that the file sizes of the pre-trained models are correct. See
the file sizes of ``*.onnx`` files below.

.. code-block:: bash

  sherpa-onnx-en-2023-03-18$ ls -lh *.onnx
  -rw-r--r-- 1 kuangfangjun root  1.3M Apr  1 07:02 decoder-epoch-99-avg-1.int8.onnx
  -rw-r--r-- 1 kuangfangjun root  2.0M Apr  1 07:02 decoder-epoch-99-avg-1.onnx
  -rw-r--r-- 1 kuangfangjun root  122M Apr  1 07:02 encoder-epoch-99-avg-1.int8.onnx
  -rw-r--r-- 1 kuangfangjun root  315M Apr  1 07:02 encoder-epoch-99-avg-1.onnx
  -rw-r--r-- 1 kuangfangjun root  254K Apr  1 07:02 joiner-epoch-99-avg-1.int8.onnx
  -rw-r--r-- 1 kuangfangjun root 1003K Apr  1 07:02 joiner-epoch-99-avg-1.onnx

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
    --tokens=./sherpa-onnx-conformer-en-2023-03-18/tokens.txt \
    --encoder=./sherpa-onnx-conformer-en-2023-03-18/encoder-epoch-99-avg-1.onnx \
    --decoder=./sherpa-onnx-conformer-en-2023-03-18/decoder-epoch-99-avg-1.onnx \
    --joiner=./sherpa-onnx-conformer-en-2023-03-18/joiner-epoch-99-avg-1.onnx \
    ./sherpa-onnx-conformer-en-2023-03-18/test_wavs/0.wav \
    ./sherpa-onnx-conformer-en-2023-03-18/test_wavs/1.wav \
    ./sherpa-onnx-conformer-en-2023-03-18/test_wavs/8k.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

You should see the following output:

.. literalinclude:: ./code-conformer/sherpa-onnx-conformer-en-2023-03-18.txt

int8
^^^^

The following code shows how to use ``int8`` models to decode wave files:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-conformer-en-2023-03-18/tokens.txt \
    --encoder=./sherpa-onnx-conformer-en-2023-03-18/encoder-epoch-99-avg-1.int8.onnx \
    --decoder=./sherpa-onnx-conformer-en-2023-03-18/decoder-epoch-99-avg-1.int8.onnx \
    --joiner=./sherpa-onnx-conformer-en-2023-03-18/joiner-epoch-99-avg-1.int8.onnx \
    ./sherpa-onnx-conformer-en-2023-03-18/test_wavs/0.wav \
    ./sherpa-onnx-conformer-en-2023-03-18/test_wavs/1.wav \
    ./sherpa-onnx-conformer-en-2023-03-18/test_wavs/8k.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

You should see the following output:

.. literalinclude:: ./code-conformer/sherpa-onnx-conformer-en-2023-03-18-int8.txt

Speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-microphone-offline \
    --tokens=./sherpa-onnx-conformer-en-2023-03-18/tokens.txt \
    --encoder=./sherpa-onnx-conformer-en-2023-03-18/encoder-epoch-99-avg-1.onnx \
    --decoder=./sherpa-onnx-conformer-en-2023-03-18/decoder-epoch-99-avg-1.onnx \
    --joiner=./sherpa-onnx-conformer-en-2023-03-18/joiner-epoch-99-avg-1.onnx
