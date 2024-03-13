.. _sherpa_onnx_offline_zipformer_transducer_models:

Zipformer-transducer-based Models
=================================

.. hint::

   Please refer to :ref:`install_sherpa_onnx` to install `sherpa-onnx`
   before you read this section.

sherpa-onnx-zipformer-cantonese-2024-03-13 (Cantonese, 粤语)
------------------------------------------------------------

Training code for this model can be found at
`<https://github.com/k2-fsa/icefall/pull/1537>`_.
It supports only Cantonese since it is trained on a `Canatonese`_ dataset.
The paper for the dataset can be found at `<https://arxiv.org/pdf/2201.02419.pdf>`_.

In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-cantonese-2024-03-13.tar.bz2
   tar xf sherpa-onnx-zipformer-cantonese-2024-03-13.tar.bz2
   ls -lh sherpa-onnx-zipformer-cantonese-2024-03-13

You should see the following output:

.. code-block:: bash

  total 340M
  -rw-r--r-- 1 1001 127 2.7M Mar 13 09:06 decoder-epoch-45-avg-35.int8.onnx
  -rw-r--r-- 1 1001 127  11M Mar 13 09:06 decoder-epoch-45-avg-35.onnx
  -rw-r--r-- 1 1001 127  67M Mar 13 09:06 encoder-epoch-45-avg-35.int8.onnx
  -rw-r--r-- 1 1001 127 248M Mar 13 09:06 encoder-epoch-45-avg-35.onnx
  -rw-r--r-- 1 1001 127 2.4M Mar 13 09:06 joiner-epoch-45-avg-35.int8.onnx
  -rw-r--r-- 1 1001 127 9.5M Mar 13 09:06 joiner-epoch-45-avg-35.onnx
  drwxr-xr-x 2 1001 127 4.0K Mar 13 09:06 test_wavs
  -rw-r--r-- 1 1001 127  42K Mar 13 09:06 tokens.txt

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
    --tokens=./sherpa-onnx-zipformer-cantonese-2024-03-13/tokens.txt \
    --encoder=./sherpa-onnx-zipformer-cantonese-2024-03-13/encoder-epoch-45-avg-35.onnx \
    --decoder=./sherpa-onnx-zipformer-cantonese-2024-03-13/decoder-epoch-45-avg-35.onnx \
    --joiner=./sherpa-onnx-zipformer-cantonese-2024-03-13/joiner-epoch-45-avg-35.onnx \
    ./sherpa-onnx-zipformer-cantonese-2024-03-13/test_wavs/test_wavs_1.wav \
    ./sherpa-onnx-zipformer-cantonese-2024-03-13/test_wavs/test_wavs_2.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

You should see the following output:

.. literalinclude:: ./code-zipformer/sherpa-onnx-zipformer-cantonese-2024-03-13.txt

int8
^^^^

The following code shows how to use ``int8`` models to decode wave files:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-zipformer-cantonese-2024-03-13/tokens.txt \
    --encoder=./sherpa-onnx-zipformer-cantonese-2024-03-13/encoder-epoch-45-avg-35.int8.onnx \
    --decoder=./sherpa-onnx-zipformer-cantonese-2024-03-13/decoder-epoch-45-avg-35.onnx \
    --joiner=./sherpa-onnx-zipformer-cantonese-2024-03-13/joiner-epoch-45-avg-35.int8.onnx \
    ./sherpa-onnx-zipformer-cantonese-2024-03-13/test_wavs/test_wavs_1.wav \
    ./sherpa-onnx-zipformer-cantonese-2024-03-13/test_wavs/test_wavs_2.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

You should see the following output:

.. literalinclude:: ./code-zipformer/sherpa-onnx-zipformer-cantonese-2024-03-13-int8.txt

Speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-microphone-offline \
    --tokens=./sherpa-onnx-zipformer-cantonese-2024-03-13/tokens.txt \
    --encoder=./sherpa-onnx-zipformer-cantonese-2024-03-13/encoder-epoch-45-avg-35.int8.onnx \
    --decoder=./sherpa-onnx-zipformer-cantonese-2024-03-13/decoder-epoch-45-avg-35.onnx \
    --joiner=./sherpa-onnx-zipformer-cantonese-2024-03-13/joiner-epoch-45-avg-35.int8.onnx


sherpa-onnx-zipformer-gigaspeech-2023-12-12 (English)
-----------------------------------------------------

Training code for this model is `<https://github.com/k2-fsa/icefall/pull/1254>`_.
It supports only English since it is trained on the `GigaSpeech`_ dataset.

In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-gigaspeech-2023-12-12.tar.bz2
   tar xf sherpa-onnx-zipformer-gigaspeech-2023-12-12.tar.bz2
   ls -lh sherpa-onnx-zipformer-gigaspeech-2023-12-12

You should see the following output:

.. code-block:: bash

  $ ls -lh sherpa-onnx-zipformer-gigaspeech-2023-12-12
  total 656184
  -rw-r--r--  1 fangjun  staff    28B Dec 12 19:00 README.md
  -rw-r--r--  1 fangjun  staff   239K Dec 12 19:00 bpe.model
  -rw-r--r--  1 fangjun  staff   528K Dec 12 19:00 decoder-epoch-30-avg-1.int8.onnx
  -rw-r--r--  1 fangjun  staff   2.0M Dec 12 19:00 decoder-epoch-30-avg-1.onnx
  -rw-r--r--  1 fangjun  staff    68M Dec 12 19:00 encoder-epoch-30-avg-1.int8.onnx
  -rw-r--r--  1 fangjun  staff   249M Dec 12 19:00 encoder-epoch-30-avg-1.onnx
  -rw-r--r--  1 fangjun  staff   253K Dec 12 19:00 joiner-epoch-30-avg-1.int8.onnx
  -rw-r--r--  1 fangjun  staff   1.0M Dec 12 19:00 joiner-epoch-30-avg-1.onnx
  drwxr-xr-x  5 fangjun  staff   160B Dec 12 19:00 test_wavs
  -rw-r--r--  1 fangjun  staff   4.9K Dec 12 19:00 tokens.txt

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
    --tokens=./sherpa-onnx-zipformer-gigaspeech-2023-12-12/tokens.txt \
    --encoder=./sherpa-onnx-zipformer-gigaspeech-2023-12-12/encoder-epoch-30-avg-1.onnx \
    --decoder=./sherpa-onnx-zipformer-gigaspeech-2023-12-12/decoder-epoch-30-avg-1.onnx \
    --joiner=./sherpa-onnx-zipformer-gigaspeech-2023-12-12/joiner-epoch-30-avg-1.onnx \
    ./sherpa-onnx-zipformer-gigaspeech-2023-12-12/test_wavs/1089-134686-0001.wav \
    ./sherpa-onnx-zipformer-gigaspeech-2023-12-12/test_wavs/1221-135766-0001.wav \
    ./sherpa-onnx-zipformer-gigaspeech-2023-12-12/test_wavs/1221-135766-0002.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

You should see the following output:

.. literalinclude:: ./code-zipformer/sherpa-onnx-zipformer-gigaspeech-2023-12-12.txt

int8
^^^^

The following code shows how to use ``int8`` models to decode wave files:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-zipformer-gigaspeech-2023-12-12/tokens.txt \
    --encoder=./sherpa-onnx-zipformer-gigaspeech-2023-12-12/encoder-epoch-30-avg-1.int8.onnx \
    --decoder=./sherpa-onnx-zipformer-gigaspeech-2023-12-12/decoder-epoch-30-avg-1.onnx \
    --joiner=./sherpa-onnx-zipformer-gigaspeech-2023-12-12/joiner-epoch-30-avg-1.int8.onnx \
    ./sherpa-onnx-zipformer-gigaspeech-2023-12-12/test_wavs/1089-134686-0001.wav \
    ./sherpa-onnx-zipformer-gigaspeech-2023-12-12/test_wavs/1221-135766-0001.wav \
    ./sherpa-onnx-zipformer-gigaspeech-2023-12-12/test_wavs/1221-135766-0002.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

You should see the following output:

.. literalinclude:: ./code-zipformer/sherpa-onnx-zipformer-gigaspeech-2023-12-12-int8.txt

Speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-microphone-offline \
    --tokens=./sherpa-onnx-zipformer-gigaspeech-2023-12-12/tokens.txt \
    --encoder=./sherpa-onnx-zipformer-gigaspeech-2023-12-12/encoder-epoch-30-avg-1.int8.onnx \
    --decoder=./sherpa-onnx-zipformer-gigaspeech-2023-12-12/decoder-epoch-30-avg-1.onnx \
    --joiner=./sherpa-onnx-zipformer-gigaspeech-2023-12-12/joiner-epoch-30-avg-1.int8.onnx

Speech recognition from a microphone with VAD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

  ./build/bin/sherpa-onnx-vad-microphone-offline-asr \
    --silero-vad-model=./silero_vad.onnx \
    --tokens=./sherpa-onnx-zipformer-gigaspeech-2023-12-12/tokens.txt \
    --encoder=./sherpa-onnx-zipformer-gigaspeech-2023-12-12/encoder-epoch-30-avg-1.int8.onnx \
    --decoder=./sherpa-onnx-zipformer-gigaspeech-2023-12-12/decoder-epoch-30-avg-1.onnx \
    --joiner=./sherpa-onnx-zipformer-gigaspeech-2023-12-12/joiner-epoch-30-avg-1.int8.onnx

zrjin/sherpa-onnx-zipformer-multi-zh-hans-2023-9-2 (Chinese)
------------------------------------------------------------

This model is from

`<https://huggingface.co/zrjin/sherpa-onnx-zipformer-multi-zh-hans-2023-9-2>`_

which supports Chinese as it is trained on whatever datasets involved in the `multi-zh_hans <https://github.com/k2-fsa/icefall/tree/master/egs/multi_zh-hans/ASR/>`_ recipe.

If you are interested in how the model is trained, please refer to
`<https://github.com/k2-fsa/icefall/pull/1238>`_.

In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/zrjin/sherpa-onnx-zipformer-multi-zh-hans-2023-9-2
  cd sherpa-onnx-zipformer-multi-zh-hans-2023-9-2

  git lfs pull --include "*.onnx"

Please check that the file sizes of the pre-trained models are correct. See
the file sizes of ``*.onnx`` files below.

.. code-block:: bash

  sherpa-onnx-zipformer-multi-zh-hans-2023-9-2 zengruijin$ ls -lh *.onnx
  -rw-rw-r--@ 1 zengruijin  staff   1.2M Sep 18 07:04 decoder-epoch-20-avg-1.int8.onnx
  -rw-rw-r--@ 1 zengruijin  staff   4.9M Sep 18 07:04 decoder-epoch-20-avg-1.onnx
  -rw-rw-r--@ 1 zengruijin  staff    66M Sep 18 07:04 encoder-epoch-20-avg-1.int8.onnx
  -rw-rw-r--@ 1 zengruijin  staff   248M Sep 18 07:05 encoder-epoch-20-avg-1.onnx
  -rw-rw-r--@ 1 zengruijin  staff   1.0M Sep 18 07:05 joiner-epoch-20-avg-1.int8.onnx
  -rw-rw-r--@ 1 zengruijin  staff   3.9M Sep 18 07:05 joiner-epoch-20-avg-1.onnx

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
    --tokens=./sherpa-onnx-zipformer-multi-zh-hans-2023-9-2/tokens.txt \
    --encoder=./sherpa-onnx-zipformer-multi-zh-hans-2023-9-2/encoder-epoch-20-avg-1.onnx \
    --decoder=./sherpa-onnx-zipformer-multi-zh-hans-2023-9-2/decoder-epoch-20-avg-1.onnx \
    --joiner=./sherpa-onnx-zipformer-multi-zh-hans-2023-9-2/joiner-epoch-20-avg-1.onnx \
    ./sherpa-onnx-zipformer-multi-zh-hans-2023-9-2/test_wavs/0.wav \
    ./sherpa-onnx-zipformer-multi-zh-hans-2023-9-2/test_wavs/1.wav \
    ./sherpa-onnx-zipformer-multi-zh-hans-2023-9-2/test_wavs/8k.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

You should see the following output:

.. literalinclude:: ./code-zipformer/sherpa-onnx-zipformer-multi-zh-hans-2023-9-2.txt

int8
^^^^

The following code shows how to use ``int8`` models to decode wave files:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-zipformer-multi-zh-hans-2023-9-2/tokens.txt \
    --encoder=./sherpa-onnx-zipformer-multi-zh-hans-2023-9-2/encoder-epoch-20-avg-1.int8.onnx \
    --decoder=./sherpa-onnx-zipformer-multi-zh-hans-2023-9-2/decoder-epoch-20-avg-1.int8.onnx \
    --joiner=./sherpa-onnx-zipformer-multi-zh-hans-2023-9-2/joiner-epoch-20-avg-1.int8.onnx \
    ./sherpa-onnx-zipformer-multi-zh-hans-2023-9-2/test_wavs/0.wav \
    ./sherpa-onnx-zipformer-multi-zh-hans-2023-9-2/test_wavs/1.wav \
    ./sherpa-onnx-zipformer-multi-zh-hans-2023-9-2/test_wavs/8k.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

You should see the following output:

.. literalinclude:: ./code-zipformer/sherpa-onnx-zipformer-multi-zh-hans-2023-9-2-int8.txt

Speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-microphone-offline \
    --tokens=./sherpa-onnx-zipformer-multi-zh-hans-2023-9-2/tokens.txt \
    --encoder=./sherpa-onnx-zipformer-multi-zh-hans-2023-9-2/encoder-epoch-20-avg-1.onnx \
    --decoder=./sherpa-onnx-zipformer-multi-zh-hans-2023-9-2/decoder-epoch-0-avg-1.onnx \
    --joiner=./sherpa-onnx-zipformer-multi-zh-hans-2023-9-2/joiner-epoch-20-avg-1.onnx


yfyeung/icefall-asr-cv-corpus-13.0-2023-03-09-en-pruned-transducer-stateless7-2023-04-17 (English)
--------------------------------------------------------------------------------------------------

This model is from

`<https://huggingface.co/yfyeung/icefall-asr-cv-corpus-13.0-2023-03-09-en-pruned-transducer-stateless7-2023-04-17>`_

which supports only English as it is trained on the `CommonVoice`_ English dataset.

If you are interested in how the model is trained, please refer to
`<https://github.com/k2-fsa/icefall/pull/997>`_.

In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/yfyeung/icefall-asr-cv-corpus-13.0-2023-03-09-en-pruned-transducer-stateless7-2023-04-17
  cd icefall-asr-cv-corpus-13.0-2023-03-09-en-pruned-transducer-stateless7-2023-04-17

  git lfs pull --include "exp/*epoch-60-avg-20*.onnx"

Please check that the file sizes of the pre-trained models are correct. See
the file sizes of ``*.onnx`` files below.

.. code-block:: bash

  icefall-asr-cv-corpus-13.0-2023-03-09-en-pruned-transducer-stateless7-2023-04-17 fangjun$ ls -lh exp/*epoch-60-avg-20*.onnx
  -rw-r--r--  1 fangjun  staff   1.2M Jun 27 09:53 exp/decoder-epoch-60-avg-20.int8.onnx
  -rw-r--r--  1 fangjun  staff   2.0M Jun 27 09:54 exp/decoder-epoch-60-avg-20.onnx
  -rw-r--r--  1 fangjun  staff   121M Jun 27 09:54 exp/encoder-epoch-60-avg-20.int8.onnx
  -rw-r--r--  1 fangjun  staff   279M Jun 27 09:55 exp/encoder-epoch-60-avg-20.onnx
  -rw-r--r--  1 fangjun  staff   253K Jun 27 09:53 exp/joiner-epoch-60-avg-20.int8.onnx
  -rw-r--r--  1 fangjun  staff   1.0M Jun 27 09:53 exp/joiner-epoch-60-avg-20.onnx

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
    --tokens=./icefall-asr-cv-corpus-13.0-2023-03-09-en-pruned-transducer-stateless7-2023-04-17/data/lang_bpe_500/tokens.txt \
    --encoder=./icefall-asr-cv-corpus-13.0-2023-03-09-en-pruned-transducer-stateless7-2023-04-17/exp/encoder-epoch-60-avg-20.onnx \
    --decoder=./icefall-asr-cv-corpus-13.0-2023-03-09-en-pruned-transducer-stateless7-2023-04-17/exp/decoder-epoch-60-avg-20.onnx \
    --joiner=./icefall-asr-cv-corpus-13.0-2023-03-09-en-pruned-transducer-stateless7-2023-04-17/exp/joiner-epoch-60-avg-20.onnx \
    ./icefall-asr-cv-corpus-13.0-2023-03-09-en-pruned-transducer-stateless7-2023-04-17/test_wavs/1089-134686-0001.wav \
    ./icefall-asr-cv-corpus-13.0-2023-03-09-en-pruned-transducer-stateless7-2023-04-17/test_wavs/1221-135766-0001.wav \
    ./icefall-asr-cv-corpus-13.0-2023-03-09-en-pruned-transducer-stateless7-2023-04-17/test_wavs/1221-135766-0002.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

You should see the following output:

.. literalinclude:: ./code-zipformer/icefall-asr-cv-corpus-13.0-2023-03-09-en-pruned-transducer-stateless7-2023-04-17.txt

int8
^^^^

The following code shows how to use ``int8`` models to decode wave files:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-offline \
    --tokens=./icefall-asr-cv-corpus-13.0-2023-03-09-en-pruned-transducer-stateless7-2023-04-17/data/lang_bpe_500/tokens.txt \
    --encoder=./icefall-asr-cv-corpus-13.0-2023-03-09-en-pruned-transducer-stateless7-2023-04-17/exp/encoder-epoch-60-avg-20.int8.onnx \
    --decoder=./icefall-asr-cv-corpus-13.0-2023-03-09-en-pruned-transducer-stateless7-2023-04-17/exp/decoder-epoch-60-avg-20.int8.onnx \
    --joiner=./icefall-asr-cv-corpus-13.0-2023-03-09-en-pruned-transducer-stateless7-2023-04-17/exp/joiner-epoch-60-avg-20.int8.onnx \
    ./icefall-asr-cv-corpus-13.0-2023-03-09-en-pruned-transducer-stateless7-2023-04-17/test_wavs/1089-134686-0001.wav \
    ./icefall-asr-cv-corpus-13.0-2023-03-09-en-pruned-transducer-stateless7-2023-04-17/test_wavs/1221-135766-0001.wav \
    ./icefall-asr-cv-corpus-13.0-2023-03-09-en-pruned-transducer-stateless7-2023-04-17/test_wavs/1221-135766-0002.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

You should see the following output:

.. literalinclude:: ./code-zipformer/icefall-asr-cv-corpus-13.0-2023-03-09-en-pruned-transducer-stateless7-2023-04-17-int8.txt

Speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-microphone-offline \
    --tokens=./icefall-asr-cv-corpus-13.0-2023-03-09-en-pruned-transducer-stateless7-2023-04-17/data/lang_bpe_500/tokens.txt \
    --encoder=./icefall-asr-cv-corpus-13.0-2023-03-09-en-pruned-transducer-stateless7-2023-04-17/exp/encoder-epoch-60-avg-20.onnx \
    --decoder=./icefall-asr-cv-corpus-13.0-2023-03-09-en-pruned-transducer-stateless7-2023-04-17/exp/decoder-epoch-60-avg-20.onnx \
    --joiner=./icefall-asr-cv-corpus-13.0-2023-03-09-en-pruned-transducer-stateless7-2023-04-17/exp/joiner-epoch-60-avg-20.onnx


pkufool/icefall-asr-zipformer-wenetspeech-20230615 (Chinese)
------------------------------------------------------------

This model is from

`<https://huggingface.co/pkufool/icefall-asr-zipformer-wenetspeech-20230615>`_

which supports only Chinese as it is trained on the `WenetSpeech`_ corpus.

If you are interested in how the model is trained, please refer to
`<https://github.com/k2-fsa/icefall/pull/1130>`_.

In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/pkufool/icefall-asr-zipformer-wenetspeech-20230615
  cd icefall-asr-zipformer-wenetspeech-20230615

  git lfs pull --include "exp/*.onnx"

Please check that the file sizes of the pre-trained models are correct. See
the file sizes of ``*.onnx`` files below.

.. code-block:: bash

  icefall-asr-zipformer-wenetspeech-20230615 fangjun$ ls -lh exp/*.onnx
  -rw-r--r--  1 fangjun  staff    11M Jun 26 14:31 exp/decoder-epoch-12-avg-4.int8.onnx
  -rw-r--r--  1 fangjun  staff    12M Jun 26 14:31 exp/decoder-epoch-12-avg-4.onnx
  -rw-r--r--  1 fangjun  staff    66M Jun 26 14:32 exp/encoder-epoch-12-avg-4.int8.onnx
  -rw-r--r--  1 fangjun  staff   248M Jun 26 14:34 exp/encoder-epoch-12-avg-4.onnx
  -rw-r--r--  1 fangjun  staff   2.7M Jun 26 14:31 exp/joiner-epoch-12-avg-4.int8.onnx
  -rw-r--r--  1 fangjun  staff    11M Jun 26 14:31 exp/joiner-epoch-12-avg-4.onnx

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
    --tokens=./icefall-asr-zipformer-wenetspeech-20230615/data/lang_char/tokens.txt \
    --encoder=./icefall-asr-zipformer-wenetspeech-20230615/exp/encoder-epoch-12-avg-4.onnx \
    --decoder=./icefall-asr-zipformer-wenetspeech-20230615/exp/decoder-epoch-12-avg-4.onnx \
    --joiner=./icefall-asr-zipformer-wenetspeech-20230615/exp/joiner-epoch-12-avg-4.onnx \
    ./icefall-asr-zipformer-wenetspeech-20230615/test_wavs/DEV_T0000000000.wav \
    ./icefall-asr-zipformer-wenetspeech-20230615/test_wavs/DEV_T0000000001.wav \
    ./icefall-asr-zipformer-wenetspeech-20230615/test_wavs/DEV_T0000000002.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

You should see the following output:

.. literalinclude:: ./code-zipformer/icefall-asr-zipformer-wenetspeech-20230615.txt

int8
^^^^

The following code shows how to use ``int8`` models to decode wave files:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-offline \
    --tokens=./icefall-asr-zipformer-wenetspeech-20230615/data/lang_char/tokens.txt \
    --encoder=./icefall-asr-zipformer-wenetspeech-20230615/exp/encoder-epoch-12-avg-4.int8.onnx \
    --decoder=./icefall-asr-zipformer-wenetspeech-20230615/exp/decoder-epoch-12-avg-4.int8.onnx \
    --joiner=./icefall-asr-zipformer-wenetspeech-20230615/exp/joiner-epoch-12-avg-4.int8.onnx \
    ./icefall-asr-zipformer-wenetspeech-20230615/test_wavs/DEV_T0000000000.wav \
    ./icefall-asr-zipformer-wenetspeech-20230615/test_wavs/DEV_T0000000001.wav \
    ./icefall-asr-zipformer-wenetspeech-20230615/test_wavs/DEV_T0000000002.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

You should see the following output:

.. literalinclude:: ./code-zipformer/icefall-asr-zipformer-wenetspeech-20230615-int8.txt

Speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-microphone-offline \
    --tokens=./icefall-asr-zipformer-wenetspeech-20230615/data/lang_char/tokens.txt \
    --encoder=./icefall-asr-zipformer-wenetspeech-20230615/exp/encoder-epoch-12-avg-4.onnx \
    --decoder=./icefall-asr-zipformer-wenetspeech-20230615/exp/decoder-epoch-12-avg-4.onnx \
    --joiner=./icefall-asr-zipformer-wenetspeech-20230615/exp/joiner-epoch-12-avg-4.onnx


csukuangfj/sherpa-onnx-zipformer-large-en-2023-06-26 (English)
--------------------------------------------------------------

This model is converted from

`<https://huggingface.co/Zengwei/icefall-asr-librispeech-zipformer-large-2023-05-16>`_

which supports only English as it is trained on the `LibriSpeech`_ corpus.

You can find the training code at

`<https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/zipformer>`_

In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/sherpa-onnx-zipformer-large-en-2023-06-26
  cd sherpa-onnx-zipformer-large-en-2023-06-26
  git lfs pull --include "*.onnx"

Please check that the file sizes of the pre-trained models are correct. See
the file sizes of ``*.onnx`` files below.

.. code-block:: bash

  sherpa-onnx-zipformer-large-en-2023-06-26 fangjun$ ls -lh *.onnx
  -rw-r--r--  1 fangjun  staff   1.2M Jun 26 13:19 decoder-epoch-99-avg-1.int8.onnx
  -rw-r--r--  1 fangjun  staff   2.0M Jun 26 13:19 decoder-epoch-99-avg-1.onnx
  -rw-r--r--  1 fangjun  staff   145M Jun 26 13:20 encoder-epoch-99-avg-1.int8.onnx
  -rw-r--r--  1 fangjun  staff   564M Jun 26 13:22 encoder-epoch-99-avg-1.onnx
  -rw-r--r--  1 fangjun  staff   253K Jun 26 13:19 joiner-epoch-99-avg-1.int8.onnx
  -rw-r--r--  1 fangjun  staff   1.0M Jun 26 13:19 joiner-epoch-99-avg-1.onnx

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
    --tokens=./sherpa-onnx-zipformer-large-en-2023-06-26/tokens.txt \
    --encoder=./sherpa-onnx-zipformer-large-en-2023-06-26/encoder-epoch-99-avg-1.onnx \
    --decoder=./sherpa-onnx-zipformer-large-en-2023-06-26/decoder-epoch-99-avg-1.onnx \
    --joiner=./sherpa-onnx-zipformer-large-en-2023-06-26/joiner-epoch-99-avg-1.onnx \
    ./sherpa-onnx-zipformer-large-en-2023-06-26/test_wavs/0.wav \
    ./sherpa-onnx-zipformer-large-en-2023-06-26/test_wavs/1.wav \
    ./sherpa-onnx-zipformer-large-en-2023-06-26/test_wavs/8k.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

You should see the following output:

.. literalinclude:: ./code-zipformer/sherpa-onnx-zipformer-large-en-2023-06-26.txt

int8
^^^^

The following code shows how to use ``int8`` models to decode wave files:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-zipformer-large-en-2023-06-26/tokens.txt \
    --encoder=./sherpa-onnx-zipformer-large-en-2023-06-26/encoder-epoch-99-avg-1.int8.onnx \
    --decoder=./sherpa-onnx-zipformer-large-en-2023-06-26/decoder-epoch-99-avg-1.int8.onnx \
    --joiner=./sherpa-onnx-zipformer-large-en-2023-06-26/joiner-epoch-99-avg-1.int8.onnx \
    ./sherpa-onnx-zipformer-large-en-2023-06-26/test_wavs/0.wav \
    ./sherpa-onnx-zipformer-large-en-2023-06-26/test_wavs/1.wav \
    ./sherpa-onnx-zipformer-large-en-2023-06-26/test_wavs/8k.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

You should see the following output:

.. literalinclude:: ./code-zipformer/sherpa-onnx-zipformer-large-en-2023-06-26-int8.txt

Speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-microphone-offline \
    --tokens=./sherpa-onnx-zipformer-large-en-2023-06-26/tokens.txt \
    --encoder=./sherpa-onnx-zipformer-large-en-2023-06-26/encoder-epoch-99-avg-1.onnx \
    --decoder=./sherpa-onnx-zipformer-large-en-2023-06-26/decoder-epoch-99-avg-1.onnx \
    --joiner=./sherpa-onnx-zipformer-large-en-2023-06-26/joiner-epoch-99-avg-1.onnx

csukuangfj/sherpa-onnx-zipformer-small-en-2023-06-26 (English)
--------------------------------------------------------------

This model is converted from

`<https://huggingface.co/Zengwei/icefall-asr-librispeech-zipformer-small-2023-05-16>`_

which supports only English as it is trained on the `LibriSpeech`_ corpus.

You can find the training code at

`<https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/zipformer>`_

In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/sherpa-onnx-zipformer-small-en-2023-06-26
  cd sherpa-onnx-zipformer-small-en-2023-06-26
  git lfs pull --include "*.onnx"

Please check that the file sizes of the pre-trained models are correct. See
the file sizes of ``*.onnx`` files below.

.. code-block:: bash

  sherpa-onnx-zipformer-small-en-2023-06-26 fangjun$ ls -lh *.onnx
  -rw-r--r--  1 fangjun  staff   1.2M Jun 26 13:04 decoder-epoch-99-avg-1.int8.onnx
  -rw-r--r--  1 fangjun  staff   2.0M Jun 26 13:04 decoder-epoch-99-avg-1.onnx
  -rw-r--r--  1 fangjun  staff    25M Jun 26 13:04 encoder-epoch-99-avg-1.int8.onnx
  -rw-r--r--  1 fangjun  staff    87M Jun 26 13:04 encoder-epoch-99-avg-1.onnx
  -rw-r--r--  1 fangjun  staff   253K Jun 26 13:04 joiner-epoch-99-avg-1.int8.onnx
  -rw-r--r--  1 fangjun  staff   1.0M Jun 26 13:04 joiner-epoch-99-avg-1.onnx

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
    --tokens=./sherpa-onnx-zipformer-small-en-2023-06-26/tokens.txt \
    --encoder=./sherpa-onnx-zipformer-small-en-2023-06-26/encoder-epoch-99-avg-1.onnx \
    --decoder=./sherpa-onnx-zipformer-small-en-2023-06-26/decoder-epoch-99-avg-1.onnx \
    --joiner=./sherpa-onnx-zipformer-small-en-2023-06-26/joiner-epoch-99-avg-1.onnx \
    ./sherpa-onnx-zipformer-small-en-2023-06-26/test_wavs/0.wav \
    ./sherpa-onnx-zipformer-small-en-2023-06-26/test_wavs/1.wav \
    ./sherpa-onnx-zipformer-small-en-2023-06-26/test_wavs/8k.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

You should see the following output:

.. literalinclude:: ./code-zipformer/sherpa-onnx-zipformer-small-en-2023-06-26.txt

int8
^^^^

The following code shows how to use ``int8`` models to decode wave files:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-zipformer-small-en-2023-06-26/tokens.txt \
    --encoder=./sherpa-onnx-zipformer-small-en-2023-06-26/encoder-epoch-99-avg-1.int8.onnx \
    --decoder=./sherpa-onnx-zipformer-small-en-2023-06-26/decoder-epoch-99-avg-1.int8.onnx \
    --joiner=./sherpa-onnx-zipformer-small-en-2023-06-26/joiner-epoch-99-avg-1.int8.onnx \
    ./sherpa-onnx-zipformer-small-en-2023-06-26/test_wavs/0.wav \
    ./sherpa-onnx-zipformer-small-en-2023-06-26/test_wavs/1.wav \
    ./sherpa-onnx-zipformer-small-en-2023-06-26/test_wavs/8k.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

You should see the following output:

.. literalinclude:: ./code-zipformer/sherpa-onnx-zipformer-small-en-2023-06-26-int8.txt

Speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-microphone-offline \
    --tokens=./sherpa-onnx-zipformer-small-en-2023-06-26/tokens.txt \
    --encoder=./sherpa-onnx-zipformer-small-en-2023-06-26/encoder-epoch-99-avg-1.onnx \
    --decoder=./sherpa-onnx-zipformer-small-en-2023-06-26/decoder-epoch-99-avg-1.onnx \
    --joiner=./sherpa-onnx-zipformer-small-en-2023-06-26/joiner-epoch-99-avg-1.onnx

.. _sherpa-onnx-zipformer-en-2023-06-26-english:

csukuangfj/sherpa-onnx-zipformer-en-2023-06-26 (English)
--------------------------------------------------------

This model is converted from

`<https://huggingface.co/Zengwei/icefall-asr-librispeech-zipformer-2023-05-15>`_

which supports only English as it is trained on the `LibriSpeech`_ corpus.

You can find the training code at

`<https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/zipformer>`_

In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-en-2023-06-26.tar.bz2
  tar xvf sherpa-onnx-zipformer-en-2023-06-26.tar.bz2

Please check that the file sizes of the pre-trained models are correct. See
the file sizes of ``*.onnx`` files below.

.. code-block:: bash

  sherpa-onnx-zipformer-en-2023-06-26 fangjun$ ls -lh *.onnx
  -rw-r--r--  1 fangjun  staff   1.2M Jun 26 12:45 decoder-epoch-99-avg-1.int8.onnx
  -rw-r--r--  1 fangjun  staff   2.0M Jun 26 12:45 decoder-epoch-99-avg-1.onnx
  -rw-r--r--  1 fangjun  staff    66M Jun 26 12:45 encoder-epoch-99-avg-1.int8.onnx
  -rw-r--r--  1 fangjun  staff   248M Jun 26 12:46 encoder-epoch-99-avg-1.onnx
  -rw-r--r--  1 fangjun  staff   253K Jun 26 12:45 joiner-epoch-99-avg-1.int8.onnx
  -rw-r--r--  1 fangjun  staff   1.0M Jun 26 12:45 joiner-epoch-99-avg-1.onnx

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
    --tokens=./sherpa-onnx-zipformer-en-2023-06-26/tokens.txt \
    --encoder=./sherpa-onnx-zipformer-en-2023-06-26/encoder-epoch-99-avg-1.onnx \
    --decoder=./sherpa-onnx-zipformer-en-2023-06-26/decoder-epoch-99-avg-1.onnx \
    --joiner=./sherpa-onnx-zipformer-en-2023-06-26/joiner-epoch-99-avg-1.onnx \
    ./sherpa-onnx-zipformer-en-2023-06-26/test_wavs/0.wav \
    ./sherpa-onnx-zipformer-en-2023-06-26/test_wavs/1.wav \
    ./sherpa-onnx-zipformer-en-2023-06-26/test_wavs/8k.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

You should see the following output:

.. literalinclude:: ./code-zipformer/sherpa-onnx-zipformer-en-2023-06-26.txt

int8
^^^^

The following code shows how to use ``int8`` models to decode wave files:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-zipformer-en-2023-06-26/tokens.txt \
    --encoder=./sherpa-onnx-zipformer-en-2023-06-26/encoder-epoch-99-avg-1.int8.onnx \
    --decoder=./sherpa-onnx-zipformer-en-2023-06-26/decoder-epoch-99-avg-1.int8.onnx \
    --joiner=./sherpa-onnx-zipformer-en-2023-06-26/joiner-epoch-99-avg-1.int8.onnx \
    ./sherpa-onnx-zipformer-en-2023-06-26/test_wavs/0.wav \
    ./sherpa-onnx-zipformer-en-2023-06-26/test_wavs/1.wav \
    ./sherpa-onnx-zipformer-en-2023-06-26/test_wavs/8k.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

You should see the following output:

.. literalinclude:: ./code-zipformer/sherpa-onnx-zipformer-en-2023-06-26-int8.txt

Speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-microphone-offline \
    --tokens=./sherpa-onnx-zipformer-en-2023-06-26/tokens.txt \
    --encoder=./sherpa-onnx-zipformer-en-2023-06-26/encoder-epoch-99-avg-1.onnx \
    --decoder=./sherpa-onnx-zipformer-en-2023-06-26/decoder-epoch-99-avg-1.onnx \
    --joiner=./sherpa-onnx-zipformer-en-2023-06-26/joiner-epoch-99-avg-1.onnx


.. _icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04-english:

icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04 (English)
--------------------------------------------------------------------------

This model is trained using GigaSpeech + LibriSpeech + Common Voice 13.0 with zipformer

See `<https://github.com/k2-fsa/icefall/pull/1010>`_ if you are interested in how
it is trained.

In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/yfyeung/icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04
  cd icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04/exp
  git lfs pull --include "*.onnx"

Please check that the file sizes of the pre-trained models are correct. See
the file sizes of ``*.onnx`` files below.

.. code-block:: bash

  $ ls -lh *.onnx
  -rw-r--r--  1 fangjun  staff   1.2M May 15 11:11 decoder-epoch-30-avg-4.int8.onnx
  -rw-r--r--  1 fangjun  staff   2.0M May 15 11:11 decoder-epoch-30-avg-4.onnx
  -rw-r--r--  1 fangjun  staff   121M May 15 11:12 encoder-epoch-30-avg-4.int8.onnx
  -rw-r--r--  1 fangjun  staff   279M May 15 11:13 encoder-epoch-30-avg-4.onnx
  -rw-r--r--  1 fangjun  staff   253K May 15 11:11 joiner-epoch-30-avg-4.int8.onnx
  -rw-r--r--  1 fangjun  staff   1.0M May 15 11:11 joiner-epoch-30-avg-4.onnx

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
    --tokens=./icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04/data/lang_bpe_500/tokens.txt \
    --encoder=./icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04/exp/encoder-epoch-30-avg-4.onnx \
    --decoder=./icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04/exp/decoder-epoch-30-avg-4.onnx \
    --joiner=./icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04/exp/joiner-epoch-30-avg-4.onnx \
    ./icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04/test_wavs/1089-134686-0001.wav \
    ./icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04/test_wavs/1221-135766-0001.wav \
    ./icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04/test_wavs/1221-135766-0002.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

You should see the following output:

.. literalinclude:: ./code-zipformer/sherpa-onnx-zipformer-multi-dataset-2023-05-04.txt

int8
^^^^

The following code shows how to use ``int8`` models to decode wave files:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-offline \
    --tokens=./icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04/data/lang_bpe_500/tokens.txt \
    --encoder=./icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04/exp/encoder-epoch-30-avg-4.int8.onnx \
    --decoder=./icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04/exp/decoder-epoch-30-avg-4.int8.onnx \
    --joiner=./icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04/exp/joiner-epoch-30-avg-4.int8.onnx \
    ./icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04/test_wavs/1089-134686-0001.wav \
    ./icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04/test_wavs/1221-135766-0001.wav \
    ./icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04/test_wavs/1221-135766-0002.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

You should see the following output:

.. literalinclude:: ./code-zipformer/sherpa-onnx-zipformer-multi-dataset-2023-05-04-int8.txt

Speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-microphone-offline \
    --tokens=./icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04/data/lang_bpe_500/tokens.txt \
    --encoder=./icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04/exp/encoder-epoch-30-avg-4.onnx \
    --decoder=./icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04/exp/decoder-epoch-30-avg-4.onnx \
    --joiner=./icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04/exp/joiner-epoch-30-avg-4.onnx

.. _sherpa_onnx_zipformer_en_2023_04_01:

csukuangfj/sherpa-onnx-zipformer-en-2023-04-01 (English)
--------------------------------------------------------

This model is converted from

`<https://huggingface.co/WeijiZhuang/icefall-asr-librispeech-pruned-transducer-stateless8-2022-12-02>`_

which supports only English as it is trained on the `LibriSpeech`_ and `GigaSpeech`_ corpus.

You can find the training code at

`<https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/pruned_transducer_stateless8>`_

In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/sherpa-onnx-zipformer-en-2023-04-01
  cd sherpa-onnx-zipformer-en-2023-04-01
  git lfs pull --include "*.onnx"

Please check that the file sizes of the pre-trained models are correct. See
the file sizes of ``*.onnx`` files below.

.. code-block:: bash

  sherpa-onnx-zipformer-en-2023-04-01$ ls -lh *.onnx
  -rw-r--r-- 1 kuangfangjun root  1.3M Apr  1 14:34 decoder-epoch-99-avg-1.int8.onnx
  -rw-r--r-- 1 kuangfangjun root  2.0M Apr  1 14:34 decoder-epoch-99-avg-1.onnx
  -rw-r--r-- 1 kuangfangjun root  180M Apr  1 14:34 encoder-epoch-99-avg-1.int8.onnx
  -rw-r--r-- 1 kuangfangjun root  338M Apr  1 14:34 encoder-epoch-99-avg-1.onnx
  -rw-r--r-- 1 kuangfangjun root  254K Apr  1 14:34 joiner-epoch-99-avg-1.int8.onnx
  -rw-r--r-- 1 kuangfangjun root 1003K Apr  1 14:34 joiner-epoch-99-avg-1.onnx

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
    --tokens=./sherpa-onnx-zipformer-en-2023-04-01/tokens.txt \
    --encoder=./sherpa-onnx-zipformer-en-2023-04-01/encoder-epoch-99-avg-1.onnx \
    --decoder=./sherpa-onnx-zipformer-en-2023-04-01/decoder-epoch-99-avg-1.onnx \
    --joiner=./sherpa-onnx-zipformer-en-2023-04-01/joiner-epoch-99-avg-1.onnx \
    ./sherpa-onnx-zipformer-en-2023-04-01/test_wavs/0.wav \
    ./sherpa-onnx-zipformer-en-2023-04-01/test_wavs/1.wav \
    ./sherpa-onnx-zipformer-en-2023-04-01/test_wavs/8k.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

You should see the following output:

.. literalinclude:: ./code-zipformer/sherpa-onnx-zipformer-en-2023-04-01.txt

int8
^^^^

The following code shows how to use ``int8`` models to decode wave files:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-zipformer-en-2023-04-01/tokens.txt \
    --encoder=./sherpa-onnx-zipformer-en-2023-04-01/encoder-epoch-99-avg-1.int8.onnx \
    --decoder=./sherpa-onnx-zipformer-en-2023-04-01/decoder-epoch-99-avg-1.int8.onnx \
    --joiner=./sherpa-onnx-zipformer-en-2023-04-01/joiner-epoch-99-avg-1.int8.onnx \
    ./sherpa-onnx-zipformer-en-2023-04-01/test_wavs/0.wav \
    ./sherpa-onnx-zipformer-en-2023-04-01/test_wavs/1.wav \
    ./sherpa-onnx-zipformer-en-2023-04-01/test_wavs/8k.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

You should see the following output:

.. literalinclude:: ./code-zipformer/sherpa-onnx-zipformer-en-2023-04-01-int8.txt

Speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-microphone-offline \
    --tokens=./sherpa-onnx-zipformer-en-2023-04-01/tokens.txt \
    --encoder=./sherpa-onnx-zipformer-en-2023-04-01/encoder-epoch-99-avg-1.onnx \
    --decoder=./sherpa-onnx-zipformer-en-2023-04-01/decoder-epoch-99-avg-1.onnx \
    --joiner=./sherpa-onnx-zipformer-en-2023-04-01/joiner-epoch-99-avg-1.onnx

.. _sherpa_onnx_zipformer_en_2023_03_30:

csukuangfj/sherpa-onnx-zipformer-en-2023-03-30 (English)
--------------------------------------------------------

This model is converted from

`<https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless7-2022-11-11>`_

which supports only English as it is trained on the `LibriSpeech`_ corpus.

You can find the training code at

`<https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/pruned_transducer_stateless7>`_

In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/sherpa-onnx-zipformer-en-2023-03-30
  cd sherpa-onnx-zipformer-en-2023-03-30
  git lfs pull --include "*.onnx"

Please check that the file sizes of the pre-trained models are correct. See
the file sizes of ``*.onnx`` files below.

.. code-block:: bash

  sherpa-onnx-zipformer-en-2023-03-30$ ls -lh *.onnx
  -rw-r--r-- 1 kuangfangjun root  1.3M Mar 31 00:37 decoder-epoch-99-avg-1.int8.onnx
  -rw-r--r-- 1 kuangfangjun root  2.0M Mar 30 20:10 decoder-epoch-99-avg-1.onnx
  -rw-r--r-- 1 kuangfangjun root  180M Mar 31 00:37 encoder-epoch-99-avg-1.int8.onnx
  -rw-r--r-- 1 kuangfangjun root  338M Mar 30 20:10 encoder-epoch-99-avg-1.onnx
  -rw-r--r-- 1 kuangfangjun root  254K Mar 31 00:37 joiner-epoch-99-avg-1.int8.onnx
  -rw-r--r-- 1 kuangfangjun root 1003K Mar 30 20:10 joiner-epoch-99-avg-1.onnx

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
    --tokens=./sherpa-onnx-zipformer-en-2023-03-30/tokens.txt \
    --encoder=./sherpa-onnx-zipformer-en-2023-03-30/encoder-epoch-99-avg-1.onnx \
    --decoder=./sherpa-onnx-zipformer-en-2023-03-30/decoder-epoch-99-avg-1.onnx \
    --joiner=./sherpa-onnx-zipformer-en-2023-03-30/joiner-epoch-99-avg-1.onnx \
    ./sherpa-onnx-zipformer-en-2023-03-30/test_wavs/0.wav \
    ./sherpa-onnx-zipformer-en-2023-03-30/test_wavs/1.wav \
    ./sherpa-onnx-zipformer-en-2023-03-30/test_wavs/8k.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

You should see the following output:

.. literalinclude:: ./code-zipformer/sherpa-onnx-zipformer-en-2023-03-30.txt

int8
^^^^

The following code shows how to use ``int8`` models to decode wave files:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-zipformer-en-2023-03-30/tokens.txt \
    --encoder=./sherpa-onnx-zipformer-en-2023-03-30/encoder-epoch-99-avg-1.int8.onnx \
    --decoder=./sherpa-onnx-zipformer-en-2023-03-30/decoder-epoch-99-avg-1.int8.onnx \
    --joiner=./sherpa-onnx-zipformer-en-2023-03-30/joiner-epoch-99-avg-1.int8.onnx \
    ./sherpa-onnx-zipformer-en-2023-03-30/test_wavs/0.wav \
    ./sherpa-onnx-zipformer-en-2023-03-30/test_wavs/1.wav \
    ./sherpa-onnx-zipformer-en-2023-03-30/test_wavs/8k.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

You should see the following output:

.. literalinclude:: ./code-zipformer/sherpa-onnx-zipformer-en-2023-03-30-int8.txt

Speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-microphone-offline \
    --tokens=./sherpa-onnx-zipformer-en-2023-03-30/tokens.txt \
    --encoder=./sherpa-onnx-zipformer-en-2023-03-30/encoder-epoch-99-avg-1.onnx \
    --decoder=./sherpa-onnx-zipformer-en-2023-03-30/decoder-epoch-99-avg-1.onnx \
    --joiner=./sherpa-onnx-zipformer-en-2023-03-30/joiner-epoch-99-avg-1.onnx
