.. _sherpa_onnx_offline_zipformer_transducer_models:

Zipformer-transducer-based Models
=================================

.. hint::

   Please refer to :ref:`install_sherpa_onnx` to install `sherpa-onnx`_
   before you read this section.

.. _icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04-english:

icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04 (Englis)
-------------------------------------------------------------------------

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
    --joiner=./icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04/exp/joiner-epoch-30-avg-4.onnx \

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
    --joiner=./sherpa-onnx-zipformer-en-2023-04-01/joiner-epoch-99-avg-1.onnx \

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
    --joiner=./sherpa-onnx-zipformer-en-2023-03-30/joiner-epoch-99-avg-1.onnx \
