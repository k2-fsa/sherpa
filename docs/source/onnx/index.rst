sherpa-onnx
===========

.. hint::

  A colab notebook is provided for you so that you can try `sherpa-onnx`_
  in the browser.

  |sherpa-onnx colab notebook|

  .. |sherpa-onnx colab notebook| image:: https://colab.research.google.com/assets/colab-badge.svg
     :target: https://colab.research.google.com/drive/1tmQbdlYeTl_klmtaGiUb7a7ZPz-AkBSH?usp=sharing


We support using `onnx`_ with `onnxruntime`_ to replace `PyTorch`_ for neural
network computation. The code is put in a separate repository `sherpa-onnx`_.

`sherpa-onnx`_ is self-contained and everything can be compiled from source.

.. hint::

   We use pre-compiled `onnxruntime`_ from
   `<https://github.com/microsoft/onnxruntime/releases>`_.

Please refer to
`<https://k2-fsa.github.io/icefall/model-export/export-onnx.html>`_
for how to export models to `onnx`_ format.

In the following, we describe how to build `sherpa-onnx`_ on Linux, macOS,
and Windows. Also, we show how to use it for speech recognition with
pretrained models.

.. caution::

   We only provide support for non-streaming conformer transducer at present.
   The work for streaming ASR is still on-going.

Build sherpa-onnx for Linux and macOS
-------------------------------------

.. code-block:: bash

  git clone https://github.com/k2-fsa/sherpa-onnx
  cd sherpa-onnx
  mkdir build
  cd build
  cmake -DCMAKE_BUILD_TYPE=Release ..
  make -j6

It will generate a binary ``sherpa-onnx`` inside ``./build/bin/``.

.. note::

   Please read below to see how to use the generated binary for speech
   recognition with pretrained models.

Build sherpa-onnx for Windows
-----------------------------

.. code-block:: bash

  git clone https://github.com/k2-fsa/sherpa-onnx
  cd sherpa-onnx
  mkdir build
  cd build
  cmake -DCMAKE_BUILD_TYPE=Release ..
  cmake --build . --config Release

It will generate a binary ``sherpa-onnx.exe`` inside ``./build/bin/Release/``.

.. note::

   Please read below to see how to use the generated binary for speech
   recognition with pretrained models.


Speech recognition with sherpa-onnx
-----------------------------------

In the following, we describe how to use the precompiled binary ``sherpa-onnx``
for offline speech recognition with pre-trained models.

We provide two examples: One is for English and the other is for Chinese.

English
^^^^^^^

First, let us download the pretrained model:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13
  cd icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13
  git lfs pull --include "exp/onnx/*.onnx"

  cd ..

.. caution::

   You have to use ``git lfs`` to download the pretrained models.

Second, we can use the following command to decode a wave file:

.. code-block:: bash

  ./build/bin/sherpa-onnx \
    ./icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13/data/lang_bpe_500/tokens.txt \
    ./icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13/exp/onnx/encoder.onnx \
    ./icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13/exp/onnx/decoder.onnx \
    ./icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13/exp/onnx/joiner.onnx \
    ./icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13/exp/onnx/joiner_encoder_proj.onnx \
    ./icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13/exp/onnx/joiner_decoder_proj.onnx \
    ./icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13/test_wavs/1089-134686-0001.wav

.. caution::

   It supports only wave format and its sampling rate has be to 16 kHz.

.. hint::

   If you are using Windows, please replace ``./build/bin/sherpa-onnx``
   with ``./build/bin/Release/sherpa-onnx``

.. note::

   Please refer to
   `<https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/RESULTS.md#librispeech-bpe-training-results-pruned-stateless-transducer-3-2022-04-29>`_
   and
   `<https://k2-fsa.github.io/icefall/model-export/export-onnx.html>`_
   if you are interested in how the model is trained and exported.

Chinese
^^^^^^^

First, let us download the pretrained model:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/luomingshuang/icefall_asr_wenetspeech_pruned_transducer_stateless2
  cd icefall_asr_wenetspeech_pruned_transducer_stateless2
  git lfs pull --include "exp/*.onnx"

  cd ..

.. caution::

   You have to use ``git lfs`` to download the pretrained models.

Second, we can use the following command to decode a wave file:

.. code-block:: bash

  ./build/bin/sherpa-onnx \
    ./icefall_asr_wenetspeech_pruned_transducer_stateless2/data/lang_char/tokens.txt \
    ./icefall_asr_wenetspeech_pruned_transducer_stateless2/exp/encoder-epoch-10-avg-2.onnx \
    ./icefall_asr_wenetspeech_pruned_transducer_stateless2/exp/decoder-epoch-10-avg-2.onnx \
    ./icefall_asr_wenetspeech_pruned_transducer_stateless2/exp/joiner-epoch-10-avg-2.onnx \
    ./icefall_asr_wenetspeech_pruned_transducer_stateless2/exp/joiner_encoder_proj-epoch-10-avg-2.onnx \
    ./icefall_asr_wenetspeech_pruned_transducer_stateless2/exp/joiner_decoder_proj-epoch-10-avg-2.onnx \
    ./icefall_asr_wenetspeech_pruned_transducer_stateless2/test_wavs/DEV_T0000000000.wav

.. caution::

   It supports only wave format and its sampling rate has be to 16 kHz.

.. hint::

   If you are using Windows, please replace ``./build/bin/sherpa-onnx``
   with ``./build/bin/Release/sherpa-onnx``

.. note::

   Please refer to
   `<https://github.com/k2-fsa/icefall/blob/master/egs/wenetspeech/ASR/RESULTS.md#2022-05-19>`_
   and
   `<https://k2-fsa.github.io/icefall/model-export/export-onnx.html>`_
   if you are interested in how the model is trained and exported.
