Transducer
==========


In this section, we describe how to use pre-trained `transducer`_
models for online (i.e., streaming) speech recognition.

.. hint::

  Please refer to :ref:`online_transducer_pretrained_models` for a list of
  available pre-trained `transducer`_ models to download.

In the following, we use the pre-trained model
:ref:`icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29`
to demonstrate how to decode sound files.

.. caution::

   Make sure you have installed `sherpa`_ before you continue.

   Please refer to :ref:`install_sherpa_from_source` to install `sherpa`_
   from source.

Download the pre-trained model
------------------------------

Please refer to :ref:`icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29`
for detailed instructions.

For ease of reference, we duplicate the download commands below:

.. code-block:: bash

  # This model is trained using LibriSpeech with streaming zipformer transducer
  #
  # See https://github.com/k2-fsa/icefall/pull/787
  #
  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/Zengwei/icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29
  cd icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29

  git lfs pull --include "exp/cpu_jit.pt"
  git lfs pull --include "data/lang_bpe_500/LG.pt"


In the following, we describe different decoding methods.

greedy search
-------------

.. code-block:: bash

    cd /path/to/sherpa

    python3 ./sherpa/bin/online_transducer_asr.py \
      --decoding-method="greedy_search" \
      --nn-model=./icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/exp/cpu_jit.pt \
      --tokens=./icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/data/lang_bpe_500/tokens.txt \
      ./icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/test_wavs/1089-134686-0001.wav \
      ./icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/test_wavs/1221-135766-0001.wav \
      ./icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/test_wavs/1221-135766-0002.wav

modified beam search
--------------------

.. code-block:: bash

    cd /path/to/sherpa

    python3 ./sherpa/bin/online_transducer_asr.py \
      --decoding-method="modified_beam_search" \
      --nn-model=./icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/exp/cpu_jit.pt \
      --tokens=./icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/data/lang_bpe_500/tokens.txt \
      ./icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/test_wavs/1089-134686-0001.wav \
      ./icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/test_wavs/1221-135766-0001.wav \
      ./icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/test_wavs/1221-135766-0002.wav

fast_beam_search
----------------

.. code-block:: bash

    cd /path/to/sherpa

    python3 ./sherpa/bin/online_transducer_asr.py \
      --decoding-method="fast_beam_search" \
      --nn-model=./icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/exp/cpu_jit.pt \
      --tokens=./icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/data/lang_bpe_500/tokens.txt \
      ./icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/test_wavs/1089-134686-0001.wav \
      ./icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/test_wavs/1221-135766-0001.wav \
      ./icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/test_wavs/1221-135766-0002.wav

fast_beam_search with LG
------------------------

.. code-block:: bash

    cd /path/to/sherpa

    python3 ./sherpa/bin/online_transducer_asr.py \
      --decoding-method="fast_beam_search" \
      --LG=./icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/data/lang_bpe_500/LG.pt \
      --nn-model=./icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/exp/cpu_jit.pt \
      --tokens=./icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/data/lang_bpe_500/tokens.txt \
      ./icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/test_wavs/1089-134686-0001.wav \
      ./icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/test_wavs/1221-135766-0001.wav \
      ./icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/test_wavs/1221-135766-0002.wav
