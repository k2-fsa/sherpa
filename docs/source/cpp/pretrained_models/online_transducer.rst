.. _online_transducer_pretrained_models:

Online transducer models
========================

.. hint::

   We use the binary ``sherpa-online`` below for demonstration.
   You can replace ``sherpa-online`` with ``sherpa-online-websocket-server``
   and ``sherpa-online-microphone``.

.. hint::

   At present, only streaming transducer models from `icefall`_ are supported.

icefall
-------

This sections lists models trained using `icefall`_.


English
^^^^^^^

.. _icefall-asr-librispeech-streaming-zipformer-2023-05-17:

icefall-asr-librispeech-streaming-zipformer-2023-05-17
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  # This model is trained using LibriSpeech with zipformer transducer
  #
  # See https://github.com/k2-fsa/icefall/pull/1058
  #
  # normal-scaled model, number of model parameters: 66110931, i.e., 66.11 M
  #
  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/Zengwei/icefall-asr-librispeech-streaming-zipformer-2023-05-17
  cd icefall-asr-librispeech-streaming-zipformer-2023-05-17

  git lfs pull --include "exp/jit_script_chunk_16_left_128.pt"

  for m in greedy_search modified_beam_search fast_beam_search; do
    sherpa-online \
      --decoding-method=$m \
      --nn-model=./exp/jit_script_chunk_16_left_128.pt \
      --tokens=./data/lang_bpe_500/tokens.txt \
      ./test_wavs/1089-134686-0001.wav \
      ./test_wavs/1221-135766-0001.wav \
      ./test_wavs/1221-135766-0002.wav
  done

  # For fast_beam_search with LG
  sherpa-online \
    --decoding-method=fast_beam_search \
    --nn-model=./exp/jit_script_chunk_16_left_128.pt \
    --lg=./data/lang_bpe_500/LG.pt \
    --tokens=./data/lang_bpe_500/tokens.txt \
    ./test_wavs/1089-134686-0001.wav \
    ./test_wavs/1221-135766-0001.wav \
    ./test_wavs/1221-135766-0002.wav

.. _icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29:

icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  # This model is trained using LibriSpeech with streaming zipformer transducer
  #
  # See https://github.com/k2-fsa/icefall/pull/787
  #
  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/Zengwei/icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29
  cd icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29

  git lfs pull --include "exp/cpu_jit.pt"
  git lfs pull --include "data/lang_bpe_500/LG.pt"

  for m in greedy_search modified_beam_search fast_beam_search; do
    sherpa-online \
      --decoding-method=$m \
      --nn-model=./exp/cpu_jit.pt \
      --tokens=./data/lang_bpe_500/tokens.txt \
      ./test_wavs/1089-134686-0001.wav \
      ./test_wavs/1221-135766-0001.wav \
      ./test_wavs/1221-135766-0002.wav
  done

  # For fast_beam_search with LG
  sherpa-online \
    --decoding-method=fast_beam_search \
    --nn-model=./exp/cpu_jit.pt \
    --lg=./data/lang_bpe_500/LG.pt \
    --tokens=./data/lang_bpe_500/tokens.txt \
    ./test_wavs/1089-134686-0001.wav \
    ./test_wavs/1221-135766-0001.wav \
    ./test_wavs/1221-135766-0002.wav

icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  # This model is trained using LibriSpeech with ConvEmformer transducer
  #
  # See https://github.com/k2-fsa/icefall/pull/440
  #
  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/Zengwei/icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05
  cd icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05

  git lfs pull --include "exp/cpu-jit-epoch-30-avg-10-torch-1.10.0.pt"
  git lfs pull --include "data/lang_bpe_500/LG.pt"
  cd exp
  ln -sv cpu-jit-epoch-30-avg-10-torch-1.10.0.pt cpu_jit.pt
  cd ..

  for m in greedy_search modified_beam_search fast_beam_search; do
    sherpa-online \
      --decoding-method=$m \
      --nn-model=./exp/cpu_jit.pt \
      --tokens=./data/lang_bpe_500/tokens.txt \
      ./test_wavs/1089-134686-0001.wav \
      ./test_wavs/1221-135766-0001.wav \
      ./test_wavs/1221-135766-0002.wav
  done

  # For fast_beam_search with LG

  ./build/bin/sherpa-online \
    --decoding-method=fast_beam_search \
    --nn-model=./exp/cpu_jit.pt \
    --lg=./data/lang_bpe_500/LG.pt \
    --tokens=./data/lang_bpe_500/tokens.txt \
    ./test_wavs/1089-134686-0001.wav \
    ./test_wavs/1221-135766-0001.wav \
    ./test_wavs/1221-135766-0002.wav

icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  # This model is trained using LibriSpeech with LSTM transducer
  #
  # See https://github.com/k2-fsa/icefall/pull/558
  #
  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03
  cd icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03

  git lfs pull --include "exp/encoder_jit_trace-iter-468000-avg-16.pt"
  git lfs pull --include "exp/decoder_jit_trace-iter-468000-avg-16.pt"
  git lfs pull --include "exp/joiner_jit_trace-iter-468000-avg-16.pt"
  git lfs pull --include "data/lang_bpe_500/LG.pt"

  cd exp
  ln -sv encoder_jit_trace-iter-468000-avg-16.pt encoder_jit_trace.pt
  ln -sv decoder_jit_trace-iter-468000-avg-16.pt decoder_jit_trace.pt
  ln -sv joiner_jit_trace-iter-468000-avg-16.pt joiner_jit_trace.pt
  cd ..

  for m in greedy_search modified_beam_search fast_beam_search; do
    sherpa-online \
      --decoding-method=$m \
      --encoder-model=./exp/encoder_jit_trace.pt \
      --decoder-model=./exp/decoder_jit_trace.pt \
      --joiner-model=./exp/joiner_jit_trace.pt \
      --tokens=./data/lang_bpe_500/tokens.txt \
      ./test_wavs/1089-134686-0001.wav \
      ./test_wavs/1221-135766-0001.wav \
      ./test_wavs/1221-135766-0002.wav
  done

  # For fast_beam_search with LG
  sherpa-online \
    --decoding-method=fast_beam_search \
    --encoder-model=./exp/encoder_jit_trace.pt \
    --decoder-model=./exp/decoder_jit_trace.pt \
    --joiner-model=./exp/joiner_jit_trace.pt \
    --lg=./data/lang_bpe_500/LG.pt \
    --tokens=./data/lang_bpe_500/tokens.txt \
    ./test_wavs/1089-134686-0001.wav \
    ./test_wavs/1221-135766-0001.wav \
    ./test_wavs/1221-135766-0002.wav

icefall-asr-librispeech-pruned-stateless-emformer-rnnt2-2022-06-01
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  # This model is trained using LibriSpeech with Emformer transducer
  #
  # See https://github.com/k2-fsa/icefall/pull/390
  #
  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-stateless-emformer-rnnt2-2022-06-01
  cd icefall-asr-librispeech-pruned-stateless-emformer-rnnt2-2022-06-01

  git lfs pull --include "exp/cpu_jit-epoch-39-avg-6-use-averaged-model-1.pt"
  git lfs pull --include "data/lang_bpe_500/LG.pt"
  cd exp
  ln -sv cpu_jit-epoch-39-avg-6-use-averaged-model-1.pt cpu_jit.pt
  cd ..

  for m in greedy_search modified_beam_search fast_beam_search; do
    sherpa-online \
      --decoding-method=$m \
      --nn-model=./exp/cpu_jit.pt \
      --tokens=./data/lang_bpe_500/tokens.txt \
      ./test_wavs/1089-134686-0001.wav \
      ./test_wavs/1221-135766-0001.wav \
      ./test_wavs/1221-135766-0002.wav
  done

  # For fast_beam_search with LG

  sherpa-online \
    --decoding-method=fast_beam_search \
    --nn-model=./exp/cpu_jit.pt \
    --lg=./data/lang_bpe_500/LG.pt \
    --tokens=./data/lang_bpe_500/tokens.txt \
    ./test_wavs/1089-134686-0001.wav \
    ./test_wavs/1221-135766-0001.wav \
    ./test_wavs/1221-135766-0002.wav


icefall_librispeech_streaming_pruned_transducer_stateless4_20220625
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  # This model is trained using LibriSpeech with Conformer transducer
  #
  # See https://github.com/k2-fsa/icefall/pull/440
  #
  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/pkufool/icefall_librispeech_streaming_pruned_transducer_stateless4_20220625
  cd icefall_librispeech_streaming_pruned_transducer_stateless4_20220625

  git lfs pull --include "exp/cpu_jit-epoch-25-avg-3.pt"
  git lfs pull --include "data/lang_bpe_500/LG.pt"
  cd exp
  ln -sv cpu_jit-epoch-25-avg-3.pt cpu_jit.pt
  cd ..

  for m in greedy_search modified_beam_search fast_beam_search; do
    sherpa-online \
      --decoding-method=$m \
      --nn-model=./exp/cpu_jit.pt \
      --tokens=./data/lang_bpe_500/tokens.txt \
      ./test_waves/1089-134686-0001.wav \
      ./test_waves/1221-135766-0001.wav \
      ./test_waves/1221-135766-0002.wav
  done

  # For fast_beam_search with LG

  sherpa-online \
    --decoding-method=fast_beam_search \
    --nn-model=./exp/cpu_jit.pt \
    --lg=./data/lang_bpe_500/LG.pt \
    --tokens=./data/lang_bpe_500/tokens.txt \
    ./test_waves/1089-134686-0001.wav \
    ./test_waves/1221-135766-0001.wav \
    ./test_waves/1221-135766-0002.wav

Chinese
^^^^^^^

icefall_asr_wenetspeech_pruned_transducer_stateless5_streaming
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  # This model is trained using WenetSpeech with Conformer transducer
  #
  # See https://github.com/k2-fsa/icefall/pull/447
  #
  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/luomingshuang/icefall_asr_wenetspeech_pruned_transducer_stateless5_streaming
  cd icefall_asr_wenetspeech_pruned_transducer_stateless5_streaming

  git lfs pull --include "exp/cpu_jit_epoch_7_avg_1_torch.1.7.1.pt"
  git lfs pull --include "data/lang_char/LG.pt"
  cd exp
  ln -sv cpu_jit_epoch_7_avg_1_torch.1.7.1.pt cpu_jit.pt
  cd ..

  for m in greedy_search modified_beam_search fast_beam_search; do
    sherpa-online \
      --decoding-method=$m \
      --nn-model=./exp/cpu_jit.pt \
      --tokens=./data/lang_char/tokens.txt \
      ./test_wavs/DEV_T0000000000.wav \
      ./test_wavs/DEV_T0000000001.wav \
      ./test_wavs/DEV_T0000000002.wav
  done

  # For fast_beam_search with LG

  sherpa-online \
    --decoding-method=fast_beam_search \
    --nn-model=./exp/cpu_jit.pt \
    --lg=./data/lang_char/LG.pt \
    --tokens=./data/lang_char/tokens.txt \
    ./test_wavs/DEV_T0000000000.wav \
    ./test_wavs/DEV_T0000000001.wav \
    ./test_wavs/DEV_T0000000002.wav

Chinese + English (all-in-one)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

pfluo/k2fsa-zipformer-chinese-english-mixed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is a `streaming zipformer model <https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/pruned_transducer_stateless7_streaming>`_

.. code-block:: bash

  # This model supports both Chinese and English
  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/pfluo/k2fsa-zipformer-chinese-english-mixed
  cd k2fsa-zipformer-chinese-english-mixed
  git lfs pull --include "exp/cpu_jit.pt"

  for m in greedy_search modified_beam_search fast_beam_search; do
    sherpa-online \
      --decoding-method=$m \
      --nn-model=./exp/cpu_jit.pt \
      --tokens=./data/lang_char_bpe/tokens.txt \
      ./test_wavs/0.wav \
      ./test_wavs/1.wav \
      ./test_wavs/2.wav \
      ./test_wavs/3.wav \
      ./test_wavs/4.wav
  done

icefall-asr-conv-emformer-transducer-stateless2-zh
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is a `ConvEmformer model <https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/conv_emformer_transducer_stateless2>`_

.. code-block:: bash

  # This model supports both Chinese and English
  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/ptrnull/icefall-asr-conv-emformer-transducer-stateless2-zh
  cd icefall-asr-conv-emformer-transducer-stateless2-zh
  git lfs pull --include "exp/cpu_jit-epoch-11-avg-1.pt"
  cd exp
  ln -sv cpu_jit-epoch-11-avg-1.pt cpu_jit.pt
  cd ..

  for m in greedy_search modified_beam_search fast_beam_search; do
    sherpa-online \
      --decoding-method=$m \
      --nn-model=./exp/cpu_jit.pt \
      --tokens=./data/lang_char_bpe/tokens.txt \
      ./test_wavs/0.wav \
      ./test_wavs/1.wav \
      ./test_wavs/2.wav \
      ./test_wavs/3.wav \
      ./test_wavs/4.wav
  done
