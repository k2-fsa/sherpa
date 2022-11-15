Offline transducer models
=========================

.. hint::

   We use the binary ``sherpa-offline`` below for demonstration.
   You can replace ``sherpa-offline`` with ``sherpa-offline-websocket-server``.

icefall
-------

This sections lists models trained using `icefall`_.

English
^^^^^^^

icefall-asr-librispeech-pruned-transducer-stateless8-2022-11-14
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  # This model is trained using GigaSpeech + LibriSpeech with zipformer
  #
  # See https://github.com/k2-fsa/icefall/pull/675
  #
  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless8-2022-11-14
  cd icefall-asr-librispeech-pruned-transducer-stateless8-2022-11-14
  git lfs pull --include "exp/cpu_jit.pt"

  sherpa-offline \
    --nn-model=./exp/cpu_jit.pt \
    --tokens=./data/lang_bpe_500/tokens.txt \
    ./test_wavs/1089-134686-0001.wav \
    ./test_wavs/1221-135766-0001.wav \
    ./test_wavs/1221-135766-0002.wav

  sherpa-offline \
    --nn-model=./exp/cpu_jit.pt \
    --tokens=./data/lang_bpe_500/tokens.txt \
    --decoding-method=modified_beam_search \
    ./test_wavs/1089-134686-0001.wav \
    ./test_wavs/1221-135766-0001.wav \
    ./test_wavs/1221-135766-0002.wav

icefall-asr-librispeech-pruned-transducer-stateless7-2022-11-11
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  # This model is trained using LibriSpeech with zipformer
  #
  # See https://github.com/k2-fsa/icefall/pull/672
  #
  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless7-2022-11-11
  cd icefall-asr-librispeech-pruned-transducer-stateless7-2022-11-11
  git lfs pull --include "exp/cpu_jit-torch-1.10.0.pt"

  sherpa-offline \
    --nn-model=./exp/cpu_jit-torch-1.10.0.pt \
    --tokens=./data/lang_bpe_500/tokens.txt \
    ./test_wavs/1089-134686-0001.wav \
    ./test_wavs/1221-135766-0001.wav \
    ./test_wavs/1221-135766-0002.wav

icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block::

  # This model is trained using LibriSpeech + GigaSpeech
  #
  # See https://github.com/k2-fsa/icefall/pull/363
  #
  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13
  cd icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13
  git lfs pull --include "exp/cpu_jit.pt"

  sherpa-offline \
    --nn-model=./exp/cpu_jit.pt \
    --tokens=./data/lang_bpe_500/tokens.txt \
    ./test_wavs/1089-134686-0001.wav \
    ./test_wavs/1221-135766-0001.wav \
    ./test_wavs/1221-135766-0002.wav

icefall-asr-gigaspeech-pruned-transducer-stateless2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block::

   # This model is trained using GigaSpeech
   #
   # See https://github.com/k2-fsa/icefall/pull/318
   #
   GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/wgb14/icefall-asr-gigaspeech-pruned-transducer-stateless2
   cd icefall-asr-gigaspeech-pruned-transducer-stateless2
   git lfs pull --include "exp/cpu_jit-iter-3488000-avg-15.pt"
   git lfs pull --include "data/lang_bpe_500/bpe.model"

   # Since this repo does not provide tokens.txt, we generate it from bpe.model
   # by ourselves
   /path/to/sherpa/scripts/bpe_model_to_tokens.py ./data/lang_bpe_500/bpe.model > ./data/lang_bpe_500/tokens.txt

   mkdir test_wavs
   cd test_wavs
   wget https://huggingface.co/csukuangfj/wav2vec2.0-torchaudio/resolve/main/test_wavs/1089-134686-0001.wav
   wget https://huggingface.co/csukuangfj/wav2vec2.0-torchaudio/resolve/main/test_wavs/1221-135766-0001.wav
   wget https://huggingface.co/csukuangfj/wav2vec2.0-torchaudio/resolve/main/test_wavs/1221-135766-0002.wav

   sherpa-offline \
      --nn-model=./exp/cpu_jit-iter-3488000-avg-15.pt \
      --tokens=./data/lang_bpe_500/tokens.txt \
      ./test_wavs/1089-134686-0001.wav \
      ./test_wavs/1221-135766-0001.wav \
      ./test_wavs/1221-135766-0002.wav

Chinese
^^^^^^^

icefall_asr_wenetspeech_pruned_transducer_stateless2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # This models is trained using WenetSpeech
   #
   # See https://github.com/k2-fsa/icefall/pull/349
   #
   GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/luomingshuang/icefall_asr_wenetspeech_pruned_transducer_stateless2

   cd icefall_asr_wenetspeech_pruned_transducer_stateless2
   git lfs pull --include "exp/cpu_jit_epoch_10_avg_2_torch_1.7.1.pt"

   sherpa-offline \
      --nn-model=./exp/cpu_jit_epoch_10_avg_2_torch_1.7.1.pt \
      --tokens=./data/lang_char/tokens.txt \
      ./test_wavs/DEV_T0000000000.wav \
      ./test_wavs/DEV_T0000000001.wav \
      ./test_wavs/DEV_T0000000002.wav
