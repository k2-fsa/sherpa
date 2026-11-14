icefall
=======

.. hint::

   We use the binary ``sherpa-offline`` below for demonstration.
   You can replace ``sherpa-offline`` with ``sherpa-offline-websocket-server``.

In this section, we list all pre-trained CTC models from `icefall`_.

icefall-asr-gigaspeech-conformer-ctc (English)
----------------------------------------------

.. code-block:: bash

  # This model is trained using GigaSpeech
  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/wgb14/icefall-asr-gigaspeech-conformer-ctc
  cd icefall-asr-gigaspeech-conformer-ctc
  git lfs pull --include "exp/cpu_jit.pt"
  git lfs pull --include "data/lang_bpe_500/HLG.pt"
  git lfs pull --include "data/lang_bpe_500/tokens.txt"
  mkdir test_wavs
  cd test_wavs
  wget https://huggingface.co/csukuangfj/wav2vec2.0-torchaudio/resolve/main/test_wavs/1089-134686-0001.wav
  wget https://huggingface.co/csukuangfj/wav2vec2.0-torchaudio/resolve/main/test_wavs/1221-135766-0001.wav
  wget https://huggingface.co/csukuangfj/wav2vec2.0-torchaudio/resolve/main/test_wavs/1221-135766-0002.wav
  cd ..

  # Decode with H
  sherpa-offline \
    --nn-model=./exp/cpu_jit.pt \
    --tokens=./data/lang_bpe_500/tokens.txt \
    ./test_wavs/1089-134686-0001.wav \
    ./test_wavs/1221-135766-0001.wav \
    ./test_wavs/1221-135766-0002.wav

  # Decode with HLG
  sherpa-offline \
    --nn-model=./exp/cpu_jit.pt \
    --hlg=./data/lang_bpe_500/HLG.pt \
    --tokens=./data/lang_bpe_500/tokens.txt \
    ./test_wavs/1089-134686-0001.wav \
    ./test_wavs/1221-135766-0001.wav \
    ./test_wavs/1221-135766-0002.wav

icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09 (English)
----------------------------------------------------------------------

.. code-block:: bash

  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09
  cd icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09

  git lfs pull --include "exp/cpu_jit.pt"
  git lfs pull --include "data/lang_bpe_500/tokens.txt"
  git lfs pull --include "data/lang_bpe_500/HLG.pt"
  git lfs pull --include "data/lang_bpe_500/HLG_modified.pt"

  # Decode with H
  sherpa-offline \
    --nn-model=./exp/cpu_jit.pt \
    --tokens=./data/lang_bpe_500/tokens.txt \
    --use-gpu=false \
    ./test_wavs/1089-134686-0001.wav \
    ./test_wavs/1221-135766-0001.wav \
    ./test_wavs/1221-135766-0002.wav

  # Decode with HLG
  sherpa-offline \
    --nn-model=./exp/cpu_jit.pt \
    --tokens=./data/lang_bpe_500/tokens.txt \
    --hlg=./data/lang_bpe_500/HLG.pt \
    --use-gpu=false \
    ./test_wavs/1089-134686-0001.wav \
    ./test_wavs/1221-135766-0001.wav \
    ./test_wavs/1221-135766-0002.wav

  # Decode with HLG (modified)
  sherpa-offline \
    --nn-model=./exp/cpu_jit.pt \
    --tokens=./data/lang_bpe_500/tokens.txt \
    --hlg=./data/lang_bpe_500/HLG_modified.pt \
    --use-gpu=false \
    ./test_wavs/1089-134686-0001.wav \
    ./test_wavs/1221-135766-0001.wav \
    ./test_wavs/1221-135766-0002.wav

icefall-asr-tedlium3-conformer-ctc2 (English)
---------------------------------------------

.. code-block:: bash

   # This model is trained using Tedlium3
   #
   # See https://github.com/k2-fsa/icefall/pull/696
   #

   GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/videodanchik/icefall-asr-tedlium3-conformer-ctc2
   cd icefall-asr-tedlium3-conformer-ctc2
   git lfs pull --include "exp/cpu_jit.pt"

   git lfs pull --include "data/lang_bpe/HLG.pt"
   git lfs pull --include "data/lang_bpe/tokens.txt"

   git lfs pull --include "test_wavs/DanBarber_2010-219.wav"
   git lfs pull --include "test_wavs/DanielKahneman_2010-157.wav"
   git lfs pull --include "test_wavs/RobertGupta_2010U-15.wav"

   # Decode with H
   sherpa-offline \
     --nn-model=./exp/cpu_jit.pt \
     --tokens=./data/lang_bpe/tokens.txt \
     ./test_wavs/DanBarber_2010-219.wav \
     ./test_wavs/DanielKahneman_2010-157.wav \
     ./test_wavs/RobertGupta_2010U-15.wav

   # Decode with HLG
   sherpa-offline \
     --nn-model=./exp/cpu_jit.pt \
     --hlg=./data/lang_bpe/HLG.pt \
     --tokens=./data/lang_bpe/tokens.txt \
     ./test_wavs/DanBarber_2010-219.wav \
     ./test_wavs/DanielKahneman_2010-157.wav \
     ./test_wavs/RobertGupta_2010U-15.wav

icefall_asr_librispeech_conformer_ctc (English)
-----------------------------------------------

.. code-block:: bash

   # This model is trained using LibriSpeech
   #
   # See https://github.com/k2-fsa/icefall/pull/13
   #

   GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/pkufool/icefall_asr_librispeech_conformer_ctc
   cd icefall_asr_librispeech_conformer_ctc

   git lfs pull --include "exp/cpu_jit.pt"
   git lfs pull --include "data/lang_bpe/HLG.pt"

   # Decode with H
   sherpa-offline \
     --nn-model=./exp/cpu_jit.pt \
     --tokens=./data/lang_bpe/tokens.txt \
     ./test_wavs/1089-134686-0001.wav \
     ./test_wavs/1221-135766-0001.wav \
     ./test_wavs/1221-135766-0002.wav

   # Decode with HLG
   sherpa-offline \
     --nn-model=./exp/cpu_jit.pt \
     --hlg=./data/lang_bpe/HLG.pt \
     --tokens=./data/lang_bpe/tokens.txt \
     ./test_wavs/1089-134686-0001.wav \
     ./test_wavs/1221-135766-0001.wav \
     ./test_wavs/1221-135766-0002.wav

.. - `<https://huggingface.co/WayneWiser/icefall-asr-librispeech-conformer-ctc2-jit-bpe-500-2022-07-21>`_

icefall_asr_aishell_conformer_ctc (Chinese)
-------------------------------------------

.. code-block:: bash

  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/pkufool/icefall_asr_aishell_conformer_ctc
  cd icefall_asr_aishell_conformer_ctc
  git lfs pull --include "exp/cpu_jit.pt"
  git lfs pull --include "data/lang_char/HLG.pt"

  # Decode with an H graph
  sherpa-offline \
    --nn-model=./exp/cpu_jit.pt \
    --tokens=./data/lang_char/tokens.txt \
    ./test_waves/BAC009S0764W0121.wav \
    ./test_waves/BAC009S0764W0122.wav \
    ./test_waves/BAC009S0764W0123.wav

  # Decode with an HLG graph
  sherpa-offline \
    --nn-model=./exp/cpu_jit.pt \
    --tokens=./data/lang_char/tokens.txt \
    --hlg=./data/lang_char/HLG.pt \
    ./test_waves/BAC009S0764W0121.wav \
    ./test_waves/BAC009S0764W0122.wav \
    ./test_waves/BAC009S0764W0123.wav


icefall-asr-mgb2-conformer_ctc-2022-27-06 (Arabic)
--------------------------------------------------

.. code-block:: bash

  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/AmirHussein/icefall-asr-mgb2-conformer_ctc-2022-27-06
  cd icefall-asr-mgb2-conformer_ctc-2022-27-06
  git lfs pull --include "exp/cpu_jit.pt"
  git lfs pull --include "data/lang_bpe_5000/HLG.pt"
  git lfs pull --include "data/lang_bpe_5000/tokens.txt"

  # Decode with an H graph
  sherpa-offline \
    --nn-model=./exp/cpu_jit.pt \
    --tokens=./data/lang_bpe_5000/tokens.txt \
    ./test_wavs/94D37D38-B203-4FC0-9F3A-538F5C174920_spk-0001_seg-0053813:0054281.wav \
    ./test_wavs/94D37D38-B203-4FC0-9F3A-538F5C174920_spk-0001_seg-0051454:0052244.wav \
    ./test_wavs/94D37D38-B203-4FC0-9F3A-538F5C174920_spk-0001_seg-0052244:0053004.wav

  # Decode with an HLG graph
  sherpa-offline \
    --nn-model=./exp/cpu_jit.pt \
    --tokens=./data/lang_bpe_5000/tokens.txt \
    --hlg=./data/lang_bpe_5000/HLG.pt \
    ./test_wavs/94D37D38-B203-4FC0-9F3A-538F5C174920_spk-0001_seg-0053813:0054281.wav \
    ./test_wavs/94D37D38-B203-4FC0-9F3A-538F5C174920_spk-0001_seg-0051454:0052244.wav \
    ./test_wavs/94D37D38-B203-4FC0-9F3A-538F5C174920_spk-0001_seg-0052244:0053004.wav
