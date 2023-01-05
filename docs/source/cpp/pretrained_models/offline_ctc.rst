Offline CTC models
==================

.. hint::

   We use the binary ``sherpa-offline`` below for demonstration.
   You can replace ``sherpa-offline`` with ``sherpa-offline-websocket-server``.

icefall
-------

This sections lists models trained using `icefall`_.

English
^^^^^^^

icefall-asr-gigaspeech-conformer-ctc
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

icefall-asr-tedlium3-conformer-ctc2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

icefall_asr_librispeech_conformer_ctc
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Chinese
^^^^^^^

icefall_asr_aishell_conformer_ctc
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Arabic
^^^^^^

icefall-asr-mgb2-conformer_ctc-2022-27-06
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

wenet
-----

This section lists models from `wenet`_.

English
^^^^^^^

wenet-english-model
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/wenet-english-model
   cd wenet-english-model
   git lfs pull --include "final.zip"

   sherpa-offline \
    --normalize-samples=false \
    --modified=true \
    --nn-model=./final.zip \
    --tokens=./units.txt \
    --use-gpu=false \
    ./test_wavs/1089-134686-0001.wav \
    ./test_wavs/1221-135766-0001.wav \
    ./test_wavs/1221-135766-0002.wav

Chinese
^^^^^^^

wenet-chinese-model
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/wenet-chinese-model
   cd wenet-chinese-model
   git lfs pull --include "final.zip"

   sherpa-offline \
     --normalize-samples=false \
     --modified=true \
     --nn-model=./final.zip \
     --tokens=./units.txt \
     ./test_wavs/BAC009S0764W0121.wav \
     ./test_wavs/BAC009S0764W0122.wav \
     ./test_wavs/BAC009S0764W0123.wav \
     ./test_wavs/DEV_T0000000000.wav \
     ./test_wavs/DEV_T0000000001.wav \
     ./test_wavs/DEV_T0000000002.wav

torchaudio
----------

This section lists models from `torchaudio`_.

wav2vec2_asr_base
~~~~~~~~~~~~~~~~~

English
^^^^^^^

.. code-block:: bash

   GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/wav2vec2.0-torchaudio
   cd wav2vec2.0-torchaudio

   # Note: There are other kinds of models fine-tuned with different
   # amount of data. We use a model that is fine-tuned with 10 minutes of data.

   git lfs pull --include "wav2vec2_asr_base_10m.pt"

   sherpa-offline \
    --nn-model=wav2vec2_asr_base_10m.pt \
    --tokens=tokens.txt \
    --use-gpu=false \
    ./test_wavs/1089-134686-0001.wav \
    ./test_wavs/1221-135766-0001.wav \
    ./test_wavs/1221-135766-0002.wav

German
^^^^^^

voxpopuli_asr_base
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/wav2vec2.0-torchaudio
   cd wav2vec2.0-torchaudio
   git lfs pull --include "voxpopuli_asr_base_10k_de.pt"

   sherpa-offline \
    --nn-model=voxpopuli_asr_base_10k_de.pt \
    --tokens=tokens-de.txt \
    --use-gpu=false \
    ./test_wavs/20120315-0900-PLENARY-14-de_20120315.wav \
    ./test_wavs/20170517-0900-PLENARY-16-de_20170517.wav
