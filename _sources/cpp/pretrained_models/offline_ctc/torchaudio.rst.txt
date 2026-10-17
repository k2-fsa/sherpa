torchaudio
==========

This section lists models from `torchaudio`_.


wav2vec2_asr_base (English)
---------------------------

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

voxpopuli_asr_base (German)
---------------------------

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
