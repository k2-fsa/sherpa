WeNet
=====

This section lists models from `WeNet`_.

wenet-english-model (English)
-----------------------------

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

wenet-chinese-model (Chinese)
-----------------------------

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
