.. _offline_transducer_pretrained_models:

Offline transducer models
=========================

.. hint::

   We use the binary ``sherpa-offline`` below for demonstration.
   You can replace ``sherpa-offline`` with ``sherpa-offline-websocket-server``.

.. hint::

   Please visit `<https://huggingface.co/spaces/k2-fsa/automatic-speech-recognition>`_
   to try the pre-trained models in your browser. You don't need to install
   anything.

icefall
-------

This sections lists models trained using `icefall`_.

English
^^^^^^^

.. _icefall-asr-librispeech-zipformer-2023-05-15:

icefall-asr-librispeech-zipformer-2023-05-15
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  # This model is trained using LibriSpeech with zipformer transducer
  #
  # See https://github.com/k2-fsa/icefall/pull/1058
  #
  # normal-scaled model, number of model parameters: 65549011, i.e., 65.55 M
  #
  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/Zengwei/icefall-asr-librispeech-zipformer-2023-05-15
  cd icefall-asr-librispeech-zipformer-2023-05-15

  git lfs pull --include "exp/jit_script.pt"
  git lfs pull --include "data/lang_bpe_500/LG.pt"

  for m in greedy_search modified_beam_search fast_beam_search; do
    sherpa-offline \
      --decoding-method=$m \
      --nn-model=./exp/jit_script.pt \
      --tokens=./data/lang_bpe_500/tokens.txt \
      ./test_wavs/1089-134686-0001.wav \
      ./test_wavs/1221-135766-0001.wav \
      ./test_wavs/1221-135766-0002.wav
  done

  sherpa-offline \
    --decoding-method=fast_beam_search \
    --nn-model=./exp/jit_script.pt \
    --lg=./data/lang_bpe_500/LG.pt \
    --tokens=./data/lang_bpe_500/tokens.txt \
    ./test_wavs/1089-134686-0001.wav \
    ./test_wavs/1221-135766-0001.wav \
    ./test_wavs/1221-135766-0002.wav

.. _icefall-asr-librispeech-zipformer-small-2023-05-16:

icefall-asr-librispeech-zipformer-small-2023-05-16
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  # This model is trained using LibriSpeech with zipformer transducer
  #
  # See https://github.com/k2-fsa/icefall/pull/1058
  #
  # small-scaled model, number of model parameters: 23285615, i.e., 23.3 M
  #
  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/Zengwei/icefall-asr-librispeech-zipformer-small-2023-05-16
  cd icefall-asr-librispeech-zipformer-small-2023-05-16

  git lfs pull --include "exp/jit_script.pt"
  git lfs pull --include "data/lang_bpe_500/LG.pt"

  for m in greedy_search modified_beam_search fast_beam_search; do
    sherpa-offline \
      --decoding-method=$m \
      --nn-model=./exp/jit_script.pt \
      --tokens=./data/lang_bpe_500/tokens.txt \
      ./test_wavs/1089-134686-0001.wav \
      ./test_wavs/1221-135766-0001.wav \
      ./test_wavs/1221-135766-0002.wav
  done

  sherpa-offline \
    --decoding-method=fast_beam_search \
    --nn-model=./exp/jit_script.pt \
    --lg=./data/lang_bpe_500/LG.pt \
    --tokens=./data/lang_bpe_500/tokens.txt \
    ./test_wavs/1089-134686-0001.wav \
    ./test_wavs/1221-135766-0001.wav \
    ./test_wavs/1221-135766-0002.wav


.. _icefall-asr-librispeech-zipformer-large-2023-05-16:

icefall-asr-librispeech-zipformer-large-2023-05-16
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  # This model is trained using LibriSpeech with zipformer transducer
  #
  # See https://github.com/k2-fsa/icefall/pull/1058
  #
  # large-scaled model, number of model parameters: 148439574, i.e., 148.4 M
  #
  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/Zengwei/icefall-asr-librispeech-zipformer-large-2023-05-16
  cd icefall-asr-librispeech-zipformer-large-2023-05-16

  git lfs pull --include "exp/jit_script.pt"
  git lfs pull --include "data/lang_bpe_500/LG.pt"

  for m in greedy_search modified_beam_search fast_beam_search; do
    sherpa-offline \
      --decoding-method=$m \
      --nn-model=./exp/jit_script.pt \
      --tokens=./data/lang_bpe_500/tokens.txt \
      ./test_wavs/1089-134686-0001.wav \
      ./test_wavs/1221-135766-0001.wav \
      ./test_wavs/1221-135766-0002.wav
  done

  sherpa-offline \
    --decoding-method=fast_beam_search \
    --nn-model=./exp/jit_script.pt \
    --lg=./data/lang_bpe_500/LG.pt \
    --tokens=./data/lang_bpe_500/tokens.txt \
    ./test_wavs/1089-134686-0001.wav \
    ./test_wavs/1221-135766-0001.wav \
    ./test_wavs/1221-135766-0002.wav

.. _icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04:

icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  # This model is trained using GigaSpeech + LibriSpeech + Common Voice 13.0 with zipformer
  #
  # See https://github.com/k2-fsa/icefall/pull/1010
  #
  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/yfyeung/icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04
  cd icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04
  git lfs pull --include "exp/cpu_jit-epoch-30-avg-4.pt"
  cd exp
  ln -s cpu_jit-epoch-30-avg-4.pt cpu_jit.pt
  cd ..

  for m in greedy_search modified_beam_search fast_beam_search; do
    sherpa-offline \
      --decoding-method=$m \
      --nn-model=./exp/cpu_jit.pt \
      --tokens=./data/lang_bpe_500/tokens.txt \
      ./test_wavs/1089-134686-0001.wav \
      ./test_wavs/1221-135766-0001.wav \
      ./test_wavs/1221-135766-0002.wav
  done

.. _icefall-asr-librispeech-pruned-transducer-stateless8-2022-12-02:

icefall-asr-librispeech-pruned-transducer-stateless8-2022-12-02
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  # This model is trained using GigaSpeech + LibriSpeech with zipformer
  #
  # See https://github.com/k2-fsa/icefall/pull/728
  #
  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/WeijiZhuang/icefall-asr-librispeech-pruned-transducer-stateless8-2022-12-02
  cd icefall-asr-librispeech-pruned-transducer-stateless8-2022-12-02
  git lfs pull --include "exp/cpu_jit-torch-1.10.pt"
  git lfs pull --include "data/lang_bpe_500/LG.pt"

  cd exp
  rm cpu_jit.pt
  ln -sv cpu_jit-torch-1.10.pt cpu_jit.pt
  cd ..

  for m in greedy_search modified_beam_search fast_beam_search; do
    sherpa-offline \
      --decoding-method=$m \
      --nn-model=./exp/cpu_jit.pt \
      --tokens=./data/lang_bpe_500/tokens.txt \
      ./test_wavs/1089-134686-0001.wav \
      ./test_wavs/1221-135766-0001.wav \
      ./test_wavs/1221-135766-0002.wav
  done

  sherpa-offline \
    --decoding-method=fast_beam_search \
    --nn-model=./exp/cpu_jit.pt \
    --lg=./data/lang_bpe_500/LG.pt \
    --tokens=./data/lang_bpe_500/tokens.txt \
    ./test_wavs/1089-134686-0001.wav \
    ./test_wavs/1221-135766-0001.wav \
    ./test_wavs/1221-135766-0002.wav

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
  git lfs pull --include "data/lang_bpe_500/LG.pt"

  for m in greedy_search modified_beam_search fast_beam_search; do
    sherpa-offline \
      --decoding-method=$m \
      --nn-model=./exp/cpu_jit.pt \
      --tokens=./data/lang_bpe_500/tokens.txt \
      ./test_wavs/1089-134686-0001.wav \
      ./test_wavs/1221-135766-0001.wav \
      ./test_wavs/1221-135766-0002.wav
  done

  sherpa-offline \
    --decoding-method=fast_beam_search \
    --nn-model=./exp/cpu_jit.pt \
    --lg=./data/lang_bpe_500/LG.pt \
    --tokens=./data/lang_bpe_500/tokens.txt \
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
  git lfs pull --include "data/lang_bpe_500/LG.pt"
  cd exp
  ln -s cpu_jit-torch-1.10.0.pt cpu_jit.pt
  cd ..

  for m in greedy_search modified_beam_search fast_beam_search; do
    sherpa-offline \
      --decoding-method=$m \
      --nn-model=./exp/cpu_jit.pt \
      --tokens=./data/lang_bpe_500/tokens.txt \
      ./test_wavs/1089-134686-0001.wav \
      ./test_wavs/1221-135766-0001.wav \
      ./test_wavs/1221-135766-0002.wav
  done

  sherpa-offline \
    --decoding-method=fast_beam_search \
    --nn-model=./exp/cpu_jit.pt \
    --lg=./data/lang_bpe_500/LG.pt \
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
  git lfs pull --include "data/lang_bpe_500/LG.pt"

  for m in greedy_search modified_beam_search fast_beam_search; do
    sherpa-offline \
      --decoding-method=$m \
      --nn-model=./exp/cpu_jit.pt \
      --tokens=./data/lang_bpe_500/tokens.txt \
      ./test_wavs/1089-134686-0001.wav \
      ./test_wavs/1221-135766-0001.wav \
      ./test_wavs/1221-135766-0002.wav
  done

  sherpa-offline \
    --decoding-method=fast_beam_search \
    --nn-model=./exp/cpu_jit.pt \
    --lg=./data/lang_bpe_500/LG.pt \
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

   cd ../exp
   ln -s cpu_jit-iter-3488000-avg-15.pt cpu_jit.pt
   cd ..

   # Since this repo does not provide tokens.txt, we generate it from bpe.model
   # by ourselves
   /path/to/sherpa/scripts/bpe_model_to_tokens.py ./data/lang_bpe_500/bpe.model > ./data/lang_bpe_500/tokens.txt

   mkdir test_wavs
   cd test_wavs
   wget https://huggingface.co/csukuangfj/wav2vec2.0-torchaudio/resolve/main/test_wavs/1089-134686-0001.wav
   wget https://huggingface.co/csukuangfj/wav2vec2.0-torchaudio/resolve/main/test_wavs/1221-135766-0001.wav
   wget https://huggingface.co/csukuangfj/wav2vec2.0-torchaudio/resolve/main/test_wavs/1221-135766-0002.wav

   for m in greedy_search modified_beam_search fast_beam_search; do
     sherpa-offline \
       --decoding-method=$m \
       --nn-model=./exp/cpu_jit.pt \
       --tokens=./data/lang_bpe_500/tokens.txt \
       ./test_wavs/1089-134686-0001.wav \
       ./test_wavs/1221-135766-0001.wav \
       ./test_wavs/1221-135766-0002.wav
   done

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
  git lfs pull --include "data/lang_char/LG.pt"
  cd exp
  ln -s cpu_jit_epoch_10_avg_2_torch_1.7.1.pt cpu_jit.pt
  cd ..

  for m in greedy_search modified_beam_search fast_beam_search; do
    sherpa-offline \
      --decoding-method=$m \
      --nn-model=./exp/cpu_jit.pt \
      --tokens=./data/lang_char/tokens.txt \
      ./test_wavs/DEV_T0000000000.wav \
      ./test_wavs/DEV_T0000000001.wav \
      ./test_wavs/DEV_T0000000002.wav
  done

  sherpa-offline \
    --decoding-method=$m \
    --nn-model=./exp/cpu_jit.pt \
    --lg=./data/lang_char/LG.pt \
    --tokens=./data/lang_char/tokens.txt \
    ./test_wavs/DEV_T0000000000.wav \
    ./test_wavs/DEV_T0000000001.wav \
    ./test_wavs/DEV_T0000000002.wav

icefall_asr_aidatatang-200zh_pruned_transducer_stateless2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  # This models is trained using aidatatang_200zh
  #
  # See https://github.com/k2-fsa/icefall/pull/355
  #
  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/luomingshuang/icefall_asr_aidatatang-200zh_pruned_transducer_stateless2
  cd icefall_asr_aidatatang-200zh_pruned_transducer_stateless2
  git lfs pull --include "exp/cpu_jit_torch.1.7.1.pt"

  cd exp
  ln -sv cpu_jit_torch.1.7.1.pt cpu_jit.pt
  cd ..

  for m in greedy_search modified_beam_search fast_beam_search; do
    sherpa-offline \
      --decoding-method=$m \
      --nn-model=./exp/cpu_jit.pt \
      --tokens=./data/lang_char/tokens.txt \
      ./test_wavs/T0055G0036S0002.wav \
      ./test_wavs/T0055G0036S0003.wav \
      ./test_wavs/T0055G0036S0004.wav
  done

icefall-asr-alimeeting-pruned-transducer-stateless7
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  # This models is trained using alimeeting (https://www.openslr.org/119/)
  #
  # See https://github.com/k2-fsa/icefall/pull/751
  #
  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/desh2608/icefall-asr-alimeeting-pruned-transducer-stateless7
  cd icefall-asr-alimeeting-pruned-transducer-stateless7

  git lfs pull --include "exp/cpu_jit.pt"

  for m in greedy_search modified_beam_search fast_beam_search; do
    sherpa-offline \
      --decoding-method=$m \
      --nn-model=./exp/cpu_jit.pt \
      --tokens=./data/lang_char/tokens.txt \
      ./test_wavs/165.wav \
      ./test_wavs/74.wav \
      ./test_wavs/209.wav
  done

Chinese + English
^^^^^^^^^^^^^^^^^

icefall_asr_tal-csasr_pruned_transducer_stateless5
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  # This models is trained using TAL_CSASR dataset from
  # https://ai.100tal.com/dataset
  # where each utterance contains both English and Chinese.
  #
  # See https://github.com/k2-fsa/icefall/pull/428
  #
  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/luomingshuang/icefall_asr_tal-csasr_pruned_transducer_stateless5
  cd icefall_asr_tal-csasr_pruned_transducer_stateless5
  git lfs pull --include "exp/cpu_jit.pt"

  for m in greedy_search modified_beam_search fast_beam_search; do
    sherpa-offline \
      --decoding-method=$m \
      --nn-model=./exp/cpu_jit.pt \
      --tokens=./data/lang_char/tokens.txt \
      ./test_wavs/210_36476_210_8341_1_1533271973_7057520_132.wav \
      ./test_wavs/210_36476_210_8341_1_1533271973_7057520_138.wav \
      ./test_wavs/210_36476_210_8341_1_1533271973_7057520_145.wav \
      ./test_wavs/210_36476_210_8341_1_1533271973_7057520_148.wav
  done

Tibetan
^^^^^^^

icefall-asr-xbmu-amdo31-pruned-transducer-stateless7-2022-12-02
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  # This model is trained using the XBMU-AMDO31 corpus
  #
  # See https://github.com/k2-fsa/icefall/pull/706
  #
  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/syzym/icefall-asr-xbmu-amdo31-pruned-transducer-stateless7-2022-12-02
  cd icefall-asr-xbmu-amdo31-pruned-transducer-stateless7-2022-12-02
  git lfs pull --include "exp/cpu_jit.pt"
  git lfs pull --include "data/lang_bpe_500/LG.pt"

  for m in greedy_search modified_beam_search fast_beam_search; do
    sherpa-offline \
      --decoding-method=$m \
      --nn-model=./exp/cpu_jit.pt \
      --tokens=./data/lang_bpe_500/tokens.txt \
      ./test_wavs/a_0_cacm-A70_31116.wav \
      ./test_wavs/a_0_cacm-A70_31117.wav \
      ./test_wavs/a_0_cacm-A70_31118.wav
  done

  sherpa-offline \
    --decoding-method=fast_beam_search \
    --nn-model=./exp/cpu_jit.pt \
    --lg=./data/lang_bpe_500/LG.pt \
    --tokens=./data/lang_bpe_500/tokens.txt \
    ./test_wavs/a_0_cacm-A70_31116.wav \
    ./test_wavs/a_0_cacm-A70_31117.wav \
    ./test_wavs/a_0_cacm-A70_31118.wav

icefall-asr-xbmu-amdo31-pruned-transducer-stateless5-2022-11-29
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  # This model is trained using the XBMU-AMDO31 corpus
  #
  # See https://github.com/k2-fsa/icefall/pull/706
  #
  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/syzym/icefall-asr-xbmu-amdo31-pruned-transducer-stateless5-2022-11-29
  cd icefall-asr-xbmu-amdo31-pruned-transducer-stateless5-2022-11-29
  git lfs pull --include "data/lang_bpe_500/LG.pt"
  git lfs pull --include "data/lang_bpe_500/tokens.txt"
  git lfs pull --include "exp/cpu_jit-epoch-28-avg-23-torch-1.10.0.pt"
  git lfs pull --include "test_wavs/a_0_cacm-A70_31116.wav"
  git lfs pull --include "test_wavs/a_0_cacm-A70_31117.wav"
  git lfs pull --include "test_wavs/a_0_cacm-A70_31118.wav"

  cd exp
  rm cpu_jit.pt
  ln -sv cpu_jit-epoch-28-avg-23-torch-1.10.0.pt cpu_jit.pt
  cd ..

  for m in greedy_search modified_beam_search fast_beam_search; do
    sherpa-offline \
      --decoding-method=$m \
      --nn-model=./exp/cpu_jit.pt \
      --tokens=./data/lang_bpe_500/tokens.txt \
      ./test_wavs/a_0_cacm-A70_31116.wav \
      ./test_wavs/a_0_cacm-A70_31117.wav \
      ./test_wavs/a_0_cacm-A70_31118.wav
  done

  sherpa-offline \
    --decoding-method=fast_beam_search \
    --nn-model=./exp/cpu_jit.pt \
    --lg=./data/lang_bpe_500/LG.pt \
    --tokens=./data/lang_bpe_500/tokens.txt \
    ./test_wavs/a_0_cacm-A70_31116.wav \
    ./test_wavs/a_0_cacm-A70_31117.wav \
    ./test_wavs/a_0_cacm-A70_31118.wav
