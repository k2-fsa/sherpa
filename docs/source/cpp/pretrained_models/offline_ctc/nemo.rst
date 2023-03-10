NeMo
====

This section lists models from `NeMo`_.


sherpa-nemo-ctc-en-citrinet-512 (English)
-----------------------------------------

This model is converted from

  `<https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_en_citrinet_512>`_

.. code-block:: bash

   GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/sherpa-nemo-ctc-en-citrinet-512
   cd sherpa-nemo-ctc-en-citrinet-512
   git lfs pull --include "model.pt"

   sherpa-offline \
      --nn-model=./model.pt \
      --tokens=./tokens.txt \
      --use-gpu=false \
      --modified=false \
      --nemo-normalize=per_feature \
      ./test_wavs/0.wav \
      ./test_wavs/1.wav \
      ./test_wavs/2.wav

.. code-block:: bash

  ls -lh model.pt
  -rw-r--r-- 1 kuangfangjun root 142M Mar  9 21:23 model.pt

.. caution::

    It is of paramount importance to specify ``--nemo-normalize=per_feature``.

sherpa-nemo-ctc-zh-citrinet-512 (Chinese)
-----------------------------------------

This model is converted from

  `<https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_zh_citrinet_512>`_

.. code-block:: bash

   GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/sherpa-nemo-ctc-zh-citrinet-512
   cd sherpa-nemo-ctc-zh-citrinet-512
   git lfs pull --include "model.pt"

   sherpa-offline \
      --nn-model=./model.pt \
      --tokens=./tokens.txt \
      --use-gpu=false \
      --modified=true \
      --nemo-normalize=per_feature \
      ./test_wavs/0.wav \
      ./test_wavs/1.wav \
      ./test_wavs/2.wav

.. code-block:: bash

  ls -lh model.pt
  -rw-r--r-- 1 kuangfangjun root 153M Mar 10 15:07 model.pt

.. hint::

    Since the vocabulary size of this model is very large, i.e, 5207, we use
    ``--modified=true`` to use a
    `modified CTC topology <https://k2-fsa.github.io/k2/python_api/api.html#k2.ctc_topo>`_

.. caution::

    It is of paramount importance to specify ``--nemo-normalize=per_feature``.

sherpa-nemo-ctc-zh-citrinet-1024-gamma-0-25 (Chinese)
-----------------------------------------------------

This model is converted from

  `<https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_zh_citrinet_1024_gamma_0_25>`_

.. code-block:: bash

   GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/sherpa-nemo-ctc-zh-citrinet-1024-gamma-0-25
   cd sherpa-nemo-ctc-zh-citrinet-1024-gamma-0-25
   git lfs pull --include "model.pt"

   sherpa-offline \
      --nn-model=./model.pt \
      --tokens=./tokens.txt \
      --use-gpu=false \
      --modified=true \
      --nemo-normalize=per_feature \
      ./test_wavs/0.wav \
      ./test_wavs/1.wav \
      ./test_wavs/2.wav

.. code-block:: bash

  ls -lh model.pt
  -rw-r--r-- 1 kuangfangjun root 557M Mar 10 16:29 model.pt

.. hint::

    Since the vocabulary size of this model is very large, i.e, 5207, we use
    ``--modified=true`` to use a
    `modified CTC topology <https://k2-fsa.github.io/k2/python_api/api.html#k2.ctc_topo>`_

.. caution::

    It is of paramount importance to specify ``--nemo-normalize=per_feature``.

sherpa-nemo-ctc-de-citrinet-1024 (German)
-----------------------------------------

This model is converted from

  `<https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_de_citrinet_1024>`_

.. code-block:: bash

   GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/sherpa-nemo-ctc-de-citrinet-1024
   cd sherpa-nemo-ctc-de-citrinet-1024
   git lfs pull --include "model.pt"

   sherpa-offline \
      --nn-model=./model.pt \
      --tokens=./tokens.txt \
      --use-gpu=false \
      --modified=false \
      --nemo-normalize=per_feature \
      ./test_wavs/0.wav \
      ./test_wavs/1.wav \
      ./test_wavs/2.wav

.. code-block:: bash

  ls -lh model.pt
  -rw-r--r-- 1 kuangfangjun root 541M Mar 10 16:55 model.pt

.. caution::

    It is of paramount importance to specify ``--nemo-normalize=per_feature``.


sherpa-nemo-ctc-en-conformer-small (English)
--------------------------------------------

This model is converted from

  `<https://registry.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_en_conformer_ctc_small>`_

.. code-block::

   GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/sherpa-nemo-ctc-en-conformer-small
   cd sherpa-nemo-ctc-en-conformer-small
   git lfs pull --include "model.pt"

   sherpa-offline \
      --nn-model=./model.pt \
      --tokens=./tokens.txt \
      --use-gpu=false \
      --modified=false \
      --nemo-normalize=per_feature \
      ./test_wavs/0.wav \
      ./test_wavs/1.wav \
      ./test_wavs/2.wav

.. code-block:: bash

  ls -lh model.pt
  -rw-r--r--  1 fangjun  staff    82M Mar 10 19:55 model.pt

.. caution::

    It is of paramount importance to specify ``--nemo-normalize=per_feature``.

sherpa-nemo-ctc-en-conformer-medium (English)
---------------------------------------------

This model is converted from

  `<https://registry.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_en_conformer_ctc_medium>`_

.. code-block::

   GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/sherpa-nemo-ctc-en-conformer-medium
   cd sherpa-nemo-ctc-en-conformer-medium
   git lfs pull --include "model.pt"

   sherpa-offline \
      --nn-model=./model.pt \
      --tokens=./tokens.txt \
      --use-gpu=false \
      --modified=false \
      --nemo-normalize=per_feature \
      ./test_wavs/0.wav \
      ./test_wavs/1.wav \
      ./test_wavs/2.wav

.. code-block:: bash

  ls -lh model.pt
  -rw-r--r--  1 fangjun  staff   152M Mar 10 20:26 model.pt

.. caution::

    It is of paramount importance to specify ``--nemo-normalize=per_feature``.

sherpa-nemo-ctc-en-conformer-large (English)
--------------------------------------------

This model is converted from

  `<https://registry.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_en_conformer_ctc_large>`_

.. hint::

   The vocabulary size is 129

.. code-block::

   GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/sherpa-nemo-ctc-en-conformer-large
   cd sherpa-nemo-ctc-en-conformer-large
   git lfs pull --include "model.pt"

   sherpa-offline \
      --nn-model=./model.pt \
      --tokens=./tokens.txt \
      --use-gpu=false \
      --modified=false \
      --nemo-normalize=per_feature \
      ./test_wavs/0.wav \
      ./test_wavs/1.wav \
      ./test_wavs/2.wav

.. code-block:: bash

  ls -lh model.pt
  -rw-r--r--  1 fangjun  staff   508M Mar 10 20:44 model.pt

.. caution::

    It is of paramount importance to specify ``--nemo-normalize=per_feature``.

sherpa-nemo-ctc-de-conformer-large (German)
-------------------------------------------

This model is converted from

  `<https://registry.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_de_conformer_ctc_large>`_

.. hint::

   The vocabulary size is 129

.. code-block::

   GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/sherpa-nemo-ctc-de-conformer-large
   cd sherpa-nemo-ctc-de-conformer-large
   git lfs pull --include "model.pt"

   sherpa-offline \
      --nn-model=./model.pt \
      --tokens=./tokens.txt \
      --use-gpu=false \
      --modified=false \
      --nemo-normalize=per_feature \
      ./test_wavs/0.wav \
      ./test_wavs/1.wav \
      ./test_wavs/2.wav

.. code-block:: bash

  ls -lh model.pt
  -rw-r--r--  1 fangjun  staff   508M Mar 10 21:34 model.pt

.. caution::

    It is of paramount importance to specify ``--nemo-normalize=per_feature``.

How to convert NeMo models to sherpa
------------------------------------

This section describes how to export `NeMo`_ pre-trained CTC models to `sherpa`_.

You can find a list of pre-trained models from `NeMo`_ by visiting:

  `<https://catalog.ngc.nvidia.com/orgs/nvidia/collections/nemo_asr>`_.

Let us take ``stt_en_conformer_ctc_small`` as an example.

You can use the following code to obtain ``model.pt`` and ``tokens.txt``:

.. code-block:: bash

  import nemo.collections.asr as nemo_asr
  m = nemo_asr.models.EncDecCTCModelBPE.from_pretrained('stt_en_conformer_ctc_small')
  m.export("model.pt")

  with open('tokens.txt', 'w', encoding='utf-8') as f:
    f.write("<blk> 0\n")
    for i, s in enumerate(m.decoder.vocabulary):
      f.write(f"{s} {i+1}\n")

One thing to note is that the blank token has the largest token ID in ``NeMo``.
However, it is always ``0`` in `sherpa`_. During network computation, we shift
the last column of the ``log_prob`` tensor to the first column so that
it matches the convention about using 0 for the blank in `sherpa`_.

You can find the exported ``model.pt`` and ``tokens.txt`` by visiting

  `<https://huggingface.co/csukuangfj/sherpa-nemo-ctc-en-conformer-small>`_
