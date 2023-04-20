English
=======

.. hint::

   Please refer to :ref:`install_sherpa_onnx` to install `sherpa-onnx`_
   before you read this section.

.. note::

   We use `./build/bin/sherpa-offline <https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/sherpa-onnx-offline.cc>`_
   as an example in this section. You can use other scripts such as

    - `./build/bin/sherpa-onnx-microphone-offline <https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/sherpa-onnx-microphone-offline.cc>`_
    - `./build/bin/sherpa-onnx-offline-websocket-server <https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/offline-websocket-server.cc>`_
    - `python-api-examples/offline-decode-files.py <https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/offline-decode-files.py>`_

This page lists offline CTC models from `NeMo`_ for English.

stt_en_citrinet_512
-------------------

This model is converted from

  `<https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_en_citrinet_512>`_

Citrinet-512 model which has been trained on the ASR Set dataset
with over 7000 hours of english speech.

In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/sherpa-onnx-nemo-ctc-en-citrinet-512
  cd sherpa-onnx-nemo-ctc-en-citrinet-512
  git lfs pull --include "*.onnx"

Please check that the file sizes of the pre-trained models are correct. See
the file sizes of ``*.onnx`` files below.

.. code-block:: bash

  sherpa-onnx-nemo-ctc-en-citrinet-512 fangjun$ ls -lh *.onnx
  -rw-r--r--  1 fangjun  staff    36M Apr  7 16:10 model.int8.onnx
  -rw-r--r--  1 fangjun  staff   142M Apr  7 14:24 model.onnx

Decode wave files
~~~~~~~~~~~~~~~~~

.. hint::

   It supports decoding only wave files of a single channel with 16-bit
   encoded samples, while the sampling rate does not need to be 16 kHz.

The following code shows how to use ``fp32`` models to decode wave files.
Please replace ``model.onnx`` with ``model.int8.onnx`` to use ``int8``
quantized model.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-nemo-ctc-en-citrinet-512/tokens.txt \
    --nemo-ctc-model=./sherpa-onnx-nemo-ctc-en-citrinet-512/model.onnx \
    --num-threads=2 \
    --decoding-method=greedy_search \
    --debug=false \
    ./sherpa-onnx-nemo-ctc-en-citrinet-512/test_wavs/0.wav \
    ./sherpa-onnx-nemo-ctc-en-citrinet-512/test_wavs/1.wav \
    ./sherpa-onnx-nemo-ctc-en-citrinet-512/test_wavs/8k.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

You should see the following output:

.. literalinclude:: ./code-english/stt_en_citrinet_512.txt

stt_en_conformer_ctc_small
--------------------------

This model is converted from

  `<https://registry.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_en_conformer_ctc_small>`_

It contains small size versions of Conformer-CTC (13M parameters) trained on
NeMo ASRSet with around 16000 hours of english speech. The model transcribes
speech in lower case english alphabet along with spaces and apostrophes.

In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/sherpa-onnx-nemo-ctc-en-conformer-small
  cd sherpa-onnx-nemo-ctc-en-conformer-small
  git lfs pull --include "*.onnx"

Please check that the file sizes of the pre-trained models are correct. See
the file sizes of ``*.onnx`` files below.

.. code-block:: bash

  sherpa-onnx-nemo-ctc-en-conformer-small fangjun$ ls -lh *.onnx
  -rw-r--r--  1 fangjun  staff    44M Apr  7 20:24 model.int8.onnx
  -rw-r--r--  1 fangjun  staff    81M Apr  7 18:56 model.onnx

Decode wave files
~~~~~~~~~~~~~~~~~

.. hint::

   It supports decoding only wave files of a single channel with 16-bit
   encoded samples, while the sampling rate does not need to be 16 kHz.

The following code shows how to use ``fp32`` models to decode wave files.
Please replace ``model.onnx`` with ``model.int8.onnx`` to use ``int8``
quantized model.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-nemo-ctc-en-conformer-small/tokens.txt \
    --nemo-ctc-model=./sherpa-onnx-nemo-ctc-en-conformer-small/model.onnx \
    --num-threads=2 \
    --decoding-method=greedy_search \
    --debug=false \
    ./sherpa-onnx-nemo-ctc-en-conformer-small/test_wavs/0.wav \
    ./sherpa-onnx-nemo-ctc-en-conformer-small/test_wavs/1.wav \
    ./sherpa-onnx-nemo-ctc-en-conformer-small/test_wavs/8k.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

You should see the following output:

.. literalinclude:: ./code-english/stt_en_conformer_ctc_small.txt

stt_en_conformer_ctc_medium
---------------------------

This model is converted from

  `<https://registry.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_en_conformer_ctc_medium>`_

It contains medium size versions of Conformer-CTC (around 30M parameters)
trained on NeMo ASRSet with around 16000 hours of english speech. The model
transcribes speech in lower case english alphabet along with spaces and apostrophes.

In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/sherpa-onnx-nemo-ctc-en-conformer-medium
  cd sherpa-onnx-nemo-ctc-en-conformer-medium
  git lfs pull --include "*.onnx"

Please check that the file sizes of the pre-trained models are correct. See
the file sizes of ``*.onnx`` files below.

.. code-block:: bash

  sherpa-onnx-nemo-ctc-en-conformer-medium fangjun$ ls -lh *.onnx
  -rw-r--r--  1 fangjun  staff    64M Apr  7 20:44 model.int8.onnx
  -rw-r--r--  1 fangjun  staff   152M Apr  7 20:43 model.onnx

Decode wave files
~~~~~~~~~~~~~~~~~

.. hint::

   It supports decoding only wave files of a single channel with 16-bit
   encoded samples, while the sampling rate does not need to be 16 kHz.

The following code shows how to use ``fp32`` models to decode wave files.
Please replace ``model.onnx`` with ``model.int8.onnx`` to use ``int8``
quantized model.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-nemo-ctc-en-conformer-medium/tokens.txt \
    --nemo-ctc-model=./sherpa-onnx-nemo-ctc-en-conformer-medium/model.onnx \
    --num-threads=2 \
    --decoding-method=greedy_search \
    --debug=false \
    ./sherpa-onnx-nemo-ctc-en-conformer-medium/test_wavs/0.wav \
    ./sherpa-onnx-nemo-ctc-en-conformer-medium/test_wavs/1.wav \
    ./sherpa-onnx-nemo-ctc-en-conformer-medium/test_wavs/8k.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

You should see the following output:

.. literalinclude:: ./code-english/stt_en_conformer_ctc_medium.txt

stt_en_conformer_ctc_large
---------------------------

This model is converted from

  `<https://registry.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_en_conformer_ctc_large>`_

It contains large size versions of Conformer-CTC (around 120M parameters)
trained on NeMo ASRSet with around 24500 hours of english speech. The model
transcribes speech in lower case english alphabet along with spaces and apostrophes

In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/sherpa-onnx-nemo-ctc-en-conformer-large
  cd sherpa-onnx-nemo-ctc-en-conformer-large
  git lfs pull --include "*.onnx"

Please check that the file sizes of the pre-trained models are correct. See
the file sizes of ``*.onnx`` files below.

.. code-block:: bash

  sherpa-onnx-nemo-ctc-en-conformer-large fangjun$ ls -lh *.onnx
  -rw-r--r--  1 fangjun  staff   162M Apr  7 22:01 model.int8.onnx
  -rw-r--r--  1 fangjun  staff   508M Apr  7 22:01 model.onnx

Decode wave files
~~~~~~~~~~~~~~~~~

.. hint::

   It supports decoding only wave files of a single channel with 16-bit
   encoded samples, while the sampling rate does not need to be 16 kHz.

The following code shows how to use ``fp32`` models to decode wave files.
Please replace ``model.onnx`` with ``model.int8.onnx`` to use ``int8``
quantized model.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-nemo-ctc-en-conformer-large/tokens.txt \
    --nemo-ctc-model=./sherpa-onnx-nemo-ctc-en-conformer-large/model.onnx \
    --num-threads=2 \
    --decoding-method=greedy_search \
    --debug=false \
    ./sherpa-onnx-nemo-ctc-en-conformer-large/test_wavs/0.wav \
    ./sherpa-onnx-nemo-ctc-en-conformer-large/test_wavs/1.wav \
    ./sherpa-onnx-nemo-ctc-en-conformer-large/test_wavs/8k.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

You should see the following output:

.. literalinclude:: ./code-english/stt_en_conformer_ctc_large.txt
