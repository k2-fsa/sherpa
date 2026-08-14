Malayalam
=========

.. hint::

   See :ref:`install_sherpa_onnx` to install `sherpa-onnx`_
   before you read this section.

This page lists offline CTC models from `NeMo`_ for Malayalam.

sherpa-onnx-nemo-ctc-indicconformer-malayalam (Malayalam, മലയാളം)
-----------------------------------------------------------------

This model is converted from the CTC branch of `AI4Bharat IndicConformer`_
and packaged for `sherpa-onnx`_ by the community. It runs fully offline on CPU
and supports Malayalam.

The weights come from the community ONNX export
`<https://huggingface.co/trysem/indicconformer-120m-onnx>`_ (a copy of
``sulabhkatiyar/indicconformer-120m-onnx``, licensed under CC-BY-4.0). They were
packaged for `sherpa-onnx`_ by adding the metadata its NeMo-CTC loader needs
(``model_type=EncDecHybridRNNTCTCBPEModel``, ``vocab_size``,
``normalize_type=per_feature``, ``subsampling_factor=8``) and generating
``tokens.txt`` from the vocabulary (tokens in vocab order, CTC blank appended
at ``id = len(vocab)``).

Model page: `<https://huggingface.co/jeswinjestin/sherpa-onnx-nemo-ctc-indicconformer-malayalam>`_

Download the model
~~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

   # Requires git-lfs (https://git-lfs.com); run `git lfs install` once first.
   GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/jeswinjestin/sherpa-onnx-nemo-ctc-indicconformer-malayalam
   cd sherpa-onnx-nemo-ctc-indicconformer-malayalam
   git lfs pull --include "model.onnx,test_wavs/0.wav"

You should see something like below after downloading::

   -rw-r--r--  model.onnx    493M
   -rw-r--r--  tokens.txt     66K
   drwxr-xr-x  test_wavs

Decode a wave file
~~~~~~~~~~~~~~~~~~~

.. hint::

   It supports decoding only wave files of a single channel with 16-bit
   encoded samples, while the sampling rate does not need to be 16 kHz.

.. code-block:: bash

   cd /path/to/sherpa-onnx

   ./build/bin/sherpa-onnx-offline \
     --nemo-ctc-model=./sherpa-onnx-nemo-ctc-indicconformer-malayalam/model.onnx \
     --tokens=./sherpa-onnx-nemo-ctc-indicconformer-malayalam/tokens.txt \
     --num-threads=2 \
     ./sherpa-onnx-nemo-ctc-indicconformer-malayalam/test_wavs/0.wav

You should see the following output:

.. code-block:: text

   ഹായ്, ഇത് ഒരു ഡെമോ ടെസ്റ്റ് റൺ ആണ്.

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your command line.

.. _AI4Bharat IndicConformer: https://github.com/AI4Bharat/IndicConformerASR
