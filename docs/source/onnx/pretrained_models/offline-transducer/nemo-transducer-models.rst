.. _sherpa_onnx_offline_nemo_transducer_models:

Transducer Models from NeMo
===========================

This section lists pre-trained models from `NeMo`_.

.. _nemo_parakeet_tdt_transducer_110m:

parakeet_tdt_transducer_110m (English)
--------------------------------------

This model is converted from

  `<https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/parakeet-tdt_ctc-110m>`_

Note that only the transducer branch is exported for the model in this section. If you want
to use the CTC branch, please see :ref:`nemo_parakeet_tdt_ctc_110m`.

This model is trained with 36000 hours of English data. It supports
both punctuations and casing.

You can find the script for exporting it to `sherpa-onnx`_ at the following address:

  `<https://github.com/k2-fsa/sherpa-onnx/tree/master/scripts/nemo/fast-conformer-hybrid-transducer-ctc>`_

In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet_tdt_transducer_110m-en-36000.tar.bz2
  tar xvf sherpa-onnx-nemo-parakeet_tdt_transducer_110m-en-36000.tar.bz2
  rm sherpa-onnx-nemo-parakeet_tdt_transducer_110m-en-36000.tar.bz2

  ls -lh sherpa-onnx-nemo-parakeet_tdt_transducer_110m-en-36000

You should see the following output:

.. code-block:: bash

  total 456M
  -rw-r--r-- 1 501 staff  16M Sep 27 06:31 decoder.onnx
  -rw-r--r-- 1 501 staff 435M Sep 27 06:31 encoder.onnx
  -rw-r--r-- 1 501 staff 5.4M Sep 27 06:31 joiner.onnx
  drwxr-xr-x 2 501 staff 4.0K Sep 27 06:31 test_wavs
  -rw-r--r-- 1 501 staff 9.8K Sep 27 06:31 tokens.txt

Decode wave files
~~~~~~~~~~~~~~~~~

.. hint::

   It supports decoding only wave files of a single channel with 16-bit
   encoded samples, while the sampling rate does not need to be 16 kHz.

fp32
^^^^

The following code shows how to use ``fp32`` models to decode wave files:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-offline \
    --encoder=sherpa-onnx-nemo-parakeet_tdt_transducer_110m-en-36000/encoder.onnx \
    --decoder=sherpa-onnx-nemo-parakeet_tdt_transducer_110m-en-36000/decoder.onnx \
    --joiner=sherpa-onnx-nemo-parakeet_tdt_transducer_110m-en-36000/joiner.onnx \
    --tokens=sherpa-onnx-nemo-parakeet_tdt_transducer_110m-en-36000/tokens.txt \
    sherpa-onnx-nemo-parakeet_tdt_transducer_110m-en-36000/test_wavs/0.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

You should see the following output:

.. literalinclude:: ./code-nemo/parakeet_tdt_transducer_110m.txt
