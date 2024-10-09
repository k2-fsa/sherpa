yesno
=====

This section describes how to use the `tdnn <https://github.com/k2-fsa/icefall/tree/master/egs/yesno/ASR/tdnn>`_
model of the `yesno`_ dataset from `icefall`_ in `sherpa-onnx`_.

.. note::

   It is a **non-streaming** model and it can only recognize
   two words in `Hebrew <https://en.wikipedia.org/wiki/Hebrew_language>`_:
   ``yes`` and ``no``.

To download the model, please use:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-tdnn-yesno.tar.bz2

  # For Chinese users, please use the following mirror
  # wget https://hub.nuaa.cf/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-tdnn-yesno.tar.bz2

  tar xvf sherpa-onnx-tdnn-yesno.tar.bz2
  rm sherpa-onnx-tdnn-yesno.tar.bz2

Please check that the file sizes of the pre-trained models are correct. See
the file sizes of ``*.onnx`` files below.

.. code-block:: bash

  sherpa-onnx-tdnn-yesno fangjun$ ls -lh *.onnx
  -rw-r--r--  1 fangjun  staff    55K Aug 12 17:02 model-epoch-14-avg-2.int8.onnx
  -rw-r--r--  1 fangjun  staff    54K Aug 12 17:02 model-epoch-14-avg-2.onnx

Decode wave files
~~~~~~~~~~~~~~~~~

.. hint::

   It supports decoding only wave files of a single channel with 16-bit
   encoded samples, while the sampling rate does not need to be 16 kHz.

The following code shows how to use ``fp32`` models to decode wave files.
Please replace ``model-epoch-14-avg-2.int8.onnx`` with ``model-epoch-14-avg-2.int8.onnx``
to use the ``int8`` quantized model.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-offline \
    --sample-rate=8000 \
    --feat-dim=23 \
    --tokens=./sherpa-onnx-tdnn-yesno/tokens.txt \
    --tdnn-model=./sherpa-onnx-tdnn-yesno/model-epoch-14-avg-2.onnx \
    ./sherpa-onnx-tdnn-yesno/test_wavs/0_0_0_1_0_0_0_1.wav \
    ./sherpa-onnx-tdnn-yesno/test_wavs/0_0_1_0_0_0_1_0.wav \
    ./sherpa-onnx-tdnn-yesno/test_wavs/0_0_1_0_0_1_1_1.wav \
    ./sherpa-onnx-tdnn-yesno/test_wavs/0_0_1_0_1_0_0_1.wav \
    ./sherpa-onnx-tdnn-yesno/test_wavs/0_0_1_1_0_0_0_1.wav \
    ./sherpa-onnx-tdnn-yesno/test_wavs/0_0_1_1_0_1_1_0.wav

The output is given below:

.. code-block:: bash

  OfflineRecognizerConfig(feat_config=OfflineFeatureExtractorConfig(sampling_rate=8000, feature_dim=23), model_config=OfflineModelConfig(transducer=OfflineTransducerModelConfig(encoder_filename="", decoder_filename="", joiner_filename=""), paraformer=OfflineParaformerModelConfig(model=""), nemo_ctc=OfflineNemoEncDecCtcModelConfig(model=""), whisper=OfflineWhisperModelConfig(encoder="", decoder=""), tdnn=OfflineTdnnModelConfig(model="./sherpa-onnx-tdnn-yesno/model-epoch-14-avg-2.onnx"), tokens="./sherpa-onnx-tdnn-yesno/tokens.txt", num_threads=2, debug=False, provider="cpu", model_type=""), lm_config=OfflineLMConfig(model="", scale=0.5), decoding_method="greedy_search", max_active_paths=4, context_score=1.5)
  Creating recognizer ...
  Started
  Done!

  ./sherpa-onnx-tdnn-yesno/test_wavs/0_0_0_1_0_0_0_1.wav
  {"text":"NNNYNNNY","timestamps":"[]","tokens":["N","N","N","Y","N","N","N","Y"]}
  ----
  ./sherpa-onnx-tdnn-yesno/test_wavs/0_0_1_0_0_0_1_0.wav
  {"text":"NNYNNNYN","timestamps":"[]","tokens":["N","N","Y","N","N","N","Y","N"]}
  ----
  ./sherpa-onnx-tdnn-yesno/test_wavs/0_0_1_0_0_1_1_1.wav
  {"text":"NNYNNYYY","timestamps":"[]","tokens":["N","N","Y","N","N","Y","Y","Y"]}
  ----
  ./sherpa-onnx-tdnn-yesno/test_wavs/0_0_1_0_1_0_0_1.wav
  {"text":"NNYNYNNY","timestamps":"[]","tokens":["N","N","Y","N","Y","N","N","Y"]}
  ----
  ./sherpa-onnx-tdnn-yesno/test_wavs/0_0_1_1_0_0_0_1.wav
  {"text":"NNYYNNNY","timestamps":"[]","tokens":["N","N","Y","Y","N","N","N","Y"]}
  ----
  ./sherpa-onnx-tdnn-yesno/test_wavs/0_0_1_1_0_1_1_0.wav
  {"text":"NNYYNYYN","timestamps":"[]","tokens":["N","N","Y","Y","N","Y","Y","N"]}
  ----
  num threads: 2
  decoding method: greedy_search
  Elapsed seconds: 0.071 s
  Real time factor (RTF): 0.071 / 38.530 = 0.002

.. note::

   In the above output, ``N`` represents ``NO``, while ``Y`` is ``YES``.
   So for the last wave, ``NNYYNYYN`` means ``NO NO YES YES NO YES YES NO``.

   In the filename of the last wave ``0_0_1_1_0_1_1_0.wav``, 0 means ``NO``
   and 1 means ``YES``. So the ground truth of the last wave is
   ``NO NO YES YES NO YES YES NO``.
