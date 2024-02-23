.. _whisper_tiny_en_sherpa_onnx:

tiny.en
=======

You can use the following command to download the exported `onnx`_ models of ``tiny.en``:

.. hint::

   Please replace ``tiny.en`` with ``base.en``, ``small.en``, or ``medium.en``
   if you want to try a different type of model.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.en.tar.bz2
  tar xvf sherpa-onnx-whisper-tiny.en.tar.bz2

Please check that the file sizes of the downloaded models are correct. See
the file size of ``*.onnx`` files below.

.. code-block:: bash

  (py38) fangjuns-MacBook-Pro:sherpa-onnx-whisper-tiny.en fangjun$ ls -lh *.onnx
  -rw-r--r--  1 fangjun  staff   105M Aug  7 16:22 tiny.en-decoder.int8.onnx
  -rw-r--r--  1 fangjun  staff   185M Aug  7 16:23 tiny.en-decoder.onnx
  -rw-r--r--  1 fangjun  staff    12M Aug  7 16:22 tiny.en-encoder.int8.onnx
  -rw-r--r--  1 fangjun  staff    36M Aug  7 16:22 tiny.en-encoder.onnx

To use the downloaded files to decode waves, please run:

.. hint::

    Please first follow :ref:`install_sherpa_onnx` to build `sherpa-onnx`_
    before you continue.

.. code-block:: bash

   cd /path/to/sherpa-onnx

   ./build/bin/sherpa-onnx-offline \
     --whisper-encoder=./sherpa-onnx-whisper-tiny.en/tiny.en-encoder.onnx \
     --whisper-decoder=./sherpa-onnx-whisper-tiny.en/tiny.en-decoder.onnx \
     --tokens=./sherpa-onnx-whisper-tiny.en/tiny.en-tokens.txt \
     ./sherpa-onnx-whisper-tiny.en/test_wavs/0.wav \
     ./sherpa-onnx-whisper-tiny.en/test_wavs/1.wav \
     ./sherpa-onnx-whisper-tiny.en/test_wavs/8k.wav

To use ``int8`` quantized models, please use:

.. code-block:: bash

   cd /path/to/sherpa-onnx

   ./build/bin/sherpa-onnx-offline \
     --whisper-encoder=./sherpa-onnx-whisper-tiny.en/tiny.en-encoder.int8.onnx \
     --whisper-decoder=./sherpa-onnx-whisper-tiny.en/tiny.en-decoder.int8.onnx \
     --tokens=./sherpa-onnx-whisper-tiny.en/tiny.en-tokens.txt \
     ./sherpa-onnx-whisper-tiny.en/test_wavs/0.wav \
     ./sherpa-onnx-whisper-tiny.en/test_wavs/1.wav \
     ./sherpa-onnx-whisper-tiny.en/test_wavs/8k.wav

Real-time factor (RTF) on Raspberry Pi 4 Model B
------------------------------------------------

One of the test command is given below:

.. code-block:: bash

  ./sherpa-onnx-offline \
    --num-threads=1 \
    --whisper-encoder=./sherpa-onnx-whisper-tiny.en/tiny.en-encoder.onnx \
    --whisper-decoder=./sherpa-onnx-whisper-tiny.en/tiny.en-decoder.onnx \
    --tokens=./sherpa-onnx-whisper-tiny.en/tiny.en-tokens.txt \
    ./sherpa-onnx-whisper-tiny.en/test_wavs/1.wav

And its output is:

.. code-block:: bash

  /root/fangjun/open-source/sherpa-onnx/sherpa-onnx/csrc/parse-options.cc:Read:361 ./sherpa-onnx-offline --num-threads=1 --whisper-encoder=./sherpa-onnx-whisper-tiny.en/tiny.en-encoder.onnx --whisper-decoder=./sherpa-onnx-whisper-tiny.en/tiny.en-decoder.onnx --tokens=./sherpa-onnx-whisper-tiny.en/tiny.en-tokens.txt ./sherpa-onnx-whisper-tiny.en/test_wavs/1.wav

  OfflineRecognizerConfig(feat_config=OfflineFeatureExtractorConfig(sampling_rate=16000, feature_dim=80), model_config=OfflineModelConfig(transducer=OfflineTransducerModelConfig(encoder_filename="", decoder_filename="", joiner_filename=""), paraformer=OfflineParaformerModelConfig(model=""), nemo_ctc=OfflineNemoEncDecCtcModelConfig(model=""), whisper=OfflineWhisperModelConfig(encoder="./sherpa-onnx-whisper-tiny.en/tiny.en-encoder.onnx", decoder="./sherpa-onnx-whisper-tiny.en/tiny.en-decoder.onnx"), tokens="./sherpa-onnx-whisper-tiny.en/tiny.en-tokens.txt", num_threads=1, debug=False, provider="cpu", model_type=""), lm_config=OfflineLMConfig(model="", scale=0.5), decoding_method="greedy_search", max_active_paths=4, context_score=1.5)
  Creating recognizer ...
  Started
  Done!

  ./sherpa-onnx-whisper-tiny.en/test_wavs/1.wav
  {"text":" God, as a direct consequence of the sin which man thus punished, had given her a lovely child, whose place was on that same dishonored bosom to connect her parent forever with the race and descent of mortals, and to be finally a blessed soul in heaven.","timestamps":"[]","tokens":[" God",","," as"," a"," direct"," consequence"," of"," the"," sin"," which"," man"," thus"," punished",","," had"," given"," her"," a"," lovely"," child",","," whose"," place"," was"," on"," that"," same"," dishon","ored"," bos","om"," to"," connect"," her"," parent"," forever"," with"," the"," race"," and"," descent"," of"," mortals",","," and"," to"," be"," finally"," a"," blessed"," soul"," in"," heaven","."]}
  ----
  num threads: 1
  decoding method: greedy_search
  Elapsed seconds: 11.454 s
  Real time factor (RTF): 11.454 / 16.715 = 0.685

The following table compares the RTF between different number of threads and types of `onnx`_ models:


.. list-table::

 * - Model type
   - Number of threads
   - RTF
 * - float32
   - 1
   - 0.685
 * - float32
   - 2
   - 0.559
 * - float32
   - 3
   - 0.526
 * - float32
   - 4
   - 0.520
 * - int8
   - 1
   - 0.547
 * - int8
   - 2
   - 0.431
 * - int8
   - 3
   - 0.398
 * - int8
   - 4
   - 0.386
