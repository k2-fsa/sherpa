.. _whisper_large_v3_sherpa_onnx:

large-v3
========

Before we start, let us
follow :ref:`install_sherpa_onnx_on_linux`
to build a CUDA-enabled version of `sherpa-onnx`_.

In the following, we assume you have run

.. code-block:: bash

  cd /content

  git clone https://github.com/k2-fsa/sherpa-onnx
  cd sherpa-onnx

  mkdir -p build
  cd build
  cmake \
    -DBUILD_SHARED_LIBS=ON \
    -DSHERPA_ONNX_ENABLE_GPU=ON ..

  make -j2 sherpa-onnx-offline

You can use the following commands to download the exported `onnx`_ models of ``large-v3``:

.. hint::

   Please replace ``large-v3`` with
   ``large``, ``large-v1``, ``large-v2``, and ``distil-large-v2``
   if you want to try a different type of model.

.. code-block:: bash

  cd /content

  git lfs install
  git clone https://huggingface.co/csukuangfj/sherpa-onnx-whisper-large-v3

  ls -lh sherpa-onnx-whisper-large-v3

The logs of the above commands are given below:

.. code-block::

  Git LFS initialized.
  Cloning into 'sherpa-onnx-whisper-large-v3'...
  remote: Enumerating objects: 26, done.
  remote: Counting objects: 100% (22/22), done.
  remote: Compressing objects: 100% (21/21), done.
  remote: Total 26 (delta 2), reused 0 (delta 0), pack-reused 4 (from 1)
  Unpacking objects: 100% (26/26), 1.00 MiB | 9.10 MiB/s, done.
  Filtering content: 100% (6/6), 7.40 GiB | 34.50 MiB/s, done.
  total 7.5G
  -rw-r--r-- 1 root root 962M Jul 13 14:19 large-v3-decoder.int8.onnx
  -rw-r--r-- 1 root root 2.8M Jul 13 14:18 large-v3-decoder.onnx
  -rw-r--r-- 1 root root 3.0G Jul 13 14:22 large-v3-decoder.weights
  -rw-r--r-- 1 root root 732M Jul 13 14:19 large-v3-encoder.int8.onnx
  -rw-r--r-- 1 root root 745K Jul 13 14:18 large-v3-encoder.onnx
  -rw-r--r-- 1 root root 2.8G Jul 13 14:21 large-v3-encoder.weights
  -rw-r--r-- 1 root root 798K Jul 13 14:18 large-v3-tokens.txt
  drwxr-xr-x 2 root root 4.0K Jul 13 14:18 test_wavs

.. caution::

   Please remember to run ``git lfs install`` before you run ``git clone``.
   If you have any issues about ``git lfs install``, please follow
   `<https://git-lfs.com/>`_ to install ``git-lfs``.

.. caution::

   Please check the file sizes are correct before proceeding. Otherwise, you would be ``SAD`` later.

.. caution::

   Please check the file sizes are correct before proceeding. Otherwise, you would be ``SAD`` later.

.. caution::

   Please check the file sizes are correct before proceeding. Otherwise, you would be ``SAD`` later.

Run with CPU (float32)
----------------------

.. code-block:: bash

  cd /content

  exe=$PWD/sherpa-onnx/build/bin/sherpa-onnx-offline

  cd sherpa-onnx-whisper-large-v3

  time $exe \
    --whisper-encoder=./large-v3-encoder.onnx \
    --whisper-decoder=./large-v3-decoder.onnx \
    --tokens=./large-v3-tokens.txt \
    --num-threads=2 \
    ./test_wavs/0.wav

The logs are given below::

    /content/sherpa-onnx/sherpa-onnx/csrc/parse-options.cc:Read:375 /content/sherpa-onnx/build/bin/sherpa-onnx-offline --whisper-encoder=./large-v3-encoder.onnx --whisper-decoder=./large-v3-decoder.onnx --tokens=./large-v3-tokens.txt --num-threads=2 ./test_wavs/0.wav

    OfflineRecognizerConfig(feat_config=FeatureExtractorConfig(sampling_rate=16000, feature_dim=80, low_freq=20, high_freq=-400, dither=0), model_config=OfflineModelConfig(transducer=OfflineTransducerModelConfig(encoder_filename="", decoder_filename="", joiner_filename=""), paraformer=OfflineParaformerModelConfig(model=""), nemo_ctc=OfflineNemoEncDecCtcModelConfig(model=""), whisper=OfflineWhisperModelConfig(encoder="./large-v3-encoder.onnx", decoder="./large-v3-decoder.onnx", language="", task="transcribe", tail_paddings=-1), tdnn=OfflineTdnnModelConfig(model=""), zipformer_ctc=OfflineZipformerCtcModelConfig(model=""), wenet_ctc=OfflineWenetCtcModelConfig(model=""), telespeech_ctc="", tokens="./large-v3-tokens.txt", num_threads=2, debug=False, provider="cpu", model_type="", modeling_unit="cjkchar", bpe_vocab=""), lm_config=OfflineLMConfig(model="", scale=0.5), ctc_fst_decoder_config=OfflineCtcFstDecoderConfig(graph="", max_active=3000), decoding_method="greedy_search", max_active_paths=4, hotwords_file="", hotwords_score=1.5, blank_penalty=0, rule_fsts="", rule_fars="")
    Creating recognizer ...
    Started
    Done!

    ./test_wavs/0.wav
    {"text": " after early nightfall the yellow lamps would light up here and there the squalid quarter of the brothels", "timestamps": [], "tokens":[" after", " early", " night", "fall", " the", " yellow", " lamps", " would", " light", " up", " here", " and", " there", " the", " squ", "alid", " quarter", " of", " the", " broth", "els"], "words": []}
    ----
    num threads: 2
    decoding method: greedy_search
    Elapsed seconds: 54.070 s
    Real time factor (RTF): 54.070 / 6.625 = 8.162

    real	1m32.107s
    user	1m39.877s
    sys	0m10.405s

Run with CPU (int8)
-------------------

.. code-block:: bash

  cd /content

  exe=$PWD/sherpa-onnx/build/bin/sherpa-onnx-offline

  cd sherpa-onnx-whisper-large-v3

  time $exe \
    --whisper-encoder=./large-v3-encoder.int8.onnx \
    --whisper-decoder=./large-v3-decoder.int8.onnx \
    --tokens=./large-v3-tokens.txt \
    --num-threads=2 \
    ./test_wavs/0.wav

The logs are given below::

  /content/sherpa-onnx/sherpa-onnx/csrc/parse-options.cc:Read:375 /content/sherpa-onnx/build/bin/sherpa-onnx-offline --whisper-encoder=./large-v3-encoder.int8.onnx --whisper-decoder=./large-v3-decoder.int8.onnx --tokens=./large-v3-tokens.txt --num-threads=2 ./test_wavs/0.wav

  OfflineRecognizerConfig(feat_config=FeatureExtractorConfig(sampling_rate=16000, feature_dim=80, low_freq=20, high_freq=-400, dither=0), model_config=OfflineModelConfig(transducer=OfflineTransducerModelConfig(encoder_filename="", decoder_filename="", joiner_filename=""), paraformer=OfflineParaformerModelConfig(model=""), nemo_ctc=OfflineNemoEncDecCtcModelConfig(model=""), whisper=OfflineWhisperModelConfig(encoder="./large-v3-encoder.int8.onnx", decoder="./large-v3-decoder.int8.onnx", language="", task="transcribe", tail_paddings=-1), tdnn=OfflineTdnnModelConfig(model=""), zipformer_ctc=OfflineZipformerCtcModelConfig(model=""), wenet_ctc=OfflineWenetCtcModelConfig(model=""), telespeech_ctc="", tokens="./large-v3-tokens.txt", num_threads=2, debug=False, provider="cpu", model_type="", modeling_unit="cjkchar", bpe_vocab=""), lm_config=OfflineLMConfig(model="", scale=0.5), ctc_fst_decoder_config=OfflineCtcFstDecoderConfig(graph="", max_active=3000), decoding_method="greedy_search", max_active_paths=4, hotwords_file="", hotwords_score=1.5, blank_penalty=0, rule_fsts="", rule_fars="")
  Creating recognizer ...
  Started
  Done!

  ./test_wavs/0.wav
  {"text": " after early nightfall the yellow lamps would light up here and there the squalid quarter of the brothels", "timestamps": [], "tokens":[" after", " early", " night", "fall", " the", " yellow", " lamps", " would", " light", " up", " here", " and", " there", " the", " squ", "alid", " quarter", " of", " the", " broth", "els"], "words": []}
  ----
  num threads: 2
  decoding method: greedy_search
  Elapsed seconds: 49.991 s
  Real time factor (RTF): 49.991 / 6.625 = 7.546

  real	1m15.555s
  user	1m41.488s
  sys	0m9.156s


Run with GPU (float32)
----------------------

.. code-block:: bash

  cd /content
  exe=$PWD/sherpa-onnx/build/bin/sherpa-onnx-offline

  cd sherpa-onnx-whisper-large-v3

  time $exe \
    --whisper-encoder=./large-v3-encoder.onnx \
    --whisper-decoder=./large-v3-decoder.onnx \
    --tokens=./large-v3-tokens.txt \
    --provider=cuda \
    --num-threads=2 \
    ./test_wavs/0.wav

The logs are given below::

  /content/sherpa-onnx/sherpa-onnx/csrc/parse-options.cc:Read:375 /content/sherpa-onnx/build/bin/sherpa-onnx-offline --whisper-encoder=./large-v3-encoder.onnx --whisper-decoder=./large-v3-decoder.onnx --tokens=./large-v3-tokens.txt --provider=cuda --num-threads=2 ./test_wavs/0.wav

  OfflineRecognizerConfig(feat_config=FeatureExtractorConfig(sampling_rate=16000, feature_dim=80, low_freq=20, high_freq=-400, dither=0), model_config=OfflineModelConfig(transducer=OfflineTransducerModelConfig(encoder_filename="", decoder_filename="", joiner_filename=""), paraformer=OfflineParaformerModelConfig(model=""), nemo_ctc=OfflineNemoEncDecCtcModelConfig(model=""), whisper=OfflineWhisperModelConfig(encoder="./large-v3-encoder.onnx", decoder="./large-v3-decoder.onnx", language="", task="transcribe", tail_paddings=-1), tdnn=OfflineTdnnModelConfig(model=""), zipformer_ctc=OfflineZipformerCtcModelConfig(model=""), wenet_ctc=OfflineWenetCtcModelConfig(model=""), telespeech_ctc="", tokens="./large-v3-tokens.txt", num_threads=2, debug=False, provider="cuda", model_type="", modeling_unit="cjkchar", bpe_vocab=""), lm_config=OfflineLMConfig(model="", scale=0.5), ctc_fst_decoder_config=OfflineCtcFstDecoderConfig(graph="", max_active=3000), decoding_method="greedy_search", max_active_paths=4, hotwords_file="", hotwords_score=1.5, blank_penalty=0, rule_fsts="", rule_fars="")
  Creating recognizer ...
  Started
  Done!

  ./test_wavs/0.wav
  {"text": " after early nightfall the yellow lamps would light up here and there the squalid quarter of the brothels", "timestamps": [], "tokens":[" after", " early", " night", "fall", " the", " yellow", " lamps", " would", " light", " up", " here", " and", " there", " the", " squ", "alid", " quarter", " of", " the", " broth", "els"], "words": []}
  ----
  num threads: 2
  decoding method: greedy_search
  Elapsed seconds: 5.910 s
  Real time factor (RTF): 5.910 / 6.625 = 0.892

  real	0m26.996s
  user	0m12.854s
  sys	0m4.486s

.. note::

   The above command is run within a colab notebook using Tesla T4 GPU.
   You can see the RTF is less than 1.

   If you has some more performant GPU, you would get an even lower RTF.

Run with GPU (int8)
-------------------

.. code-block:: bash

  cd /content
  exe=$PWD/sherpa-onnx/build/bin/sherpa-onnx-offline

  cd sherpa-onnx-whisper-large-v3

  time $exe \
    --whisper-encoder=./large-v3-encoder.int8.onnx \
    --whisper-decoder=./large-v3-decoder.int8.onnx \
    --tokens=./large-v3-tokens.txt \
    --provider=cuda \
    --num-threads=2 \
    ./test_wavs/0.wav

The logs are given below::

  /content/sherpa-onnx/sherpa-onnx/csrc/parse-options.cc:Read:375 /content/sherpa-onnx/build/bin/sherpa-onnx-offline --whisper-encoder=./large-v3-encoder.int8.onnx --whisper-decoder=./large-v3-decoder.int8.onnx --tokens=./large-v3-tokens.txt --provider=cuda --num-threads=2 ./test_wavs/0.wav

  OfflineRecognizerConfig(feat_config=FeatureExtractorConfig(sampling_rate=16000, feature_dim=80, low_freq=20, high_freq=-400, dither=0), model_config=OfflineModelConfig(transducer=OfflineTransducerModelConfig(encoder_filename="", decoder_filename="", joiner_filename=""), paraformer=OfflineParaformerModelConfig(model=""), nemo_ctc=OfflineNemoEncDecCtcModelConfig(model=""), whisper=OfflineWhisperModelConfig(encoder="./large-v3-encoder.int8.onnx", decoder="./large-v3-decoder.int8.onnx", language="", task="transcribe", tail_paddings=-1), tdnn=OfflineTdnnModelConfig(model=""), zipformer_ctc=OfflineZipformerCtcModelConfig(model=""), wenet_ctc=OfflineWenetCtcModelConfig(model=""), telespeech_ctc="", tokens="./large-v3-tokens.txt", num_threads=2, debug=False, provider="cuda", model_type="", modeling_unit="cjkchar", bpe_vocab=""), lm_config=OfflineLMConfig(model="", scale=0.5), ctc_fst_decoder_config=OfflineCtcFstDecoderConfig(graph="", max_active=3000), decoding_method="greedy_search", max_active_paths=4, hotwords_file="", hotwords_score=1.5, blank_penalty=0, rule_fsts="", rule_fars="")
  Creating recognizer ...
  Started
  Done!

  ./test_wavs/0.wav
  {"text": " after early nightfall the yellow lamps would light up here and there the squalid quarter of the brothels", "timestamps": [], "tokens":[" after", " early", " night", "fall", " the", " yellow", " lamps", " would", " light", " up", " here", " and", " there", " the", " squ", "alid", " quarter", " of", " the", " broth", "els"], "words": []}
  ----
  num threads: 2
  decoding method: greedy_search
  Elapsed seconds: 19.190 s
  Real time factor (RTF): 19.190 / 6.625 = 2.897

  real	0m46.850s
  user	0m50.007s
  sys	0m8.013s

Fix issues about running on GPU
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you get errors like below::

    what():  /onnxruntime_src/onnxruntime/core/session/provider_bridge_ort.cc:1426
    onnxruntime::Provider& onnxruntime::ProviderLibrary::Get()
    [ONNXRuntimeError] : 1 : FAIL :
    Failed to load library libonnxruntime_providers_cuda.so with error:
    libcublasLt.so.11: cannot open shared object file: No such file or directory

please follow `<https://www.google.com/url?q=https%3A%2F%2Fk2-fsa.github.io%2Fk2%2Finstallation%2Fcuda-cudnn.html>`_
to install CUDA toolkit.

To determine which version of CUDA toolkit to install, please read
`<https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html>`_
to figure it out.

For instance, if onnxruntime v1.18.1 is used in `sherpa-onnx`_, we have to install
CUDA 11.8 according to `<https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html>`_

colab
-----

Please see the following colab notebook
|sherpa-onnx with whisper large-v3 colab notebook|.

It walks you step by step to try the exported large-v3 onnx model with `sherpa-onnx`_
on CPU as well as on GPU.

.. |sherpa-onnx with whisper large-v3 colab notebook| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://github.com/k2-fsa/colab/blob/master/sherpa-onnx/sherpa_onnx_whisper_large_v3.ipynb

