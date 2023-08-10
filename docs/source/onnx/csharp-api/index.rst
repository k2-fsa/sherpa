.. _sherpa-onnx-csharp-api:

C# API
======

In this section, we describe how to use the ``C#``
API examples of `sherpa-onnx`_.

The ``C#`` API of `sherpa-onnx`_ supports both streaming and non-streaming speech recognition.

The following table lists some ``C#`` API examples:

.. list-table::

 * - Description
   - URL
 * - Decode a file with **non-streaming** models
   - `<https://github.com/k2-fsa/sherpa-onnx/tree/master/dotnet-examples/offline-decode-files>`_
 * - Decode a file with **streaming** models
   - `<https://github.com/k2-fsa/sherpa-onnx/tree/master/dotnet-examples/online-decode-files>`_
 * - **Real-time** speech recognition from a ``microphone``
   - `<https://github.com/k2-fsa/sherpa-onnx/tree/master/dotnet-examples/speech-recognition-from-microphone>`_

You can find the implementation in the following files:

  - API for **streaming** speech recognition

    `<https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/dotnet/online.cs>`_

  - API for **non-streaming** speech recognition

    `<https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/dotnet/offline.cs>`_

We also provide a nuget package for `sherpa-onnx`_:

  `<https://www.nuget.org/packages/org.k2fsa.sherpa.onnx>`_

You can use the following statement in your ``csproj`` file to introduce
the dependency on `sherpa-onnx`_:

.. code-block:: bash

   <PackageReference Include="org.k2fsa.sherpa.onnx" Version="*" />

One thing to note is that we have provided pre-built libraries for ``C#`` so that you don't need
to build `sherpa-onnx`_ by yourself when using the ``C#`` API.

In the following, we describe how to run our provided ``C#`` API examples.

.. note::

   Before you continue, please make sure you have installed `.Net <https://en.wikipedia.org/wiki/.NET>`_.
   If not, please follow `<https://dotnet.microsoft.com/en-us/download>`_ to install ``.Net``.

.. hint::

    ``.Net`` supports Windows, macOS, and Linux.

Decode files with non-streaming models
--------------------------------------

First, let us build the example:

.. code-block:: bash

  git clone https://github.com/k2-fsa/sherpa-onnx
  cd sherpa-onnx/dotnet-examples/offline-decode-files/
  dotnet build -c Release
  ./bin/Release/net6.0/offline-decode-files --help

You will find the following output:

.. code-block:: bash

  # Zipformer

  dotnet run \
    --tokens=./sherpa-onnx-zipformer-en-2023-04-01/tokens.txt \
    --encoder=./sherpa-onnx-zipformer-en-2023-04-01/encoder-epoch-99-avg-1.onnx \
    --decoder=./sherpa-onnx-zipformer-en-2023-04-01/decoder-epoch-99-avg-1.onnx \
    --joiner=./sherpa-onnx-zipformer-en-2023-04-01/joiner-epoch-99-avg-1.onnx \
    --files ./sherpa-onnx-zipformer-en-2023-04-01/test_wavs/0.wav \
    ./sherpa-onnx-zipformer-en-2023-04-01/test_wavs/1.wav \
    ./sherpa-onnx-zipformer-en-2023-04-01/test_wavs/8k.wav

  Please refer to
  https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/index.html
  to download pre-trained non-streaming zipformer models.

  # Paraformer

  dotnet run \
    --tokens=./sherpa-onnx-paraformer-zh-2023-03-28/tokens.txt \
    --paraformer=./sherpa-onnx-paraformer-zh-2023-03-28/model.onnx \
    --files ./sherpa-onnx-zipformer-en-2023-04-01/test_wavs/0.wav \
    ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/0.wav \
    ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/1.wav \
    ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/2.wav \
    ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/8k.wav
  Please refer to
  https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-paraformer/index.html
  to download pre-trained paraformer models

  # NeMo CTC

  dotnet run \
    --tokens=./sherpa-onnx-nemo-ctc-en-conformer-medium/tokens.txt \
    --nemo-ctc=./sherpa-onnx-nemo-ctc-en-conformer-medium/model.onnx \
    --num-threads=1 \
    --files ./sherpa-onnx-nemo-ctc-en-conformer-medium/test_wavs/0.wav \
    ./sherpa-onnx-nemo-ctc-en-conformer-medium/test_wavs/1.wav \
    ./sherpa-onnx-nemo-ctc-en-conformer-medium/test_wavs/8k.wav

  Please refer to
  https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-ctc/index.html
  to download pre-trained paraformer models

  Copyright (c) 2023 Xiaomi Corporation

    --tokens              Path to tokens.txt
    --encoder             Path to encoder.onnx. Used only for transducer models
    --decoder             Path to decoder.onnx. Used only for transducer models
    --joiner              Path to joiner.onnx. Used only for transducer models
    --paraformer          Path to model.onnx. Used only for paraformer models
    --nemo-ctc            Path to model.onnx. Used only for NeMo CTC models
    --num-threads         (Default: 1) Number of threads for computation
    --decoding-method     (Default: greedy_search) Valid decoding methods are:
                          greedy_search, modified_beam_search
    --max-active-paths    (Default: 4) Used only when --decoding--method is
                          modified_beam_search.
                          It specifies number of active paths to keep during the
                          search
    --files               Required. Audio files for decoding
    --help                Display this help screen.
    --version             Display version information.

Now let us refer to :ref:`sherpa-onnx-pre-trained-models` to download a non-streaming model.

We give several examples below for demonstration.

Non-streaming transducer
^^^^^^^^^^^^^^^^^^^^^^^^

We will use :ref:`sherpa-onnx-zipformer-en-2023-06-26-english` as an example.

First, let us download it:

.. code-block:: bash

  cd sherpa-onnx/dotnet-examples/offline-decode-files
  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/sherpa-onnx-zipformer-en-2023-06-26
  cd sherpa-onnx-zipformer-en-2023-06-26
  git lfs pull --include "*.onnx"
  cd ..

Now we can use:

.. code-block:: bash

  dotnet run -c Release \
    --encoder ./sherpa-onnx-zipformer-en-2023-06-26/encoder-epoch-99-avg-1.onnx \
    --decoder ./sherpa-onnx-zipformer-en-2023-06-26/decoder-epoch-99-avg-1.onnx \
    --joiner ./sherpa-onnx-zipformer-en-2023-06-26/joiner-epoch-99-avg-1.onnx \
    --tokens ./sherpa-onnx-zipformer-en-2023-06-26/tokens.txt \
    --files ./sherpa-onnx-zipformer-en-2023-06-26/test_wavs/0.wav \
    ./sherpa-onnx-zipformer-en-2023-06-26/test_wavs/1.wav \
    ./sherpa-onnx-zipformer-en-2023-06-26/test_wavs/8k.wav

It should give you the following output:

.. code-block:: bash

  /Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/offline-stream.cc:AcceptWaveformImpl:117 Creating a resampler:
     in_sample_rate: 8000
     output_sample_rate: 16000

  --------------------
  ./sherpa-onnx-zipformer-en-2023-06-26/test_wavs/0.wav
   AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE THE SQUALID QUARTER OF THE BROTHELS
  --------------------
  ./sherpa-onnx-zipformer-en-2023-06-26/test_wavs/1.wav
   GOD AS A DIRECT CONSEQUENCE OF THE SIN WHICH MAN THUS PUNISHED HAD GIVEN HER A LOVELY CHILD WHOSE PLACE WAS ON THAT SAME DISHONORED BOSOM TO CONNECT HER PARENT FOREVER WITH THE RACE AND DESCENT OF MORTALS AND TO BE FINALLY A BLESSED SOUL IN HEAVEN
  --------------------
  ./sherpa-onnx-zipformer-en-2023-06-26/test_wavs/8k.wav
   YET THESE THOUGHTS AFFECTED HESTER PRYNNE LESS WITH HOPE THAN APPREHENSION
  --------------------

Non-streaming paraformer
^^^^^^^^^^^^^^^^^^^^^^^^

We will use :ref:`sherpa_onnx_offline_paraformer_zh_2023_03_28_chinese` as an example.

First, let us download it:

.. code-block:: bash

  cd sherpa-onnx/dotnet-examples/offline-decode-files
  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/sherpa-onnx-paraformer-zh-2023-03-28
  cd sherpa-onnx-paraformer-zh-2023-03-28
  git lfs pull --include "*.onnx"
  cd ..

Now we can use:

.. code-block:: bash

  dotnet run -c Release \
    --paraformer ./sherpa-onnx-paraformer-zh-2023-03-28/model.int8.onnx \
    --tokens ./sherpa-onnx-paraformer-zh-2023-03-28/tokens.txt \
    --files ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/0.wav \
    ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/1.wav \
    ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/8k.wav

It should give you the following output:

.. code-block:: bash

  /Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/offline-stream.cc:AcceptWaveformImpl:117 Creating a resampler:
     in_sample_rate: 8000
     output_sample_rate: 16000

  --------------------
  ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/0.wav
  对我做了介绍啊那么我想说的是呢大家如果对我的研究感兴趣呢你
  --------------------
  ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/1.wav
  重点呢想谈三个问题首先呢就是这一轮全球金融动荡的表现
  --------------------
  ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/8k.wav
  甚至出现交易几乎停滞的情况
  --------------------

Non-streaming CTC model from NeMo
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We will use :ref:`stt-en-conformer-ctc-medium-nemo-sherpa-onnx` as an example.

First, let us download it:

.. code-block:: bash

  cd sherpa-onnx/dotnet-examples/offline-decode-files
  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/sherpa-onnx-nemo-ctc-en-conformer-medium
  cd sherpa-onnx-nemo-ctc-en-conformer-medium
  git lfs pull --include "*.onnx"
  cd ..

Now we can use:

.. code-block:: bash

  dotnet run -c Release \
    --nemo-ctc ./sherpa-onnx-nemo-ctc-en-conformer-medium/model.onnx \
    --tokens ./sherpa-onnx-nemo-ctc-en-conformer-medium/tokens.txt \
    --files ./sherpa-onnx-nemo-ctc-en-conformer-medium/test_wavs/0.wav \
    ./sherpa-onnx-nemo-ctc-en-conformer-medium/test_wavs/1.wav \
    ./sherpa-onnx-nemo-ctc-en-conformer-medium/test_wavs/8k.wav

It should give you the following output:

.. code-block:: bash

  /Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/offline-stream.cc:AcceptWaveformImpl:117 Creating a resampler:
     in_sample_rate: 8000
     output_sample_rate: 16000

  --------------------
  ./sherpa-onnx-nemo-ctc-en-conformer-medium/test_wavs/0.wav
   after early nightfall the yellow lamps would light up here and there the squalid quarter of the brothels
  --------------------
  ./sherpa-onnx-nemo-ctc-en-conformer-medium/test_wavs/1.wav
   god as a direct consequence of the sin which man thus punished had given her a lovely child whose place was on that same dishonored bosom to connect her parent for ever with the race and descent of mortals and to be finally a blessed soul in heaven
  --------------------
  ./sherpa-onnx-nemo-ctc-en-conformer-medium/test_wavs/8k.wav
   yet these thoughts affected hester pryne less with hope than apprehension
  --------------------

Decode files with streaming models
----------------------------------

First, let us build the example:

.. code-block:: bash

  git clone https://github.com/k2-fsa/sherpa-onnx
  cd sherpa-onnx/dotnet-examples/online-decode-files
  dotnet build -c Release
  ./bin/Release/net6.0/online-decode-files --help

You will find the following output:

.. code-block:: bash

    dotnet run \
      --tokens=./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt \
      --encoder=./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.onnx \
      --decoder=./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx \
      --joiner=./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.onnx \
      --num-threads=2 \
      --decoding-method=modified_beam_search \
      --debug=false \
      --files ./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/test_wavs/0.wav \
      ./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/test_wavs/1.wav

    Please refer to
    https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/index.html
    to download pre-trained streaming models.

    Copyright (c) 2023 Xiaomi Corporation

      --tokens                        Required. Path to tokens.txt
      --provider                      (Default: cpu) Provider, e.g., cpu, coreml
      --encoder                       Required. Path to encoder.onnx
      --decoder                       Required. Path to decoder.onnx
      --joiner                        Required. Path to joiner.onnx
      --num-threads                   (Default: 1) Number of threads for computation
      --decoding-method               (Default: greedy_search) Valid decoding
                                      methods are: greedy_search,
                                      modified_beam_search
      --debug                         (Default: false) True to show model info
                                      during loading
      --sample-rate                   (Default: 16000) Sample rate of the data used
                                      to train the model
      --max-active-paths              (Default: 4) Used only when --decoding--method
                                      is modified_beam_search.
                                      It specifies number of active paths to keep
                                      during the search
      --enable-endpoint               (Default: false) True to enable endpoint
                                      detection.
      --rule1-min-trailing-silence    (Default: 2.4) An endpoint is detected if
                                      trailing silence in seconds is
                                      larger than this value even if nothing has
                                      been decoded. Used only when --enable-endpoint
                                      is true.
      --rule2-min-trailing-silence    (Default: 1.2) An endpoint is detected if
                                      trailing silence in seconds is
                                      larger than this value after something that is
                                      not blank has been decoded. Used
                                      only when --enable-endpoint is true.
      --rule3-min-utterance-length    (Default: 20) An endpoint is detected if the
                                      utterance in seconds is
                                      larger than this value. Used only when
                                      --enable-endpoint is true.
      --files                         Required. Audio files for decoding
      --help                          Display this help screen.
      --version                       Display version information.

Now let us refer to :ref:`sherpa-onnx-pre-trained-models` to download a streaming model.

We give one example below for demonstration.

Streaming transducer
^^^^^^^^^^^^^^^^^^^^

We will use :ref:`sherpa-onnx-streaming-zipformer-en-2023-06-26-english` as an example.

First, let us download it:

.. code-block:: bash

  cd sherpa-onnx/dotnet-examples/online-decode-files/
  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26
  cd sherpa-onnx-streaming-zipformer-en-2023-06-26
  git lfs pull --include "*.onnx"
  cd ..

Now we can use:

.. code-block:: bash

  dotnet run -c Release \
    --encoder ./sherpa-onnx-streaming-zipformer-en-2023-06-26/encoder-epoch-99-avg-1-chunk-16-left-128.onnx \
    --decoder ./sherpa-onnx-streaming-zipformer-en-2023-06-26/decoder-epoch-99-avg-1-chunk-16-left-128.onnx \
    --joiner ./sherpa-onnx-streaming-zipformer-en-2023-06-26/joiner-epoch-99-avg-1-chunk-16-left-128.onnx \
    --tokens ./sherpa-onnx-streaming-zipformer-en-2023-06-26/tokens.txt \
    --files ./sherpa-onnx-streaming-zipformer-en-2023-06-26/test_wavs/0.wav \
    ./sherpa-onnx-streaming-zipformer-en-2023-06-26/test_wavs/1.wav \
    ./sherpa-onnx-streaming-zipformer-en-2023-06-26/test_wavs/8k.wav

You will find the following output:

.. code-block:: bash

    /Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/features.cc:AcceptWaveform:76 Creating a resampler:
       in_sample_rate: 8000
       output_sample_rate: 16000

    --------------------
    ./sherpa-onnx-streaming-zipformer-en-2023-06-26/test_wavs/0.wav
     AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE THE SQUALID QUARTER OF THE BROTHELS
    --------------------
    ./sherpa-onnx-streaming-zipformer-en-2023-06-26/test_wavs/1.wav
     GOD AS A DIRECT CONSEQUENCE OF THE SIN WHICH MAN THUS PUNISHED HAD GIVEN HER A LOVELY CHILD WHOSE PLACE WAS ON THAT SAME DISHONOURED BOSOM TO CONNECT HER PARENT FOR EVER WITH THE RACE AND DESCENT OF MORTALS AND TO BE FINALLY A BLESSED SOUL IN HEAVEN
    --------------------
    ./sherpa-onnx-streaming-zipformer-en-2023-06-26/test_wavs/8k.wav
     YET THESE THOUGHTS AFFECTED HESTER PRYNNE LESS WITH HOPE THAN APPREHENSION
    --------------------

Real-time speech recognition from microphone
--------------------------------------------

First, let us build the example:

.. code-block:: bash

  git clone https://github.com/k2-fsa/sherpa-onnx
  cd sherpa-onnx/dotnet-examples/speech-recognition-from-microphone
  dotnet build -c Release
  ./bin/Release/net6.0/speech-recognition-from-microphone --help

You will find the following output:

.. code-block:: bash

    dotnet run -c Release \
      --tokens ./icefall-asr-zipformer-streaming-wenetspeech-20230615/data/lang_char/tokens.txt \
      --encoder ./icefall-asr-zipformer-streaming-wenetspeech-20230615/exp/encoder-epoch-12-avg-4-chunk-16-left-128.onnx \
      --decoder ./icefall-asr-zipformer-streaming-wenetspeech-20230615/exp/decoder-epoch-12-avg-4-chunk-16-left-128.onnx \
      --joiner ./icefall-asr-zipformer-streaming-wenetspeech-20230615/exp/joiner-epoch-12-avg-4-chunk-16-left-128.onnx \

    Please refer to
    https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/index.html
    to download pre-trained streaming models.

    Copyright (c) 2023 Xiaomi Corporation

      --tokens                        Required. Path to tokens.txt
      --provider                      (Default: cpu) Provider, e.g., cpu, coreml
      --encoder                       Required. Path to encoder.onnx
      --decoder                       Required. Path to decoder.onnx
      --joiner                        Required. Path to joiner.onnx
      --num-threads                   (Default: 1) Number of threads for computation
      --decoding-method               (Default: greedy_search) Valid decoding
                                      methods are: greedy_search,
                                      modified_beam_search
      --debug                         (Default: false) True to show model info
                                      during loading
      --sample-rate                   (Default: 16000) Sample rate of the data used
                                      to train the model
      --max-active-paths              (Default: 4) Used only when --decoding--method
                                      is modified_beam_search.
                                      It specifies number of active paths to keep
                                      during the search
      --enable-endpoint               (Default: true) True to enable endpoint
                                      detection.
      --rule1-min-trailing-silence    (Default: 2.4) An endpoint is detected if
                                      trailing silence in seconds is
                                      larger than this value even if nothing has
                                      been decoded. Used only when --enable-endpoint
                                      is true.
      --rule2-min-trailing-silence    (Default: 0.8) An endpoint is detected if
                                      trailing silence in seconds is
                                      larger than this value after something that is
                                      not blank has been decoded. Used
                                      only when --enable-endpoint is true.
      --rule3-min-utterance-length    (Default: 20) An endpoint is detected if the
                                      utterance in seconds is
                                      larger than this value. Used only when
                                      --enable-endpoint is true.
      --help                          Display this help screen.
      --version                       Display version information.

Now let us refer to :ref:`sherpa-onnx-pre-trained-models` to download a streaming model.

We give one example below for demonstration.

Streaming transducer
^^^^^^^^^^^^^^^^^^^^

We will use :ref:`sherpa-onnx-streaming-zipformer-en-2023-06-26-english` as an example.

First, let us download it:

.. code-block:: bash

  cd sherpa-onnx/dotnet-examples/speech-recognition-from-microphone
  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26
  cd sherpa-onnx-streaming-zipformer-en-2023-06-26
  git lfs pull --include "*.onnx"
  cd ..

Now we can use:

.. code-block:: bash

  dotnet run -c Release \
    --encoder ./sherpa-onnx-streaming-zipformer-en-2023-06-26/encoder-epoch-99-avg-1-chunk-16-left-128.onnx \
    --decoder ./sherpa-onnx-streaming-zipformer-en-2023-06-26/decoder-epoch-99-avg-1-chunk-16-left-128.onnx \
    --joiner ./sherpa-onnx-streaming-zipformer-en-2023-06-26/joiner-epoch-99-avg-1-chunk-16-left-128.onnx \
    --tokens ./sherpa-onnx-streaming-zipformer-en-2023-06-26/tokens.txt

You will find the following output:

.. code-block:: bash

    PortAudio V19.7.0-devel, revision 147dd722548358763a8b649b3e4b41dfffbcfbb6
    Number of devices: 5
     Device 0
       Name: Background Music
       Max input channels: 2
       Default sample rate: 44100
     Device 1
       Name: Background Music (UI Sounds)
       Max input channels: 2
       Default sample rate: 44100
     Device 2
       Name: MacBook Pro Microphone
       Max input channels: 1
       Default sample rate: 48000
     Device 3
       Name: MacBook Pro Speakers
       Max input channels: 0
       Default sample rate: 48000
     Device 4
       Name: WeMeet Audio Device
       Max input channels: 2
       Default sample rate: 48000

    Use default device 2 (MacBook Pro Microphone)
    StreamParameters [
      device=2
      channelCount=1
      sampleFormat=Float32
      suggestedLatency=0.034520833333333334
      hostApiSpecificStreamInfo?=[False]
    ]
    Started! Please speak

    0:  THIS IS A TEST
    1:  THIS IS A SECOND TEST

colab
-----

We provide a colab notebook
|Sherpa-onnx csharp api example colab notebook|
for you to try the ``C#`` API examples of `sherpa-onnx`_.

.. |Sherpa-onnx csharp api example colab notebook| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://github.com/k2-fsa/colab/blob/master/sherpa-onnx/sherpa_onnx_csharp_api_example.ipynb
