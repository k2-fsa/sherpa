.. _sherpa-onnx-go-api:

Go API
======

In this section, we describe how to use the `Go`_
API of `sherpa-onnx`_.

The `Go`_ API of `sherpa-onnx`_ supports both streaming and non-streaming speech recognition.

The following table lists some `Go`_ API examples:

.. list-table::

 * - Description
   - URL
 * - Decode a file with **non-streaming** models
   - `<https://github.com/k2-fsa/sherpa-onnx/tree/master/go-api-examples/non-streaming-decode-files>`_
 * - Decode a file with **streaming** models
   - `<https://github.com/k2-fsa/sherpa-onnx/tree/master/go-api-examples/streaming-decode-files>`_
 * - **Real-time** speech recognition from a ``microphone``
   - `<https://github.com/k2-fsa/sherpa-onnx/tree/master/go-api-examples/real-time-speech-recognition-from-microphone>`_

One thing to note is that we have provided pre-built libraries for `Go`_ so that you don't need
to build `sherpa-onnx`_ by yourself when using the `Go`_ API.

To make supporting multiple platforms easier, we split the `Go`_ API of `sherpa-onnx`_ into
multiple packages, as listed in the following table:

.. list-table::

 * - OS
   - Package name
   - Supported Arch
   - Doc
 * - Linux
   - `sherpa-onnx-go-linux <https://github.com/k2-fsa/sherpa-onnx-go-linux>`_
   - ``x86_64``, ``aarch64``, ``arm``
   - `<https://pkg.go.dev/github.com/k2-fsa/sherpa-onnx-go-linux>`_
 * - macOS
   - `sherpa-onnx-go-macos <https://github.com/k2-fsa/sherpa-onnx-go-macos>`_
   - ``x86_64``, ``aarch64``
   - `<https://pkg.go.dev/github.com/k2-fsa/sherpa-onnx-go-macos>`_
 * - Windows
   - `sherpa-onnx-go-windows <https://github.com/k2-fsa/sherpa-onnx-go-windows>`_
   - ``x86_64``, ``x86``
   - `<https://pkg.go.dev/github.com/k2-fsa/sherpa-onnx-go-windows>`_

To simplify the usage, we have provided a single `Go`_ package for `sherpa-onnx`_ that
supports multiple operating systems. It can be found at

  `<https://github.com/k2-fsa/sherpa-onnx-go>`_

.. hint::

   Such a design is insipred by the following article:

    `Cross platform Go modules for giants <https://kobi.one/2021/08/22/cross-platform-go-modules-for-giants.html>`_.

You can use the following ``import`` to import `sherpa-onnx-go`_
into your `Go`_ project:

.. code-block:: go

  import (
    sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
  )

In the following, we describe how to run our provided `Go`_ API examples.

.. note::

   Before you continue, please make sure you have installed `Go`_.
   If not, please follow `<https://go.dev/doc/install>`_ to install `Go`_.

.. hint::

   You need to enable `cgo <https://pkg.go.dev/cmd/cgo>`_ to build `sherpa-onnx-go`_.

Decode files with non-streaming models
--------------------------------------

First, let us build the example:

.. code-block:: bash

  git clone https://github.com/k2-fsa/sherpa-onnx
  cd sherpa-onnx/go-api-examples/non-streaming-decode-files
  go mod tidy
  go build
  ./non-streaming-decode-files --help

You will find the following output:

.. code-block:: bash

  Usage of ./non-streaming-decode-files:
        --debug int                Whether to show debug message
        --decoder string           Path to the decoder model
        --decoding-method string   Decoding method. Possible values: greedy_search, modified_beam_search (default "greedy_search")
        --encoder string           Path to the encoder model
        --joiner string            Path to the joiner model
        --lm-model string          Optional. Path to the LM model
        --lm-scale float32         Optional. Scale for the LM model (default 1)
        --max-active-paths int     Used only when --decoding-method is modified_beam_search (default 4)
        --model-type string        Optional. Used for loading the model in a faster way
        --nemo-ctc string          Path to the NeMo CTC model
        --num-threads int          Number of threads for computing (default 1)
        --paraformer string        Path to the paraformer model
        --provider string          Provider to use (default "cpu")
        --tokens string            Path to the tokens file
  pflag: help requested

Congratulations! You have successfully built your first `Go`_ API example for speech recognition.

.. note::

   If you are using Windows and don't see any output after running ``./non-streaming-decode-files --help``,
   please copy ``*.dll`` from `<https://github.com/k2-fsa/sherpa-onnx-go-windows/tree/master/lib/x86_64-pc-windows-gnu>`_ (for Win64)
   or `<https://github.com/k2-fsa/sherpa-onnx-go-windows/tree/master/lib/i686-pc-windows-gnu>`_ (for Win32)
   to the directory ``sherpa-onnx/go-api-examples/non-streaming-decode-files``.

Now let us refer to :ref:`sherpa-onnx-pre-trained-models` to download a non-streaming model.

We give several examples below for demonstration.

Non-streaming transducer
^^^^^^^^^^^^^^^^^^^^^^^^

We will use :ref:`sherpa-onnx-zipformer-en-2023-06-26-english` as an example.

First, let us download it:

.. code-block:: bash

  cd sherpa-onnx/go-api-examples/non-streaming-decode-files
  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/sherpa-onnx-zipformer-en-2023-06-26
  cd sherpa-onnx-zipformer-en-2023-06-26
  git lfs pull --include "*.onnx"
  cd ..

Now we can use:

.. code-block:: bash

  ./non-streaming-decode-files \
    --encoder ./sherpa-onnx-zipformer-en-2023-06-26/encoder-epoch-99-avg-1.onnx \
    --decoder ./sherpa-onnx-zipformer-en-2023-06-26/decoder-epoch-99-avg-1.onnx \
    --joiner ./sherpa-onnx-zipformer-en-2023-06-26/joiner-epoch-99-avg-1.onnx \
    --tokens ./sherpa-onnx-zipformer-en-2023-06-26/tokens.txt \
    --model-type transducer \
    ./sherpa-onnx-zipformer-en-2023-06-26/test_wavs/0.wav

It should give you the following output:

.. code-block:: bash

  2023/08/10 14:52:48.723098 Reading ./sherpa-onnx-zipformer-en-2023-06-26/test_wavs/0.wav
  2023/08/10 14:52:48.741042 Initializing recognizer (may take several seconds)
  2023/08/10 14:52:51.998848 Recognizer created!
  2023/08/10 14:52:51.998870 Start decoding!
  2023/08/10 14:52:52.258818 Decoding done!
  2023/08/10 14:52:52.258847  after early nightfall the yellow lamps would light up here and there the squalid quarter of the brothels
  2023/08/10 14:52:52.258952 Wave duration: 6.625 seconds

Non-streaming paraformer
^^^^^^^^^^^^^^^^^^^^^^^^

We will use :ref:`sherpa_onnx_offline_paraformer_zh_2023_03_28_chinese` as an example.

First, let us download it:

.. code-block:: bash

  cd sherpa-onnx/go-api-examples/non-streaming-decode-files
  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/sherpa-onnx-paraformer-zh-2023-03-28
  cd sherpa-onnx-paraformer-zh-2023-03-28
  git lfs pull --include "*.onnx"
  cd ..

Now we can use:

.. code-block:: bash

  ./non-streaming-decode-files \
    --paraformer ./sherpa-onnx-paraformer-zh-2023-03-28/model.int8.onnx \
    --tokens ./sherpa-onnx-paraformer-zh-2023-03-28/tokens.txt \
    --model-type paraformer \
    ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/0.wav

It should give you the following output:

.. code-block:: bash

  2023/08/10 15:07:10.745412 Reading ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/0.wav
  2023/08/10 15:07:10.758414 Initializing recognizer (may take several seconds)
  2023/08/10 15:07:13.992424 Recognizer created!
  2023/08/10 15:07:13.992441 Start decoding!
  2023/08/10 15:07:14.382157 Decoding done!
  2023/08/10 15:07:14.382847 对我做了介绍啊那么我想说的是呢大家如果对我的研究感兴趣呢你
  2023/08/10 15:07:14.382898 Wave duration: 5.614625 seconds

Non-streaming CTC model from NeMo
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We will use :ref:`stt-en-conformer-ctc-medium-nemo-sherpa-onnx` as an example.

First, let us download it:

.. code-block:: bash

  cd sherpa-onnx/go-api-examples/non-streaming-decode-files
  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/sherpa-onnx-nemo-ctc-en-conformer-medium
  cd sherpa-onnx-nemo-ctc-en-conformer-medium
  git lfs pull --include "*.onnx"
  cd ..

Now we can use:

.. code-block:: bash

  ./non-streaming-decode-files \
    --nemo-ctc ./sherpa-onnx-nemo-ctc-en-conformer-medium/model.onnx \
    --tokens ./sherpa-onnx-nemo-ctc-en-conformer-medium/tokens.txt \
    --model-type nemo_ctc \
    ./sherpa-onnx-nemo-ctc-en-conformer-medium/test_wavs/0.wav

It should give you the following output:

.. code-block:: bash

    2023/08/10 15:11:48.667693 Reading ./sherpa-onnx-nemo-ctc-en-conformer-medium/test_wavs/0.wav
    2023/08/10 15:11:48.680855 Initializing recognizer (may take several seconds)
    2023/08/10 15:11:51.900852 Recognizer created!
    2023/08/10 15:11:51.900869 Start decoding!
    2023/08/10 15:11:52.125605 Decoding done!
    2023/08/10 15:11:52.125630  after early nightfall the yellow lamps would light up here and there the squalid quarter of the brothels
    2023/08/10 15:11:52.125645 Wave duration: 6.625 seconds

Decode files with streaming models
----------------------------------

First, let us build the example:

.. code-block:: bash

  git clone https://github.com/k2-fsa/sherpa-onnx
  cd sherpa-onnx/go-api-examples/streaming-decode-files
  go mod tidy
  go build
  ./streaming-decode-files --help

You will find the following output:

.. code-block:: bash

  Usage of ./streaming-decode-files:
        --debug int                Whether to show debug message
        --decoder string           Path to the decoder model
        --decoding-method string   Decoding method. Possible values: greedy_search, modified_beam_search (default "greedy_search")
        --encoder string           Path to the encoder model
        --joiner string            Path to the joiner model
        --max-active-paths int     Used only when --decoding-method is modified_beam_search (default 4)
        --model-type string        Optional. Used for loading the model in a faster way
        --num-threads int          Number of threads for computing (default 1)
        --provider string          Provider to use (default "cpu")
        --tokens string            Path to the tokens file
  pflag: help requested

.. note::

   If you are using Windows and don't see any output after running ``./streaming-decode-files --help``,
   please copy ``*.dll`` from `<https://github.com/k2-fsa/sherpa-onnx-go-windows/tree/master/lib/x86_64-pc-windows-gnu>`_ (for Win64)
   or `<https://github.com/k2-fsa/sherpa-onnx-go-windows/tree/master/lib/i686-pc-windows-gnu>`_ (for Win32)
   to the directory ``sherpa-onnx/go-api-examples/streaming-decode-files``.

Now let us refer to :ref:`sherpa-onnx-pre-trained-models` to download a streaming model.

We give one example below for demonstration.

Streaming transducer
^^^^^^^^^^^^^^^^^^^^

We will use :ref:`sherpa-onnx-streaming-zipformer-en-2023-06-26-english` as an example.

First, let us download it:

.. code-block:: bash

  cd sherpa-onnx/go-api-examples/streaming-decode-files
  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26
  cd sherpa-onnx-streaming-zipformer-en-2023-06-26
  git lfs pull --include "*.onnx"
  cd ..

Now we can use:

.. code-block:: bash

  ./streaming-decode-files \
    --encoder ./sherpa-onnx-streaming-zipformer-en-2023-06-26/encoder-epoch-99-avg-1-chunk-16-left-128.onnx \
    --decoder ./sherpa-onnx-streaming-zipformer-en-2023-06-26/decoder-epoch-99-avg-1-chunk-16-left-128.onnx \
    --joiner ./sherpa-onnx-streaming-zipformer-en-2023-06-26/joiner-epoch-99-avg-1-chunk-16-left-128.onnx \
    --tokens ./sherpa-onnx-streaming-zipformer-en-2023-06-26/tokens.txt \
    --model-type zipformer2 \
    ./sherpa-onnx-streaming-zipformer-en-2023-06-26/test_wavs/0.wav

It should give you the following output:

.. code-block:: bash

    2023/08/10 15:17:00.226228 Reading ./sherpa-onnx-streaming-zipformer-en-2023-06-26/test_wavs/0.wav
    2023/08/10 15:17:00.241024 Initializing recognizer (may take several seconds)
    2023/08/10 15:17:03.352697 Recognizer created!
    2023/08/10 15:17:03.352711 Start decoding!
    2023/08/10 15:17:04.057130 Decoding done!
    2023/08/10 15:17:04.057215  after early nightfall the yellow lamps would light up here and there the squalid quarter of the brothels
    2023/08/10 15:17:04.057235 Wave duration: 6.625 seconds

Real-time speech recognition from microphone
--------------------------------------------

.. hint::

   You need to install ``portaudio`` for this example.

   .. code-block:: bash

      # for macOS
      brew install portaudio
      export PKG_CONFIG_PATH=/usr/local/Cellar/portaudio/19.7.0

      # for Ubuntu
      sudo apt-get install libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0

   To check that you have installed ``portaudio`` successfully, please run:

    .. code-block:: bash

      pkg-config --cflags --libs portaudio-2.0

   It should give you something like below:

    .. code-block:: bash

      # for macOS
      -I/usr/local/Cellar/portaudio/19.7.0/include -L/usr/local/Cellar/portaudio/19.7.0/lib -lportaudio -framework CoreAudio -framework AudioToolbox -framework AudioUnit -framework CoreFoundation -framework CoreServices

      # for Ubuntu
      -pthread -lportaudio -lasound -lm -lpthread


First, let us build the example:

.. code-block:: bash

  git clone https://github.com/k2-fsa/sherpa-onnx
  cd sherpa-onnx/go-api-examples/real-time-speech-recognition-from-microphone
  go mod tidy
  go build
  ./real-time-speech-recognition-from-microphone --help

You will find the following output:

.. code-block:: bash

  Select default input device: MacBook Pro Microphone
  Usage of ./real-time-speech-recognition-from-microphone:
        --debug int                            Whether to show debug message
        --decoder string                       Path to the decoder model
        --decoding-method string               Decoding method. Possible values: greedy_search, modified_beam_search (default "greedy_search")
        --enable-endpoint int                  Whether to enable endpoint (default 1)
        --encoder string                       Path to the encoder model
        --joiner string                        Path to the joiner model
        --max-active-paths int                 Used only when --decoding-method is modified_beam_search (default 4)
        --model-type string                    Optional. Used for loading the model in a faster way
        --num-threads int                      Number of threads for computing (default 1)
        --provider string                      Provider to use (default "cpu")
        --rule1-min-trailing-silence float32   Threshold for rule1 (default 2.4)
        --rule2-min-trailing-silence float32   Threshold for rule2 (default 1.2)
        --rule3-min-utterance-length float32   Threshold for rule3 (default 20)
        --tokens string                        Path to the tokens file
  pflag: help requested

Now let us refer to :ref:`sherpa-onnx-pre-trained-models` to download a streaming model.

We give one example below for demonstration.

Streaming transducer
^^^^^^^^^^^^^^^^^^^^

We will use :ref:`sherpa-onnx-streaming-zipformer-en-2023-06-26-english` as an example.

First, let us download it:

.. code-block:: bash

  cd sherpa-onnx/go-api-examples/real-time-speech-recognition-from-microphone
  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26
  cd sherpa-onnx-streaming-zipformer-en-2023-06-26
  git lfs pull --include "*.onnx"
  cd ..

Now we can use:

.. code-block:: bash

  ./real-time-speech-recognition-from-microphone \
    --encoder ./sherpa-onnx-streaming-zipformer-en-2023-06-26/encoder-epoch-99-avg-1-chunk-16-left-128.onnx \
    --decoder ./sherpa-onnx-streaming-zipformer-en-2023-06-26/decoder-epoch-99-avg-1-chunk-16-left-128.onnx \
    --joiner ./sherpa-onnx-streaming-zipformer-en-2023-06-26/joiner-epoch-99-avg-1-chunk-16-left-128.onnx \
    --tokens ./sherpa-onnx-streaming-zipformer-en-2023-06-26/tokens.txt \
    --model-type zipformer2

It should give you the following output:

.. code-block:: bash

  Select default input device: MacBook Pro Microphone
  2023/08/10 15:22:00 Initializing recognizer (may take several seconds)
  2023/08/10 15:22:03 Recognizer created!
  Started! Please speak
  0:  this is the first test
  1:  this is the second

colab
-----

We provide a colab notebook
|Sherpa-onnx go api example colab notebook|
for you to try the `Go`_ API examples of `sherpa-onnx`_.

.. |Sherpa-onnx go api example colab notebook| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://github.com/k2-fsa/colab/blob/master/sherpa-onnx/sherpa_onnx_go_api_example.ipynb
