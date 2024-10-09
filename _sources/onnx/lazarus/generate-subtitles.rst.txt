Generate subtitles
==================

This page describes how to run the code in the following directory:

  `<https://github.com/k2-fsa/sherpa-onnx/tree/master/lazarus-examples/generate_subtitles>`_

.. hint::

   Before you continue, we assume you have installed `Lazarus`_.

Screenshots on different platforms
----------------------------------

The same code can be compiled without any modifications for different operating systems
and architectures.

That is `WOCA <https://en.wikipedia.org/wiki/Write_once,_compile_anywhere>`_,

  Write once, compile anywhere.

The following screenshots give an example about that.

.. tabs::

   .. tab:: Linux x64 screenshot

      .. figure:: ./pic/generate-subtitles/linux-x64.jpg
         :alt: Windows-x64
         :width: 90%

         Linux-x64 screenshot


   .. tab:: Windows x64 screenshot

      .. figure:: ./pic/generate-subtitles/windows-x64.jpg
         :alt: Windows-x64
         :width: 90%

         Windows-x64 screenshot

   .. tab:: macOS x64 screenshot

      .. figure:: ./pic/generate-subtitles/macos-x64.jpg
         :alt: macos-x64
         :width: 90%

         macOS-x64 screenshot

Get sherpa-onnx libraries
-------------------------

`sherpa-onnx`_ is implemented in C++. To use it with Object Pascal, we have to
get either the static library or the dynamic library for `sherpa-onnx`_.

To achieve that, you can either build `sherpa-onnx`_ from source ``or`` download
pre-built libraries from

  `<https://github.com/k2-fsa/sherpa-onnx/releases>`_

1. Build sherpa-onnx from source
::::::::::::::::::::::::::::::::

The following code builds shared libraries for `sherpa-onnx`_:

.. code-block:: bash

    mkdir -p $HOME/open-source/
    cd $HOME/open-source/
    git clone https://github.com/k2-fsa/sherpa-onnx
    cd sherpa-onnx

    # The build directory must be named as "build"
    # for shared libraries

    mkdir build
    cd build

    cmake \
      -DBUILD_SHARED_LIBS=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=./install \
      ..
    cmake --build . --target install --config Release

The following code builds static libraries for `sherpa-onnx`_:

.. code-block:: bash

    mkdir -p $HOME/open-source/
    cd $HOME/open-source/
    git clone https://github.com/k2-fsa/sherpa-onnx
    cd sherpa-onnx

    # The build directory must be named as "build-static"
    # for shared libraries

    mkdir build-static
    cd build-static

    cmake \
      -DBUILD_SHARED_LIBS=OFF \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=./install \
      ..
    cmake --build . --target install --config Release

.. caution::

   - For building shared libraries, the build directory must be ``build``.

   - For building static libraries, the build directory must be ``build-static``.

   If you want to learn why there are such constraints, please search for
   ``build-static`` in the file `generate_subtitles.lpi <https://github.com/k2-fsa/sherpa-onnx/blob/master/lazarus-examples/generate_subtitles/generate_subtitles.lpi>`_

2. Download pre-built libraries
:::::::::::::::::::::::::::::::

If you don't want to build `sherpa-onnx`_ from source, please download pre-built libraries
from `<https://github.com/k2-fsa/sherpa-onnx/releases>`_.

We suggest that you always use the latest release.

.. list-table::

 * - ****
   - Required dynamic library files
   - Required static library files
 * - Windows
   - ``sherpa-onnx-c-api.dll``, ``onnxruntime.dll``
   - ``N/A`` (We only support dynamic linking with sherpa-onnx in Lazarus on Windows)
 * - Linux
   -  - ``libsherpa-onnx-c-api.so``
      - ``libonnxruntime.so``
   -  - ``libsherpa-onnx-c-api.a``
      - ``libsherpa-onnx-core.a``
      - ``libkaldi-decoder-core.a``
      - ``libsherpa-onnx-kaldifst-core.a``
      - ``libsherpa-onnx-fstfar.a``
      - ``libsherpa-onnx-fst.a``
      - ``libkaldi-native-fbank-core.a``
      - ``libpiper_phonemize``
      - ``liblibespeak-ng.a``
      - ``libucd.a``
      - ``liblibonnxruntime.a``
      - ``libssentencepiece_core.a``
 * - macOS
   -  - ``libsherpa-onnx-c-api.dylib``
      - ``libonnxruntime.1.17.1.dylib``
   -  - ``libsherpa-onnx-c-api.a``
      - ``libsherpa-onnx-core.a``
      - ``libkaldi-decoder-core.a``
      - ``libsherpa-onnx-kaldifst-core.a``
      - ``libsherpa-onnx-fstfar.a``
      - ``libsherpa-onnx-fst.a``
      - ``libkaldi-native-fbank-core.a``
      - ``libpiper_phonemize``
      - ``liblibespeak-ng.a``
      - ``libucd.a``
      - ``liblibonnxruntime.a``
      - ``libssentencepiece_core.a``

If you download ``shared`` libraries, please create a ``build`` directory
inside the ``sherpa-onnx`` project directory and put the library files into
``build/install/lib``. An example on my macOS is given below::

  (py38) fangjuns-MacBook-Pro:sherpa-onnx fangjun$ pwd
  /Users/fangjun/open-source/sherpa-onnx
  (py38) fangjuns-MacBook-Pro:sherpa-onnx fangjun$ ls -lh build/install/lib
  total 59696
  -rw-r--r--  1 fangjun  staff    25M Aug 14 14:09 libonnxruntime.1.17.1.dylib
  lrwxr-xr-x  1 fangjun  staff    27B Aug 14 14:18 libonnxruntime.dylib -> libonnxruntime.1.17.1.dylib
  -rwxr-xr-x  1 fangjun  staff   3.9M Aug 15 15:01 libsherpa-onnx-c-api.dylib


If you download ``static`` libraries, please create a ``build-static`` directory
inside the ``sherpa-onnx`` project directory and put the library files into
``build-static/install/lib``. An example on my macOS is given below::

  (py38) fangjuns-MacBook-Pro:sherpa-onnx fangjun$ pwd
  /Users/fangjun/open-source/sherpa-onnx
  (py38) fangjuns-MacBook-Pro:sherpa-onnx fangjun$ ls -lh build-static/install/lib
  total 138176
  -rw-r--r--  1 fangjun  staff   438K Aug 15 15:03 libespeak-ng.a
  -rw-r--r--  1 fangjun  staff   726K Aug 15 15:03 libkaldi-decoder-core.a
  -rw-r--r--  1 fangjun  staff   198K Aug 15 15:03 libkaldi-native-fbank-core.a
  -rw-r--r--  1 fangjun  staff    56M Aug 14 14:25 libonnxruntime.a
  -rw-r--r--  1 fangjun  staff   421K Aug 15 15:03 libpiper_phonemize.a
  -rw-r--r--  1 fangjun  staff    87K Aug 15 15:03 libsherpa-onnx-c-api.a
  -rw-r--r--  1 fangjun  staff   5.7M Aug 15 15:03 libsherpa-onnx-core.a
  -rw-r--r--  1 fangjun  staff   2.3M Aug 15 15:03 libsherpa-onnx-fst.a
  -rw-r--r--  1 fangjun  staff    30K Aug 15 15:03 libsherpa-onnx-fstfar.a
  -rw-r--r--  1 fangjun  staff   1.6M Aug 15 15:03 libsherpa-onnx-kaldifst-core.a
  -rw-r--r--  1 fangjun  staff   131K Aug 15 15:03 libsherpa-onnx-portaudio_static.a
  -rw-r--r--  1 fangjun  staff   147K Aug 15 15:03 libssentencepiece_core.a
  -rw-r--r--  1 fangjun  staff   197K Aug 15 15:03 libucd.a

Build the generate_subtitles project
------------------------------------

Now you can start Lazarus and open `generate_subtitles.lpi <https://github.com/k2-fsa/sherpa-onnx/blob/master/lazarus-examples/generate_subtitles/generate_subtitles.lpi>`_ .

Click the menu ``Run`` -> ``Compile``. It should be able to build the project without any errors.

.. hint::

   Please ignore warnings, if there are any.

After building, you should find the following files inside the directory `generate_subtitles <https://github.com/k2-fsa/sherpa-onnx/blob/master/lazarus-examples/generate_subtitles>`_:

.. tabs::

   .. tab:: macOS

      .. code-block::

        (py38) fangjuns-MacBook-Pro:generate_subtitles fangjun$ pwd
        /Users/fangjun/open-source/sherpa-onnx/lazarus-examples/generate_subtitles
        (py38) fangjuns-MacBook-Pro:generate_subtitles fangjun$ ls -lh generate_subtitles generate_subtitles.app/
        -rwxr-xr-x  1 fangjun  staff    25M Aug 15 20:44 generate_subtitles

        generate_subtitles.app/:
        total 0
        drwxr-xr-x  6 fangjun  staff   192B Aug 14 23:01 Contents

   .. tab:: Windows

      .. code-block:: bash

          fangjun@M-0LQSDCC2RV398 C:\Users\fangjun\open-source\sherpa-onnx\lazarus-examples\generate_subtitles>dir generate_subtitles.exe
           Volume in drive C is 系统
           Volume Serial Number is 8E17-A21F

           Directory of C:\Users\fangjun\open-source\sherpa-onnx\lazarus-examples\generate_subtitles

          08/15/2024  09:39 PM         2,897,408 generate_subtitles.exe
                         1 File(s)      2,897,408 bytes
                         0 Dir(s)  38,524,039,168 bytes free

   .. tab:: Linux

      .. code-block:: bash

        cd lazarus-examples/generate_subtitles
        ls -lh generate_subtitles

        -rwxr-xr-x 1 runner docker 3.1M Aug 16 03:37 generate_subtitles

Now you can start the generated executable ``generate_subtitles`` and you should
get the screenshot like the one listed at the start of this section.

If you get any issues about ``shared libraries not found``, please copy the shared
library files from ``build/install/lib`` to the directory ``lazarus-examples/generate_subtitles``
or you can set the environment variable ``DYLD_LIBRARY_PATH`` (for macOS) and ``LD_LIBRARY_PATH`` (for Linux).

Download models
---------------

The generated executable expects that there are model files located in the same directory.

Download the VAD model
::::::::::::::::::::::

.. code-block:: bash

   cd lazarus-examples/generate_subtitles
   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

and put ``silero_vad.onnx`` into ``lazarus-examples/generate_subtitles``.

.. hint::

   If you are using macOS, please put it into ``lazarus-examples/generate_subtitles/generate_subtitles.app/Contents/Resources/``


Download a speech recognition model
:::::::::::::::::::::::::::::::::::

The executable expects a non-streaming speech recognition model. Currently, we have supported the following
types of models

  - Whisper
  - Zipformer transducer
  - NeMo transducer
  - SenseVoice
  - Paraformer
  - TeleSpeech CTC

You can download them from `<https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models>`_

Note that you have to rename the model files after downloading.

.. list-table::

 * - ****
   - Expected filenames
 * - Whisper
   - - tokens.txt
     - whisper-encoder.onnx
     - whisper-decoder.onnx
 * - Zipformer transducer
   - - tokens.txt
     - transducer-encoder.onnx
     - transducer-decoder.onnx
     - transducer-joiner.onnx
 * - NeMo transducer
   - - tokens.txt
     - nemo-transducer-encoder.onnx
     - nemo-transducer-decoder.onnx
     - nemo-transducer-joiner.onnx
 * - SenseVoice
   - - tokens.txt
     - sense-voice.onnx
 * - Paraformer
   - - tokens.txt
     - paraformer.onnx
 * - TeleSpeech
   - - tokens.txt
     - telespeech.onnx

We give several examples below.

1. Wisper
:::::::::

.. code-block:: bash

    cd lazarus-examples/generate_subtitles
    curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.en.tar.bz2

    tar xvf sherpa-onnx-whisper-tiny.en.tar.bz2
    rm sherpa-onnx-whisper-tiny.en.tar.bz2

    cd sherpa-onnx-whisper-tiny.en

    mv -v tiny.en-encoder.int8.onnx ../whisper-encoder.onnx
    mv -v tiny.en-decoder.int8.onnx ../whisper-decoder.onnx
    mv -v tiny.en-tokens.txt ../tokens.txt

    cd ..
    rm -rf sherpa-onnx-whisper-tiny.en

You can replace ``tiny.en`` with other types of Whisper models, e.g., ``tiny``, ``base``, etc.

2. Zipformer transducer
:::::::::::::::::::::::

We give two examples below for Zipformer transducer

**Example 1**

.. code-block:: bash

    cd lazarus-examples/generate_subtitles
    curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/icefall-asr-zipformer-wenetspeech-20230615.tar.bz2
    tar xvf icefall-asr-zipformer-wenetspeech-20230615.tar.bz2
    rm icefall-asr-zipformer-wenetspeech-20230615.tar.bz2

    cd icefall-asr-zipformer-wenetspeech-20230615

    mv -v data/lang_char/tokens.txt ../

    mv -v exp/encoder-epoch-12-avg-4.int8.onnx ../transducer-encoder.onnx
    mv -v exp/decoder-epoch-12-avg-4.onnx ../transducer-decoder.onnx
    mv -v exp/joiner-epoch-12-avg-4.int8.onnx ../transducer-joiner.onnx

    cd ..
    rm icefall-asr-zipformer-wenetspeech-20230615

**Example 2**

.. code-block:: bash

    cd lazarus-examples/generate_subtitles
    curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01.tar.bz2
    tar xvf sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01.tar.bz2
    rm sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01.tar.bz2

    cd sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01

    mv ./tokens.txt ../
    mv encoder-epoch-99-avg-1.int8.onnx ../transducer-encoder.onnx
    mv decoder-epoch-99-avg-1.onnx ../transducer-decoder.onnx
    mv joiner-epoch-99-avg-1.int8.onnx ../transducer-joiner.onnx

    cd ../

    rm -rf sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01

3. NeMo transducer
::::::::::::::::::

.. code-block:: bash

    cd lazarus-examples/generate_subtitles
    curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-fast-conformer-transducer-be-de-en-es-fr-hr-it-pl-ru-uk-20k.tar.bz2
    tar xvf sherpa-onnx-nemo-fast-conformer-transducer-be-de-en-es-fr-hr-it-pl-ru-uk-20k.tar.bz2
    rm sherpa-onnx-nemo-fast-conformer-transducer-be-de-en-es-fr-hr-it-pl-ru-uk-20k.tar.bz2

    cd sherpa-onnx-nemo-fast-conformer-transducer-be-de-en-es-fr-hr-it-pl-ru-uk-20k

    mv tokens.txt ../
    mv encoder.onnx ../nemo-transducer-encoder.onnx
    mv decoder.onnx ../nemo-transducer-decoder.onnx
    mv joiner.onnx ../nemo-transducer-joiner.onnx

    cd ../
    rm -rf  sherpa-onnx-nemo-fast-conformer-transducer-be-de-en-es-fr-hr-it-pl-ru-uk-20k

4. SenseVoice
:::::::::::::

.. code-block:: bash

    cd lazarus-examples/generate_subtitles
    curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
    tar xvf sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
    rm sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2

    cd sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17

    mv tokens.txt ../
    mv model.int8.onnx ../sense-voice.onnx

    cd ../
    rm -rf sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17

5. Paraformer
:::::::::::::

.. code-block:: bash

    cd lazarus-examples/generate_subtitles
    curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2

    tar xvf sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2
    rm sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2

    cd sherpa-onnx-paraformer-zh-2023-09-14

    mv tokens.txt ../
    mv model.int8.onnx ../paraformer.onnx

    cd ../
    rm -rf sherpa-onnx-paraformer-zh-2023-09-14

6. TeleSpeech
:::::::::::::

.. code-block:: bash

    cd lazarus-examples/generate_subtitles
    curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-telespeech-ctc-int8-zh-2024-06-04.tar.bz2

    tar xvf sherpa-onnx-telespeech-ctc-int8-zh-2024-06-04.tar.bz2
    rm sherpa-onnx-telespeech-ctc-int8-zh-2024-06-04.tar.bz2

    cd sherpa-onnx-telespeech-ctc-int8-zh-2024-06-04

    mv tokens.txt ../
    mv model.int8.onnx ../telespeech.onnx

    cd ../
    rm -rf sherpa-onnx-telespeech-ctc-int8-zh-2024-06-04

For the more curious
--------------------

If you want to find out how we generate the APPs in
`<https://k2-fsa.github.io/sherpa/onnx/lazarus/download-generated-subtitles.html>`_,
please have a look at

  - `<https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/lazarus/generate-subtitles.py>`_
  - `<https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/lazarus/build-generate-subtitles.sh.in>`_
  - `<https://github.com/k2-fsa/sherpa-onnx/blob/master/.github/workflows/lazarus.yaml>`_
