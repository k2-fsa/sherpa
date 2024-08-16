.. _sherpa-onnx-pascal-api:

Pascal API
==========

We provide APIs for `Object Pascal <https://en.wikipedia.org/wiki/Object_Pascal>`_.

In other words, you can develop the following types of applications using Object Pascal:

  - Voice activity detection
  - Streaming speech recognition (i.e., real-time speech recognition)
  - Non-streaming speech recognition

on Windows, Linux, and macOS.

.. hint::

   For macOS, both Apple Silicon (i.e., macOS arm64, M1/M2/M3) and Intel chips
   are supported.

.. note::

   We will support text-to-speech, audio tagging, keyword spotting,
   speaker recognition, speech identification, and spoken language identification
   with object pascal later.

In the following, we describe how to use the object pascal API to decode files.

We use macOS below as an example. You can adapt it for Linux and Windows.

.. hint::

   We support both static link and dynamic link; the example below uses
   dynamic link. You can pass ``-DBUILD_SHARED_LIBS=OFF`` to ``cmake`` if you
   want to use static link.


   On the Windows platform, it supports only dynamic link though.

Install free pascal
-------------------

Please visit
`<https://wiki.freepascal.org/Installing_the_Free_Pascal_Compiler>`_
for installation.

To check that you have installed ``fpc`` successfully, please run::

  fpc -h

which should print the usage information of ``fpc``.

Build sherpa-onnx
-----------------

.. code-block:: bash

   mkdir -p $HOME/open-source
   cd $HOME/open-source
   git clone https://github.com/k2-fsa/sherpa-onnx
   cd sherpa-onnx

   mkdir build
   cd build

   cmake \
     -DBUILD_SHARED_LIBS=ON \
     -DCMAKE_BUILD_TYPE=Release \
     -DCMAKE_INSTALL_PREFIX=./install \
     ..

   cmake --build . --target install --config Release

   ls -lh install/lib

You should get the following two shared library files::

  (py38) fangjuns-MacBook-Pro:build fangjun$ ls -lh install/lib/
  total 59696
  -rw-r--r--  1 fangjun  staff    25M Aug 14 14:09 libonnxruntime.1.17.1.dylib
  lrwxr-xr-x  1 fangjun  staff    27B Aug 14 14:18 libonnxruntime.dylib -> libonnxruntime.1.17.1.dylib
  -rwxr-xr-x  1 fangjun  staff   3.9M Aug 15 15:01 libsherpa-onnx-c-api.dylib

Non-streaming speech recognition from files
-------------------------------------------

We use the ``Whisper tiny.en`` model below as an example.

.. hint::

   We have hardcoded the model filenames in the code.

.. code-block:: bash

  cd $HOME/open-source/sherpa-onnx

  cd pascal-api-examples/non-streaming-asr/

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.en.tar.bz2
  tar xvf sherpa-onnx-whisper-tiny.en.tar.bz2
  rm sherpa-onnx-whisper-tiny.en.tar.bz2

  fpc \
    -dSHERPA_ONNX_USE_SHARED_LIBS \
    -Fu$HOME/open-source/sherpa-onnx/sherpa-onnx/pascal-api \
    -Fl$HOME/open-source/sherpa-onnx/build/install/lib \
    ./whisper.pas

  # It will generate a file ./whisper

The output logs of the above ``fpc`` command are given below::

  Free Pascal Compiler version 3.2.2 [2021/05/16] for x86_64
  Copyright (c) 1993-2021 by Florian Klaempfl and others
  Target OS: Darwin for x86_64
  Compiling ./whisper.pas
  Compiling /Users/fangjun/open-source/sherpa-onnx/sherpa-onnx/pascal-api/sherpa_onnx.pas
  Assembling sherpa_onnx
  Assembling whisper
  Linking whisper
  ld: warning: dylib (/Users/fangjun/open-source/sherpa-onnx/build/install/lib//libsherpa-onnx-c-api.dylib) was built for newer macOS version (10.14) tha
  n being linked (10.8)
  1530 lines compiled, 3.8 sec

Explanation of the options for the ``fpc`` command:

 - ``-dSHERPA_ONNX_USE_SHARED_LIBS``

   It defines a symbol ``SHERPA_ONNX_USE_SHARED_LIBS``, which means
   we want to use dynamic link in the code. If you omit it, it will use static link.
   Please search for the string ``SHERPA_ONNX_USE_SHARED_LIBS`` in the file
   `<https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/pascal-api/sherpa_onnx.pas>`_
   if you want to learn more.

 - ``-Fu$HOME/open-source/sherpa-onnx/pascal-api``

   It specifies the unit search path.
   Our `sherpa_onnx.pas <https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/pascal-api/sherpa_onnx.pas>`_
   is inside the directory ``$HOME/open-source/sherpa-onnx/pascal-api`` and we have to
   tell ``fpc`` where to find it.

 - ``-Fl$HOME/sherpa-onnx/build/install/lib``

   It tells ``fpc`` where to look for ``libsherpa-onnx-c-api.dylib``.

After running the above ``fpc`` command, we will find an executable file ``whisper``
in the current directory, i.e., ``$HOME/open-source/sherpa-onnx/pascal-api-examples/non-streaming-asr/whisper``::

  (py38) fangjuns-MacBook-Pro:non-streaming-asr fangjun$ ls -lh ./whisper
  -rwxr-xr-x  1 fangjun  staff   2.3M Aug 16 12:13 ./whisper

If we run it::

  (py38) fangjuns-MacBook-Pro:non-streaming-asr fangjun$ ./whisper
  dyld[23162]: Library not loaded: @rpath/libsherpa-onnx-c-api.dylib
    Referenced from: <3AE58F60-4925-335D-89A5-B30FD7D97D7E> /Users/fangjun/open-source/sherpa-onnx/pascal-api-examples/non-streaming-asr/whisper
    Reason: tried: '/Users/fangjun/py38/lib/python3.8/site-packages/libsherpa-onnx-c-api.dylib' (no such file), '/usr/local/Cellar/ghostscript/9.55.0/lib/libsherpa-onnx-c-api.dylib' (no such file), '/Users/fangjun/py38/lib/python3.8/site-packages/libsherpa-onnx-c-api.dylib' (no such file), '/usr/local/Cellar/ghostscript/9.55.0/lib/libsherpa-onnx-c-api.dylib' (no such file), '/libsherpa-onnx-c-api.dylib' (no such file), '/System/Volumes/Preboot/Cryptexes/OS@rpath/libsherpa-onnx-c-api.dylib' (no such file), '/usr/local/lib/libsherpa-onnx-c-api.dylib' (no such file), '/usr/lib/libsherpa-onnx-c-api.dylib' (no such file, not in dyld cache)
  Abort trap: 6

You can see it cannot find ``libsherpa-onnx-c-api.dylib``.

At the compilation time, we have used ``-Fl$HOME/sherpa-onnx/build/install/lib``
to tell the compiler ``fpc`` where to find ``libsherpa-onnx-c-api.dylib``.

At the runtime, we also need to do something to tell the executable where to look
for ``libsherpa-onnx-c-api.dylib``.

The following command does exactly that::

  (py38) fangjuns-MacBook-Pro:non-streaming-asr fangjun$ export DYLD_LIBRARY_PATH=$HOME/open-source/sherpa-onnx/build/install/lib:$DYLD_LIBRARY_PATH
  (py38) fangjuns-MacBook-Pro:non-streaming-asr fangjun$ ./whisper
  TSherpaOnnxOfflineRecognizerResult(Text :=  After early nightfall, the yellow lamps would light up here and there the squalid quarter of the brothels., Tokens := [ After,  early,  night, fall, ,,  the,  yellow,  lamps,  would,  light,  up,  here,  and,  there,  the,  squ, alid,  quarter,  of,  the,  bro, the, ls, .], Timestamps := [])
  NumThreads 1
  Elapsed 0.803 s
  Wave duration 6.625 s
  RTF = 0.803/6.625 = 0.121

.. hint::

   If you are using Linux, please replace ``DYLD_LIBRARY_PATH`` with ``LD_LIBRARY_PATH``.

Congratulations! You have successfully managed to use the object pascal API with
Whisper for speech recognition!

You can find more examples at:

  `<https://github.com/k2-fsa/sherpa-onnx/tree/master/pascal-api-examples>`_

Colab notebook
--------------

We provide a colab notebook
|use sherpa-onnx for pascal colab notebook|
for you to try this section step by step.

.. |use sherpa-onnx for pascal colab notebook| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://github.com/k2-fsa/colab/blob/master/sherpa-onnx/sherpa_onnx_pascal_api_example.ipynb
