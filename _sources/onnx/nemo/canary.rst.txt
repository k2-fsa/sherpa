.. _nemo-non-streaming-canary-models:

Non-streaming Canary models
============================

sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8 (English + Spanish + German + French, 英语+西班牙语+德语+法语)
------------------------------------------------------------------------------------------------------------------------

This model is converted from `<https://huggingface.co/nvidia/canary-180m-flash>`_.

As described in its huggingface model repo::

  It supports automatic speech-to-text recognition (ASR) in 4 languages
  (English, German, French, Spanish) and translation from English to
  German/French/Spanish and from German/French/Spanish to English with or
  without punctuation and capitalization (PnC).

You can find the conversion script at

  `<https://github.com/k2-fsa/sherpa-onnx/tree/master/scripts/nemo/canary>`_

In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8.tar.bz2
   tar xvf sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8.tar.bz2
   rm sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8.tar.bz2

.. hint::

   If you want to try the non-quantized model, please use `sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr.tar.bz2 <https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr.tar.bz2>`_

You should see something like below after downloading::

  ls -lh sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/
  total 428208
  -rw-r--r--  1 fangjun  staff    71M Jul  7 16:03 decoder.int8.onnx
  -rw-r--r--  1 fangjun  staff   127M Jul  7 16:03 encoder.int8.onnx
  drwxr-xr-x  4 fangjun  staff   128B Jul  7 16:03 test_wavs
  -rw-r--r--  1 fangjun  staff    52K Jul  7 16:03 tokens.txt

Decode wave files
~~~~~~~~~~~~~~~~~

.. hint::

   It supports decoding only wave files of a single channel with 16-bit
   encoded samples, while the sampling rate does not need to be 16 kHz.

Input English, output English
::::::::::::::::::::::::::::::

We use ``--canary-src-lang=en`` to indicate that the input audio contains
English speech. ``--canary-tgt-lang=en`` means the recognition result should
be in English.

.. hint::

   If the input audio is English, we can select whether to output English,
   French, German, or Spanish.

   If the input audio is German, we can select whether to output English
   or German.

   If the input audio is French, we can select whether to output English
   or French.

   If the input audio is Spanish, we can select whether to output English
   or Spanish.

.. warning::

   ``--src-lang`` and ``--tgt-lang`` have the same default value ``en``.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-offline \
    --canary-encoder=./sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/encoder.int8.onnx \
    --canary-decoder=./sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/decoder.int8.onnx \
    --tokens=./sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/tokens.txt \
    --canary-src-lang=en \
    --canary-tgt-lang=en \
    ./sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/test_wavs/en.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

You should see the following output:

.. literalinclude:: ./code-canary/180m-en-en-int8.txt


Input English, output German
::::::::::::::::::::::::::::

We use ``--canary-src-lang=en`` to indicate that the input audio contains
English speech. ``--canary-tgt-lang=de`` means the recognition result should
be in German.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-offline \
    --canary-encoder=./sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/encoder.int8.onnx \
    --canary-decoder=./sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/decoder.int8.onnx \
    --tokens=./sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/tokens.txt \
    --canary-src-lang=en \
    --canary-tgt-lang=de \
    ./sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/test_wavs/en.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

You should see the following output:

.. literalinclude:: ./code-canary/180m-en-de-int8.txt


Input English, output Spanish
:::::::::::::::::::::::::::::

We use ``--canary-src-lang=en`` to indicate that the input audio contains
English speech. ``--canary-tgt-lang=es`` means the recognition result should
be in Spanish.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-offline \
    --canary-encoder=./sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/encoder.int8.onnx \
    --canary-decoder=./sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/decoder.int8.onnx \
    --tokens=./sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/tokens.txt \
    --canary-src-lang=en \
    --canary-tgt-lang=es \
    ./sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/test_wavs/en.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

You should see the following output:

.. literalinclude:: ./code-canary/180m-en-es-int8.txt

Input English, output French
::::::::::::::::::::::::::::

We use ``--canary-src-lang=en`` to indicate that the input audio contains
English speech. ``--canary-tgt-lang=fr`` means the recognition result should
be in French.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-offline \
    --canary-encoder=./sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/encoder.int8.onnx \
    --canary-decoder=./sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/decoder.int8.onnx \
    --tokens=./sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/tokens.txt \
    --canary-src-lang=en \
    --canary-tgt-lang=fr \
    ./sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/test_wavs/en.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

You should see the following output:

.. literalinclude:: ./code-canary/180m-en-fr-int8.txt

Input German, output English
::::::::::::::::::::::::::::

We use ``--canary-src-lang=de`` to indicate that the input audio contains
German speech. ``--canary-tgt-lang=en`` means the recognition result should
be in English.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-offline \
    --canary-encoder=./sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/encoder.int8.onnx \
    --canary-decoder=./sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/decoder.int8.onnx \
    --tokens=./sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/tokens.txt \
    --canary-src-lang=de \
    --canary-tgt-lang=en \
    ./sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/test_wavs/de.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

You should see the following output:

.. literalinclude:: ./code-canary/180m-de-en-int8.txt

Input German, output German
:::::::::::::::::::::::::::

We use ``--canary-src-lang=de`` to indicate that the input audio contains
German speech. ``--canary-tgt-lang=de`` means the recognition result should
be in German.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-offline \
    --canary-encoder=./sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/encoder.int8.onnx \
    --canary-decoder=./sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/decoder.int8.onnx \
    --tokens=./sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/tokens.txt \
    --canary-src-lang=de \
    --canary-tgt-lang=de \
    ./sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/test_wavs/de.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

You should see the following output:

.. literalinclude:: ./code-canary/180m-de-de-int8.txt

Python API examples
~~~~~~~~~~~~~~~~~~~

Please see `<https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/offline-nemo-canary-decode-files.py>`_
