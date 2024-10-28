Models
======

We provide 8-bit quantized ONNX models for `Moonshine`_.

You can find scripts for model quantization at

  `<https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/moonshine/export-onnx.py>`_.

In the following, we describe how to use `Moonshine`_ models with pre-built executables
in `sherpa-onnx`_.


sherpa-onnx-moonshine-tiny-en-int8
----------------------------------

Please use the following commands to download it.

.. code-block:: bash

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-moonshine-tiny-en-int8.tar.bz2
   tar xvf sherpa-onnx-moonshine-tiny-en-int8.tar.bz2
   rm sherpa-onnx-moonshine-tiny-en-int8.tar.bz2

You should see something like below after downloading::

  ls -lh sherpa-onnx-moonshine-tiny-en-int8/
  total 242160
  -rw-r--r--  1 fangjun  staff   1.0K Oct 26 09:42 LICENSE
  -rw-r--r--  1 fangjun  staff   175B Oct 26 09:42 README.md
  -rw-r--r--  1 fangjun  staff    43M Oct 26 09:42 cached_decode.int8.onnx
  -rw-r--r--  1 fangjun  staff    17M Oct 26 09:42 encode.int8.onnx
  -rw-r--r--  1 fangjun  staff   6.5M Oct 26 09:42 preprocess.onnx
  drwxr-xr-x  6 fangjun  staff   192B Oct 26 09:42 test_wavs
  -rw-r--r--  1 fangjun  staff   426K Oct 26 09:42 tokens.txt
  -rw-r--r--  1 fangjun  staff    51M Oct 26 09:42 uncached_decode.int8.onnx

Decode wave files
~~~~~~~~~~~~~~~~~

.. hint::

   It supports decoding only wave files of a single channel with 16-bit
   encoded samples, while the sampling rate does not need to be 16 kHz.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-offline \
    --moonshine-preprocessor=./sherpa-onnx-moonshine-tiny-en-int8/preprocess.onnx \
    --moonshine-encoder=./sherpa-onnx-moonshine-tiny-en-int8/encode.int8.onnx \
    --moonshine-uncached-decoder=./sherpa-onnx-moonshine-tiny-en-int8/uncached_decode.int8.onnx \
    --moonshine-cached-decoder=./sherpa-onnx-moonshine-tiny-en-int8/cached_decode.int8.onnx \
    --tokens=./sherpa-onnx-moonshine-tiny-en-int8/tokens.txt \
    --num-threads=1 \
    ./sherpa-onnx-moonshine-tiny-en-int8/test_wavs/0.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

You should see the following output:

.. literalinclude:: ./code/sherpa-onnx-moonshine-tiny-en-int8.txt

Speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-microphone-offline \
    --moonshine-preprocessor=./sherpa-onnx-moonshine-tiny-en-int8/preprocess.onnx \
    --moonshine-encoder=./sherpa-onnx-moonshine-tiny-en-int8/encode.int8.onnx \
    --moonshine-uncached-decoder=./sherpa-onnx-moonshine-tiny-en-int8/uncached_decode.int8.onnx \
    --moonshine-cached-decoder=./sherpa-onnx-moonshine-tiny-en-int8/cached_decode.int8.onnx \
    --tokens=./sherpa-onnx-moonshine-tiny-en-int8/tokens.txt

Speech recognition from a microphone with VAD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

  ./build/bin/sherpa-onnx-vad-microphone-offline-asr \
    --silero-vad-model=./silero_vad.onnx \
    --moonshine-preprocessor=./sherpa-onnx-moonshine-tiny-en-int8/preprocess.onnx \
    --moonshine-encoder=./sherpa-onnx-moonshine-tiny-en-int8/encode.int8.onnx \
    --moonshine-uncached-decoder=./sherpa-onnx-moonshine-tiny-en-int8/uncached_decode.int8.onnx \
    --moonshine-cached-decoder=./sherpa-onnx-moonshine-tiny-en-int8/cached_decode.int8.onnx \
    --tokens=./sherpa-onnx-moonshine-tiny-en-int8/tokens.txt

sherpa-onnx-moonshine-base-en-int8
----------------------------------

Please use the following commands to download it.

.. code-block:: bash

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-moonshine-base-en-int8.tar.bz2
   tar xvf sherpa-onnx-moonshine-base-en-int8.tar.bz2
   rm sherpa-onnx-moonshine-base-en-int8.tar.bz2

You should see something like below after downloading::

  ls -lh sherpa-onnx-moonshine-base-en-int8/
  total 560448
  -rw-r--r--  1 fangjun  staff   1.0K Oct 26 09:42 LICENSE
  -rw-r--r--  1 fangjun  staff   175B Oct 26 09:42 README.md
  -rw-r--r--  1 fangjun  staff    95M Oct 26 09:42 cached_decode.int8.onnx
  -rw-r--r--  1 fangjun  staff    48M Oct 26 09:42 encode.int8.onnx
  -rw-r--r--  1 fangjun  staff    13M Oct 26 09:42 preprocess.onnx
  drwxr-xr-x  6 fangjun  staff   192B Oct 26 09:42 test_wavs
  -rw-r--r--  1 fangjun  staff   426K Oct 26 09:42 tokens.txt
  -rw-r--r--  1 fangjun  staff   116M Oct 26 09:42 uncached_decode.int8.onnx


Decode wave files
~~~~~~~~~~~~~~~~~

.. hint::

   It supports decoding only wave files of a single channel with 16-bit
   encoded samples, while the sampling rate does not need to be 16 kHz.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-offline \
    --moonshine-preprocessor=./sherpa-onnx-moonshine-base-en-int8/preprocess.onnx \
    --moonshine-encoder=./sherpa-onnx-moonshine-base-en-int8/encode.int8.onnx \
    --moonshine-uncached-decoder=./sherpa-onnx-moonshine-base-en-int8/uncached_decode.int8.onnx \
    --moonshine-cached-decoder=./sherpa-onnx-moonshine-base-en-int8/cached_decode.int8.onnx \
    --tokens=./sherpa-onnx-moonshine-base-en-int8/tokens.txt \
    --num-threads=1 \
    ./sherpa-onnx-moonshine-base-en-int8/test_wavs/0.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

You should see the following output:

.. literalinclude:: ./code/sherpa-onnx-moonshine-base-en-int8.txt

Speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-microphone-offline \
    --moonshine-preprocessor=./sherpa-onnx-moonshine-base-en-int8/preprocess.onnx \
    --moonshine-encoder=./sherpa-onnx-moonshine-base-en-int8/encode.int8.onnx \
    --moonshine-uncached-decoder=./sherpa-onnx-moonshine-base-en-int8/uncached_decode.int8.onnx \
    --moonshine-cached-decoder=./sherpa-onnx-moonshine-base-en-int8/cached_decode.int8.onnx \
    --tokens=./sherpa-onnx-moonshine-base-en-int8/tokens.txt

Speech recognition from a microphone with VAD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

  ./build/bin/sherpa-onnx-vad-microphone-offline-asr \
    --silero-vad-model=./silero_vad.onnx \
    --moonshine-preprocessor=./sherpa-onnx-moonshine-base-en-int8/preprocess.onnx \
    --moonshine-encoder=./sherpa-onnx-moonshine-base-en-int8/encode.int8.onnx \
    --moonshine-uncached-decoder=./sherpa-onnx-moonshine-base-en-int8/uncached_decode.int8.onnx \
    --moonshine-cached-decoder=./sherpa-onnx-moonshine-base-en-int8/cached_decode.int8.onnx \
    --tokens=./sherpa-onnx-moonshine-base-en-int8/tokens.txt
