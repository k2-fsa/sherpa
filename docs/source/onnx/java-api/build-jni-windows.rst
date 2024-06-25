.. _sherpa-onnx-jni-windows-build:

Build JNI interface (Windows)
=============================

If you want to build ``JNI`` libs by yourself, please see `<https://github.com/k2-fsa/sherpa-onnx/issues/882>`_.

.. hint::

   The PDFs in the above link are in Chinese.

If you want to download pre-built ``JNI`` libs, please see below.

Download pre-built JNI libs
---------------------------

If you don't want to build ``JNI`` libs by yourself, please download pre-built ``JNI``
libs from

    `<https://huggingface.co/csukuangfj/sherpa-onnx-libs/tree/main/jni>`_

For Chinese users, please use

  `<https://hf-mirror.com/csukuangfj/sherpa-onnx-libs/tree/main/jni>`_

Please always use the latest version. In the following, we describe how to download
the version ``1.10.2``.

.. code-block:: bash

   wget https://huggingface.co/csukuangfj/sherpa-onnx-libs/resolve/main/jni/sherpa-onnx-v1.10.2-win-x64-jni.tar.bz2

   # For Chinese users
   # wget https://hf-mirror.com/csukuangfj/sherpa-onnx-libs/resolve/main/jni/sherpa-onnx-v1.10.2-win-x64-jni.tar.bz2

   tar xf sherpa-onnx-v1.10.2-win-x64-jni.tar.bz2
   rm sherpa-onnx-v1.10.2-win-x64-jni.tar.bz2

You should find the following files:

.. code-block:: bash

  ls -lh sherpa-onnx-v1.10.2-win-x64-jni/lib/

  -rwxr-xr-x  1 fangjun  staff    12K Jun 25 11:43 cargs.dll
  -rw-r--r--  1 fangjun  staff   3.1K Jun 25 11:43 cargs.lib
  -rwxr-xr-x  1 fangjun  staff   271K Jun 25 11:43 espeak-ng.dll
  -rw-r--r--  1 fangjun  staff    49K Jun 25 11:43 espeak-ng.lib
  -rwxr-xr-x  1 fangjun  staff   533K Jun 25 11:44 kaldi-decoder-core.dll
  -rw-r--r--  1 fangjun  staff   1.1M Jun 25 11:44 kaldi-decoder-core.lib
  -rwxr-xr-x  1 fangjun  staff   109K Jun 25 11:43 kaldi-native-fbank-core.dll
  -rw-r--r--  1 fangjun  staff   103K Jun 25 11:43 kaldi-native-fbank-core.lib
  -rwxr-xr-x  1 fangjun  staff    10M Jun 25 11:43 onnxruntime.dll
  -rwxr-xr-x  1 fangjun  staff    22K Jun 25 11:43 onnxruntime_providers_shared.dll
  -rwxr-xr-x  1 fangjun  staff   439K Jun 25 11:43 piper_phonemize.dll
  -rw-r--r--  1 fangjun  staff   212K Jun 25 11:43 piper_phonemize.lib
  -rwxr-xr-x  1 fangjun  staff    76K Jun 25 11:47 sherpa-onnx-c-api.dll
  -rw-r--r--  1 fangjun  staff    63K Jun 25 11:47 sherpa-onnx-c-api.lib
  -rwxr-xr-x  1 fangjun  staff   1.7M Jun 25 11:47 sherpa-onnx-core.dll
  -rw-r--r--  1 fangjun  staff   2.5M Jun 25 11:47 sherpa-onnx-core.lib
  -rwxr-xr-x  1 fangjun  staff   1.4M Jun 25 11:43 sherpa-onnx-fst.dll
  -rw-r--r--  1 fangjun  staff   3.8M Jun 25 11:43 sherpa-onnx-fst.lib
  -rwxr-xr-x  1 fangjun  staff    30K Jun 25 11:43 sherpa-onnx-fstfar.dll
  -rw-r--r--  1 fangjun  staff    24K Jun 25 11:43 sherpa-onnx-fstfar.lib
  -rwxr-xr-x  1 fangjun  staff   109K Jun 25 11:47 sherpa-onnx-jni.dll
  -rw-r--r--  1 fangjun  staff    92K Jun 25 11:47 sherpa-onnx-jni.lib
  -rwxr-xr-x  1 fangjun  staff   1.1M Jun 25 11:44 sherpa-onnx-kaldifst-core.dll
  -rw-r--r--  1 fangjun  staff   2.8M Jun 25 11:44 sherpa-onnx-kaldifst-core.lib
  -rwxr-xr-x  1 fangjun  staff   173K Jun 25 11:43 sherpa-onnx-portaudio.dll
  -rw-r--r--  1 fangjun  staff    43K Jun 25 11:43 sherpa-onnx-portaudio.lib
  -rwxr-xr-x  1 fangjun  staff   135K Jun 25 11:43 ssentencepiece_core.dll
  -rw-r--r--  1 fangjun  staff   174K Jun 25 11:43 ssentencepiece_core.lib
  -rwxr-xr-x  1 fangjun  staff   168K Jun 25 11:43 ucd.dll
  -rw-r--r--  1 fangjun  staff   5.9K Jun 25 11:43 ucd.lib

.. hint::

   Only ``*.dll`` files are needed during runtime.
