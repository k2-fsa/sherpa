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
the version ``1.10.23``.

.. code-block:: bash

   wget https://huggingface.co/csukuangfj/sherpa-onnx-libs/resolve/main/jni/sherpa-onnx-v1.10.23-win-x64-jni.tar.bz2

   # For Chinese users
   # wget https://hf-mirror.com/csukuangfj/sherpa-onnx-libs/resolve/main/jni/sherpa-onnx-v1.10.23-win-x64-jni.tar.bz2

   tar xf sherpa-onnx-v1.10.23-win-x64-jni.tar.bz2
   rm sherpa-onnx-v1.10.23-win-x64-jni.tar.bz2

You should find the following files:

.. code-block:: bash

  ls -lh  sherpa-onnx-v1.10.23-win-x64-jni/lib/
  total 14M
  -rwxr-xr-x 1 fangjun fangjun  11M Aug 24 15:41 onnxruntime.dll
  -rwxr-xr-x 1 fangjun fangjun  23K Aug 24 15:41 onnxruntime_providers_shared.dll
  -rwxr-xr-x 1 fangjun fangjun 3.1M Aug 24 15:48 sherpa-onnx-jni.dll
  -rw-r--r-- 1 fangjun fangjun  51K Aug 24 15:47 sherpa-onnx-jni.lib

.. hint::

   Only ``*.dll`` files are needed during runtime.

.. note::

   You can also download it from

    `<https://github.com/k2-fsa/sherpa-onnx/releases>`_
