.. _build-sherpa-onx-for-qualcomm-npu:

Build sherpa-onnx for Qualcomm NPU
==================================

Before you continue, we assume you have followed :ref:`onnx-download-qnn`
to download QNN SDK and setup the environment variable ``QNN_SDK_ROOT``.

You shoud run::

  echo $QNN_SDK_ROOT

to check that it points to the QNN SDK directory.

Also, we assume you have installed Android NDK and setup the environment variable ``ANDROID_NDK``.

You can use the following commands to build `sherpa-onnx`_ for Qualcomm NPU.

.. code-block::

   git clone https://github.com/k2-fsa/sherpa-onnx

   cd sherpa-onnx

   export SHERPA_ONNX_ENABLE_QNN=ON

   export SHERPA_ONNX_ENABLE_BINARY=ON

   ./build-android-arm64-v8a.sh

After building, you should get the following shared libraries and executables.

.. _qnn-build-shared-libs:

Shared libraries
-----------------

.. code-block::

  ls -lh build-android-arm64-v8a/install/lib/
  total 40752

  -rw-r--r--@ 1 fangjun  staff    15M 20 Nov 17:05 libonnxruntime.so
  -rwxr-xr-x@ 1 fangjun  staff   4.6M 20 Nov 17:05 libsherpa-onnx-jni.so

  file build-android-arm64-v8a/install/lib/libonnxruntime.so

  build-android-arm64-v8a/install/lib/libonnxruntime.so: ELF 64-bit LSB shared object, ARM aarch64, version 1 (SYSV), dynamically linked, BuildID[sha1]=d1713f0e89b18cfa4f3102dd39c3fa8500f32772, stripped

  file build-android-arm64-v8a/install/lib/libsherpa-onnx-jni.so

  build-android-arm64-v8a/install/lib/libsherpa-onnx-jni.so: ELF 64-bit LSB shared object, ARM aarch64, version 1 (SYSV), dynamically linked, BuildID[sha1]=a262171712e69be58a2ab281f3e9980d4bc37e29, stripped

Please copy all of the shared libraries to the ``jniLibs/arm64-v8a`` directory in our Android examples.
For instance, for the demo `SherpaOnnxSimulateStreamingAsr <https://github.com/k2-fsa/sherpa-onnx/tree/master/android/SherpaOnnxSimulateStreamingAsr/app/src/main/jniLibs/arm64-v8a>`_::

  cp  -v build-android-arm64-v8a/install/lib/lib* android/SherpaOnnxSimulateStreamingAsr/app/src/main/jniLibs/arm64-v8a/

Executable files
----------------

.. code-block::

  ls -lh build-android-arm64-v8a/install/bin/sherpa-onnx-offline

  -rwxr-xr-x@ 1 fangjun  staff   2.2M 20 Nov 17:05 build-android-arm64-v8a/install/bin/sherpa-onnx-offline

  file build-android-arm64-v8a/install/bin/sherpa-onnx-offline

  build-android-arm64-v8a/install/bin/sherpa-onnx-offline: ELF 64-bit LSB pie executable, ARM aarch64, version 1 (SYSV), dynamically linked, interpreter /system/bin/linker64, BuildID[sha1]=6d693f987dea91ad36931a1709315ae88f3b7090, stripped


We describe how to run the executable files on your phone with adb in :ref:`run-exe-on-your-phone-with-qnn`.

See :ref:`build-android-examples-with-qnn` for how to use the generated shared libraries.
