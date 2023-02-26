.. _sherpa-onnx-install-android-studio:

Build sherpa-onnx for Android
=============================

Install Android Studio
----------------------

The first step is to download and install Android Studio.

Please refer to `<https://developer.android.com/studio>`_ for how to install
Android Studio.

.. hint::

  Any recent version of Android Studio should work fine. Also, you can use
  the default settings of Android Studio during installation.

  For reference, we post the version we are using below:

  .. image:: ./pic/android-studio-version.png
     :alt: screenshot of my version of Android Studio
     :width: 600


Download sherpa-onnx
--------------------

Next, download the source code of `sherpa-onnx`_:

.. code-block:: bash

  git clone https://github.com/k2-fsa/sherpa-onnx

Install NDK
-----------

Step 1, start Android Studio.

  .. figure:: ./pic/start-android-studio.png
     :alt: Start Android Studio
     :width: 600

     Step 1: Click ``Open`` to select ``sherpa-onnx/android/SherpaOnnx``

Step 2, Open ``sherpa-onnx/android/SherpaOnnx``.

  .. figure:: ./pic/open-sherpa-onnx.png
     :alt: Open SherpaOnnx
     :width: 600

     Step 2: Open ``SherpaOnnx``.


Step 3, Select ``Tools -> SDK Manager``.

  .. figure:: ./pic/select-sdk-manager.png
     :alt: Select Tools -> SDK Manager
     :width: 600

     Step 3: Select ``Tools -> SDK Manager``.

Step 4, ``Install NDK``.

  .. figure:: ./pic/ndk-tools.png
     :alt: Install NDK
     :width: 600

     Step 4: Install NDK.

In the following, we assume ``Android SDK location`` was set to
``/Users/fangjun/software/my-android``. You can change it accordingly below.

After installing NDK, you can find it in

.. code-block::

  /Users/fangjun/software/my-android/ndk/22.1.7171670

.. warning::

    If you selected a different version of NDK, please replace ``22.1.7171670``
    accordingly.

Next, let us set the environment variable ``ANDROID_NDK`` for later use.

.. code-block:: bash

    export ANDROID_NDK=/Users/fangjun/software/my-android/ndk/22.1.7171670

.. note::

  Note from https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-android

  (Important) remove the hardcoded debug flag in Android NDK to fix
  the android-ndk issue: https://github.com/android/ndk/issues/243

  1. open ``$ANDROID_NDK/build/cmake/android.toolchain.cmake`` for ndk < r23
  or ``$ANDROID_NDK/build/cmake/android-legacy.toolchain.cmake`` for ndk >= r23

  2. delete the line containing "-g"

    .. code-block::

      list(APPEND ANDROID_COMPILER_FLAGS
      -g
      -DANDROID

Build sherpa-onnx (C++ code)
----------------------------

After installing ``NDK``, it is time to build the C++ code of `sherpa-onnx`_.

In the following, we show how to build `sherpa-onnx`_ for the following
Android ABIs:

  - ``arm64-v8a``
  - ``x86_64``

.. caution::

  You only need to select one and only one ABI. ``arm64-v8a`` is probably the
  most common one.

  If you want to test the app on an emulator, you probably need ``x86_64``.

Build for arm64-v8a
^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  cd sherpa-onnx # Go to the root repo
  ./build-android-arm64-v8a.sh

After building, you will find the following shared libraries:

.. code-block:: bash

  $ ls -lh build-android-arm64-v8a/install/lib/lib*.so
  -rwxr-xr-x  1 fangjun  staff   848K Feb 26 15:54 build-android-arm64-v8a/install/lib/libkaldi-native-fbank-core.so
  -rw-r--r--@ 1 fangjun  staff    13M Feb 26 15:54 build-android-arm64-v8a/install/lib/libonnxruntime.so
  -rwxr-xr-x  1 fangjun  staff    29K Feb 26 15:54 build-android-arm64-v8a/install/lib/libsherpa-onnx-c-api.so
  -rwxr-xr-x  1 fangjun  staff   313K Feb 26 15:54 build-android-arm64-v8a/install/lib/libsherpa-onnx-core.so
  -rwxr-xr-x  1 fangjun  staff    34K Feb 26 15:54 build-android-arm64-v8a/install/lib/libsherpa-onnx-jni.so

Please copy them to ``android/SherpaOnnx/app/src/main/jniLibs/arm64-v8a/``:

.. code-block:: bash

  $ cp build-android-arm64-v8a/install/lib/lib*.so  android/SherpaOnnx/app/src/main/jniLibs/arm64-v8a/

You should see the following screen shot after running the above copy ``cp`` command.

.. figure:: ./pic/so-libs-for-arm64-v8a.png
   :alt: Generated shared libraries for arm64-v8a
   :width: 600

Build for x86_64
^^^^^^^^^^^^^^^^

.. code-block:: bash

  cd sherpa-onnx # Go to the root repo
  ./build-android-x86-64.sh

After building, you will find the following shared libraries:

.. code-block:: bash

  $ ls -lh build-android-x86-64/install/lib/lib*.so
  -rwxr-xr-x  1 fangjun  staff   901K Feb 26 16:00 build-android-x86-64/install/lib/libkaldi-native-fbank-core.so
  -rw-r--r--@ 1 fangjun  staff    15M Feb 26 16:00 build-android-x86-64/install/lib/libonnxruntime.so
  -rwxr-xr-x  1 fangjun  staff   347K Feb 26 16:00 build-android-x86-64/install/lib/libsherpa-onnx-core.so
  -rwxr-xr-x  1 fangjun  staff    32K Feb 26 16:00 build-android-x86-64/install/lib/libsherpa-onnx-jni.so

Please copy them to ``android/SherpaOnnx/app/src/main/jniLibs/x86_64/``:

.. code-block:: bash

  $ cp build-android-x86-64/install/lib/lib*.so android/SherpaOnnx/app/src/main/jniLibs/x86_64/

You should see the following screen shot after running the above copy ``cp`` command.

.. figure:: ./pic/so-libs-for-x86-64.png
   :alt: Generated shared libraries for x86_64
   :width: 600

Download pre-trained models
---------------------------

Please read :ref:`sherpa-onnx-pre-trained-models` for all available pre-trained
models.

In the following, we use a pre-trained model :ref:`sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20`,
which supports both Chinese and English.

.. hint::

  The model is trained using `icefall`_ and the original torchscript model
  is from `<https://huggingface.co/pfluo/k2fsa-zipformer-chinese-english-mixed>`_.

Use the following command to download the pre-trained model and place it into
``android/SherpaOnnx/app/src/main/assets/``:

.. code-block:: bash

  cd android/SherpaOnnx/app/src/main/assets/

  sudo apt-get install git-lfs

  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20
  cd sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20
  git lfs pull --include "*.onnx"

  # Now, remove extra files to reduce the file size of the generated apk
  rm -rf .git test_wavs
  rm *.sh README.md

In the end, you should have the following files:

.. code-block:: bash

  $ ls -lh
  total 696984
  -rw-r--r--  1 fangjun  staff    13M Feb 21 21:45 decoder-epoch-99-avg-1.onnx
  -rw-r--r--  1 fangjun  staff   315M Feb 23 21:18 encoder-epoch-99-avg-1.onnx
  -rw-r--r--  1 fangjun  staff    12M Feb 21 21:45 joiner-epoch-99-avg-1.onnx
  -rw-r--r--  1 fangjun  staff    55K Feb 21 21:45 tokens.txt

  $ du -h .
  340M    .

You should see the following screen shot after downloading the pre-trained model:

.. figure:: ./pic/pre-trained-model-2023-02-20.png
   :alt: Files after downloading the pre-trained model
   :width: 600

.. hint::

  If you select a different pre-trained model, make sure that you also change the
  corresponding code listed in the following screen shot:

  .. figure:: ./pic/type-for-pre-trained-model-2023-02-20.png
     :alt: Change code if you select a different model
     :width: 600

Generate APK
------------

Finally, it is time to build `sherpa-onnx`_ to generate an APK package.

Select ``Build -> Make Project``, as shown in the following screen shot.

.. figure:: ./pic/build-make-project.png
   :alt: Select ``Build -> Make Project``
   :width: 600

You can find the generated APK in ``android/SherpaOnnx/app/build/outputs/apk/debug/app-debug.apk``:

.. code-block:: bash

  $ ls -lh android/SherpaOnnx/app/build/outputs/apk/debug/app-debug.apk
  -rw-r--r--  1 fangjun  staff   331M Feb 26 16:17 android/SherpaOnnx/app/build/outputs/apk/debug/app-debug.apk

Congratulations! You have successfully built an APK for Android.

Read below to learn more.

Analyze the APK
---------------

.. figure:: ./pic/analyze-apk.png
   :alt: Select ``Build -> Analyze APK ...``
   :width: 600

Select ``Build -> Analyze APK ...`` in the above screen shot, in the
popped-up dialog select the generated APK ``app-debug.apk``,
and you will see the following screen shot:

.. figure:: ./pic/analyze-apk-result.png
   :alt: Result of analyzing apk
   :width: 700

You can see from the above screen shot that most part of the APK
is occupied by the pre-trained model, while the runtime, including the shared
libraries, is only ``5.4 MB``.

.. caution::

  You can see that ``libonnxruntime.so`` alone occupies ``5MB`` out of ``5.4MB``.

  We use a so-called ``Full build`` instead of ``Mobile build``, so the file
  size of the library is somewhat a bit larger.

  ``libonnxruntime.so`` is donwloaded from

    `<https://mvnrepository.com/artifact/com.microsoft.onnxruntime/onnxruntime-android/1.14.0>`_

  Please refer to `<https://onnxruntime.ai/docs/build/custom.html>`_ for a
  custom build to reduce the file size of ``libonnxruntime.so``.

.. hint::

  We recommend you to use `sherpa-ncnn`_. Please see
  :ref:`sherpa-ncnn-analyze-apk-result` for `sherpa-ncnn`_. The total runtime of
  `sherpa-ncnn`_ is only ``1.6 MB``, which is much smaller than `sherpa-onnx`_.