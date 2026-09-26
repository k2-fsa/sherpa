Java for Android
================

We provide two Android demos for streaming speech recognition.
Both use `JitPack <https://jitpack.io/>`_ to fetch the sherpa-onnx dependency, so there is
no need to build native libraries manually.

Demo 1: SherpaOnnxJavaDemo (Groovy DSL)
----------------------------------------

A Java-based Android demo for streaming speech recognition.

  `<https://github.com/k2-fsa/sherpa-onnx/tree/master/android/SherpaOnnxJavaDemo>`_

JitPack is configured in `settings.gradle <https://github.com/k2-fsa/sherpa-onnx/blob/master/android/SherpaOnnxJavaDemo/settings.gradle>`_ (Groovy DSL):

.. code-block:: groovy

   dependencyResolutionManagement {
       repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
       repositories {
           google()
           mavenCentral()
           maven { url 'https://jitpack.io' }
       }
   }

The dependency in `app/build.gradle <https://github.com/k2-fsa/sherpa-onnx/blob/master/android/SherpaOnnxJavaDemo/app/build.gradle>`_:

.. code-block:: groovy

   implementation 'com.github.k2-fsa.sherpa-onnx:sherpa-onnx:v1.13.5'

Download model files
::::::::::::::::::::

Before building, download the model files into the ``assets`` directory:

.. code-block:: bash

   # Assume we are inside the SherpaOnnxJavaDemo directory
   cd app/src/main/assets/

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2

   tar xvf sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
   rm sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2

   mv sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.int8.onnx ./
   mv sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx ./
   mv sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.int8.onnx ./
   mv sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt ./

   rm -rf sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/*

   mv encoder-epoch-99-avg-1.int8.onnx sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/
   mv decoder-epoch-99-avg-1.onnx sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/
   mv joiner-epoch-99-avg-1.int8.onnx sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/
   mv tokens.txt sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/

You should have the following directory structure::

   app/src/main/assets/
   └── sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20
       ├── decoder-epoch-99-avg-1.onnx
       ├── encoder-epoch-99-avg-1.int8.onnx
       ├── joiner-epoch-99-avg-1.int8.onnx
       └── tokens.txt

   1 directory, 4 files

Remember to remove unused files to reduce the file size of the final APK.

Build and run
:::::::::::::

Open the project in Android Studio, or build from the command line:

.. code-block:: bash

   cd android/SherpaOnnxJavaDemo
   ./gradlew assembleDebug

Install the APK on your device:

.. code-block:: bash

   adb install app/build/outputs/apk/debug/app-debug.apk

Demo 2: SherpaOnnxSimulateStreamingAsrWearOs (Kotlin DSL)
---------------------------------------------------------

A Kotlin/Compose-based Wear OS demo that simulates streaming ASR.

  `<https://github.com/k2-fsa/sherpa-onnx/tree/master/android/SherpaOnnxSimulateStreamingAsrWearOs>`_

JitPack is configured in `settings.gradle.kts <https://github.com/k2-fsa/sherpa-onnx/blob/master/android/SherpaOnnxSimulateStreamingAsrWearOs/settings.gradle.kts>`_ (Kotlin DSL):

.. code-block:: kotlin

   dependencyResolutionManagement {
       repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
       repositories {
           google()
           mavenCentral()
           maven { url = uri("https://jitpack.io") }
       }
   }

The dependency in `app/build.gradle.kts <https://github.com/k2-fsa/sherpa-onnx/blob/master/android/SherpaOnnxSimulateStreamingAsrWearOs/app/build.gradle.kts>`_:

.. code-block:: kotlin

   implementation("com.github.k2-fsa.sherpa-onnx:sherpa-onnx:v1.13.5")

Build and run
:::::::::::::

.. code-block:: bash

   cd android/SherpaOnnxSimulateStreamingAsrWearOs
   ./gradlew assembleDebug

Install the APK on your Wear OS device:

.. code-block:: bash

   adb install app/build/outputs/apk/debug/app-debug.apk
