.. _java-gradle-kts-asr:

Gradle KTS: Offline ASR
=======================

This example shows how to use an `X-ASR <https://github.com/Gilgamesh-J/X-ASR>`_ non-streaming zipformer transducer model for speech recognition. It reads a wave file and prints the recognition result.

`X-ASR <https://github.com/Gilgamesh-J/X-ASR>`_ supports both Chinese and English. The punct model used here also adds punctuations to the result. Please see :ref:`onnx-x-asr` for more `X-ASR <https://github.com/Gilgamesh-J/X-ASR>`_ models and `java-api-examples <https://github.com/k2-fsa/sherpa-onnx/tree/master/java-api-examples>`_ for more offline ASR models, e.g., `NonStreamingDecodeFileParaformer.java <https://github.com/k2-fsa/sherpa-onnx/blob/master/java-api-examples/NonStreamingDecodeFileParaformer.java>`_, `NonStreamingDecodeFileWhisper.java <https://github.com/k2-fsa/sherpa-onnx/blob/master/java-api-examples/NonStreamingDecodeFileWhisper.java>`_, `NonStreamingDecodeFileSenseVoice.java <https://github.com/k2-fsa/sherpa-onnx/blob/master/java-api-examples/NonStreamingDecodeFileSenseVoice.java>`_. The API is the same; only the model config changes.

It is the Java version of `NonStreamingDecodeFileXasr.java <https://github.com/k2-fsa/sherpa-onnx/blob/master/java-api-examples/NonStreamingDecodeFileXasr.java>`_ in `sherpa-onnx/java-api-examples <https://github.com/k2-fsa/sherpa-onnx/tree/master/java-api-examples>`_.

.. contents:: On this page
   :local:
   :depth: 2

Requirements
------------

- JDK 11 or above
- Gradle 8.x+ (or use the included Gradle wrapper, no separate install needed)

.. warning::

   The sherpa-onnx Java packages are hosted on `JitPack <https://jitpack.io/>`_, so you need to add the JitPack repository.

Step 1. Create the project directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   mkdir sherpa-onnx-java-demo
   cd sherpa-onnx-java-demo

Step 2. Create the build files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Save the following as ``build.gradle.kts``:

.. code-block:: kotlin

   plugins {
       application
       java
   }

   application {
       mainClass.set("NonStreamingDecodeFileXasr")
   }

   repositories {
       mavenCentral()
       // sherpa-onnx packages are hosted on JitPack
       maven { url = uri("https://jitpack.io") }
   }

   dependencies {
       implementation("com.github.k2-fsa:sherpa-onnx:v2.0.1")
   }

   java {
       sourceCompatibility = JavaVersion.VERSION_11
       targetCompatibility = JavaVersion.VERSION_11
   }

Save the following as ``settings.gradle.kts``:

.. code-block:: kotlin

   rootProject.name = "sherpa-onnx-java-example"

Step 3. Download the model
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03.tar.bz2
   tar xf sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03.tar.bz2
   rm sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03.tar.bz2

Step 4. Save the code
^^^^^^^^^^^^^^^^^^^^^

Save the following as ``src/main/java/NonStreamingDecodeFileXasr.java``:

.. code-block:: java

   // Copyright 2026 Xiaomi Corporation

   // This file shows how to use an X-ASR non-streaming zipformer transducer
   // model to decode files.
   //
   // X-ASR is from https://github.com/Gilgamesh-J/X-ASR and supports both
   // Chinese and English. The punct model used here also adds punctuations to
   // the recognition result.
   //
   // Please see
   // https://k2-fsa.github.io/sherpa/onnx/x-asr/index.html
   // to download model files
   import com.k2fsa.sherpa.onnx.*;

   public class NonStreamingDecodeFileXasr {
     public static void main(String[] args) {
       String modelDir = "./sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03";
       String encoder = modelDir + "/encoder-epoch-99-avg-1.int8.onnx";
       String decoder = modelDir + "/decoder-epoch-99-avg-1.onnx";
       String joiner = modelDir + "/joiner-epoch-99-avg-1.int8.onnx";
       String tokens = modelDir + "/tokens.txt";

       String waveFilename = modelDir + "/test_wavs/0.wav";

       WaveReader reader = new WaveReader(waveFilename);

       OfflineTransducerModelConfig transducer =
           OfflineTransducerModelConfig.builder()
               .setEncoder(encoder)
               .setDecoder(decoder)
               .setJoiner(joiner)
               .build();

       OfflineModelConfig modelConfig =
           OfflineModelConfig.builder()
               .setTransducer(transducer)
               .setTokens(tokens)
               .setNumThreads(2)
               .setDebug(false)
               .build();

       OfflineRecognizerConfig config =
           OfflineRecognizerConfig.builder()
               .setOfflineModelConfig(modelConfig)
               .setDecodingMethod("greedy_search")
               .build();

       OfflineRecognizer recognizer = new OfflineRecognizer(config);
       long start = System.currentTimeMillis();
       OfflineStream stream = recognizer.createStream();
       stream.acceptWaveform(reader.getSamples(), reader.getSampleRate());

       recognizer.decode(stream);

       String text = recognizer.getResult(stream).getText();

       float timeElapsedSeconds = (System.currentTimeMillis() - start) / 1000.0f;
       System.out.printf("filename:%s\nresult:%s\n", waveFilename, text);
       float audioDuration = reader.getSamples().length / (float) reader.getSampleRate();
       System.out.printf("audio duration: %.3f s\n", audioDuration);
       System.out.printf("elapsed seconds: %.3f s\n", timeElapsedSeconds);
       System.out.printf(
           "RTF = %.3f/%.3f = %.3f\n", timeElapsedSeconds, audioDuration, timeElapsedSeconds / audioDuration);


       stream.release();
       recognizer.release();
     }
   }

Step 5. Check the project layout
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Your project directory should now look like this:

.. code-block:: text

   $ tree .
   .
   ├── build.gradle.kts
   ├── settings.gradle.kts
   ├── sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03
   │   ├── bpe.model
   │   ├── decoder-epoch-99-avg-1.onnx
   │   ├── encoder-epoch-99-avg-1.int8.onnx
   │   ├── joiner-epoch-99-avg-1.int8.onnx
   │   ├── README.md
   │   ├── test_onnx.py
   │   ├── test_wavs
   │   │   ├── 0.wav
   │   │   ├── 1.wav
   │   │   ├── 2.wav
   │   │   └── 3.wav
   │   └── tokens.txt
   └── src
       └── main
           └── java
               └── NonStreamingDecodeFileXasr.java

   6 directories, 14 files

   $ ls -lh
   total 16
   -rw-r--r--@  1 fangjun  wheel   419B 19 Nov 14:00 build.gradle.kts
   -rw-r--r--@  1 fangjun  wheel    45B 19 Nov 14:00 settings.gradle.kts
   drwxr-xr-x@ 10 fangjun  wheel   320B 29 Jun 16:31 sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03
   drwxr-xr-x@  3 fangjun  wheel    96B 19 Nov 14:00 src

Step 6. Build and run
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Generate the Gradle wrapper (only once; requires Gradle to be installed)
   gradle wrapper

   ./gradlew run

Output
^^^^^^

.. code-block:: text

   1 actionable task: 1 executed
   Consider enabling configuration cache to speed up this build: https://docs.gradle.org/9.6.1/userguide/configuration_cache_enabling.html

   filename:./sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03/test_wavs/0.wav
   result:昨天是 Monday, today is 礼拜二， the day after tomorrow 是星期三。
   audio duration: 10.053 s
   elapsed seconds: 0.124 s
   RTF = 0.124/10.053 = 0.012

   2 actionable tasks: 2 executed
   Consider enabling configuration cache to speed up this build: https://docs.gradle.org/9.6.1/userguide/configuration_cache_enabling.html

