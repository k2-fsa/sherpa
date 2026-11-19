.. _java-gradle-kts-streaming-asr:

Gradle KTS: Streaming ASR
=========================

This example shows how to use an `X-ASR <https://github.com/Gilgamesh-J/X-ASR>`_ streaming zipformer transducer model for speech recognition. The audio is fed to the recognizer in small chunks and partial results are printed while the audio is being processed.

The model used here has a chunk size of 480 ms. Please see :ref:`onnx-x-asr` for the other chunk sizes (160 ms, 960 ms, 1920 ms) and the fp32 and no-punct variants, and `java-api-examples <https://github.com/k2-fsa/sherpa-onnx/tree/master/java-api-examples>`_ for more streaming ASR models, e.g., `StreamingDecodeFileParaformer.java <https://github.com/k2-fsa/sherpa-onnx/blob/master/java-api-examples/StreamingDecodeFileParaformer.java>`_, `StreamingDecodeFileCtc.java <https://github.com/k2-fsa/sherpa-onnx/blob/master/java-api-examples/StreamingDecodeFileCtc.java>`_.

It is the Java version of `StreamingDecodeFileXasr.java <https://github.com/k2-fsa/sherpa-onnx/blob/master/java-api-examples/StreamingDecodeFileXasr.java>`_ in `sherpa-onnx/java-api-examples <https://github.com/k2-fsa/sherpa-onnx/tree/master/java-api-examples>`_.

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
       mainClass.set("StreamingDecodeFileXasr")
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

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05.tar.bz2
   tar xf sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05.tar.bz2
   rm sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05.tar.bz2

Step 4. Save the code
^^^^^^^^^^^^^^^^^^^^^

Save the following as ``src/main/java/StreamingDecodeFileXasr.java``:

.. code-block:: java

   // Copyright 2026 Xiaomi Corporation

   // This file shows how to use an X-ASR streaming zipformer transducer model
   // to decode files.
   //
   // X-ASR is from https://github.com/Gilgamesh-J/X-ASR and supports both
   // Chinese and English. The punct model used here also adds punctuations to
   // the recognition result.
   //
   // Please see
   // https://k2-fsa.github.io/sherpa/onnx/x-asr/index.html
   // to download model files
   import com.k2fsa.sherpa.onnx.*;

   public class StreamingDecodeFileXasr {
     public static void main(String[] args) {
       String modelDir =
           "./sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05";
       String encoder = modelDir + "/encoder.int8.onnx";
       String decoder = modelDir + "/decoder.onnx";
       String joiner = modelDir + "/joiner.int8.onnx";
       String tokens = modelDir + "/tokens.txt";

       String waveFilename = modelDir + "/test_wavs/0.wav";

       WaveReader reader = new WaveReader(waveFilename);

       OnlineTransducerModelConfig transducer =
           OnlineTransducerModelConfig.builder()
               .setEncoder(encoder)
               .setDecoder(decoder)
               .setJoiner(joiner)
               .build();

       OnlineModelConfig modelConfig =
           OnlineModelConfig.builder()
               .setTransducer(transducer)
               .setTokens(tokens)
               .setNumThreads(2)
               .setDebug(false)
               .build();

       OnlineRecognizerConfig config =
           OnlineRecognizerConfig.builder()
               .setOnlineModelConfig(modelConfig)
               .setDecodingMethod("greedy_search")
               .build();

       OnlineRecognizer recognizer = new OnlineRecognizer(config);
       long start = System.currentTimeMillis();
       OnlineStream stream = recognizer.createStream();
       stream.acceptWaveform(reader.getSamples(), reader.getSampleRate());

       float[] tailPaddings = new float[(int) (0.8 * reader.getSampleRate())];
       stream.acceptWaveform(tailPaddings, reader.getSampleRate());

       while (recognizer.isReady(stream)) {
         recognizer.decode(stream);
       }

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
   ├── sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05
   │   ├── bpe.model
   │   ├── decoder.onnx
   │   ├── encoder.int8.onnx
   │   ├── joiner.int8.onnx
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
               └── StreamingDecodeFileXasr.java

   6 directories, 14 files

   $ ls -lh
   total 16
   -rw-r--r--@  1 fangjun  wheel   416B 19 Nov 14:00 build.gradle.kts
   -rw-r--r--@  1 fangjun  wheel    45B 19 Nov 14:00 settings.gradle.kts
   drwxr-xr-x@ 10 fangjun  wheel   320B 29 Jun 16:30 sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05
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

   filename:./sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05/test_wavs/0.wav
   result:昨天是 Monday， today is 礼拜二， the day after tomorrow 是星期三
   audio duration: 10.053 s
   elapsed seconds: 0.332 s
   RTF = 0.332/10.053 = 0.033

   2 actionable tasks: 2 executed
   Consider enabling configuration cache to speed up this build: https://docs.gradle.org/9.6.1/userguide/configuration_cache_enabling.html

