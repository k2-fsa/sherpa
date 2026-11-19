.. _java-gradle-kws:

Gradle: Keyword spotting
========================

This example shows how to use a streaming keyword spotting (KWS) model. The audio is read from a file and is fed to the keyword spotter in chunks; the detected keywords are printed.

It is the Java version of `KeywordSpotterFromFile.java <https://github.com/k2-fsa/sherpa-onnx/blob/master/java-api-examples/KeywordSpotterFromFile.java>`_ in `sherpa-onnx/java-api-examples <https://github.com/k2-fsa/sherpa-onnx/tree/master/java-api-examples>`_.

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

Save the following as ``build.gradle``:

.. code-block:: groovy

   plugins {
       id 'application'
       id 'java'
   }

   application {
       mainClass = 'KeywordSpotterFromFile'
   }

   repositories {
       mavenCentral()
       // sherpa-onnx packages are hosted on JitPack
       maven { url 'https://jitpack.io' }
   }

   dependencies {
       implementation 'com.github.k2-fsa:sherpa-onnx:v2.0.1'
   }

   java {
       sourceCompatibility = JavaVersion.VERSION_11
       targetCompatibility = JavaVersion.VERSION_11
   }

Save the following as ``settings.gradle``:

.. code-block:: groovy

   rootProject.name = 'sherpa-onnx-java-example'

Step 3. Download the model
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2
   tar xf sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2
   rm sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2

Step 4. Save the code
^^^^^^^^^^^^^^^^^^^^^

Save the following as ``src/main/java/KeywordSpotterFromFile.java``:

.. code-block:: java

   // Copyright 2024 Xiaomi Corporation

   // This file shows how to use a keyword spotter model to spot keywords from
   // a file.

   import com.k2fsa.sherpa.onnx.*;

   public class KeywordSpotterFromFile {
     public static void main(String[] args) {
       // please download test files from https://github.com/k2-fsa/sherpa-onnx/releases/tag/kws-models
       String encoder =
           "./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/encoder-epoch-12-avg-2-chunk-16-left-64.onnx";
       String decoder =
           "./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/decoder-epoch-12-avg-2-chunk-16-left-64.onnx";
       String joiner =
           "./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/joiner-epoch-12-avg-2-chunk-16-left-64.onnx";
       String tokens = "./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/tokens.txt";

       String keywordsFile =
           "./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/test_wavs/test_keywords.txt";

       String waveFilename = "./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/test_wavs/3.wav";

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
               .setNumThreads(1)
               .setDebug(true)
               .build();

       KeywordSpotterConfig config =
           KeywordSpotterConfig.builder()
               .setOnlineModelConfig(modelConfig)
               .setKeywordsFile(keywordsFile)
               .build();

       KeywordSpotter kws = new KeywordSpotter(config);
       OnlineStream stream = kws.createStream();

       WaveReader reader = new WaveReader(waveFilename);

       stream.acceptWaveform(reader.getSamples(), reader.getSampleRate());

       float[] tailPaddings = new float[(int) (0.8 * reader.getSampleRate())];
       stream.acceptWaveform(tailPaddings, reader.getSampleRate());
       while (kws.isReady(stream)) {
         kws.decode(stream);

         String keyword = kws.getResult(stream).getKeyword();
         if (!keyword.isEmpty()) {
           // Remember to reset the stream right after detecting a keyword
           kws.reset(stream);
           System.out.printf("Detected keyword: %s\n", keyword);
         }
       }

       kws.release();
     }
   }

Step 5. Check the project layout
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Your project directory should now look like this:

.. code-block:: text

   $ tree .
   .
   ├── build.gradle
   ├── settings.gradle
   ├── sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01
   │   ├── configuration.json
   │   ├── decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx
   │   ├── decoder-epoch-12-avg-2-chunk-16-left-64.onnx
   │   ├── decoder-epoch-99-avg-1-chunk-16-left-64.int8.onnx
   │   ├── decoder-epoch-99-avg-1-chunk-16-left-64.onnx
   │   ├── encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx
   │   ├── encoder-epoch-12-avg-2-chunk-16-left-64.onnx
   │   ├── encoder-epoch-99-avg-1-chunk-16-left-64.int8.onnx
   │   ├── encoder-epoch-99-avg-1-chunk-16-left-64.onnx
   │   ├── joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx
   │   ├── joiner-epoch-12-avg-2-chunk-16-left-64.onnx
   │   ├── joiner-epoch-99-avg-1-chunk-16-left-64.int8.onnx
   │   ├── joiner-epoch-99-avg-1-chunk-16-left-64.onnx
   │   ├── keywords_raw.txt
   │   ├── keywords.txt
   │   ├── README.md
   │   ├── test_wavs
   │   │   ├── 0.wav
   │   │   ├── 1.wav
   │   │   ├── 2.wav
   │   │   ├── 3.wav
   │   │   ├── 4.wav
   │   │   ├── 5.wav
   │   │   ├── 6.wav
   │   │   └── test_keywords.txt
   │   └── tokens.txt
   └── src
       └── main
           └── java
               └── KeywordSpotterFromFile.java

   6 directories, 28 files

   $ ls -lh
   total 16
   -rw-r--r--@  1 fangjun  wheel   414B 19 Nov 14:00 build.gradle
   -rw-r--r--@  1 fangjun  wheel    45B 19 Nov 14:00 settings.gradle
   drwxr-xr-x@ 20 fangjun  wheel   640B 11 Sep  2024 sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01
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

   [Incubating] Problems report is available at: file:///private/tmp/java-doc-work/layouts-java/gradle-kws/build/reports/problems/problems-report.html

   You can use '--warning-mode all' to show the individual deprecation warnings and determine if they come from your own scripts or plugins.

   For more on this, please refer to https://docs.gradle.org/9.6.1/userguide/command_line_interface.html#sec:command_line_warnings in the Gradle documentation.

   1 actionable task: 1 up-to-date
   Consider enabling configuration cache to speed up this build: https://docs.gradle.org/9.6.1/userguide/configuration_cache_enabling.html

   encoder_dims=128,128,128,128,128,128
   query_head_dims=32,32,32,32,32,32
   num_encoder_layers=1,1,1,1,1,1
   left_context_len=64,32,16,8,16,32
   model_type=zipformer2
   cnn_module_kernels=31,31,15,15,15,31
   decode_chunk_len=32
   value_head_dims=12,12,12,12,12,12
   T=45
   model_author=k2-fsa
   version=1
   comment=streaming zipformer2

   num_heads=4,4,4,8,4,4
   encoder_dims=128,128,128,128,128,128
   query_head_dims=32,32,32,32,32,32
   num_encoder_layers=1,1,1,1,1,1
   left_context_len=64,32,16,8,16,32
   model_type=zipformer2
   cnn_module_kernels=31,31,15,15,15,31
   decode_chunk_len=32
   value_head_dims=12,12,12,12,12,12
   T=45
   model_author=k2-fsa
   version=1
   comment=streaming zipformer2

   vocab_size=197
   context_size=2

   joiner_dim=320

   Detected keyword: 文森特卡索
   Detected keyword: 法国

   [Incubating] Problems report is available at: file:///private/tmp/java-doc-work/layouts-java/gradle-kws/build/reports/problems/problems-report.html

   You can use '--warning-mode all' to show the individual deprecation warnings and determine if they come from your own scripts or plugins.

   For more on this, please refer to https://docs.gradle.org/9.6.1/userguide/command_line_interface.html#sec:command_line_warnings in the Gradle documentation.

   2 actionable tasks: 2 executed
   Consider enabling configuration cache to speed up this build: https://docs.gradle.org/9.6.1/userguide/configuration_cache_enabling.html

