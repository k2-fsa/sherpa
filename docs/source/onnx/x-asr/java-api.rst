.. _onnx-x-asr-java-api:

Java
====

This page shows how to use `X-ASR <https://github.com/Gilgamesh-J/X-ASR>`_ models with the Java API of `sherpa-onnx`_. Two demos are given: one with a non-streaming model and one with a streaming model.

.. contents:: On this page
   :local:
   :depth: 2

Requirements
------------

- JDK 11 or above
- Maven 3.x

Non-streaming X-ASR
-------------------

Step 1. Create the project directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   mkdir x-asr-demo
   cd x-asr-demo

Step 2. Create the build file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Save the following as ``pom.xml``:

.. code-block:: xml

   <?xml version="1.0" encoding="UTF-8"?>
   <project xmlns="http://maven.apache.org/POM/4.0.0"
            xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
            xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
       <modelVersion>4.0.0</modelVersion>

       <groupId>com.example</groupId>
       <artifactId>sherpa-onnx-java-example</artifactId>
       <version>1.0</version>
       <packaging>jar</packaging>

       <properties>
           <maven.compiler.release>11</maven.compiler.release>
           <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
           <sherpa-onnx.version>v2.0.1</sherpa-onnx.version>
       </properties>

       <repositories>
           <!-- sherpa-onnx packages are hosted on JitPack -->
           <repository>
               <id>jitpack.io</id>
               <url>https://jitpack.io</url>
           </repository>
       </repositories>

       <dependencies>
           <dependency>
               <groupId>com.github.k2-fsa</groupId>
               <artifactId>sherpa-onnx</artifactId>
               <version>${sherpa-onnx.version}</version>
           </dependency>
       </dependencies>

       <build>
           <plugins>
               <plugin>
                   <groupId>org.apache.maven.plugins</groupId>
                   <artifactId>maven-jar-plugin</artifactId>
                   <version>3.3.0</version>
                   <configuration>
                       <archive>
                           <manifest>
                               <mainClass>NonStreamingDecodeFileXasr</mainClass>
                           </manifest>
                       </archive>
                   </configuration>
               </plugin>
               <plugin>
                   <groupId>org.apache.maven.plugins</groupId>
                   <artifactId>maven-shade-plugin</artifactId>
                   <version>3.5.1</version>
                   <executions>
                       <execution>
                           <phase>package</phase>
                           <goals>
                               <goal>shade</goal>
                           </goals>
                       </execution>
                   </executions>
               </plugin>
           </plugins>
       </build>
   </project>

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
   ├── pom.xml
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

   6 directories, 13 files

   $ ls -lh
   total 8
   -rw-r--r--@  1 fangjun  wheel   2.1K 19 Nov 12:23 pom.xml
   drwxr-xr-x@ 10 fangjun  wheel   320B 29 Jun 16:31 sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03
   drwxr-xr-x@  3 fangjun  wheel    96B 19 Nov 12:23 src

Step 6. Build and run
^^^^^^^^^^^^^^^^^^^^^

From the project directory, run:

.. code-block:: bash

   mvn -q package
   java -jar target/sherpa-onnx-java-example-1.0.jar

Output
^^^^^^

.. code-block:: text

   filename:./sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03/test_wavs/0.wav
   result:昨天是 Monday, today is 礼拜二， the day after tomorrow 是星期三。
   audio duration: 10.053 s
   elapsed seconds: 0.116 s
   RTF = 0.116/10.053 = 0.012

.. note::

   The timing above was measured on a MacBook Pro (Apple M4) using 2 threads. Please increase the number of threads for a smaller RTF on your machine.

Streaming X-ASR
---------------

Step 1. Create the project directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   mkdir x-asr-demo
   cd x-asr-demo

Step 2. Create the build file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Save the following as ``pom.xml``:

.. code-block:: xml

   <?xml version="1.0" encoding="UTF-8"?>
   <project xmlns="http://maven.apache.org/POM/4.0.0"
            xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
            xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
       <modelVersion>4.0.0</modelVersion>

       <groupId>com.example</groupId>
       <artifactId>sherpa-onnx-java-example</artifactId>
       <version>1.0</version>
       <packaging>jar</packaging>

       <properties>
           <maven.compiler.release>11</maven.compiler.release>
           <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
           <sherpa-onnx.version>v2.0.1</sherpa-onnx.version>
       </properties>

       <repositories>
           <!-- sherpa-onnx packages are hosted on JitPack -->
           <repository>
               <id>jitpack.io</id>
               <url>https://jitpack.io</url>
           </repository>
       </repositories>

       <dependencies>
           <dependency>
               <groupId>com.github.k2-fsa</groupId>
               <artifactId>sherpa-onnx</artifactId>
               <version>${sherpa-onnx.version}</version>
           </dependency>
       </dependencies>

       <build>
           <plugins>
               <plugin>
                   <groupId>org.apache.maven.plugins</groupId>
                   <artifactId>maven-jar-plugin</artifactId>
                   <version>3.3.0</version>
                   <configuration>
                       <archive>
                           <manifest>
                               <mainClass>StreamingDecodeFileXasr</mainClass>
                           </manifest>
                       </archive>
                   </configuration>
               </plugin>
               <plugin>
                   <groupId>org.apache.maven.plugins</groupId>
                   <artifactId>maven-shade-plugin</artifactId>
                   <version>3.5.1</version>
                   <executions>
                       <execution>
                           <phase>package</phase>
                           <goals>
                               <goal>shade</goal>
                           </goals>
                       </execution>
                   </executions>
               </plugin>
           </plugins>
       </build>
   </project>

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
   ├── pom.xml
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

   6 directories, 13 files

   $ ls -lh
   total 8
   -rw-r--r--@  1 fangjun  wheel   2.1K 19 Nov 12:23 pom.xml
   drwxr-xr-x@ 10 fangjun  wheel   320B 29 Jun 16:30 sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05
   drwxr-xr-x@  3 fangjun  wheel    96B 19 Nov 12:23 src

Step 6. Build and run
^^^^^^^^^^^^^^^^^^^^^

From the project directory, run:

.. code-block:: bash

   mvn -q package
   java -jar target/sherpa-onnx-java-example-1.0.jar

Output
^^^^^^

.. code-block:: text

   filename:./sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05/test_wavs/0.wav
   result:昨天是 Monday， today is 礼拜二， the day after tomorrow 是星期三
   audio duration: 10.053 s
   elapsed seconds: 0.326 s
   RTF = 0.326/10.053 = 0.032

.. note::

   The timing above was measured on a MacBook Pro (Apple M4) using 2 threads. Please increase the number of threads for a smaller RTF on your machine.

Notes
-----

Please see :ref:`sherpa-onnx-java-api` if you want to use Gradle or Gradle KTS instead of Maven.

