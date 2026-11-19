.. _onnx-x-asr-kotlin-api:

Kotlin
======

This page shows how to use `X-ASR <https://github.com/Gilgamesh-J/X-ASR>`_ models with the Kotlin API of `sherpa-onnx`_. Two demos are given: one with a non-streaming model and one with a streaming model. You can build the project with either Maven or Gradle (Kotlin DSL).

.. contents:: On this page
   :local:
   :depth: 2

Requirements
------------

- JDK 11 or above
- Maven 3.x or Gradle 8.x+

Non-streaming X-ASR
-------------------

Step 1. Create the project directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   mkdir x-asr-demo
   cd x-asr-demo

Step 2. Create the build files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you use Maven, create ``pom.xml``. If you use Gradle (Kotlin DSL), create ``build.gradle.kts`` and ``settings.gradle.kts``.

Save the following as ``pom.xml``:

.. code-block:: xml

   <?xml version="1.0" encoding="UTF-8"?>
   <project xmlns="http://maven.apache.org/POM/4.0.0"
            xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
            xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
       <modelVersion>4.0.0</modelVersion>

       <groupId>com.example</groupId>
       <artifactId>x-asr-kotlin-example</artifactId>
       <version>1.0</version>
       <packaging>jar</packaging>

       <properties>
           <kotlin.version>2.2.20</kotlin.version>
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
               <groupId>org.jetbrains.kotlin</groupId>
               <artifactId>kotlin-stdlib</artifactId>
               <version>${kotlin.version}</version>
           </dependency>
           <dependency>
               <groupId>com.github.k2-fsa</groupId>
               <artifactId>sherpa-onnx</artifactId>
               <version>${sherpa-onnx.version}</version>
           </dependency>
       </dependencies>

       <build>
           <sourceDirectory>${project.basedir}/src/main/kotlin</sourceDirectory>
           <plugins>
               <plugin>
                   <groupId>org.jetbrains.kotlin</groupId>
                   <artifactId>kotlin-maven-plugin</artifactId>
                   <version>${kotlin.version}</version>
                   <executions>
                       <execution>
                           <id>compile</id>
                           <phase>compile</phase>
                           <goals>
                               <goal>compile</goal>
                           </goals>
                       </execution>
                   </executions>
                   <configuration>
                       <jvmTarget>11</jvmTarget>
                   </configuration>
               </plugin>
               <plugin>
                   <groupId>org.apache.maven.plugins</groupId>
                   <artifactId>maven-jar-plugin</artifactId>
                   <version>3.3.0</version>
                   <configuration>
                       <archive>
                           <manifest>
                               <mainClass>XasrOfflineDemoKt</mainClass>
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

Save the following as ``build.gradle.kts``:

.. code-block:: kotlin

   plugins {
       kotlin("jvm") version "2.2.20"
       application
   }

   application {
       mainClass.set("XasrOfflineDemoKt")
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

   kotlin {
       compilerOptions {
           jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_11)
       }
   }

Save the following as ``settings.gradle.kts``:

.. code-block:: kotlin

   pluginManagement {
       repositories {
           mavenCentral()
           gradlePluginPortal()
       }
   }

   rootProject.name = "x-asr-kotlin-example"


Step 3. Download the model
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03.tar.bz2
   tar xf sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03.tar.bz2
   rm sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03.tar.bz2

Step 4. Save the code
^^^^^^^^^^^^^^^^^^^^^

Save the following as ``src/main/kotlin/XasrOfflineDemo.kt``:

.. code-block:: kotlin

   // This file shows how to use a non-streaming X-ASR model with sherpa-onnx
   // in Kotlin.
   //
   // Before you run it, please download the model:
   //
   //   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03.tar.bz2
   //   tar xf sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03.tar.bz2
   //
   // Note: this file has a top-level main function, so the generated main
   // class is XasrOfflineDemoKt (the file name plus the suffix Kt).
   import com.k2fsa.sherpa.onnx.*

   fun main() {
     val modelDir = "./sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03"

     // Step 1: Create the recognizer from the X-ASR model files.
     val transducer = OfflineTransducerModelConfig.builder()
       .setEncoder("$modelDir/encoder-epoch-99-avg-1.int8.onnx")
       .setDecoder("$modelDir/decoder-epoch-99-avg-1.onnx")
       .setJoiner("$modelDir/joiner-epoch-99-avg-1.int8.onnx")
       .build()
     val modelConfig = OfflineModelConfig.builder()
       .setTransducer(transducer)
       .setTokens("$modelDir/tokens.txt")
       .setNumThreads(2)
       .setDebug(false)
       .build()
     val config = OfflineRecognizerConfig.builder()
       .setOfflineModelConfig(modelConfig)
       .setDecodingMethod("greedy_search")
       .build()
     val recognizer = OfflineRecognizer(config)

     // Step 2: Read the wave file you want to transcribe.
     val filename = "$modelDir/test_wavs/0.wav"
     val reader = WaveReader(filename)

     // Step 3: Create a stream, send the whole audio to it, and decode.
     val start = System.nanoTime()
     val stream = recognizer.createStream()
     stream.acceptWaveform(reader.getSamples(), reader.getSampleRate())
     recognizer.decode(stream)
     val elapsed = (System.nanoTime() - start) / 1e9

     // Step 4: Print the result. X-ASR punct models also add punctuations.
     println(filename)
     println(recognizer.getResult(stream).getText())

     val audioDuration = reader.getSamples().size.toDouble() / reader.getSampleRate()
     println(String.format("audio duration: %.3f s", audioDuration))
     println(String.format("elapsed seconds: %.3f s", elapsed))
     println(String.format("RTF = %.3f/%.3f = %.3f", elapsed, audioDuration, elapsed / audioDuration))

     stream.release()
     recognizer.release()
   }

Step 5. Check the project layout
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Your project directory should now look like this:

.. code-block:: text

   $ tree .
   .
   ├── build.gradle.kts
   ├── pom.xml
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
           └── kotlin
               └── XasrOfflineDemo.kt

   6 directories, 15 files

   $ ls -lh
   total 24
   -rw-r--r--@  1 fangjun  wheel   548B 19 Nov 12:23 build.gradle.kts
   -rw-r--r--@  1 fangjun  wheel   3.1K 19 Nov 12:23 pom.xml
   -rw-r--r--@  1 fangjun  wheel   141B 19 Nov 12:23 settings.gradle.kts
   drwxr-xr-x@ 10 fangjun  wheel   320B 29 Jun 16:31 sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03
   drwxr-xr-x@  3 fangjun  wheel    96B 19 Nov 12:23 src

Step 6. Build and run
^^^^^^^^^^^^^^^^^^^^^

From the project directory, run:

.. code-block:: bash

   # Maven
   mvn -q package
   java -jar target/x-asr-kotlin-example-1.0.jar

   # or Gradle KTS
   gradle wrapper
   ./gradlew run

Output
^^^^^^

.. code-block:: text

   WARNING: A restricted method in java.lang.System has been called
   WARNING: java.lang.System::load has been called by com.k2fsa.sherpa.onnx.LibraryUtils in an unnamed module (file:/private/tmp/java-doc-work/test-kt-maven/target/x-asr-kotlin-example-1.0.jar)
   WARNING: Use --enable-native-access=ALL-UNNAMED to avoid a warning for callers in this module
   WARNING: Restricted methods will be blocked in a future release unless native access is enabled

   ./sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03/test_wavs/0.wav
   昨天是 Monday, today is 礼拜二， the day after tomorrow 是星期三。
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

Step 2. Create the build files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you use Maven, create ``pom.xml``. If you use Gradle (Kotlin DSL), create ``build.gradle.kts`` and ``settings.gradle.kts``.

Save the following as ``pom.xml``:

.. code-block:: xml

   <?xml version="1.0" encoding="UTF-8"?>
   <project xmlns="http://maven.apache.org/POM/4.0.0"
            xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
            xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
       <modelVersion>4.0.0</modelVersion>

       <groupId>com.example</groupId>
       <artifactId>x-asr-kotlin-example</artifactId>
       <version>1.0</version>
       <packaging>jar</packaging>

       <properties>
           <kotlin.version>2.2.20</kotlin.version>
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
               <groupId>org.jetbrains.kotlin</groupId>
               <artifactId>kotlin-stdlib</artifactId>
               <version>${kotlin.version}</version>
           </dependency>
           <dependency>
               <groupId>com.github.k2-fsa</groupId>
               <artifactId>sherpa-onnx</artifactId>
               <version>${sherpa-onnx.version}</version>
           </dependency>
       </dependencies>

       <build>
           <sourceDirectory>${project.basedir}/src/main/kotlin</sourceDirectory>
           <plugins>
               <plugin>
                   <groupId>org.jetbrains.kotlin</groupId>
                   <artifactId>kotlin-maven-plugin</artifactId>
                   <version>${kotlin.version}</version>
                   <executions>
                       <execution>
                           <id>compile</id>
                           <phase>compile</phase>
                           <goals>
                               <goal>compile</goal>
                           </goals>
                       </execution>
                   </executions>
                   <configuration>
                       <jvmTarget>11</jvmTarget>
                   </configuration>
               </plugin>
               <plugin>
                   <groupId>org.apache.maven.plugins</groupId>
                   <artifactId>maven-jar-plugin</artifactId>
                   <version>3.3.0</version>
                   <configuration>
                       <archive>
                           <manifest>
                               <mainClass>XasrStreamingDemoKt</mainClass>
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

Save the following as ``build.gradle.kts``:

.. code-block:: kotlin

   plugins {
       kotlin("jvm") version "2.2.20"
       application
   }

   application {
       mainClass.set("XasrStreamingDemoKt")
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

   kotlin {
       compilerOptions {
           jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_11)
       }
   }

Save the following as ``settings.gradle.kts``:

.. code-block:: kotlin

   pluginManagement {
       repositories {
           mavenCentral()
           gradlePluginPortal()
       }
   }

   rootProject.name = "x-asr-kotlin-example"


Step 3. Download the model
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05.tar.bz2
   tar xf sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05.tar.bz2
   rm sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05.tar.bz2

Step 4. Save the code
^^^^^^^^^^^^^^^^^^^^^

Save the following as ``src/main/kotlin/XasrStreamingDemo.kt``:

.. code-block:: kotlin

   // This file shows how to use a streaming X-ASR model with sherpa-onnx
   // in Kotlin.
   //
   // Before you run it, please download the model:
   //
   //   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05.tar.bz2
   //   tar xf sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05.tar.bz2
   //
   // Note: this file has a top-level main function, so the generated main
   // class is XasrStreamingDemoKt (the file name plus the suffix Kt).
   import com.k2fsa.sherpa.onnx.*

   fun main() {
     val modelDir =
       "./sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05"

     // Step 1: Create the recognizer from the X-ASR model files.
     val transducer = OnlineTransducerModelConfig.builder()
       .setEncoder("$modelDir/encoder.int8.onnx")
       .setDecoder("$modelDir/decoder.onnx")
       .setJoiner("$modelDir/joiner.int8.onnx")
       .build()
     val modelConfig = OnlineModelConfig.builder()
       .setTransducer(transducer)
       .setTokens("$modelDir/tokens.txt")
       .setNumThreads(2)
       .setDebug(false)
       .build()
     val config = OnlineRecognizerConfig.builder()
       .setOnlineModelConfig(modelConfig)
       .setDecodingMethod("greedy_search")
       .build()
     val recognizer = OnlineRecognizer(config)

     // Step 2: Read the wave file you want to transcribe.
     val filename = "$modelDir/test_wavs/0.wav"
     val reader = WaveReader(filename)
     val samples = reader.getSamples()
     val sampleRate = reader.getSampleRate()

     // Step 3: Create a stream and feed the audio in small chunks, just like
     // a microphone would do in a real-time application.
     val start = System.nanoTime()
     val stream = recognizer.createStream()

     // 0.2 second for 16 kHz audio
     val chunk = 3200

     var last = ""
     var i = 0
     while (i < samples.size) {
       val end = minOf(i + chunk, samples.size)
       stream.acceptWaveform(samples.copyOfRange(i, end), sampleRate)

       while (recognizer.isReady(stream)) {
         recognizer.decode(stream)
       }

       // Print the latest partial result when it changes.
       val text = recognizer.getResult(stream).getText()
       if (text.isNotEmpty() && text != last) {
         last = text
         println(text)
       }
       i += chunk
     }

     // Send some tail padding so the last words are flushed out.
     stream.acceptWaveform(FloatArray((0.5 * sampleRate).toInt()), sampleRate)
     while (recognizer.isReady(stream)) {
       recognizer.decode(stream)
     }

     // Step 4: Print the final result.
     val elapsed = (System.nanoTime() - start) / 1e9
     println("Final result: " + recognizer.getResult(stream).getText())

     val audioDuration = samples.size.toDouble() / sampleRate
     println(String.format("audio duration: %.3f s", audioDuration))
     println(String.format("elapsed seconds: %.3f s", elapsed))
     println(String.format("RTF = %.3f/%.3f = %.3f", elapsed, audioDuration, elapsed / audioDuration))

     stream.release()
     recognizer.release()
   }

Step 5. Check the project layout
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Your project directory should now look like this:

.. code-block:: text

   $ tree .
   .
   ├── build.gradle.kts
   ├── pom.xml
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
           └── kotlin
               └── XasrStreamingDemo.kt

   6 directories, 15 files

   $ ls -lh
   total 24
   -rw-r--r--@  1 fangjun  wheel   550B 19 Nov 12:23 build.gradle.kts
   -rw-r--r--@  1 fangjun  wheel   3.1K 19 Nov 12:23 pom.xml
   -rw-r--r--@  1 fangjun  wheel   141B 19 Nov 12:23 settings.gradle.kts
   drwxr-xr-x@ 10 fangjun  wheel   320B 29 Jun 16:30 sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05
   drwxr-xr-x@  3 fangjun  wheel    96B 19 Nov 12:23 src

Step 6. Build and run
^^^^^^^^^^^^^^^^^^^^^

From the project directory, run:

.. code-block:: bash

   # Maven
   mvn -q package
   java -jar target/x-asr-kotlin-example-1.0.jar

   # or Gradle KTS
   gradle wrapper
   ./gradlew run

Output
^^^^^^

.. code-block:: text

   WARNING: A restricted method in java.lang.System has been called
   WARNING: java.lang.System::load has been called by com.k2fsa.sherpa.onnx.LibraryUtils in an unnamed module (file:/private/tmp/java-doc-work/test-kt-maven-st/target/x-asr-kotlin-example-1.0.jar)
   WARNING: Use --enable-native-access=ALL-UNNAMED to avoid a warning for callers in this module
   WARNING: Restricted methods will be blocked in a future release unless native access is enabled

   昨
   昨天是
   昨天是 Mon
   昨天是 Monday
   昨天是 Monday，
   昨天是 Monday， today
   昨天是 Monday， today is 礼
   昨天是 Monday， today is 礼拜二
   昨天是 Monday， today is 礼拜二，
   昨天是 Monday， today is 礼拜二， the day
   昨天是 Monday， today is 礼拜二， the day after
   昨天是 Monday， today is 礼拜二， the day after tomorro
   昨天是 Monday， today is 礼拜二， the day after tomorrow
   昨天是 Monday， today is 礼拜二， the day after tomorrow 是
   Final result: 昨天是 Monday， today is 礼拜二， the day after tomorrow 是星期
   audio duration: 10.053 s
   elapsed seconds: 0.314 s
   RTF = 0.314/10.053 = 0.031

.. note::

   The timing above was measured on a MacBook Pro (Apple M4) using 2 threads. Please increase the number of threads for a smaller RTF on your machine.

Notes
-----

A Kotlin file with a top-level ``main`` function is compiled to a class named after the file plus the suffix ``Kt``, e.g., ``XasrOfflineDemo.kt`` -> ``XasrOfflineDemoKt``. This is why the main class in the build files above ends with ``Kt``.

