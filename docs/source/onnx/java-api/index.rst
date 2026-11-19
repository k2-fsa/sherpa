.. _sherpa-onnx-java-api:

Java API
========

In this section, we describe how to use the ``Java`` API of `sherpa-onnx`_ on
non-Android platforms, i.e., Linux, macOS and Windows.

The examples used in this section come from
`java-api-examples <https://github.com/k2-fsa/sherpa-onnx/tree/master/java-api-examples>`_.
For each function, e.g., ASR, TTS, VAD, forced alignment, etc., there is one
page for each build tool: Maven, Gradle, and Gradle KTS. Every page is
self-contained: it shows the project files, the Java code, how to download the
model, how to run the example, and the expected output.

Please use the table of contents below (or at the right) to navigate to the
page you are interested in.

Requirements
------------

- JDK 11 or above
- One of

  - Maven 3.x
  - Gradle 8.x+ (or use the Gradle wrapper included in the example projects)

Supported platforms
-------------------

  - Linux (x64)
  - Linux (arm64)
  - macOS (x64)
  - macOS (arm64)
  - Windows (x64)
  - Windows (arm64)

Dependencies
------------

The sherpa-onnx Java packages are hosted on `JitPack <https://jitpack.io/>`_,
so you need to add the JitPack repository in your project. There are two ways
to declare the dependencies.

Approach 1: simple (single dependency)
::::::::::::::::::::::::::::::::::::::

This is the easiest way — one dependency pulls in everything. The pages below
use this approach to keep the build files simple:

.. code-block:: xml

   <dependency>
       <groupId>com.github.k2-fsa</groupId>
       <artifactId>sherpa-onnx</artifactId>
       <version>v2.0.1</version>
   </dependency>

Note that it bundles the native libraries for every platform, so the
resulting jar is very large (about 95 MB compressed).

Approach 2: multi-module (smaller jars)
:::::::::::::::::::::::::::::::::::::::

This is useful for production use. It splits the JVM API and the native
libraries so you only ship the platform you need:

.. code-block:: xml

   <!-- 1. JVM core API -->
   <dependency>
       <groupId>com.github.k2-fsa.sherpa-onnx</groupId>
       <artifactId>sherpa-onnx-jvm</artifactId>
       <version>v2.0.1</version>
   </dependency>

   <!-- 2. Platform native lib, pick ONE for your target platform -->
   <dependency>
       <groupId>com.github.k2-fsa.sherpa-onnx</groupId>
       <artifactId>sherpa-onnx-native-lib-osx-aarch64</artifactId>
       <version>v2.0.1</version>
   </dependency>

The available native library artifacts are:

.. list-table::
   :header-rows: 1

   * - Platform
     - Artifact
   * - macOS ARM64 (Apple Silicon)
     - ``sherpa-onnx-native-lib-osx-aarch64``
   * - macOS x64 (Intel)
     - ``sherpa-onnx-native-lib-osx-x64``
   * - Linux x64
     - ``sherpa-onnx-native-lib-linux-x64``
   * - Linux ARM64
     - ``sherpa-onnx-native-lib-linux-aarch64``
   * - Windows x64
     - ``sherpa-onnx-native-lib-win-x64``
   * - Windows ARM64
     - ``sherpa-onnx-native-lib-win-arm64``

The pages below use approach 1. If you want to ship only one platform and
keep the jar small, please switch to approach 2.

Examples
--------

.. toctree::
   :maxdepth: 1
   :caption: Android

   ./android.rst

.. toctree::
   :maxdepth: 1
   :caption: Maven

   ./maven-version-test.rst
   ./maven-asr.rst
   ./maven-streaming-asr.rst
   ./maven-tts.rst
   ./maven-vad.rst
   ./maven-forced-alignment.rst
   ./maven-kws.rst
   ./maven-punctuation.rst
   ./maven-diacritics.rst
   ./maven-speaker-identification.rst
   ./maven-audio-tagging.rst
   ./maven-speech-enhancement.rst
   ./maven-spoken-language-identification.rst
   ./maven-speaker-diarization.rst
   ./maven-itn.rst
   ./maven-websocket.rst

.. toctree::
   :maxdepth: 1
   :caption: Gradle

   ./gradle-version-test.rst
   ./gradle-asr.rst
   ./gradle-streaming-asr.rst
   ./gradle-tts.rst
   ./gradle-vad.rst
   ./gradle-forced-alignment.rst
   ./gradle-kws.rst
   ./gradle-punctuation.rst
   ./gradle-diacritics.rst
   ./gradle-speaker-identification.rst
   ./gradle-audio-tagging.rst
   ./gradle-speech-enhancement.rst
   ./gradle-spoken-language-identification.rst
   ./gradle-speaker-diarization.rst
   ./gradle-itn.rst
   ./gradle-websocket.rst

.. toctree::
   :maxdepth: 1
   :caption: Gradle KTS

   ./gradle-kts-version-test.rst
   ./gradle-kts-asr.rst
   ./gradle-kts-streaming-asr.rst
   ./gradle-kts-tts.rst
   ./gradle-kts-vad.rst
   ./gradle-kts-forced-alignment.rst
   ./gradle-kts-kws.rst
   ./gradle-kts-punctuation.rst
   ./gradle-kts-diacritics.rst
   ./gradle-kts-speaker-identification.rst
   ./gradle-kts-audio-tagging.rst
   ./gradle-kts-speech-enhancement.rst
   ./gradle-kts-spoken-language-identification.rst
   ./gradle-kts-speaker-diarization.rst
   ./gradle-kts-itn.rst
   ./gradle-kts-websocket.rst

Useful links
------------

- Java API examples: `<https://github.com/k2-fsa/sherpa-onnx/tree/master/java-api-examples>`_

  - Maven examples: `<https://github.com/k2-fsa/sherpa-onnx/tree/master/java-api-examples/maven-examples>`_
  - Gradle examples: `<https://github.com/k2-fsa/sherpa-onnx/tree/master/java-api-examples/gradle-examples>`_
  - Gradle KTS examples: `<https://github.com/k2-fsa/sherpa-onnx/tree/master/java-api-examples/gradle-kts-examples>`_

- Java API source files: `<https://github.com/k2-fsa/sherpa-onnx/tree/master/sherpa-onnx/java-api/src/main/java/com/k2fsa/sherpa/onnx>`_
