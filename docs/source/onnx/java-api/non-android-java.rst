Non-Android Java
====================

We provide plenty of examples about using non-Android Java API of `sherpa-onnx`_
in `<https://github.com/k2-fsa/sherpa-onnx/tree/master/java-api-examples>`_.

In this section, we describe how to run the examples in the following platforms:

  - Linux (x64)
  - Linux (arm64)
  - macOS (x64)
  - macOS (arm64)
  - Windows (x64)
  - Windows (arm64)

Prerequisites
-------------

- JDK 8 or above
- Gradle 8.x+ (or use the included Gradle wrapper, no separate install needed)

Create a Gradle project
-----------------------

Create a new directory for your project and add the following files.

.. warning::

  The sherpa-onnx Java packages are hosted on `JitPack <https://jitpack.io/>`_, so you need
  to add the JitPack repository.

``build.gradle.kts``
::::::::::::::::::::

.. code-block:: kotlin

   plugins {
       application
       java
   }

   application {
       mainClass.set("com.k2fsa.sherpa.onnx.example.YourApp")
   }

   repositories {
       mavenCentral()
       // sherpa-onnx packages are hosted on JitPack
       maven { url = uri("https://jitpack.io") }
   }

   // Auto-detect current OS and architecture
   val osName = System.getProperty("os.name").lowercase()
   val osArch = System.getProperty("os.arch").lowercase()

   val targetNativeClassifier = when {
       osName.contains("mac") || osName.contains("darwin") -> {
           if (osArch == "aarch64" || osArch == "arm64") "osx-aarch64" else "osx-x64"
       }
       osName.contains("linux") -> {
           if (osArch == "aarch64" || osArch == "arm64") "linux-aarch64" else "linux-x64"
       }
       osName.contains("win") -> {
           if (osArch == "aarch64" || osArch == "arm64") "win-arm64" else "win-x64"
       }
       else -> throw GradleException("Unsupported OS: $osName, Arch: $osArch")
   }

   logger.lifecycle("--> Auto-detected platform native lib: $targetNativeClassifier")

   dependencies {
       // 1. JVM core API
       implementation("com.github.k2-fsa.sherpa-onnx:sherpa-onnx-jvm:v1.13.5")

       // 2. Platform native lib (auto-detected)
       implementation("com.github.k2-fsa.sherpa-onnx:sherpa-onnx-native-lib-$targetNativeClassifier:v1.13.5")
   }

   java {
       sourceCompatibility = JavaVersion.VERSION_1_8
       targetCompatibility = JavaVersion.VERSION_1_8
   }

``settings.gradle.kts``
:::::::::::::::::::::::

.. code-block:: kotlin

   rootProject.name = "sherpa-onnx-java-example"

Generate the Gradle wrapper
:::::::::::::::::::::::::::

.. code-block:: bash

   gradle wrapper

This creates ``gradlew``, ``gradlew.bat``, and the ``gradle/`` directory so that
anyone can build the project without installing Gradle separately.

Build and run
-------------

.. code-block:: bash

   # Build
   ./gradlew build

   # Run
   ./gradlew run

On Windows, use ``gradlew.bat`` instead:

.. code-block:: cmd

   gradlew.bat build
   gradlew.bat run

The build output will show the auto-detected platform::

   --> Auto-detected platform native lib: osx-aarch64

.. note::

   The Gradle Kotlin DSL build file **automatically detects** your OS and architecture
   at build time, so there is no need to manually select platform-specific native
   libraries.

Useful links
------------

- Java API examples: `<https://github.com/k2-fsa/sherpa-onnx/tree/master/java-api-examples>`_

  - Maven examples: `<https://github.com/k2-fsa/sherpa-onnx/tree/master/java-api-examples/maven-examples>`_
  - Gradle examples: `<https://github.com/k2-fsa/sherpa-onnx/tree/master/java-api-examples/gradle-examples>`_
  - Gradle KTS examples: `<https://github.com/k2-fsa/sherpa-onnx/tree/master/java-api-examples/gradle-kts-examples>`_

- Java API source files: `<https://github.com/k2-fsa/sherpa-onnx/tree/master/sherpa-onnx/java-api/src/com/k2fsa/sherpa/onnx>`_
