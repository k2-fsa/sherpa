.. _java-gradle-version-test:

Gradle: Version test
====================

This example prints the version of `sherpa-onnx`_ and can be used to verify that the Java API and the native libraries are installed correctly.

It is the Java version of `VersionTest.java <https://github.com/k2-fsa/sherpa-onnx/blob/master/java-api-examples/VersionTest.java>`_ in `sherpa-onnx/java-api-examples <https://github.com/k2-fsa/sherpa-onnx/tree/master/java-api-examples>`_.

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
       mainClass = 'VersionTest'
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

Step 3. Save the code
^^^^^^^^^^^^^^^^^^^^^

Save the following as ``src/main/java/VersionTest.java``:

.. code-block:: java

   // Copyright 2025 Xiaomi Corporation

   import com.k2fsa.sherpa.onnx.*;

   public class VersionTest {
     public static void main(String[] args) {
       System.out.printf("sherpa-onnx version: %s\n", VersionInfo.getVersion());
       System.out.printf("sherpa-onnx gitSha1: %s\n", VersionInfo.getGitSha1());
       System.out.printf("sherpa-onnx gitDate: %s\n", VersionInfo.getGitDate());
       System.out.printf("onnxruntime version: %s\n", VersionInfo.getOnnxruntimeVersion());
     }
   }

Step 4. Check the project layout
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Your project directory should now look like this:

.. code-block:: text

   $ tree .
   .
   ├── build.gradle
   ├── settings.gradle
   └── src
       └── main
           └── java
               └── VersionTest.java

   4 directories, 3 files

   $ ls -lh
   total 16
   -rw-r--r--@ 1 fangjun  wheel   403B 19 Nov 14:00 build.gradle
   -rw-r--r--@ 1 fangjun  wheel    45B 19 Nov 14:00 settings.gradle
   drwxr-xr-x@ 3 fangjun  wheel    96B 19 Nov 14:00 src

Step 5. Build and run
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Generate the Gradle wrapper (only once; requires Gradle to be installed)
   gradle wrapper

   ./gradlew run

Output
^^^^^^

.. code-block:: text

   [Incubating] Problems report is available at: file:///private/tmp/java-doc-work/layouts-java/gradle-version-test/build/reports/problems/problems-report.html

   You can use '--warning-mode all' to show the individual deprecation warnings and determine if they come from your own scripts or plugins.

   For more on this, please refer to https://docs.gradle.org/9.6.1/userguide/command_line_interface.html#sec:command_line_warnings in the Gradle documentation.

   1 actionable task: 1 executed
   Consider enabling configuration cache to speed up this build: https://docs.gradle.org/9.6.1/userguide/configuration_cache_enabling.html

   sherpa-onnx version: 2.0.1
   sherpa-onnx gitSha1: af2d8d81
   sherpa-onnx gitDate: Mon Oct 26 14:57:12 2026
   onnxruntime version: 1.28.2

   [Incubating] Problems report is available at: file:///private/tmp/java-doc-work/layouts-java/gradle-version-test/build/reports/problems/problems-report.html

   You can use '--warning-mode all' to show the individual deprecation warnings and determine if they come from your own scripts or plugins.

   For more on this, please refer to https://docs.gradle.org/9.6.1/userguide/command_line_interface.html#sec:command_line_warnings in the Gradle documentation.

   2 actionable tasks: 2 executed
   Consider enabling configuration cache to speed up this build: https://docs.gradle.org/9.6.1/userguide/configuration_cache_enabling.html

