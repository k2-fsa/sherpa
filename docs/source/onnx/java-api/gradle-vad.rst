.. _java-gradle-vad:

Gradle: VAD
===========

This example shows how to use silero VAD to detect speech segments in a wave file. It removes the silences and saves the remaining speech to a new wave file.

Please see also `TenVadRemoveSilence.java <https://github.com/k2-fsa/sherpa-onnx/blob/master/java-api-examples/TenVadRemoveSilence.java>`_ for the ten-vad model and `VadNonStreamingSenseVoice.java <https://github.com/k2-fsa/sherpa-onnx/blob/master/java-api-examples/VadNonStreamingSenseVoice.java>`_ for an example combining VAD with a non-streaming recognizer.

It is the Java version of `VadRemoveSilence.java <https://github.com/k2-fsa/sherpa-onnx/blob/master/java-api-examples/VadRemoveSilence.java>`_ in `sherpa-onnx/java-api-examples <https://github.com/k2-fsa/sherpa-onnx/tree/master/java-api-examples>`_.

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
       mainClass = 'VadRemoveSilence'
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

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/lei-jun-test.wav

Step 4. Save the code
^^^^^^^^^^^^^^^^^^^^^

Save the following as ``src/main/java/VadRemoveSilence.java``:

.. code-block:: java

   // Copyright 2024 Xiaomi Corporation

   // This file shows how to use a silero_vad model to remove silences from
   // a wave file.

   import com.k2fsa.sherpa.onnx.*;
   import java.util.ArrayList;
   import java.util.Arrays;

   public class VadRemoveSilence {
     public static void main(String[] args) {
       // please download ./silero_vad.onnx from
       // https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
       String model = "./silero_vad.onnx";
       SileroVadModelConfig sileroVad =
           SileroVadModelConfig.builder()
               .setModel(model)
               .setThreshold(0.5f)
               .setMinSilenceDuration(0.25f)
               .setMinSpeechDuration(0.5f)
               .setWindowSize(512)
               .setMaxSpeechDuration(5.0f)
               .build();

       VadModelConfig config =
           VadModelConfig.builder()
               .setSileroVadModelConfig(sileroVad)
               .setSampleRate(16000)
               .setNumThreads(1)
               .setDebug(true)
               .setProvider("cpu")
               .build();

       Vad vad = new Vad(config);

       // You can download the test file from
       // https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
       String testWaveFilename = "./lei-jun-test.wav";
       WaveReader reader = new WaveReader(testWaveFilename);

       int numSamples = reader.getSamples().length;
       int numIter = numSamples / 512;

       ArrayList<float[]> segments = new ArrayList<float[]>();

       for (int i = 0; i != numIter; ++i) {
         int start = i * 512;
         int end = start + 512;
         float[] samples = Arrays.copyOfRange(reader.getSamples(), start, end);
         vad.acceptWaveform(samples);
         if (vad.isSpeechDetected()) {
           while (!vad.empty()) {

             // if you want to get the starting time of this segment, you can use
             /* double startTime = vad.front().getStart() / 16000.0; */

             segments.add(vad.front().getSamples());
             vad.pop();
           }
         }
       }

       vad.flush();
       while (!vad.empty()) {

         // if you want to get the starting time of this segment, you can use
         /* double startTime = vad.front().getStart() / 16000.0; */

         segments.add(vad.front().getSamples());
         vad.pop();
       }

       // get total number of samples
       int n = 0;
       for (float[] s : segments) {
         n += s.length;
       }

       float[] allSamples = new float[n];
       int i = 0;
       for (float[] s : segments) {
         System.arraycopy(s, 0, allSamples, i, s.length);
         i += s.length;
       }

       String outFilename = "lei-jun-test-no-silence.wav";
       WaveWriter.write(outFilename, allSamples, 16000);
       System.out.printf("Saved to %s\n", outFilename);

       vad.release();
     }
   }

Step 5. Check the project layout
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Your project directory should now look like this:

.. code-block:: text

   $ tree .
   .
   ├── build.gradle
   ├── lei-jun-test.wav
   ├── settings.gradle
   ├── silero_vad.onnx
   └── src
       └── main
           └── java
               └── VadRemoveSilence.java

   4 directories, 5 files

   $ ls -lh
   total 20584
   -rw-r--r--@ 1 fangjun  wheel   408B 19 Nov 14:00 build.gradle
   -rw-r--r--@ 1 fangjun  wheel   8.3M 19 Nov 14:00 lei-jun-test.wav
   -rw-r--r--@ 1 fangjun  wheel    45B 19 Nov 14:00 settings.gradle
   -rw-r--r--@ 1 fangjun  wheel   1.7M 19 Nov 14:00 silero_vad.onnx
   drwxr-xr-x@ 3 fangjun  wheel    96B 19 Nov 14:00 src

Step 6. Build and run
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Generate the Gradle wrapper (only once; requires Gradle to be installed)
   gradle wrapper

   ./gradlew run

Output
^^^^^^

.. code-block:: text

   [Incubating] Problems report is available at: file:///private/tmp/java-doc-work/layouts-java/gradle-vad/build/reports/problems/problems-report.html

   You can use '--warning-mode all' to show the individual deprecation warnings and determine if they come from your own scripts or plugins.

   For more on this, please refer to https://docs.gradle.org/9.6.1/userguide/command_line_interface.html#sec:command_line_warnings in the Gradle documentation.

   1 actionable task: 1 executed
   Consider enabling configuration cache to speed up this build: https://docs.gradle.org/9.6.1/userguide/configuration_cache_enabling.html

   VadModelConfig(silero_vad=SileroVadModelConfig(model="./silero_vad.onnx", threshold=0.5, min_silence_duration=0.25, min_speech_duration=0.5, max_speech_duration=5, window_size=512, neg_threshold=-1), ten_vad=TenVadModelConfig(model="", threshold=0.5, min_silence_duration=0.25, min_speech_duration=0.25, max_speech_duration=5, window_size=256), sample_rate=16000, num_threads=1, provider="cpu", debug=True)
   Saved to lei-jun-test-no-silence.wav

   [Incubating] Problems report is available at: file:///private/tmp/java-doc-work/layouts-java/gradle-vad/build/reports/problems/problems-report.html

   You can use '--warning-mode all' to show the individual deprecation warnings and determine if they come from your own scripts or plugins.

   For more on this, please refer to https://docs.gradle.org/9.6.1/userguide/command_line_interface.html#sec:command_line_warnings in the Gradle documentation.

   2 actionable tasks: 2 executed
   Consider enabling configuration cache to speed up this build: https://docs.gradle.org/9.6.1/userguide/configuration_cache_enabling.html

.. note::

   The generated wave file contains only the speech segments of the input.

