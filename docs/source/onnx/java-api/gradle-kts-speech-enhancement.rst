.. _java-gradle-kts-speech-enhancement:

Gradle KTS: Speech enhancement
==============================

This example shows how to use the GTCRN model for speech enhancement (noise suppression). It reads a noisy wave file and saves the enhanced audio to a new wave file.

It is the Java version of `NonStreamingSpeechEnhancementGtcrn.java <https://github.com/k2-fsa/sherpa-onnx/blob/master/java-api-examples/NonStreamingSpeechEnhancementGtcrn.java>`_ in `sherpa-onnx/java-api-examples <https://github.com/k2-fsa/sherpa-onnx/tree/master/java-api-examples>`_.

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
       mainClass.set("NonStreamingSpeechEnhancementGtcrn")
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

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/gtcrn_simple.onnx
   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/inp_16k.wav

Step 4. Save the code
^^^^^^^^^^^^^^^^^^^^^

Save the following as ``src/main/java/NonStreamingSpeechEnhancementGtcrn.java``:

.. code-block:: java

   // Copyright 2025 Xiaomi Corporation

   // This file shows how to use speech enhancement models in sherpa-onnx
   //
   // Download GTCRN models and sample test waves from:
   // https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models

   import com.k2fsa.sherpa.onnx.*;

   public class NonStreamingSpeechEnhancementGtcrn {
     public static void main(String[] args) {
       String model = "./gtcrn_simple.onnx";
       OfflineSpeechDenoiserModelConfig.Builder builder =
           OfflineSpeechDenoiserModelConfig.builder()
               .setNumThreads(1)
               .setDebug(true)
               .setProvider("cpu");

       builder.setGtcrn(OfflineSpeechDenoiserGtcrnModelConfig.builder().setModel(model).build());

       OfflineSpeechDenoiserModelConfig modelConfig = builder.build();
       OfflineSpeechDenoiserConfig config =
           OfflineSpeechDenoiserConfig.builder().setModel(modelConfig).build();

       OfflineSpeechDenoiser speechDenoiser = new OfflineSpeechDenoiser(config);

       String testWaveFilename = "./inp_16k.wav";
       WaveReader reader = new WaveReader(testWaveFilename);

       DenoisedAudio denoised = speechDenoiser.run(reader.getSamples(), reader.getSampleRate());
       String outFilename = "enhanced.wav";
       WaveWriter.write(outFilename, denoised.getSamples(), denoised.getSampleRate());
       System.out.printf("Saved to %s\n", outFilename);

       speechDenoiser.release();
     }
   }

Step 5. Check the project layout
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Your project directory should now look like this:

.. code-block:: text

   $ tree .
   .
   ├── build.gradle.kts
   ├── gtcrn_simple.onnx
   ├── inp_16k.wav
   ├── settings.gradle.kts
   └── src
       └── main
           └── java
               └── NonStreamingSpeechEnhancementGtcrn.java

   4 directories, 5 files

   $ ls -lh
   total 1680
   -rw-r--r--@ 1 fangjun  wheel   427B 19 Nov 14:00 build.gradle.kts
   -rw-r--r--@ 1 fangjun  wheel   523K 19 Nov 14:00 gtcrn_simple.onnx
   -rw-r--r--@ 1 fangjun  wheel   305K 19 Nov 14:00 inp_16k.wav
   -rw-r--r--@ 1 fangjun  wheel    45B 19 Nov 14:00 settings.gradle.kts
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

   1 actionable task: 1 executed
   Consider enabling configuration cache to speed up this build: https://docs.gradle.org/9.6.1/userguide/configuration_cache_enabling.html

   OfflineSpeechDenoiserConfig(model=OfflineSpeechDenoiserModelConfig(gtcrn=OfflineSpeechDenoiserGtcrnModelConfig(model="./gtcrn_simple.onnx"), dpdfnet=OfflineSpeechDenoiserDpdfNetModelConfig(model="", attenuation_limit_db=0), voxlock=OfflineSpeechDenoiserVoxLockModelConfig(enroll_model="", chunk_model="", voice_sample=""), num_threads=1, debug=True, provider="cpu"))
   hop_length=256
   n_fft=512
   conv_cache_shape=2,1,16,16,33
   window_type=hann_sqrt
   tra_cache_shape=2,3,1,1,16
   model_url=https://github.com/Xiaobin-Rong/gtcrn/blob/main/stream/onnx_models/gtcrn_simple.onnx
   comment2=Please see also https://github.com/Xiaobin-Rong/gtcrn
   sample_rate=16000
   window_length=512
   version=1
   maintainer=k2-fsa
   inter_cache_shape=2,1,33,16
   comment=gtcrn_simple
   model_type=gtcrn
   ----------input names----------
   0 mix
   1 conv_cache
   2 tra_cache
   3 inter_cache
   ----------output names----------
   0 enh
   1 conv_cache_out
   2 tra_cache_out
   3 inter_cache_out

   Saved to enhanced.wav

   2 actionable tasks: 2 executed
   Consider enabling configuration cache to speed up this build: https://docs.gradle.org/9.6.1/userguide/configuration_cache_enabling.html

.. note::

   The enhanced audio is saved to ``enhanced.wav``.

