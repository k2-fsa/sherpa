.. _java-gradle-diacritics:

Gradle: Diacritics
==================

This example shows how to add diacritics to text. The model used here is the CATT Esperanto model.

It is the Java version of `OfflineAddDiacritics.java <https://github.com/k2-fsa/sherpa-onnx/blob/master/java-api-examples/OfflineAddDiacritics.java>`_ in `sherpa-onnx/java-api-examples <https://github.com/k2-fsa/sherpa-onnx/tree/master/java-api-examples>`_.

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
       mainClass = 'OfflineAddDiacritics'
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

   wget https://github.com/abjadai/catt/releases/download/v2/eo_model_onnx.zip
   unzip eo_model_onnx.zip -d catt_eo_model_onnx
   rm eo_model_onnx.zip

Step 4. Save the code
^^^^^^^^^^^^^^^^^^^^^

Save the following as ``src/main/java/OfflineAddDiacritics.java``:

.. code-block:: java

   // Copyright 2026 Matias Lin

   // This file shows how to use a CATT diacritization model to add diacritics to Arabic text.
   import com.k2fsa.sherpa.onnx.*;

   public class OfflineAddDiacritics {
     public static void main(String[] args) {
       // please download the model from
       // https://github.com/abjadai/catt/releases/download/v2/eo_model_onnx.zip
       String cattEncoder = "./catt_eo_model_onnx/encoder.onnx";
       String cattDecoder = "./catt_eo_model_onnx/decoder.onnx";
       OfflineDiacritizationModelConfig modelConfig =
           OfflineDiacritizationModelConfig.builder()
               .setCattEncoder(cattEncoder)
               .setCattDecoder(cattDecoder)
               .setNumThreads(1)
               .setDebug(true)
               .build();
       OfflineDiacritizationConfig config =
           OfflineDiacritizationConfig.builder().setModel(modelConfig).build();

       OfflineDiacritization diacrt = new OfflineDiacritization(config);

       String[] sentences =
           new String[] {
             "وقالت مجلة نيوزويك الأمريكية التحديث الجديد ل إنستجرام يمكن أن يساهم في إيقاف وكشف الحسابات المزورة بسهولة شديدة",
             "اللغة العربية من أقدم اللغات السامية",
           };

       System.out.println("---");
       for (String text : sentences) {
         String out = diacrt.addDiacritics(text);
         System.out.printf("Input: %s\n", text);
         System.out.printf("Output: %s\n", out);
         System.out.println("---");
       }
     }
   }

Step 5. Check the project layout
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Your project directory should now look like this:

.. code-block:: text

   $ tree .
   .
   ├── build.gradle
   ├── catt_eo_model_onnx
   │   ├── decoder.onnx
   │   └── encoder.onnx
   ├── settings.gradle
   └── src
       └── main
           └── java
               └── OfflineAddDiacritics.java

   5 directories, 5 files

   $ ls -lh
   total 16
   -rw-r--r--@ 1 fangjun  wheel   412B 19 Nov 14:00 build.gradle
   drwxr-xr-x@ 4 fangjun  wheel   128B 19 Nov 10:25 catt_eo_model_onnx
   -rw-r--r--@ 1 fangjun  wheel    45B 19 Nov 14:00 settings.gradle
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

   [Incubating] Problems report is available at: file:///private/tmp/java-doc-work/layouts-java/gradle-diacritics/build/reports/problems/problems-report.html

   You can use '--warning-mode all' to show the individual deprecation warnings and determine if they come from your own scripts or plugins.

   For more on this, please refer to https://docs.gradle.org/9.6.1/userguide/command_line_interface.html#sec:command_line_warnings in the Gradle documentation.

   1 actionable task: 1 executed
   Consider enabling configuration cache to speed up this build: https://docs.gradle.org/9.6.1/userguide/configuration_cache_enabling.html

   OfflineDiacritizationConfig(model=OfflineDiacritizationModelConfig(catt_encoder="./catt_eo_model_onnx/encoder.onnx", catt_decoder="./catt_eo_model_onnx/decoder.onnx", num_threads=1, debug=True, provider="cpu"))
   CATT encoder inputs: src src_mask
   CATT encoder outputs: encoder_output

   CATT decoder inputs: enc_src
   CATT decoder outputs: decoder_output

   ---
   Input: وقالت مجلة نيوزويك الأمريكية التحديث الجديد ل إنستجرام يمكن أن يساهم في إيقاف وكشف الحسابات المزورة بسهولة شديدة
   Output: وَقَالَتْ مَجَلَّةُ نِيُوزْوِيكَ الْأَمْرِيكِيَّةُ التَّحْدِيثَ الْجَدِيدُ لِ إِنسِتِجْرَامٍ يُمْكِنُ أنْ يُسَاهِمَ فِي إِيقَافِ وَكَشْفِ الْحِسَابَاتِ الْمُزَوَّرَةِ بِسُهُولَةٍ شَدِيدَةٍ
   ---
   Input: اللغة العربية من أقدم اللغات السامية
   Output: اللُّغَةُ الْعَرَبِيَّةُ مِنْ أقْدَمِ اللُّغَاتِ السَّامِيَةِ
   ---

   [Incubating] Problems report is available at: file:///private/tmp/java-doc-work/layouts-java/gradle-diacritics/build/reports/problems/problems-report.html

   You can use '--warning-mode all' to show the individual deprecation warnings and determine if they come from your own scripts or plugins.

   For more on this, please refer to https://docs.gradle.org/9.6.1/userguide/command_line_interface.html#sec:command_line_warnings in the Gradle documentation.

   2 actionable tasks: 2 executed
   Consider enabling configuration cache to speed up this build: https://docs.gradle.org/9.6.1/userguide/configuration_cache_enabling.html

