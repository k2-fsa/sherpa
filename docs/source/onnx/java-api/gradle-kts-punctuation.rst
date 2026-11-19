.. _java-gradle-kts-punctuation:

Gradle KTS: Punctuation
=======================

This example shows how to add punctuation to text, e.g., the output of an ASR system. The model used here is ``sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12``.

It is the Java version of `OfflineAddPunctuation.java <https://github.com/k2-fsa/sherpa-onnx/blob/master/java-api-examples/OfflineAddPunctuation.java>`_ in `sherpa-onnx/java-api-examples <https://github.com/k2-fsa/sherpa-onnx/tree/master/java-api-examples>`_.

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
       mainClass.set("OfflineAddPunctuation")
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

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2
   tar xf sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2
   rm sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2

Step 4. Save the code
^^^^^^^^^^^^^^^^^^^^^

Save the following as ``src/main/java/OfflineAddPunctuation.java``:

.. code-block:: java

   // Copyright 2024 Xiaomi Corporation

   // This file shows how to use a punctuation model to add punctuations to text.
   //
   // The model supports both English and Chinese.
   import com.k2fsa.sherpa.onnx.*;

   public class OfflineAddPunctuation {
     public static void main(String[] args) {
       // please download the model from
       // https://github.com/k2-fsa/sherpa-onnx/releases/tag/punctuation-models
       String model = "./sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12/model.onnx";
       OfflinePunctuationModelConfig modelConfig =
           OfflinePunctuationModelConfig.builder()
               .setCtTransformer(model)
               .setNumThreads(1)
               .setDebug(true)
               .build();
       OfflinePunctuationConfig config =
           OfflinePunctuationConfig.builder().setModel(modelConfig).build();

       OfflinePunctuation punct = new OfflinePunctuation(config);

       String[] sentences =
           new String[] {
             "这是一个测试你好吗How are you我很好thank you are you ok谢谢你",
             "我们都是木头人不会说话不会动",
             "The African blogosphere is rapidly expanding bringing more voices online in the form of"
                 + " commentaries opinions analyses rants and poetry",
           };

       System.out.println("---");
       for (String text : sentences) {
         String out = punct.addPunctuation(text);
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
   ├── build.gradle.kts
   ├── settings.gradle.kts
   ├── sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12
   │   ├── add-model-metadata.py
   │   ├── config.yaml
   │   ├── model.onnx
   │   ├── README.md
   │   ├── show-model-input-output.py
   │   ├── test.py
   │   └── tokens.json
   └── src
       └── main
           └── java
               └── OfflineAddPunctuation.java

   5 directories, 10 files

   $ ls -lh
   total 16
   -rw-r--r--@ 1 fangjun  wheel   414B 19 Nov 14:00 build.gradle.kts
   -rw-r--r--@ 1 fangjun  wheel    45B 19 Nov 14:00 settings.gradle.kts
   drwxr-xr-x@ 9 fangjun  wheel   288B 18 Jun  2025 sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12
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

   OfflinePunctuationConfig(model=OfflinePunctuationModelConfig(ct_transformer="./sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12/model.onnx", num_threads=1, debug=True, provider="cpu"))
   vocab_size: 272727
   num_punctuations: 6
   punctuations: <unk> _ ， 。 ？ 、

   ---
   Input: 这是一个测试你好吗How are you我很好thank you are you ok谢谢你
   Output: 这是一个测试，你好吗？How are you？我很好？thank you，are you ok，谢谢你。
   ---
   Input: 我们都是木头人不会说话不会动
   Output: 我们都是木头人，不会说话不会动。
   ---
   Input: The African blogosphere is rapidly expanding bringing more voices online in the form of commentaries opinions analyses rants and poetry
   Output: The African blogosphere is rapidly expanding，bringing more voices online in the form of commentaries，opinions，analyses，rants and poetry。
   ---

   2 actionable tasks: 2 executed
   Consider enabling configuration cache to speed up this build: https://docs.gradle.org/9.6.1/userguide/configuration_cache_enabling.html

