.. _java-gradle-kts-itn:

Gradle KTS: Inverse text normalization
======================================

This example shows how to use inverse text normalization (ITN) together with a paraformer model. The raw ASR output is converted to normalized text, e.g., numbers are written in numeric form.

It is the Java version of `InverseTextNormalizationNonStreamingParaformer.java <https://github.com/k2-fsa/sherpa-onnx/blob/master/java-api-examples/InverseTextNormalizationNonStreamingParaformer.java>`_ in `sherpa-onnx/java-api-examples <https://github.com/k2-fsa/sherpa-onnx/tree/master/java-api-examples>`_.

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
       mainClass.set("InverseTextNormalizationNonStreamingParaformer")
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

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2
   tar xf sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2
   rm sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/itn-zh-number.wav
   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/itn_zh_number.fst

Step 4. Save the code
^^^^^^^^^^^^^^^^^^^^^

Save the following as ``src/main/java/InverseTextNormalizationNonStreamingParaformer.java``:

.. code-block:: java

   // Copyright 2024 Xiaomi Corporation

   // This file shows how to use an offline paraformer, i.e., non-streaming paraformer,
   // to decode files with inverse text normalization.
   import com.k2fsa.sherpa.onnx.*;

   public class InverseTextNormalizationNonStreamingParaformer {
     public static void main(String[] args) {
       // please refer to
       // https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-paraformer/paraformer-models.html#csukuangfj-sherpa-onnx-paraformer-zh-2023-09-14-chinese-english
       // to download model files
       String model = "./sherpa-onnx-paraformer-zh-2023-09-14/model.int8.onnx";
       String tokens = "./sherpa-onnx-paraformer-zh-2023-09-14/tokens.txt";

       // https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/itn-zh-number.wav
       String waveFilename = "./itn-zh-number.wav";

       // https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/itn_zh_number.fst
       String ruleFsts = "./itn_zh_number.fst";

       WaveReader reader = new WaveReader(waveFilename);

       OfflineParaformerModelConfig paraformer =
           OfflineParaformerModelConfig.builder().setModel(model).build();

       OfflineModelConfig modelConfig =
           OfflineModelConfig.builder()
               .setParaformer(paraformer)
               .setTokens(tokens)
               .setNumThreads(1)
               .setDebug(true)
               .build();

       OfflineRecognizerConfig config =
           OfflineRecognizerConfig.builder()
               .setOfflineModelConfig(modelConfig)
               .setDecodingMethod("greedy_search")
               .setRuleFsts(ruleFsts)
               .build();

       OfflineRecognizer recognizer = new OfflineRecognizer(config);
       OfflineStream stream = recognizer.createStream();
       stream.acceptWaveform(reader.getSamples(), reader.getSampleRate());

       recognizer.decode(stream);

       String text = recognizer.getResult(stream).getText();

       System.out.printf("filename:%s\nresult:%s\n", waveFilename, text);

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
   ├── build.gradle.kts
   ├── itn_zh_number.fst
   ├── itn-zh-number.wav
   ├── settings.gradle.kts
   ├── sherpa-onnx-paraformer-zh-2023-09-14
   │   ├── add-model-metadata.py
   │   ├── am.mvn
   │   ├── config.yaml
   │   ├── configuration.json
   │   ├── convert-tokens.py
   │   ├── download-model.py
   │   ├── model.int8.onnx
   │   ├── README.md
   │   ├── test_wavs
   │   │   ├── 0.wav
   │   │   ├── 1.wav
   │   │   ├── 2.wav
   │   │   ├── 3-sichuan.wav
   │   │   ├── 4-tianjin.wav
   │   │   ├── 5-henan.wav
   │   │   ├── 6-zh-en.wav
   │   │   └── 8k.wav
   │   ├── test-paraformer-onnx.py
   │   └── tokens.txt
   └── src
       └── main
           └── java
               └── InverseTextNormalizationNonStreamingParaformer.java

   6 directories, 23 files

   $ ls -lh
   total 640
   -rw-r--r--@  1 fangjun  wheel   439B 19 Nov 14:00 build.gradle.kts
   -rw-r--r--@  1 fangjun  wheel    25K 19 Nov 14:00 itn_zh_number.fst
   -rw-r--r--@  1 fangjun  wheel   281K 19 Nov 14:00 itn-zh-number.wav
   -rw-r--r--@  1 fangjun  wheel    45B 19 Nov 14:00 settings.gradle.kts
   drwxr-xr-x@ 13 fangjun  wheel   416B 10 Mar  2024 sherpa-onnx-paraformer-zh-2023-09-14
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

   1 actionable task: 1 executed
   Consider enabling configuration cache to speed up this build: https://docs.gradle.org/9.6.1/userguide/configuration_cache_enabling.html

   model_type=paraformer
   lfr_window_shift=6
   vocab_size=8404
   lfr_window_size=7
   model_author=damo
   comment=speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx
   onnx.infer=onnxruntime.quant

   filename:./itn-zh-number.wav
   result:中文数字转阿拉伯数字的测试365天

   2 actionable tasks: 2 executed
   Consider enabling configuration cache to speed up this build: https://docs.gradle.org/9.6.1/userguide/configuration_cache_enabling.html

