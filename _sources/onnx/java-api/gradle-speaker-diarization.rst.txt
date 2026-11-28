.. _java-gradle-speaker-diarization:

Gradle: Speaker diarization
===========================

This example shows how to perform speaker diarization, i.e., to figure out who spoke when. It needs a speaker segmentation model and a speaker embedding model.

It is the Java version of `OfflineSpeakerDiarizationDemo.java <https://github.com/k2-fsa/sherpa-onnx/blob/master/java-api-examples/OfflineSpeakerDiarizationDemo.java>`_ in `sherpa-onnx/java-api-examples <https://github.com/k2-fsa/sherpa-onnx/tree/master/java-api-examples>`_.

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
       mainClass = 'OfflineSpeakerDiarizationDemo'
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

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
   tar xf sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
   rm sherpa-onnx-pyannote-segmentation-3-0.tar.bz2

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/0-four-speakers-zh.wav

Step 4. Save the code
^^^^^^^^^^^^^^^^^^^^^

Save the following as ``src/main/java/OfflineSpeakerDiarizationDemo.java``:

.. code-block:: java

   // Copyright 2024 Xiaomi Corporation

   // This file shows how to use sherpa-onnx Java API for speaker diarization,
   import com.k2fsa.sherpa.onnx.*;

   public class OfflineSpeakerDiarizationDemo {
     public static void main(String[] args) {
       /* Please use the following commands to download files used in this file
       Step 1: Download a speaker segmentation model

       Please visit https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-segmentation-models
       for a list of available models. The following is an example

         wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
         tar xvf sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
         rm sherpa-onnx-pyannote-segmentation-3-0.tar.bz2

       Step 2: Download a speaker embedding extractor model

       Please visit https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models
       for a list of available models. The following is an example

         wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx

       Step 3. Download test wave files

       Please visit https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-segmentation-models
       for a list of available test wave files. The following is an example

         wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/0-four-speakers-zh.wav

       Step 4. Run it
           */

       String segmentationModel = "./sherpa-onnx-pyannote-segmentation-3-0/model.onnx";
       String embeddingModel = "./3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx";
       String waveFilename = "./0-four-speakers-zh.wav";

       WaveReader reader = new WaveReader(waveFilename);

       OfflineSpeakerSegmentationPyannoteModelConfig pyannote =
           OfflineSpeakerSegmentationPyannoteModelConfig.builder()
               .setModel(segmentationModel)
               .setWindowShiftRatio(0.1f)
               .build();

       OfflineSpeakerSegmentationModelConfig segmentation =
           OfflineSpeakerSegmentationModelConfig.builder()
               .setPyannote(pyannote)
               .setDebug(true)
               .build();

       SpeakerEmbeddingExtractorConfig embedding =
           SpeakerEmbeddingExtractorConfig.builder().setModel(embeddingModel).setDebug(true).build();

       // The test wave file ./0-four-speakers-zh.wav contains four speakers, so
       // we use numClusters=4 here. If you don't know the number of speakers
       // in the test wave file, please set the numClusters to -1 and provide
       // threshold for clustering
       FastClusteringConfig clustering =
           FastClusteringConfig.builder()
               .setNumClusters(4) // set it to -1 if you don't know the actual number
               .setThreshold(0.5f)
               .setComputeConfidence(true) // enable confidence computation
               .build();

       OfflineSpeakerDiarizationConfig config =
           OfflineSpeakerDiarizationConfig.builder()
               .setSegmentation(segmentation)
               .setEmbedding(embedding)
               .setClustering(clustering)
               .setMinDurationOn(0.2f)
               .setMinDurationOff(0.5f)
               .build();

       OfflineSpeakerDiarization sd = new OfflineSpeakerDiarization(config);
       if (sd.getSampleRate() != reader.getSampleRate()) {
         System.out.printf(
             "Expected sample rate: %d, given: %d\n", sd.getSampleRate(), reader.getSampleRate());
         sd.release();
         return;
       }

       // OfflineSpeakerDiarizationSegment[] segments = sd.process(reader.getSamples());
       // without callback is also ok

       // or you can use a callback to show the progress
       OfflineSpeakerDiarizationSegment[] segments =
           sd.processWithCallback(
               reader.getSamples(),
               (int numProcessedChunks, int numTotalChunks, long arg) -> {
                 float progress = 100.0f * numProcessedChunks / numTotalChunks;
                 System.out.printf("Progress: %.2f%%\n", progress);

                 return 0;
               });

       for (OfflineSpeakerDiarizationSegment s : segments) {
         System.out.printf("%.3f -- %.3f speaker_%02d confidence=%.3f\n", s.getStart(), s.getEnd(), s.getSpeaker(), s.getConfidence());
       }

       sd.release();
     }
   }

Step 5. Check the project layout
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Your project directory should now look like this:

.. code-block:: text

   $ tree .
   .
   ├── 0-four-speakers-zh.wav
   ├── 3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx
   ├── build.gradle
   ├── settings.gradle
   ├── sherpa-onnx-pyannote-segmentation-3-0
   │   ├── export-onnx.py
   │   ├── LICENSE
   │   ├── model.int8.onnx
   │   ├── model.onnx
   │   ├── README.md
   │   ├── run.sh
   │   ├── show-onnx.py
   │   ├── speaker-diarization-onnx.py
   │   ├── speaker-diarization-torch.py
   │   ├── vad-onnx.py
   │   └── vad-torch.py
   └── src
       └── main
           └── java
               └── OfflineSpeakerDiarizationDemo.java

   5 directories, 16 files

   $ ls -lh
   total 80912
   -rw-r--r--@  1 fangjun  wheel   1.7M 19 Nov 14:00 0-four-speakers-zh.wav
   -rw-r--r--@  1 fangjun  wheel    38M 19 Nov 14:00 3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx
   -rw-r--r--@  1 fangjun  wheel   421B 19 Nov 14:00 build.gradle
   -rw-r--r--@  1 fangjun  wheel    45B 19 Nov 14:00 settings.gradle
   drwxr-xr-x@ 13 fangjun  wheel   416B  8 Oct  2024 sherpa-onnx-pyannote-segmentation-3-0
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

   [Incubating] Problems report is available at: file:///private/tmp/java-doc-work/layouts-java/gradle-speaker-diarization/build/reports/problems/problems-report.html

   You can use '--warning-mode all' to show the individual deprecation warnings and determine if they come from your own scripts or plugins.

   For more on this, please refer to https://docs.gradle.org/9.6.1/userguide/command_line_interface.html#sec:command_line_warnings in the Gradle documentation.

   1 actionable task: 1 executed
   Consider enabling configuration cache to speed up this build: https://docs.gradle.org/9.6.1/userguide/configuration_cache_enabling.html

   version=1
   model_author=pyannote
   num_classes=7
   maintainer=k2-fsa
   model_type=pyannote-segmentation-3.0
   receptive_field_shift=270
   window_size=160000
   license=https://huggingface.co/pyannote/segmentation-3.0/blob/main/LICENSE
   receptive_field_size=991
   onnx.quant.pre_process=onnxruntime.quant
   sample_rate=16000
   num_speakers=3
   powerset_max_classes=2
   url_1=https://huggingface.co/pyannote/segmentation-3.0

   normalize_samples=1
   output_dim=512
   language=Chinese
   sample_rate=16000
   comment=This model is from damo/speech_eres2net_base_sv_zh-cn_3dspeaker_16k
   url=https://www.modelscope.cn/models/damo/speech_eres2net_base_sv_zh-cn_3dspeaker_16k/summary
   framework=3d-speaker

   normalize_samples=1
   output_dim=512
   language=Chinese
   sample_rate=16000
   comment=This model is from damo/speech_eres2net_base_sv_zh-cn_3dspeaker_16k
   url=https://www.modelscope.cn/models/damo/speech_eres2net_base_sv_zh-cn_3dspeaker_16k/summary
   framework=3d-speaker

   Progress: 1.30%
   Progress: 2.60%
   Progress: 3.90%
   Progress: 5.19%
   Progress: 6.49%
   Progress: 7.79%
   Progress: 9.09%
   Progress: 10.39%
   Progress: 11.69%
   Progress: 12.99%
   Progress: 14.29%
   Progress: 15.58%
   Progress: 16.88%
   Progress: 18.18%
   Progress: 19.48%
   Progress: 20.78%
   Progress: 22.08%
   Progress: 23.38%
   Progress: 24.68%
   Progress: 25.97%
   Progress: 27.27%
   Progress: 28.57%
   Progress: 29.87%
   Progress: 31.17%
   Progress: 32.47%
   Progress: 33.77%
   Progress: 35.06%
   Progress: 36.36%
   Progress: 37.66%
   Progress: 38.96%
   Progress: 40.26%
   Progress: 41.56%
   Progress: 42.86%
   Progress: 44.16%
   Progress: 45.45%
   Progress: 46.75%
   Progress: 48.05%
   Progress: 49.35%
   Progress: 50.65%
   Progress: 51.95%
   Progress: 53.25%
   Progress: 54.55%
   Progress: 55.84%
   Progress: 57.14%
   Progress: 58.44%
   Progress: 59.74%
   Progress: 61.04%
   Progress: 62.34%
   Progress: 63.64%
   Progress: 64.94%
   Progress: 66.23%
   Progress: 67.53%
   Progress: 68.83%
   Progress: 70.13%
   Progress: 71.43%
   Progress: 72.73%
   Progress: 74.03%
   Progress: 75.32%
   Progress: 76.62%
   Progress: 77.92%
   Progress: 79.22%
   Progress: 80.52%
   Progress: 81.82%
   Progress: 83.12%
   Progress: 84.42%
   Progress: 85.71%
   Progress: 87.01%
   Progress: 88.31%
   Progress: 89.61%
   Progress: 90.91%
   Progress: 92.21%
   Progress: 93.51%
   Progress: 94.81%
   Progress: 96.10%
   Progress: 97.40%
   Progress: 98.70%
   Progress: 100.00%
   0.318 -- 6.865 speaker_00 confidence=0.621
   7.017 -- 10.747 speaker_01 confidence=0.826
   11.455 -- 13.632 speaker_01 confidence=0.756
   13.750 -- 17.041 speaker_02 confidence=0.507
   22.137 -- 24.837 speaker_00 confidence=0.672
   27.638 -- 29.478 speaker_03 confidence=0.628
   30.001 -- 31.553 speaker_03 confidence=0.694
   33.680 -- 37.932 speaker_03 confidence=0.669
   48.040 -- 50.470 speaker_02 confidence=0.631
   52.529 -- 54.605 speaker_00 confidence=0.570

   [Incubating] Problems report is available at: file:///private/tmp/java-doc-work/layouts-java/gradle-speaker-diarization/build/reports/problems/problems-report.html

   You can use '--warning-mode all' to show the individual deprecation warnings and determine if they come from your own scripts or plugins.

   For more on this, please refer to https://docs.gradle.org/9.6.1/userguide/command_line_interface.html#sec:command_line_warnings in the Gradle documentation.

   2 actionable tasks: 2 executed
   Consider enabling configuration cache to speed up this build: https://docs.gradle.org/9.6.1/userguide/configuration_cache_enabling.html

