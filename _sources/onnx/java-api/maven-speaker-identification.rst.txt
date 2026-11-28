.. _java-maven-speaker-identification:

Maven: Speaker identification
=============================

This example shows how to compute speaker embeddings and how to identify speakers with the embedding manager. Several wave files are enrolled as different speakers and then test files are matched against the enrolled speakers.

It is the Java version of `SpeakerIdentification.java <https://github.com/k2-fsa/sherpa-onnx/blob/master/java-api-examples/SpeakerIdentification.java>`_ in `sherpa-onnx/java-api-examples <https://github.com/k2-fsa/sherpa-onnx/tree/master/java-api-examples>`_.

.. contents:: On this page
   :local:
   :depth: 2

Requirements
------------

- JDK 11 or above
- Maven 3.x

.. warning::

   The sherpa-onnx Java packages are hosted on `JitPack <https://jitpack.io/>`_, so you need to add the JitPack repository.

Step 1. Create the project directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   mkdir sherpa-onnx-java-demo
   cd sherpa-onnx-java-demo

Step 2. Create the build file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Save the following as ``pom.xml``:

.. code-block:: xml

   <?xml version="1.0" encoding="UTF-8"?>
   <project xmlns="http://maven.apache.org/POM/4.0.0"
            xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
            xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
       <modelVersion>4.0.0</modelVersion>

       <groupId>com.example</groupId>
       <artifactId>sherpa-onnx-java-example</artifactId>
       <version>1.0</version>
       <packaging>jar</packaging>

       <properties>
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
               <groupId>com.github.k2-fsa</groupId>
               <artifactId>sherpa-onnx</artifactId>
               <version>${sherpa-onnx.version}</version>
           </dependency>
       </dependencies>

       <build>
           <plugins>
               <plugin>
                   <groupId>org.apache.maven.plugins</groupId>
                   <artifactId>maven-jar-plugin</artifactId>
                   <version>3.3.0</version>
                   <configuration>
                       <archive>
                           <manifest>
                               <mainClass>SpeakerIdentification</mainClass>
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

Step 3. Download the model
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx

   wget https://github.com/csukuangfj/sr-data/archive/refs/tags/v1.0.0.tar.gz
   tar xf v1.0.0.tar.gz
   mv sr-data-1.0.0 sr-data
   rm v1.0.0.tar.gz

Step 4. Save the code
^^^^^^^^^^^^^^^^^^^^^

Save the following as ``src/main/java/SpeakerIdentification.java``:

.. code-block:: java

   // Copyright 2024 Xiaomi Corporation

   // This file shows how to use a speaker embedding extractor model for speaker
   // identification.
   import com.k2fsa.sherpa.onnx.*;

   public class SpeakerIdentification {
     public static float[] computeEmbedding(SpeakerEmbeddingExtractor extractor, String filename) {
       WaveReader reader = new WaveReader(filename);

       OnlineStream stream = extractor.createStream();
       stream.acceptWaveform(reader.getSamples(), reader.getSampleRate());
       stream.inputFinished();

       float[] embedding = extractor.compute(stream);
       stream.release();

       return embedding;
     }

     public static void main(String[] args) {
       // Please download the model from
       // https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models
       String model = "./3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx";
       SpeakerEmbeddingExtractorConfig config =
           SpeakerEmbeddingExtractorConfig.builder()
               .setModel(model)
               .setNumThreads(1)
               .setDebug(true)
               .build();
       SpeakerEmbeddingExtractor extractor = new SpeakerEmbeddingExtractor(config);
       SpeakerEmbeddingManager manager = new SpeakerEmbeddingManager(extractor.getDim());

       try {
         String[] spk1Files =
             new String[] {
               "./sr-data/enroll/fangjun-sr-1.wav",
               "./sr-data/enroll/fangjun-sr-2.wav",
               "./sr-data/enroll/fangjun-sr-3.wav",
             };

         float[][] spk1Vec = new float[spk1Files.length][];

         for (int i = 0; i < spk1Files.length; ++i) {
           spk1Vec[i] = computeEmbedding(extractor, spk1Files[i]);
         }

         String[] spk2Files =
             new String[] {
               "./sr-data/enroll/leijun-sr-1.wav", "./sr-data/enroll/leijun-sr-2.wav",
             };

         float[][] spk2Vec = new float[spk2Files.length][];

         for (int i = 0; i < spk2Files.length; ++i) {
           spk2Vec[i] = computeEmbedding(extractor, spk2Files[i]);
         }

         if (!manager.add("fangjun", spk1Vec)) {
           System.out.println("Failed to register fangjun");
           return;
         }

         if (!manager.add("leijun", spk2Vec)) {
           System.out.println("Failed to register leijun");
           return;
         }

         if (manager.getNumSpeakers() != 2) {
           System.out.println("There should be two speakers");
           return;
         }

         if (!manager.contains("fangjun")) {
           System.out.println("It should contain the speaker fangjun");
           return;
         }

         if (!manager.contains("leijun")) {
           System.out.println("It should contain the speaker leijun");
           return;
         }

         System.out.println("---All speakers---");
         String[] allSpeakers = manager.getAllSpeakerNames();
         for (String s : allSpeakers) {
           System.out.println(s);
         }
         System.out.println("------------");

         String[] testFiles =
             new String[] {
               "./sr-data/test/fangjun-test-sr-1.wav",
               "./sr-data/test/leijun-test-sr-1.wav",
               "./sr-data/test/liudehua-test-sr-1.wav"
             };

         float threshold = 0.6f;
         for (String file : testFiles) {
           float[] embedding = computeEmbedding(extractor, file);

           String name = manager.search(embedding, threshold);
           if (name.isEmpty()) {
             name = "<Unknown>";
           }
           System.out.printf("%s: %s\n", file, name);
         }

         // test verify
         if (!manager.verify("fangjun", computeEmbedding(extractor, testFiles[0]), threshold)) {
           System.out.printf("%s should match fangjun!\n", testFiles[0]);
           return;
         }

         if (!manager.remove("fangjun")) {
           System.out.println("Failed to remove fangjun");
           return;
         }

         if (manager.verify("fangjun", computeEmbedding(extractor, testFiles[0]), threshold)) {
           System.out.printf("%s should match no one!\n", testFiles[0]);
           return;
         }

         if (manager.getNumSpeakers() != 1) {
           System.out.println("There should only 1 speaker left.");
           return;
         }
       } finally {
         extractor.release();
         manager.release();
       }
     }
   }

Step 5. Check the project layout
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Your project directory should now look like this:

.. code-block:: text

   $ tree .
   .
   ├── 3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx
   ├── pom.xml
   ├── sr-data
   │   ├── enroll
   │   │   ├── fangjun-sr-1.wav
   │   │   ├── fangjun-sr-2.wav
   │   │   ├── fangjun-sr-3.wav
   │   │   ├── leijun-sr-1.wav
   │   │   ├── leijun-sr-2.wav
   │   │   ├── liudehua-sr-1.wav
   │   │   └── liudehua-sr-2.wav
   │   ├── LICENSE
   │   ├── README.md
   │   └── test
   │       ├── 3d-speaker
   │       │   ├── README.md
   │       │   ├── speaker1_a_cn_16k.wav
   │       │   ├── speaker1_a_en_16k.wav
   │       │   ├── speaker1_b_cn_16k.wav
   │       │   ├── speaker1_b_en_16k.wav
   │       │   ├── speaker2_a_cn_16k.wav
   │       │   └── speaker2_a_en_16k.wav
   │       ├── fangjun-test-sr-1.wav
   │       ├── fangjun-test-sr-2.wav
   │       ├── leijun-test-sr-1.wav
   │       ├── leijun-test-sr-2.wav
   │       ├── leijun-test-sr-3.wav
   │       ├── liudehua-test-sr-1.wav
   │       └── liudehua-test-sr-2.wav
   └── src
       └── main
           └── java
               └── SpeakerIdentification.java

   8 directories, 26 files

   $ ls -lh
   total 77344
   -rw-r--r--@ 1 fangjun  wheel    38M 19 Nov 14:00 3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx
   -rw-r--r--@ 1 fangjun  wheel   2.1K 19 Nov 14:00 pom.xml
   drwxr-xr-x@ 6 fangjun  wheel   192B 15 Jan  2024 sr-data
   drwxr-xr-x@ 3 fangjun  wheel    96B 19 Nov 14:00 src

Step 6. Build and run
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   mvn -q package
   java -jar target/sherpa-onnx-java-example-1.0.jar

Output
^^^^^^

.. code-block:: text

   SpeakerEmbeddingExtractorConfig(model="./3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx", num_threads=1, debug=True, provider="cpu")
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

   ---All speakers---
   fangjun
   leijun
   ------------
   ./sr-data/test/fangjun-test-sr-1.wav: fangjun
   ./sr-data/test/leijun-test-sr-1.wav: leijun
   ./sr-data/test/liudehua-test-sr-1.wav: <Unknown>

