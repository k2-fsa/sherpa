.. _java-maven-audio-tagging:

Maven: Audio tagging
====================

This example shows how to use a zipformer model for audio tagging. It reads a wave file and prints the detected audio events.

It is the Java version of `AudioTaggingZipformerFromFile.java <https://github.com/k2-fsa/sherpa-onnx/blob/master/java-api-examples/AudioTaggingZipformerFromFile.java>`_ in `sherpa-onnx/java-api-examples <https://github.com/k2-fsa/sherpa-onnx/tree/master/java-api-examples>`_.

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
                               <mainClass>AudioTaggingZipformerFromFile</mainClass>
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

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/audio-tagging-models/sherpa-onnx-zipformer-small-audio-tagging-2024-04-15.tar.bz2
   tar xf sherpa-onnx-zipformer-small-audio-tagging-2024-04-15.tar.bz2
   rm sherpa-onnx-zipformer-small-audio-tagging-2024-04-15.tar.bz2

Step 4. Save the code
^^^^^^^^^^^^^^^^^^^^^

Save the following as ``src/main/java/AudioTaggingZipformerFromFile.java``:

.. code-block:: java

   // Copyright 2024 Xiaomi Corporation

   // This file shows how to use a zipformer audio tagging model to tag
   // input audio files.
   import com.k2fsa.sherpa.onnx.*;

   public class AudioTaggingZipformerFromFile {
     public static void main(String[] args) {
       // please download the model from
       // https://github.com/k2-fsa/sherpa-onnx/releases/tag/audio-tagging-models
       String model = "./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/model.int8.onnx";
       String labels =
           "./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/class_labels_indices.csv";
       int topK = 5;

       String[] testWaves =
           new String[] {
             "./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/test_wavs/1.wav",
             "./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/test_wavs/2.wav",
             "./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/test_wavs/3.wav",
             "./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/test_wavs/4.wav",
             "./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/test_wavs/5.wav",
             "./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/test_wavs/6.wav",
             "./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/test_wavs/7.wav",
             "./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/test_wavs/8.wav",
             "./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/test_wavs/9.wav",
             "./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/test_wavs/10.wav",
             "./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/test_wavs/11.wav",
             "./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/test_wavs/12.wav",
             "./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/test_wavs/13.wav",
           };

       OfflineZipformerAudioTaggingModelConfig zipformer =
           OfflineZipformerAudioTaggingModelConfig.builder().setModel(model).build();

       AudioTaggingModelConfig modelConfig =
           AudioTaggingModelConfig.builder()
               .setZipformer(zipformer)
               .setNumThreads(1)
               .setDebug(true)
               .build();

       AudioTaggingConfig config =
           AudioTaggingConfig.builder().setModel(modelConfig).setLabels(labels).setTopK(topK).build();

       AudioTagging tagger = new AudioTagging(config);
       System.out.println("------");
       for (String filename : testWaves) {
         WaveReader reader = new WaveReader(filename);

         OfflineStream stream = tagger.createStream();
         stream.acceptWaveform(reader.getSamples(), reader.getSampleRate());

         AudioEvent[] events = tagger.compute(stream);

         stream.release();

         System.out.printf("input file: %s\n", filename);
         System.out.printf("Probability\t\tName\n");
         for (AudioEvent e : events) {
           System.out.printf("%.3f\t\t\t%s\n", e.getProb(), e.getName());
         }
         System.out.println("------");
       }

       tagger.release();
     }
   }

Step 5. Check the project layout
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Your project directory should now look like this:

.. code-block:: text

   $ tree .
   .
   ├── pom.xml
   ├── sherpa-onnx-zipformer-small-audio-tagging-2024-04-15
   │   ├── class_labels_indices.csv
   │   ├── model.int8.onnx
   │   ├── model.onnx
   │   ├── README.md
   │   └── test_wavs
   │       ├── 1.wav
   │       ├── 10.wav
   │       ├── 11.wav
   │       ├── 12.wav
   │       ├── 13.wav
   │       ├── 2.wav
   │       ├── 3.wav
   │       ├── 4.wav
   │       ├── 5.wav
   │       ├── 6.wav
   │       ├── 7.wav
   │       ├── 8.wav
   │       └── 9.wav
   └── src
       └── main
           └── java
               └── AudioTaggingZipformerFromFile.java

   6 directories, 19 files

   $ ls -lh
   total 8
   -rw-r--r--@ 1 fangjun  wheel   2.1K 19 Nov 14:00 pom.xml
   drwxr-xr-x@ 7 fangjun  wheel   224B 15 Apr  2024 sherpa-onnx-zipformer-small-audio-tagging-2024-04-15
   drwxr-xr-x@ 3 fangjun  wheel    96B 19 Nov 14:00 src

Step 6. Build and run
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   mvn -q package
   java -jar target/sherpa-onnx-java-example-1.0.jar

Output
^^^^^^

.. code-block:: text

   AudioTaggingConfig(model=AudioTaggingModelConfig(zipformer=OfflineZipformerAudioTaggingModelConfig(model="./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/model.int8.onnx"), ced="", num_threads=1, debug=True, provider="cpu"), labels="./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/class_labels_indices.csv", top_k=5)
   comment=zipformer2 audio tagger
   model_author=k2-fsa
   version=1
   model_type=zipformer2
   onnx.infer=onnxruntime.quant

   ------
   input file: ./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/test_wavs/1.wav
   Probability		Name
   0.942			Animal
   0.925			Cat
   0.922			Domestic animals, pets
   0.152			Meow
   0.140			Caterwaul
   ------
   input file: ./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/test_wavs/2.wav
   Probability		Name
   0.896			Whistling
   0.249			Music
   0.232			Piano
   0.112			Keyboard (musical)
   0.052			Musical instrument
   ------
   input file: ./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/test_wavs/3.wav
   Probability		Name
   0.790			Music
   0.756			A capella
   0.464			Singing
   0.440			Vocal music
   0.436			Choir
   ------
   input file: ./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/test_wavs/4.wav
   Probability		Name
   0.907			Laughter
   0.256			Snicker
   0.118			Giggle
   0.107			Inside, small room
   0.078			Speech
   ------
   input file: ./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/test_wavs/5.wav
   Probability		Name
   0.640			Finger snapping
   0.425			Slap, smack
   0.149			Clapping
   0.092			Sound effect
   0.028			Whack, thwack
   ------
   input file: ./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/test_wavs/6.wav
   Probability		Name
   0.879			Baby cry, infant cry
   0.586			Crying, sobbing
   0.134			Whimper
   0.029			Inside, small room
   0.009			Sound effect
   ------
   input file: ./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/test_wavs/7.wav
   Probability		Name
   0.800			Smoke detector, smoke alarm
   0.689			Beep, bleep
   0.253			Buzzer
   0.147			Fire alarm
   0.013			Alarm
   ------
   input file: ./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/test_wavs/8.wav
   Probability		Name
   0.896			Siren
   0.765			Civil defense siren
   0.011			Vehicle
   0.006			Inside, small room
   0.003			Outside, urban or manmade
   ------
   input file: ./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/test_wavs/9.wav
   Probability		Name
   0.976			Siren
   0.808			Civil defense siren
   0.006			Sound effect
   0.006			Police car (siren)
   0.004			Vehicle
   ------
   input file: ./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/test_wavs/10.wav
   Probability		Name
   0.359			Stream
   0.308			Gurgling
   0.308			Water
   0.192			Trickle, dribble
   0.099			Liquid
   ------
   input file: ./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/test_wavs/11.wav
   Probability		Name
   0.822			Meow
   0.717			Cat
   0.566			Domestic animals, pets
   0.544			Animal
   0.105			Music
   ------
   input file: ./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/test_wavs/12.wav
   Probability		Name
   0.658			Animal
   0.584			Dog
   0.552			Bark
   0.471			Bow-wow
   0.446			Domestic animals, pets
   ------
   input file: ./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/test_wavs/13.wav
   Probability		Name
   0.848			Oink
   0.170			Animal
   0.162			Pig
   0.032			Speech
   0.017			Snort
   ------

