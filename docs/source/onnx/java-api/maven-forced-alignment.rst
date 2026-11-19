.. _java-maven-forced-alignment:

Maven: Forced alignment
=======================

This example shows how to compute forced alignments for a wave file given its transcript. It writes a Praat TextGrid file that you can open with `Praat <https://www.fon.hum.uva.nl/praat/>`_ to visualize the phone and word boundaries.

The model used here is ``english-mfa-v3.1.0``. Please see `ForcedAlignmentChinese.java <https://github.com/k2-fsa/sherpa-onnx/blob/master/java-api-examples/ForcedAlignmentChinese.java>`_ for a Chinese example.

It is the Java version of `ForcedAlignmentEnglish.java <https://github.com/k2-fsa/sherpa-onnx/blob/master/java-api-examples/ForcedAlignmentEnglish.java>`_ in `sherpa-onnx/java-api-examples <https://github.com/k2-fsa/sherpa-onnx/tree/master/java-api-examples>`_.

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
           <sherpa-onnx.version>v2.0.2</sherpa-onnx.version>
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
                               <mainClass>ForcedAlignmentEnglish</mainClass>
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

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/english-mfa-v3.1.0.tar.bz2
   tar xf english-mfa-v3.1.0.tar.bz2
   rm english-mfa-v3.1.0.tar.bz2

Step 4. Save the code
^^^^^^^^^^^^^^^^^^^^^

Save the following as ``src/main/java/ForcedAlignmentEnglish.java``:

.. code-block:: java

   // Copyright (c)  2026  Xiaomi Corporation

   /*
    * Forced alignment of English speech with the sherpa-onnx Java API.
    *
    * It aligns the test wav of the english-mfa-v3.1.0 model package with its
    * transcript and writes a Praat TextGrid. The model directory, its
    * pronunciation lexicon and the test wav are hard-coded below.
    *
    * How to download the model (the package contains the model files, the
    * lexicon and the test wav used here):
    *
    *   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/english-mfa-v3.1.0.tar.bz2
    *   tar xf english-mfa-v3.1.0.tar.bz2
    *
    * How to run:
    *
    *   ./run-forced-alignment-english.sh
    *
    * Output:
    *
    *   en.TextGrid in ForcedAlignmentEnglish/. Open it with Praat
    *   (https://www.fon.hum.uva.nl/praat/) to visualize the phone and word
    *   boundaries.
    *
    * See also ForcedAlignmentChinese.java and the Python examples
    * python-api-examples/forced-alignment-english.py.
    */

   import com.k2fsa.sherpa.onnx.AlignmentInterval;
   import com.k2fsa.sherpa.onnx.AlignmentTier;
   import com.k2fsa.sherpa.onnx.ForcedAligner;
   import com.k2fsa.sherpa.onnx.ForcedAlignerConfig;
   import com.k2fsa.sherpa.onnx.ForcedAlignmentResult;
   import com.k2fsa.sherpa.onnx.OfflineStream;
   import com.k2fsa.sherpa.onnx.WaveReader;

   import java.io.IOException;
   import java.nio.file.Files;
   import java.nio.file.Path;

   public class ForcedAlignmentEnglish {
       public static void main(String[] args) throws IOException {
           String modelDir = "./english-mfa-v3.1.0";
           String wav = modelDir + "/en.wav";
           String text =
                   "ask not what your country can do for you, "
                           + "ask what you can do for your country.";

           ForcedAlignerConfig config = new ForcedAlignerConfig();
           config.modelDir = modelDir;
           config.lexicon = modelDir + "/lexicon.txt";

           long beginInit = System.nanoTime();
           ForcedAligner aligner = new ForcedAligner(config);
           OfflineStream stream = aligner.createStream();
           long endInit = System.nanoTime();

           WaveReader reader = new WaveReader(wav);
           // samples: float32 normalized to the range [-1, 1]
           stream.acceptWaveform(reader.getSamples(), reader.getSampleRate());

           long begin = System.nanoTime();
           ForcedAlignmentResult result = aligner.compute(stream, text);
           long end = System.nanoTime();

           for (AlignmentTier tier : result.getTiers()) {
               System.out.println("--- tier: " + tier.getName());
               for (AlignmentInterval interval : tier.getIntervals()) {
                   System.out.printf("%.2f - %.2f  %s%n", interval.getStart(),
                           interval.getEnd(), interval.getText());
               }
           }

           Files.writeString(Path.of("en.TextGrid"), result.toTextGrid());
           System.out.println("wrote en.TextGrid");

           // time statistics, like sherpa-onnx-fa prints
           double initSeconds = (endInit - beginInit) / 1e9;
           double elapsedSeconds = (end - begin) / 1e9;
           double audioDuration =
                   reader.getSamples().length / (double) reader.getSampleRate();
           System.out.printf("init is done in %.3f s%n", initSeconds);
           System.out.printf("Elapsed seconds: %.3f s%n", elapsedSeconds);
           System.out.printf(
                   "Real time factor (RTF): %.3f / %.3f = %.3f%n",
                   elapsedSeconds, audioDuration, elapsedSeconds / audioDuration);

           aligner.release();
       }
   }

Step 5. Check the project layout
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Your project directory should now look like this:

.. code-block:: text

   $ tree .
   .
   ├── english-mfa-v3.1.0
   │   ├── en.wav
   │   ├── final.alimdl
   │   ├── lda.mat
   │   ├── lexicon.txt
   │   ├── meta.json
   │   ├── phones.txt
   │   ├── README.md
   │   └── tree
   ├── pom.xml
   └── src
       └── main
           └── java
               └── ForcedAlignmentEnglish.java

   5 directories, 10 files

   $ ls -lh
   total 8
   drwxr-xr-x@ 10 fangjun  wheel   320B 13 Nov 11:16 english-mfa-v3.1.0
   -rw-r--r--@  1 fangjun  wheel   2.1K 19 Nov 14:00 pom.xml
   drwxr-xr-x@  3 fangjun  wheel    96B 19 Nov 14:00 src

Step 6. Build and run
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   mvn -q package
   java -jar target/sherpa-onnx-java-example-1.0.jar

Output
^^^^^^

.. code-block:: text

      in_sample_rate: 24000
      output_sample_rate: 16000

   --- tier: phones
   0.00 - 0.11  sil
   0.11 - 0.25  a
   0.25 - 0.28  k
   0.28 - 0.39  s
   0.39 - 0.44  sil
   0.44 - 0.51  n
   0.51 - 0.59  a
   0.59 - 0.63  t
   0.63 - 0.66  ʋ
   0.66 - 0.70  ɒ
   0.70 - 0.73  ʈ
   0.73 - 0.78  j
   0.78 - 0.81  ɒ
   0.81 - 0.84  ɹ
   0.84 - 0.96  k
   0.96 - 1.01  ə
   1.01 - 1.05  n
   1.05 - 1.13  ʈ
   1.13 - 1.17  ɹ
   1.17 - 1.22  i
   1.22 - 1.31  cʰ
   1.31 - 1.35  æ
   1.35 - 1.41  n
   1.41 - 1.48  d
   1.48 - 1.56  əw
   1.56 - 1.66  f
   1.66 - 1.69  a
   1.69 - 1.81  j
   1.81 - 1.87  ə
   1.87 - 2.05  sil
   2.05 - 2.25  a
   2.25 - 2.28  k
   2.28 - 2.37  s
   2.37 - 2.46  ʋ
   2.46 - 2.49  ɒ
   2.49 - 2.53  ʈ
   2.53 - 2.63  j
   2.63 - 2.64  ə
   2.64 - 2.72  cʰ
   2.72 - 2.76  æ
   2.76 - 2.83  n
   2.83 - 2.89  d
   2.89 - 2.97  əw
   2.97 - 3.07  f
   3.07 - 3.10  a
   3.10 - 3.14  j
   3.14 - 3.17  ɒ
   3.17 - 3.21  ɹ
   3.21 - 3.32  k
   3.32 - 3.38  ə
   3.38 - 3.45  n
   3.45 - 3.55  ʈ
   3.55 - 3.60  ɹ
   3.60 - 3.79  i
   3.79 - 3.85  sil
   --- tier: words
   0.00 - 0.11
   0.11 - 0.39  ask
   0.39 - 0.44
   0.44 - 0.63  not
   0.63 - 0.73  what
   0.73 - 0.84  your
   0.84 - 1.22  country
   1.22 - 1.41  can
   1.41 - 1.56  do
   1.56 - 1.69  for
   1.69 - 1.87  you
   1.87 - 2.05
   2.05 - 2.37  ask
   2.37 - 2.53  what
   2.53 - 2.64  you
   2.64 - 2.83  can
   2.83 - 2.97  do
   2.97 - 3.10  for
   3.10 - 3.21  your
   3.21 - 3.79  country
   3.79 - 3.85
   wrote en.TextGrid
   init is done in 0.249 s
   Elapsed seconds: 0.007 s
   Real time factor (RTF): 0.007 / 3.845 = 0.002

.. note::

   The generated TextGrid file is written to ``en.TextGrid``.

   The forced alignment API is available since v2.0.2.

