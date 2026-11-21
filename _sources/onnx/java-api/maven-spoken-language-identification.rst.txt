.. _java-maven-spoken-language-identification:

Maven: Spoken language identification
=====================================

This example shows how to use a whisper model to identify the spoken language of a wave file. Note that it needs a multilingual whisper model, e.g., ``tiny`` works while ``tiny.en`` does not.

It is the Java version of `SpokenLanguageIdentificationWhisper.java <https://github.com/k2-fsa/sherpa-onnx/blob/master/java-api-examples/SpokenLanguageIdentificationWhisper.java>`_ in `sherpa-onnx/java-api-examples <https://github.com/k2-fsa/sherpa-onnx/tree/master/java-api-examples>`_.

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
                               <mainClass>SpokenLanguageIdentificationWhisper</mainClass>
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

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.tar.bz2
   tar xf sherpa-onnx-whisper-tiny.tar.bz2
   rm sherpa-onnx-whisper-tiny.tar.bz2

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/spoken-language-identification-test-wavs.tar.bz2
   tar xf spoken-language-identification-test-wavs.tar.bz2
   rm spoken-language-identification-test-wavs.tar.bz2

Step 4. Save the code
^^^^^^^^^^^^^^^^^^^^^

Save the following as ``src/main/java/SpokenLanguageIdentificationWhisper.java``:

.. code-block:: java

   // Copyright 2024 Xiaomi Corporation

   // This file shows how to use a multilingual whisper model for
   // spoken language identification.
   //
   // Note that it needs a multilingual whisper model. For instance,
   // tiny works, but tiny.en doesn't.
   import com.k2fsa.sherpa.onnx.*;

   public class SpokenLanguageIdentificationWhisper {
     public static void main(String[] args) {
       // please download model and test files from
       // https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
       String encoder = "./sherpa-onnx-whisper-tiny/tiny-encoder.int8.onnx";
       String decoder = "./sherpa-onnx-whisper-tiny/tiny-decoder.int8.onnx";

       String[] testFiles =
           new String[] {
             "./spoken-language-identification-test-wavs/en-english.wav",
             "./spoken-language-identification-test-wavs/de-german.wav",
             "./spoken-language-identification-test-wavs/zh-chinese.wav",
             "./spoken-language-identification-test-wavs/es-spanish.wav",
             "./spoken-language-identification-test-wavs/fa-persian.wav",
             "./spoken-language-identification-test-wavs/ko-korean.wav",
             "./spoken-language-identification-test-wavs/ja-japanese.wav",
             "./spoken-language-identification-test-wavs/ru-russian.wav",
             "./spoken-language-identification-test-wavs/uk-ukrainian.wav",
           };

       SpokenLanguageIdentificationWhisperConfig whisper =
           SpokenLanguageIdentificationWhisperConfig.builder()
               .setEncoder(encoder)
               .setDecoder(decoder)
               .build();

       SpokenLanguageIdentificationConfig config =
           SpokenLanguageIdentificationConfig.builder()
               .setWhisper(whisper)
               .setNumThreads(1)
               .setDebug(true)
               .build();

       SpokenLanguageIdentification slid = new SpokenLanguageIdentification(config);
       for (String filename : testFiles) {
         WaveReader reader = new WaveReader(filename);

         OfflineStream stream = slid.createStream();
         stream.acceptWaveform(reader.getSamples(), reader.getSampleRate());

         String lang = slid.compute(stream);
         System.out.println("---");
         System.out.printf("filename: %s\n", filename);
         System.out.printf("lang: %s\n", lang);

         stream.release();
       }
       System.out.println("---");

       slid.release();
     }
   }

Step 5. Check the project layout
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Your project directory should now look like this:

.. code-block:: text

   $ tree .
   .
   ├── pom.xml
   ├── sherpa-onnx-whisper-tiny
   │   ├── test_wavs
   │   │   ├── 0.wav
   │   │   ├── 1.wav
   │   │   ├── 8k.wav
   │   │   └── trans.txt
   │   ├── tiny-decoder.int8.onnx
   │   ├── tiny-decoder.onnx
   │   ├── tiny-encoder.int8.onnx
   │   ├── tiny-encoder.onnx
   │   └── tiny-tokens.txt
   ├── spoken-language-identification-test-wavs
   │   ├── ar-arabic.wav
   │   ├── bg-bulgarian.wav
   │   ├── cs-czech.wav
   │   ├── da-danish.wav
   │   ├── de-german.wav
   │   ├── el-greek.wav
   │   ├── en-english.wav
   │   ├── es-spanish.wav
   │   ├── fa-persian.wav
   │   ├── fi-finnish.wav
   │   ├── fr-french.wav
   │   ├── hi-hindi.wav
   │   ├── hr-croatian.wav
   │   ├── id-indonesian.wav
   │   ├── it-italian.wav
   │   ├── ja-japanese.wav
   │   ├── ko-korean.wav
   │   ├── nl-dutch.wav
   │   ├── no-norwegian.wav
   │   ├── po-polish.wav
   │   ├── pt-portuguese.wav
   │   ├── ro-romanian.wav
   │   ├── ru-russian.wav
   │   ├── sk-slovak.wav
   │   ├── sv-swedish.wav
   │   ├── ta-tamil.wav
   │   ├── tl-tagalog.wav
   │   ├── tr-turkish.wav
   │   ├── uk-ukrainian.wav
   │   └── zh-chinese.wav
   └── src
       └── main
           └── java
               └── SpokenLanguageIdentificationWhisper.java

   7 directories, 41 files

   $ ls -lh
   total 8
   -rw-r--r--@  1 fangjun  wheel   2.1K 19 Nov 14:00 pom.xml
   drwxr-xr-x@  8 fangjun  wheel   256B 13 Jul  2024 sherpa-onnx-whisper-tiny
   drwxr-xr-x@ 32 fangjun  wheel   1.0K  9 May  2024 spoken-language-identification-test-wavs
   drwxr-xr-x@  3 fangjun  wheel    96B 19 Nov 14:00 src

Step 6. Build and run
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   mvn -q package
   java -jar target/sherpa-onnx-java-example-1.0.jar

Output
^^^^^^

.. code-block:: text

   SpokenLanguageIdentificationConfig(whisper=SpokenLanguageIdentificationWhisperConfig(encoder="./sherpa-onnx-whisper-tiny/tiny-encoder.int8.onnx", decoder="./sherpa-onnx-whisper-tiny/tiny-decoder.int8.onnx", tail_paddings=1000), num_threads=1, debug=True, provider="cpu")
   sot_lm=50360
   sot_prev=50361
   transcribe=50359
   blank_id=220
   eot=50257
   n_text_state=384
   sot_index=0
   sot=50258
   sot_sequence=50258,50259,50359
   n_text_head=6
   n_text_layer=4
   onnx.infer=onnxruntime.quant
   n_text_ctx=448
   n_audio_head=6
   n_audio_ctx=1500
   is_multilingual=1
   non_speech_tokens=1,2,7,8,9,10,14,25,26,27,28,29,31,58,59,60,61,62,63,90,91,92,93,359,503,522,542,873,893,902,918,922,931,1350,1853,1982,2460,2627,3246,3253,3268,3536,3846,3961,4183,4667,6585,6647,7273,9061,9383,10428,10929,11938,12033,12331,12562,13793,14157,14635,15265,15618,16553,16604,18362,18956,20075,21675,22520,26130,26161,26435,28279,29464,31650,32302,32470,36865,42863,47425,49870,50254
   all_language_codes=sq,ml,de,hi,uz,zh,nl,he,sk,gl,lv,ar,tr,haw,gu,si,be,bo,tk,sr,mr,pt,mn,sa,tg,pa,sv,as,sl,su,ba,hu,ln,br,es,ca,bg,hy,th,yo,it,da,oc,ro,ta,is,bs,cs,lb,my,sd,fo,ne,ha,te,mi,cy,el,mg,af,yi,kn,ms,so,nn,tt,en,tl,pl,ko,az,hr,am,id,uk,ps,no,eu,kk,km,lo,sn,fr,mk,la,jw,fi,bn,lt,vi,ka,mt,sw,ur,ru,ja,fa,ht,et
   n_vocab=51865
   version=1
   n_audio_state=384
   n_mels=80
   no_speech=50362
   model_type=whisper-tiny
   maintainer=k2-fsa
   n_audio_layer=4
   translate=50358

   ---
   filename: ./spoken-language-identification-test-wavs/en-english.wav
   lang: English
   ---
   filename: ./spoken-language-identification-test-wavs/de-german.wav
   lang: German
   ---
   filename: ./spoken-language-identification-test-wavs/zh-chinese.wav
   lang: Chinese
   ---
   filename: ./spoken-language-identification-test-wavs/es-spanish.wav
   lang: Spanish
      in_sample_rate: 22050
      output_sample_rate: 16000

   ---
   filename: ./spoken-language-identification-test-wavs/fa-persian.wav
   lang: Persian
   ---
   filename: ./spoken-language-identification-test-wavs/ko-korean.wav
   lang: Korean
   ---
   filename: ./spoken-language-identification-test-wavs/ja-japanese.wav
   lang: Japanese
   ---
   filename: ./spoken-language-identification-test-wavs/ru-russian.wav
   lang: Russian
   ---
   filename: ./spoken-language-identification-test-wavs/uk-ukrainian.wav
   lang: Ukrainian
   ---

