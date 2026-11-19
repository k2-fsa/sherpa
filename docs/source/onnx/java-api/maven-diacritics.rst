.. _java-maven-diacritics:

Maven: Diacritics
=================

This example shows how to add diacritics to text. The model used here is the CATT Esperanto model.

It is the Java version of `OfflineAddDiacritics.java <https://github.com/k2-fsa/sherpa-onnx/blob/master/java-api-examples/OfflineAddDiacritics.java>`_ in `sherpa-onnx/java-api-examples <https://github.com/k2-fsa/sherpa-onnx/tree/master/java-api-examples>`_.

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
                               <mainClass>OfflineAddDiacritics</mainClass>
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
   ├── catt_eo_model_onnx
   │   ├── decoder.onnx
   │   └── encoder.onnx
   ├── pom.xml
   └── src
       └── main
           └── java
               └── OfflineAddDiacritics.java

   5 directories, 4 files

   $ ls -lh
   total 8
   drwxr-xr-x@ 4 fangjun  wheel   128B 19 Nov 10:25 catt_eo_model_onnx
   -rw-r--r--@ 1 fangjun  wheel   2.1K 19 Nov 14:00 pom.xml
   drwxr-xr-x@ 3 fangjun  wheel    96B 19 Nov 14:00 src

Step 6. Build and run
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   mvn -q package
   java -jar target/sherpa-onnx-java-example-1.0.jar

Output
^^^^^^

.. code-block:: text

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

