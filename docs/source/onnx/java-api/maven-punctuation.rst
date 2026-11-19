.. _java-maven-punctuation:

Maven: Punctuation
==================

This example shows how to add punctuation to text, e.g., the output of an ASR system. The model used here is ``sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12``.

It is the Java version of `OfflineAddPunctuation.java <https://github.com/k2-fsa/sherpa-onnx/blob/master/java-api-examples/OfflineAddPunctuation.java>`_ in `sherpa-onnx/java-api-examples <https://github.com/k2-fsa/sherpa-onnx/tree/master/java-api-examples>`_.

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
                               <mainClass>OfflineAddPunctuation</mainClass>
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
   ├── pom.xml
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

   5 directories, 9 files

   $ ls -lh
   total 8
   -rw-r--r--@ 1 fangjun  wheel   2.1K 19 Nov 14:00 pom.xml
   drwxr-xr-x@ 9 fangjun  wheel   288B 18 Jun  2025 sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12
   drwxr-xr-x@ 3 fangjun  wheel    96B 19 Nov 14:00 src

Step 6. Build and run
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   mvn -q package
   java -jar target/sherpa-onnx-java-example-1.0.jar

Output
^^^^^^

.. code-block:: text

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

