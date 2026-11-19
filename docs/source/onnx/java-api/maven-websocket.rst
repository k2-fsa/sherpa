.. _java-maven-websocket:

Maven: WebSocket client
=======================

This example is a WebSocket client for the Python server `non_streaming_server.py <https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/non_streaming_server.py>`_. The client sends a wave file to the server and prints the recognition result returned by the server.

It is the Java version of `NonStreamingWebsocketClient.java <https://github.com/k2-fsa/sherpa-onnx/blob/master/java-api-examples/NonStreamingWebsocketClient.java>`_ in `sherpa-onnx/java-api-examples <https://github.com/k2-fsa/sherpa-onnx/tree/master/java-api-examples>`_.

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
                               <mainClass>NonStreamingWebsocketClient</mainClass>
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

   wget https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main/test_wavs/zh.wav

Step 4. Save the code
^^^^^^^^^^^^^^^^^^^^^

Save the following as ``src/main/java/NonStreamingWebsocketClient.java``:

.. code-block:: java

   // Refer to
   // https://stackoverflow.com/questions/55380813/require-assistance-with-simple-pure-java-11-websocket-client-example
   //
   //
   // This is a WebSocketClient client for ../python-api-examples/non_streaming_server.py
   //
   // Please see ./run-non-streaming-websocket-client.sh
   import com.k2fsa.sherpa.onnx.*;
   import java.net.URI;
   import java.net.http.HttpClient;
   import java.net.http.WebSocket;
   import java.nio.*;
   import java.util.concurrent.CompletionStage;
   import java.util.concurrent.CountDownLatch;

   public class NonStreamingWebsocketClient {
     public static void main(String[] args) throws Exception {
       CountDownLatch latch = new CountDownLatch(1);

       WebSocket ws =
           HttpClient.newHttpClient()
               .newWebSocketBuilder()
               .buildAsync(URI.create("ws://localhost:6006"), new WebSocketClient(latch))
               .join();

       // Please use a 16-bit, single channel wav for testing.
       // the sample rate does not need to be 16kHz
       String waveFilename = "./zh.wav";
       WaveReader reader = new WaveReader(waveFilename);
       int sampleRate = reader.getSampleRate();
       int numSamples = reader.getSamples().length;

       // Here is the format of the message
       // byte 0-3 in little endian: sampleRate
       // byte 4-7 in little endian: number of bytes for samples
       // remaining bytes: samples. Each sample is a float32
       ByteBuffer buffer = ByteBuffer.allocate(8 + 4 * numSamples).order(ByteOrder.LITTLE_ENDIAN);
       buffer.putInt(sampleRate);
       buffer.putInt(numSamples * 4); // each sample has 4 bytes

       for (float s : reader.getSamples()) {
         buffer.putFloat(s);
       }

       buffer.rewind();
       buffer.flip();
       buffer.order(ByteOrder.LITTLE_ENDIAN);

       ws.sendBinary(ByteBuffer.wrap(buffer.array()), true).join();

       // Send Done to the server to indicate that we don't have new wave files to decode
       ws.sendText("Done", true).join();

       latch.await();
     }

     private static class WebSocketClient implements WebSocket.Listener {
       private final CountDownLatch latch;

       public WebSocketClient(CountDownLatch latch) {
         this.latch = latch;
       }

       @Override
       public CompletionStage<?> onText(WebSocket webSocket, CharSequence data, boolean last) {
         System.out.println("Result is " + data);
         latch.countDown();
         return WebSocket.Listener.super.onText(webSocket, data, last);
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
   ├── src
   │   └── main
   │       └── java
   │           └── NonStreamingWebsocketClient.java
   └── zh.wav

   4 directories, 3 files

   $ ls -lh
   total 360
   -rw-r--r--@ 1 fangjun  wheel   2.1K 19 Nov 14:00 pom.xml
   drwxr-xr-x@ 3 fangjun  wheel    96B 19 Nov 14:00 src
   -rw-r--r--@ 1 fangjun  wheel   175K 19 Nov 14:00 zh.wav

Step 6. Build and run
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   mvn -q package
   java -jar target/sherpa-onnx-java-example-1.0.jar

Output
^^^^^^

.. code-block:: text

   Result is 开放时间早上九点至下午五点

.. note::

   Please start the server first in a separate terminal. For instance::

      python3 /path/to/sherpa-onnx/python-api-examples/non_streaming_server.py \
        --paraformer=./sherpa-onnx-paraformer-zh-2023-09-14/model.int8.onnx \
        --tokens=./sherpa-onnx-paraformer-zh-2023-09-14/tokens.txt \
        --port=6006 \
        --doc-root=/path/to/sherpa-onnx/python-api-examples/web

   Run it in the project directory, so the model paths above are those downloaded in this project. The client connects to ``ws://localhost:6006``. Please see `non_streaming_server.py <https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/non_streaming_server.py>`_ for more options.

