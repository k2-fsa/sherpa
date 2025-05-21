Build the jar package
=====================

.. note::

   Please see the end of this page for how to download pre-built ``jar``.

.. code-block:: bash

  cd sherpa-onnx/sherpa-onnx/java-api
  ls -lh

You should see the following output::

  (py311) fangjun@ubuntu23-04:/mnt/sdb/shared/sherpa-onnx/sherpa-onnx/java-api$ ls -lh

  total 8.0K
  -rw-rw-r-- 1 fangjun fangjun 2.5K May  8 06:17 Makefile
  drwxrwxr-x 3 fangjun fangjun 4.0K Mar  1 04:29 src

Please run the following command in the directory ``sherpa-onnx/java-api``:

.. code-block:: bash

   make

You should see the following output after running ``make``::

  (py311) fangjun@ubuntu23-04:/mnt/sdb/shared/sherpa-onnx/sherpa-onnx/java-api$ ls -lh
  total 12K
  drwxrwxr-x 3 fangjun fangjun 4.0K May 15 03:59 build
  -rw-rw-r-- 1 fangjun fangjun 2.5K May  8 06:17 Makefile
  drwxrwxr-x 3 fangjun fangjun 4.0K Mar  1 04:29 src
  (py311) fangjun@ubuntu23-04:/mnt/sdb/shared/sherpa-onnx/sherpa-onnx/java-api$ ls -lh build/
  total 60K
  drwxrwxr-x 3 fangjun fangjun 4.0K May 15 03:58 com
  -rw-rw-r-- 1 fangjun fangjun  53K May 15 03:59 sherpa-onnx.jar

Congratulations! You have generated ``sherpa-onnx.jar`` successfully.

.. hint::

   You can find the Java API source files at

    `<https://github.com/k2-fsa/sherpa-onnx/tree/master/sherpa-onnx/java-api/src/com/k2fsa/sherpa/onnx>`_

Download pre-built jar
----------------------

If you don't want to build ``jar`` by yourself, you can download pre-built ``jar`` from
from

    `<https://github.com/k2-fsa/sherpa-onnx/releases>`_

Please always use the latest version. In the following, we describe how to download
the version `1.11.1 <https://github.com/k2-fsa/sherpa-onnx/releases/tag/v1.11.1>`_.

.. code-block:: bash

  # For java 8 or java 1.8
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.11.1/sherpa-onnx-v1.11.1-java8.jar

  # For Java 11
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.11.1/sherpa-onnx-v1.11.1-java11.jar

  # For Java 16
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.11.1/sherpa-onnx-v1.11.1-java16.jar

  # For Java 17
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.11.1/sherpa-onnx-v1.11.1-java17.jar

  # For Java 18
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.11.1/sherpa-onnx-v1.11.1-java18.jar

  # For Java 19
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.11.1/sherpa-onnx-v1.11.1-java19.jar

  # For Java 20
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.11.1/sherpa-onnx-v1.11.1-java20.jar

  # For Java 21
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.11.1/sherpa-onnx-v1.11.1-java21.jar

  # For Java 22
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.11.1/sherpa-onnx-v1.11.1-java22.jar

  # For Java 23
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.11.1/sherpa-onnx-v1.11.1-java23.jar


