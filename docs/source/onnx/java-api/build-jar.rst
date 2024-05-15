Build the jar package
=====================

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
