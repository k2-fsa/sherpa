Non-android Java
====================

We provide plenty of examples about using non-Android Java API of `sherpa-onnx`_
in `<https://github.com/k2-fsa/sherpa-onnx/tree/master/java-api-examples>`_.

In this section, we describe how to run the examples in the following platforms:

  - Linux (x64)
  - Linux (arm64)
  - macOS (x64)
  - macOS (arm64)
  - Windows (x64)

Download jar files
------------------

You need to download two ``jar`` files.

The first ``jar`` file is shared by all platforms built from pure Java source code.
You can download it from our GitHub release page at

  `<https://github.com/k2-fsa/sherpa-onnx/releases>`_

We recommend always using the latest version. For instance, to download the version
``v1.12.10``, visit `<https://github.com/k2-fsa/sherpa-onnx/releases/tag/v1.12.10>`_,
and run::

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.12.10/sherpa-onnx-v1.12.10.jar

The second ``jar`` file contains shared libraries for different platforms built from
C++ code. The following table lists the download links for version ``v1.12.10``.

.. list-table::

 * - Platform
   - URL
 * - Linux x64
   - `sherpa-onnx-native-lib-linux-x64-v1.12.10.jar <https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.12.10/sherpa-onnx-native-lib-linux-x64-v1.12.10.jar>`_
 * - Linux arm64
   - `sherpa-onnx-native-lib-linux-aarch64-v1.12.10.jar <https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.12.10/sherpa-onnx-native-lib-linux-aarch64-v1.12.10.jar>`_
 * - macOS x64
   - `sherpa-onnx-native-lib-osx-x64-v1.12.10.jar <https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.12.10/sherpa-onnx-native-lib-osx-x64-v1.12.10.jar>`_
 * - macOS arm64
   - `sherpa-onnx-native-lib-osx-aarch64-v1.12.10.jar <https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.12.10/sherpa-onnx-native-lib-osx-aarch64-v1.12.10.jar>`_
 * - Windows x64
   - `sherpa-onnx-native-lib-win-x64-v1.12.10.jar <https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.12.10/sherpa-onnx-native-lib-win-x64-v1.12.10.jar>`_

Usage
-----

Linux x64
:::::::::

.. code-block::

   java -cp "./sherpa-onnx-v1.12.10.jar:./sherpa-onnx-native-lib-linux-x64-v1.12.10.jar"  SomeExample.java


Linux arm64
:::::::::::

.. code-block::

   java -cp "./sherpa-onnx-v1.12.10.jar:./sherpa-onnx-native-lib-linux-aarch64-v1.12.10.jar"  SomeExample.java

macOS x64
:::::::::

.. code-block::

   java -cp "./sherpa-onnx-v1.12.10.jar:./sherpa-onnx-native-lib-osx-x64-v1.12.10.jar"  SomeExample.java


macOS arm64
:::::::::::

.. code-block::

   java -cp "./sherpa-onnx-v1.12.10.jar:./sherpa-onnx-native-lib-osx-aarch64-v1.12.10.jar"  SomeExample.java

Windows x64
:::::::::::

.. code-block::

   java -cp "./sherpa-onnx-v1.12.10.jar;./sherpa-onnx-native-lib-win-x64-v1.12.10.jar"  SomeExample.java

.. caution::

   It uses ``;`` to separate the two ``jar`` files for Windows.

.. caution::

   It uses ``;`` to separate the two ``jar`` files for Windows.

.. caution::

   It uses ``;`` to separate the two ``jar`` files for Windows.

Colab notebook example
----------------------

We provide a colab notebook to guide you step by step to run ``sherpa-onnx`` with its Java API.
Please see

  `<https://github.com/k2-fsa/colab/blob/master/sherpa-onnx/sherpa_onnx_java_api_linux_example.ipynb>`_
