.. _sherpa-onnx-java-api:

Java API
========

In this section, we describe how to use the ``Java`` API of `sherpa-onnx`_.

The core part of `sherpa-onnx`_ is written in C++. We have provided
`JNI <https://docs.oracle.com/javase/8/docs/technotes/guides/jni/spec/intro.html>`_
interface for `sherpa-onnx`_ so that you can use it in Java.

Before using the Java API of `sherpa-onnx`_, you have to build the ``JNI`` interface.

.. toctree::
   :maxdepth: 5

   ./build-jni-macos.rst
   ./build-jni-linux.rst
   ./build-jni-windows.rst
   ./build-jar.rst
   ./examples.rst
