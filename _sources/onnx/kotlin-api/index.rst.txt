.. _sherpa-onnx-kotlin-api:

Kotlin API
==========

In this section, we describe how to use the ``Kotlin`` API of `sherpa-onnx`_.

The core part of `sherpa-onnx`_ is written in C++. We have provided
`JNI <https://docs.oracle.com/javase/8/docs/technotes/guides/jni/spec/intro.html>`_
interface for `sherpa-onnx`_ so that you can use it in Kotlin.

Before using the Kotlin API of `sherpa-onnx`_, you have to build the ``JNI`` interface.

.. toctree::
   :maxdepth: 5

   ./build-jni.rst
   ./examples.rst
