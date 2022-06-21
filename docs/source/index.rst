.. sherpa documentation master file, created by
   sphinx-quickstart on Sun Jun  5 08:31:41 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

sherpa
======

`sherpa`_ is a framework for streaming and non-streaming
automatic speech recognition (ASR).

CPU-bound tasks, such as neural network computation, are implemented in
C++; while IO-bound tasks, such as socket communication, are implemented
in Python with `asyncio`_.

Python is responsible for managing threads, which call into C++ extensions
with the `global interpreter lock (GIL) <https://docs.python.org/3/glossary.html#term-global-interpreter-lock>`_
released so that multiple threads can run concurrently.

The following content describes how to install `sherpa`_ and its usage
for both streaming ASR and offline ASR (i.e., non-streaming ASR).

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   ./installation/index
   ./streaming_asr/index
   ./offline_asr/index
   faq
