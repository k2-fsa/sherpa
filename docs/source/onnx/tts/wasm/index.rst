WebAssembly
===========

In this section, we describe how to build text-to-speech from `sherpa-onnx`_ for `WebAssembly`_
so that you can run text-to-speech with `WebAssembly`_.

Please follow the steps below to build and run `sherpa-onnx`_ for `WebAssembly`_.

.. hint::

   We provide a colab notebook
   |build sherpa-onnx WebAssembly TTS colab|
   for you to try this section step by step.

   If you are using Windows or you don't want to setup your local environment
   to build WebAssembly support, please use the above colab notebook.

.. |build sherpa-onnx WebAssembly TTS colab| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://github.com/k2-fsa/colab/blob/master/sherpa-onnx/sherpa_onnx_wasm_tts.ipynb

.. toctree::
   :maxdepth: 3

   ./install-emscripten.rst
   ./build.rst
   ./hf-spaces.rst
