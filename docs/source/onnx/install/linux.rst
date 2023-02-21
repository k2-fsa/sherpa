.. _install_sherpa_onnx_on_linux:

Linux
=====

This page describes how to build `sherpa-onnx`_ on Linux.

All you need is to run:

.. code-block:: bash

  git clone https://github.com/k2-fsa/sherpa-onnx
  cd sherpa-onnx
  mkdir build
  cd build
  cmake -DCMAKE_BUILD_TYPE=Release ..
  make -j6

After building, you will find an executable ``sherpa-onnx`` inside the ``bin`` directory.

That's it!

Please refer to :ref:`sherpa-onnx-pre-trained-models` for a list of pre-trained
models.
