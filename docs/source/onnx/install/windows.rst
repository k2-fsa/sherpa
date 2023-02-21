Windows
=======

This page describes how to build `sherpa-onnx`_ on Windows.

64-bit Windows (x64)
--------------------

All you need is to run:

.. code-block:: bash

  git clone https://github.com/k2-fsa/sherpa-onnx
  cd sherpa-onnx
  mkdir build
  cd build
  cmake -DCMAKE_BUILD_TYPE=Release ..
  cmake --build . --config Release -- -m:6

After building, you will find an executable ``sherpa-onnx.exe`` inside the ``bin/Release`` directory.

That's it!

Please refer to :ref:`sherpa-onnx-pre-trained-models` for a list of pre-trained
models.
