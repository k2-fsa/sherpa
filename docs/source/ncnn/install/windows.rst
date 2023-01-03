Windows
=======

This page describes how to build `sherpa-ncnn`_ on Windows.

All you need is to run:

.. code-block:: bash

  git clone https://github.com/k2-fsa/sherpa-ncnn
  cd sherpa-ncnn
  mkdir build
  cd build
  cmake -DCMAKE_BUILD_TYPE=Release ..
  cmake --build . --config Release -- -m:6

It will generate two executables inside ``./bin/Release/``:

  - ``sherpa-ncnn.exe``: For decoding a single wave file.
  - ``sherpa-ncnn-microphone.exe``: For real-time speech recognition from a microphone

That's it!

Please read :ref:`sherpa-ncnn-pre-trained-models` for usages about
the generated binaries.

Please create an issue at `<https://github.com/k2-fsa/sherpa-ncnn/issues>`_
if you have any problems.
