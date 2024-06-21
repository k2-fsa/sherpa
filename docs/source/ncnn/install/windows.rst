Windows
=======

This page describes how to build `sherpa-ncnn`_ on Windows.

.. hint::

  For the Python API, please refer to :ref:`sherpa-ncnn-python-api`.

.. hint::

   MinGW is known not to work.
   Please install ``Visual Studio`` before you continue.

64-bit Windows (x64)
--------------------

All you need is to run:

.. code-block:: bash

  git clone https://github.com/k2-fsa/sherpa-ncnn
  cd sherpa-ncnn
  mkdir build
  cd build
  cmake -DCMAKE_BUILD_TYPE=Release ..
  cmake --build . --config Release

It will generate two executables inside ``./bin/Release/``:

  - ``sherpa-ncnn.exe``: For decoding a single wave file.
  - ``sherpa-ncnn-microphone.exe``: For real-time speech recognition from a microphone

That's it!

Please read :ref:`sherpa-ncnn-pre-trained-models` for usages about
the generated binaries.

Please create an issue at `<https://github.com/k2-fsa/sherpa-ncnn/issues>`_
if you have any problems.

32-bit Windows (x86)
--------------------

All you need is to run:

.. code-block:: bash

  git clone https://github.com/k2-fsa/sherpa-ncnn
  cd sherpa-ncnn
  mkdir build
  cd build

  # Please select one toolset among VS 2015, 2017, 2019, and 2022 below
  # We use VS 2022 as an example.

  # For Visual Studio 2015
  # cmake -T v140,host=x64 -A Win32 -D CMAKE_BUILD_TYPE=Release ..

  # For Visual Studio 2017
  # cmake -T v141,host=x64 -A Win32 -D CMAKE_BUILD_TYPE=Release ..

  # For Visual Studio 2019
  # cmake -T v142,host=x64 -A Win32 -D CMAKE_BUILD_TYPE=Release ..

  # For Visual Studio 2022
  cmake -T v143,host=x64 -A Win32 -D CMAKE_BUILD_TYPE=Release ..

  cmake --build . --config Release -- -m:6

It will generate two executables inside ``./bin/Release/``:

  - ``sherpa-ncnn.exe``: For decoding a single wave file.
  - ``sherpa-ncnn-microphone.exe``: For real-time speech recognition from a microphone

That's it!

Please read :ref:`sherpa-ncnn-pre-trained-models` for usages about
the generated binaries.

Please create an issue at `<https://github.com/k2-fsa/sherpa-ncnn/issues>`_
if you have any problems.
