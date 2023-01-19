macOS
=====

This page describes how to build `sherpa-ncnn`_ on macOS.

.. hint::

  For the Python API, please refer to :ref:`sherpa-ncnn-python-api`.

All you need is to run:

.. code-block:: bash

  git clone https://github.com/k2-fsa/sherpa-ncnn
  cd sherpa-ncnn
  mkdir build
  cd build
  cmake -DCMAKE_BUILD_TYPE=Release ..
  make -j6

After building, you will find two executables inside the ``bin`` directory:

.. code-block:: bash

  $ ls -lh bin/
  total 24232
  -rwxr-xr-x  1 fangjun  staff   5.9M Dec 18 12:39 sherpa-ncnn
  -rwxr-xr-x  1 fangjun  staff   6.0M Dec 18 12:39 sherpa-ncnn-microphone

That's it!

Please read :ref:`sherpa-ncnn-pre-trained-models` for usages about
the generated binaries.

Read below if you want to learn more.

You can strip the binaries by

.. code-block:: bash

  $ strip bin/sherpa-ncnn
  $ strip bin/sherpa-ncnn-microphone

After stripping, the file size of each binary is:

.. code-block:: bash

  $ ls -lh bin/
  total 23000
  -rwxr-xr-x  1 fangjun  staff   5.6M Dec 18 12:40 sherpa-ncnn
  -rwxr-xr-x  1 fangjun  staff   5.6M Dec 18 12:40 sherpa-ncnn-microphone

.. hint::

  By default, all external dependencies are statically linked. That means,
  the generated binaries are self-contained.

  You can use the following commands to check that and you will find
  they depend only on system libraries.

    .. code-block::

      $ otool -L bin/sherpa-ncnn
      bin/sherpa-ncnn:
              /usr/local/opt/libomp/lib/libomp.dylib (compatibility version 5.0.0, current version 5.0.0)
              /usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 902.1.0)
              /usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1281.100.1)

      $ otool -L bin/sherpa-ncnn-microphone
      bin/sherpa-ncnn-microphone:
              /System/Library/Frameworks/CoreAudio.framework/Versions/A/CoreAudio (compatibility version 1.0.0, current version 1.0.0)
              /System/Library/Frameworks/AudioToolbox.framework/Versions/A/AudioToolbox (compatibility version 1.0.0, current version 1000.0.0)
              /System/Library/Frameworks/AudioUnit.framework/Versions/A/AudioUnit (compatibility version 1.0.0, current version 1.0.0)
              /System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation (compatibility version 150.0.0, current version 1677.104.0)
              /System/Library/Frameworks/CoreServices.framework/Versions/A/CoreServices (compatibility version 1.0.0, current version 1069.24.0)
              /usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1281.100.1)
              /usr/local/opt/libomp/lib/libomp.dylib (compatibility version 5.0.0, current version 5.0.0)
              /usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 902.1.0)

Please create an issue at `<https://github.com/k2-fsa/sherpa-ncnn/issues>`_
if you have any problems.
