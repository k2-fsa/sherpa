.. _install_sherpa_ncnn_on_linux:

Linux
=====

This page describes how to build `sherpa-ncnn`_ on Linux.

.. hint::

   You can follow this section if you want to build `sherpa-ncnn`_ directly
   on your board.

.. hint::

  For the Python API, please refer to :ref:`sherpa-ncnn-python-api`.

All you need is to run:

.. tabs::

   .. tab:: x86/x86_64

      .. code-block:: bash

        git clone https://github.com/k2-fsa/sherpa-ncnn
        cd sherpa-ncnn
        mkdir build
        cd build
        cmake -DCMAKE_BUILD_TYPE=Release ..
        make -j6

   .. tab:: 32-bit ARM

     .. code-block:: bash

        git clone https://github.com/k2-fsa/sherpa-ncnn
        cd sherpa-ncnn
        mkdir build
        cd build
        cmake \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_C_FLAGS="-march=armv7-a -mfloat-abi=hard -mfpu=neon" \
          -DCMAKE_CXX_FLAGS="-march=armv7-a -mfloat-abi=hard -mfpu=neon" \
          ..
        make -j6

   .. tab:: 64-bit ARM

     .. code-block:: bash

        git clone https://github.com/k2-fsa/sherpa-ncnn
        cd sherpa-ncnn
        mkdir build
        cd build
        cmake \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_C_FLAGS="-march=armv8-a" \
          -DCMAKE_CXX_FLAGS="-march=armv8-a" \
          ..
        make -j6


After building, you will find two executables inside the ``bin`` directory:

.. code-block:: bash

  $ ls -lh bin/
  total 13M
  -rwxr-xr-x 1 kuangfangjun root 6.5M Dec 18 11:31 sherpa-ncnn
  -rwxr-xr-x 1 kuangfangjun root 6.5M Dec 18 11:31 sherpa-ncnn-microphone

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
  total 12M
  -rwxr-xr-x 1 kuangfangjun root 5.8M Dec 18 11:35 sherpa-ncnn
  -rwxr-xr-x 1 kuangfangjun root 5.8M Dec 18 11:36 sherpa-ncnn-microphone

.. hint::

  By default, all external dependencies are statically linked. That means,
  the generated binaries are self-contained.

  You can use the following commands to check that and you will find
  they depend only on system libraries.

    .. code-block::

      $ readelf -d bin/sherpa-ncnn

      Dynamic section at offset 0x5c0650 contains 34 entries:
        Tag        Type                         Name/Value
       0x0000000000000001 (NEEDED)             Shared library: [libgomp.so.1]
       0x0000000000000001 (NEEDED)             Shared library: [libpthread.so.0]
       0x0000000000000001 (NEEDED)             Shared library: [libstdc++.so.6]
       0x0000000000000001 (NEEDED)             Shared library: [libm.so.6]
       0x0000000000000001 (NEEDED)             Shared library: [libmvec.so.1]
       0x0000000000000001 (NEEDED)             Shared library: [libgcc_s.so.1]
       0x0000000000000001 (NEEDED)             Shared library: [libc.so.6]
       0x000000000000001d (RUNPATH)            Library runpath: [$ORIGIN:]

      $ readelf -d bin/sherpa-ncnn-microphone

      Dynamic section at offset 0x5c45d0 contains 34 entries:
        Tag        Type                         Name/Value
       0x0000000000000001 (NEEDED)             Shared library: [libpthread.so.0]
       0x0000000000000001 (NEEDED)             Shared library: [libgomp.so.1]
       0x0000000000000001 (NEEDED)             Shared library: [libstdc++.so.6]
       0x0000000000000001 (NEEDED)             Shared library: [libm.so.6]
       0x0000000000000001 (NEEDED)             Shared library: [libmvec.so.1]
       0x0000000000000001 (NEEDED)             Shared library: [libgcc_s.so.1]
       0x0000000000000001 (NEEDED)             Shared library: [libc.so.6]
       0x000000000000001d (RUNPATH)            Library runpath: [$ORIGIN:]

Please create an issue at `<https://github.com/k2-fsa/sherpa-ncnn/issues>`_
if you have any problems.
