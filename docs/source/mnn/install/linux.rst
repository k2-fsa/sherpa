.. _install_sherpa_mnn_on_linux:

Linux
=====

This page describes how to build `sherpa-mnn`_ on Linux.

.. hint::

   You can follow this section if you want to build `sherpa-mnn`_ directly
   on your board.

All you need is to run:

.. tabs::

   .. tab:: x86/x86_64

      .. code-block:: bash

        git clone https://github.com/k2-fsa/sherpa-mnn
        cd sherpa-mnn
        mkdir build
        cd build
        cmake -DCMAKE_BUILD_TYPE=Release ..
        make -j6

   .. tab:: 32-bit ARM

     .. code-block:: bash

        git clone https://github.com/k2-fsa/sherpa-mnn
        cd sherpa-mnn
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

        git clone https://github.com/k2-fsa/sherpa-mnn
        cd sherpa-mnn
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
  total 8.4M
  -rwxrwxr-x 1 meixu meixu 4.2M Oct 15 19:46 sherpa-mnn
  -rwxrwxr-x 1 meixu meixu 4.2M Oct 15 19:46 sherpa-mnn-microphone

That's it!

Please read :ref:`sherpa-mnn-pre-trained-models` for usages about
the generated binaries.

Read below if you want to learn more.

You can strip the binaries by

.. code-block:: bash

  $ strip bin/sherpa-mnn
  $ strip bin/sherpa-mnn-microphone

After stripping, the file size of each binary is:

.. code-block:: bash

  $ ls -lh bin/
  total 7.3M
  -rwxrwxr-x 1 meixu meixu 3.6M Oct 15 19:50 sherpa-mnn
  -rwxrwxr-x 1 meixu meixu 3.7M Oct 15 19:50 sherpa-mnn-microphone

.. hint::

  By default, all external dependencies are statically linked. That means,
  the generated binaries are self-contained.

  You can use the following commands to check that and you will find
  they depend only on system libraries.

    .. code-block::

      $ readelf -d bin/sherpa-mnn

      Dynamic section at offset 0x3965b0 contains 33 entries:
        Tag        Type                         Name/Value
       0x0000000000000001 (NEEDED)             Shared library: [libstdc++.so.6]
       0x0000000000000001 (NEEDED)             Shared library: [libm.so.6]
       0x0000000000000001 (NEEDED)             Shared library: [libmvec.so.1]
       0x0000000000000001 (NEEDED)             Shared library: [libgcc_s.so.1]
       0x0000000000000001 (NEEDED)             Shared library: [libc.so.6]
       0x0000000000000001 (NEEDED)             Shared library: [ld-linux-x86-64.so.2]
       0x000000000000001d (RUNPATH)            Library runpath: [$ORIGIN:]

      $ readelf -d bin/sherpa-mnn-microphone

      Dynamic section at offset 0x39a510 contains 33 entries:
        Tag        Type                         Name/Value
       0x0000000000000001 (NEEDED)             Shared library: [libstdc++.so.6]
       0x0000000000000001 (NEEDED)             Shared library: [libm.so.6]
       0x0000000000000001 (NEEDED)             Shared library: [libmvec.so.1]
       0x0000000000000001 (NEEDED)             Shared library: [libgcc_s.so.1]
       0x0000000000000001 (NEEDED)             Shared library: [libc.so.6]
       0x0000000000000001 (NEEDED)             Shared library: [ld-linux-x86-64.so.2]
       0x000000000000001d (RUNPATH)            Library runpath: [$ORIGIN]

Please create an issue at `<https://github.com/k2-fsa/sherpa-mnn/issues>`_
if you have any problems.
