Embedded Linux (aarch64)
========================

This page describes how to build `sherpa-ncnn`_ for embedded Linux (aarch64, 64-bit)
with cross-compiling on an x86 machine with Ubuntu OS.

Install toolchain
-----------------

The first step is to install a toolchain for cross-compiling.

.. warning::

  You can use any toolchain that is suitable for your platform. The toolchain
  we use below is just an example.

Visit `<https://releases.linaro.org/components/toolchain/binaries/latest-7/aarch64-linux-gnu/>`_
to download the toolchain.

We are going to download ``gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu.tar.xz``,
which has been uploaded to `<https://huggingface.co/csukuangfj/sherpa-ncnn-toolchains>`_.

Assume you want to install it in the folder ``$HOME/software``:

.. code-block:: bash

   mkdir -p $HOME/software
   cd $HOME/software
   wget https://huggingface.co/csukuangfj/sherpa-ncnn-toolchains/resolve/main/gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu.tar.xz
   tar xvf gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu.tar.xz

Next, we need to set the following environment variable:

.. code-block:: bash

   export PATH=$HOME/software/gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu/bin:$PATH

To check that we have installed the cross-compiling toolchain successfully, please
run:

.. code-block:: bash

  aarch64-linux-gnu-gcc --version

which should print the following log:

.. code-block::

  aarch64-linux-gnu-gcc (Linaro GCC 7.5-2019.12) 7.5.0
  Copyright (C) 2017 Free Software Foundation, Inc.
  This is free software; see the source for copying conditions.  There is NO
  warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

Congratulations! You have successfully installed a toolchain for cross-compiling
`sherpa-ncnn`_.

Build sherpa-ncnn
-----------------

Finally, let us build `sherpa-ncnn`_.

.. code-block:: bash

  git clone https://github.com/k2-fsa/sherpa-ncnn
  cd sherpa-ncnn
  ./build-aarch64-linux-gnu.sh

After building, you will get two binaries:

.. code-block:: bash

  $ ls -lh build-aarch64-linux-gnu/install/bin/
  total 6.1M
  -rwxr-xr-x 1 kuangfangjun root 3.1M Dec 18 14:53 sherpa-ncnn
  -rwxr-xr-x 1 kuangfangjun root 3.1M Dec 18 14:53 sherpa-ncnn-microphone

That's it!

Please read :ref:`sherpa-ncnn-pre-trained-models` for usages about
the generated binaries.

Read below if you want to learn more.

.. hint::

  By default, all external dependencies are statically linked. That means,
  the generated binaries are self-contained.

  You can use the following commands to check that and you will find
  they depend only on system libraries.

    .. code-block::

      $ readelf -d build-aarch64-linux-gnu/install/bin/sherpa-ncnn

      Dynamic section at offset 0x302a80 contains 30 entries:
        Tag        Type                         Name/Value
       0x0000000000000001 (NEEDED)             Shared library: [libgomp.so.1]
       0x0000000000000001 (NEEDED)             Shared library: [libpthread.so.0]
       0x0000000000000001 (NEEDED)             Shared library: [libstdc++.so.6]
       0x0000000000000001 (NEEDED)             Shared library: [libm.so.6]
       0x0000000000000001 (NEEDED)             Shared library: [libgcc_s.so.1]
       0x0000000000000001 (NEEDED)             Shared library: [libc.so.6]
       0x000000000000000f (RPATH)              Library rpath: [$ORIGIN]

      $ readelf -d build-aarch64-linux-gnu/install/bin/sherpa-ncnn-microphone

      Dynamic section at offset 0x301a98 contains 30 entries:
        Tag        Type                         Name/Value
       0x0000000000000001 (NEEDED)             Shared library: [libpthread.so.0]
       0x0000000000000001 (NEEDED)             Shared library: [libgomp.so.1]
       0x0000000000000001 (NEEDED)             Shared library: [libstdc++.so.6]
       0x0000000000000001 (NEEDED)             Shared library: [libm.so.6]
       0x0000000000000001 (NEEDED)             Shared library: [libgcc_s.so.1]
       0x0000000000000001 (NEEDED)             Shared library: [libc.so.6]
       0x000000000000000f (RPATH)              Library rpath: [$ORIGIN]

Please create an issue at `<https://github.com/k2-fsa/sherpa-ncnn/issues>`_
if you have any problems.
