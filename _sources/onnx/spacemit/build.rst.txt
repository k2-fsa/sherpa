.. _build-sherpa-onnx-for-spacemit:

Build sherpa-onnx for SpacemiT CPUs
===================================

This page documents the basic build flow for enabling the SpacemiT execution
provider in `sherpa-onnx`_.

Cross-compilation prerequisites
-------------------------------

The provided build script targets SpacemiT RISC-V Linux.

If ``RISCV_ROOT_PATH`` is already set to a valid SpacemiT toolchain directory,
the script will reuse it. Otherwise, it will download a toolchain archive
automatically.

Quick start
-----------

The repository already provides a dedicated build script:

.. code-block:: bash

   git clone https://github.com/k2-fsa/sherpa-onnx
   cd sherpa-onnx
   ./build-riscv64-linux-gnu-spacemit.sh

If you want to use a local toolchain, set:

.. code-block:: bash

   export RISCV_ROOT_PATH=/path/to/spacemit-toolchain
   ./build-riscv64-linux-gnu-spacemit.sh

The script enables the following CMake option:

.. code-block:: bash

   -DSHERPA_ONNX_ENABLE_SPACEMIT=ON

It also uses the SpacemiT RISC-V toolchain file:

.. code-block:: bash

   -DCMAKE_TOOLCHAIN_FILE=../toolchains/riscv64-linux-gnu-spacemit.toolchain.cmake

Build notes
-----------

- The build script cross-compiles ``alsa-lib`` when needed.
- The default build directory is ``build-riscv64-linux-gnu-spacemit``.
- The default install prefix is ``build-riscv64-linux-gnu-spacemit/install``.
- Shared libraries are enabled by default in the build script.
- ``SHERPA_ONNX_ENABLE_C_API`` and ``SHERPA_ONNX_ENABLE_WEBSOCKET`` are enabled by the script.
- ``SHERPA_ONNX_ENABLE_PYTHON`` and tests are disabled in the script.

Manual CMake flow
-----------------

If you prefer to drive CMake manually, the skeleton command looks like this:

.. code-block:: bash

   cmake \
     -DCMAKE_INSTALL_PREFIX=./install \
     -DCMAKE_BUILD_TYPE=Release \
     -DBUILD_SHARED_LIBS=ON \
     -DSHERPA_ONNX_ENABLE_SPACEMIT=ON \
     -DCMAKE_TOOLCHAIN_FILE=../toolchains/riscv64-linux-gnu-spacemit.toolchain.cmake \
     ..

   make -j4
   make install

Build outputs
-------------

After a successful build, you can usually find artifacts in both of the
following locations:

- ``build-riscv64-linux-gnu-spacemit/bin`` and ``build-riscv64-linux-gnu-spacemit/lib``
- ``build-riscv64-linux-gnu-spacemit/install/bin`` and ``build-riscv64-linux-gnu-spacemit/install/lib``

What to verify after build
--------------------------

After the build finishes, you can start by checking the generated binaries:

.. code-block:: bash

   ls -lh build-riscv64-linux-gnu-spacemit/install/bin/
   ls -lh build-riscv64-linux-gnu-spacemit/install/lib/

Typical binaries to verify first:

- ``sherpa-onnx``
- ``sherpa-onnx-offline``
- ``sherpa-onnx-offline-tts``

Board-side run
--------------

If you copy the install directory to your board, a minimal smoke test looks
like this:

.. code-block:: bash

   export LD_LIBRARY_PATH=./lib
   ./bin/sherpa-onnx --help
   ./bin/sherpa-onnx-offline --help
   ./bin/sherpa-onnx-offline-tts --help

If ``--help`` works, it usually means the binary and runtime libraries are in a
good state for the next step of model testing.

QEMU smoke test
---------------

The CI workflow also validates the generated binaries with ``qemu-riscv64``.
The minimal pattern is:

.. code-block:: bash

   export PATH=/path/to/toolchain/bin:$PATH
   export PATH=/path/to/qemu/bin:$PATH
   export QEMU_LD_PREFIX=/path/to/toolchain/sysroot
   export LD_LIBRARY_PATH=/path/to/toolchain/sysroot/lib
   export QEMU_ARGS="-cpu max,vlen=256,elen=64,vext_spec=v1.0"

   qemu-riscv64 ${QEMU_ARGS} ./build-riscv64-linux-gnu-spacemit/bin/sherpa-onnx --help
   qemu-riscv64 ${QEMU_ARGS} ./build-riscv64-linux-gnu-spacemit/bin/sherpa-onnx-offline --help
   qemu-riscv64 ${QEMU_ARGS} ./build-riscv64-linux-gnu-spacemit/bin/sherpa-onnx-offline-tts --help

This is a convenient way to do a quick verification on the host before copying
artifacts to a board.
