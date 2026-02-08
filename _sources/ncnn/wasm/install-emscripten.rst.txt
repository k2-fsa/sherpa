Install Emscripten
==================

We need to compile the C/C++ files in `sherpa-ncnn`_ with the help of
`emscripten`_.

Please refer to `<https://emscripten.org/docs/getting_started/downloads>`_
for detailed installation instructions.

The following is an example to show you how to install it on Linux/macOS.

.. code-block:: bash

  git clone https://github.com/emscripten-core/emsdk.git
  cd emsdk
  git pull
  ./emsdk install latest
  ./emsdk activate latest
  source ./emsdk_env.sh

To check that you have installed `emscripten`_ successfully, please run:

.. code-block:: bash

  emcc -v

The above command should print something like below:

.. code-block::

  emcc (Emscripten gcc/clang-like replacement + linker emulating GNU ld) 3.1.48 (e967e20b4727956a30592165a3c1cde5c67fa0a8)
  shared:INFO: (Emscripten: Running sanity checks)
  (py38) fangjuns-MacBook-Pro:open-source fangjun$ emcc -v
  emcc (Emscripten gcc/clang-like replacement + linker emulating GNU ld) 3.1.48 (e967e20b4727956a30592165a3c1cde5c67fa0a8)
  clang version 18.0.0 (https://github.com/llvm/llvm-project a54545ba6514802178cf7cf1c1dd9f7efbf3cde7)
  Target: wasm32-unknown-emscripten
  Thread model: posix
  InstalledDir: /Users/fangjun/open-source/emsdk/upstream/bin

Congratulations! You have successfully installed `emscripten`_.
