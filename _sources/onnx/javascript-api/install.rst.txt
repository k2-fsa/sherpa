.. _install_sherpa_onnx_node:

Install
=======

We provide npm packages for `sherpa-onnx-node`_.

It can be found at

  `<https://www.npmjs.com/package/sherpa-onnx-node>`_

.. hint::

   It requires ``Node >= v16``.

Please always use the latest version.

To install it, please run::

  npm install sherpa-onnx-node

It supports the following platforms:

  - Linux x64
  - Linux arm64
  - macOS x64
  - macOS arm64
  - Windows x64

.. hint::

   You don't need to pre-install anything in order to install ``sherpa-onnx-node``.

   That is, you don't need to install a C/C++ compiler. You don't need to install Python.
   You don't need to install CMake, etc.

Set up the library path
-----------------------

The native addon requires shared libraries at runtime. You must set the
appropriate environment variable before running your script.

macOS
^^^^^

.. code-block:: bash

   export DYLD_LIBRARY_PATH=$(npm root)/sherpa-onnx-node/lib:$DYLD_LIBRARY_PATH

Linux
^^^^^

.. code-block:: bash

   export LD_LIBRARY_PATH=$(npm root)/sherpa-onnx-node/lib:$LD_LIBRARY_PATH

Windows
^^^^^^^

No extra setup is needed. The DLLs are located inside ``node_modules`` and
are found automatically.

.. tip::

   You can add the ``export`` command to your shell profile (e.g., ``~/.bashrc``
   or ``~/.zshrc``) so you don't have to run it every time.

Optional dependencies
---------------------

Microphone support
^^^^^^^^^^^^^^^^^^

For examples that record from a microphone (VAD, streaming ASR from mic),
install `node-cpal <https://www.npmjs.com/package/node-cpal>`_::

  npm install node-cpal

Speaker playback
^^^^^^^^^^^^^^^^

For examples that play audio through speakers (TTS with real-time playback),
install `speaker <https://www.npmjs.com/package/speaker>`_::

  npm install speaker

Verify the installation
-----------------------

Run the following command to verify that the package is installed correctly:

.. code-block:: bash

   node -e "const sherpa_onnx = require('sherpa-onnx-node'); console.log('sherpa-onnx-node is installed successfully')"

If you see ``sherpa-onnx-node is installed successfully``, the installation
is working.

Where to go next
----------------

See :doc:`./examples/index` for runnable examples covering TTS, ASR, VAD,
speaker diarization, and more.
