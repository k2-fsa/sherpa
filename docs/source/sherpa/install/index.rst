.. _sherpa_installation:

Installation
============

Before installing `k2-fsa/sherpa`_, we assume you have installed:

- `PyTorch`_
- `k2`_
- `kaldifeat`_

You can use the following commands to install `k2-fsa/sherpa`_:

.. code-block:: bash

   git clone http://github.com/k2-fsa/sherpa
   cd sherpa
   mkdir build
   cd build
   cmake ..
   make -j

   export PATH=$PWD/bin:$PATH
   export PYTHONPATH=$PWD/lib:$PWD/../sherpa/python:$PYTHONPATH

To test that you have installed `k2-fsa/sherpa`_ successfully, you can run the
following commands:

.. code-block:: bash

   ./bin/sherpa-version

   ./bin/sherpa-offline --help
   ./bin/sherpa-online --help
   ./bin/sherpa-online-microphone --help

   ./bin/sherpa-offline-websocket-server --help
   ./bin/sherpa-offline-websocket-client --help

   ./bin/sherpa-online-websocket-server --help
   ./bin/sherpa-online-websocket-client --help
   ./bin/sherpa-online-websocket-client-microphone --help

If you have any issues about the installation, please create an issue
at the following address:

  `<https://github.com/k2-fsa/sherpa/issues>`_
