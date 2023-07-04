.. _sherpa_installation:

Installation
============

Requirements
------------

Before installing `k2-fsa/sherpa`_, we assume you have installed:

- `PyTorch`_
- `k2`_
- `kaldifeat`_


Normal
------

You can use the following commands to install `k2-fsa/sherpa`_:

.. code-block:: bash

   # Please make sure you have installed PyTorch, k2, and kaldifeat
   # before you continue
   #
   git clone http://github.com/k2-fsa/sherpa
   cd sherpa
   python3 -m pip install --verbose .

The above commands install the Python package ``k2-sherpa`` as well as
generated ``executables``.

.. hint::

   Please don't use ``python3 setup.py install``.

   Use ``python3 -m pip install --verbose .`` instead.

To check that you have installed `k2-fsa/sherpa`_ successfully, please run:

.. code-block:: bash

   # Test for Python
   #
   python3 -c "import sherpa; print(sherpa.__file__)"
   python3 -c "import sherpa; print(sherpa.__version__)"

   # Test for generated executables
   #
   which sherpa-version
   sherpa-version

   which sherpa-online
   sherpa-online --help

   which sherpa-offline
   sherpa-offline --help

   which sherpa-online-microphone
   sherpa-online-microphone --help

   which sherpa-offline-microphone
   sherpa-offline-microphone --help

   which sherpa-online-websocket-server
   sherpa-online-websocket-server --help

   which sherpa-online-websocket-client
   sherpa-online-websocket-client --help

   which sherpa-online-websocket-client-microphone
   sherpa-online-websocket-client-microphone --help

   which sherpa-offline-websocket-server
   sherpa-offline-websocket-server --help

   which sherpa-offline-websocket-client
   sherpa-offline-websocket-client --help


To uninstall `k2-fsa/sherpa`_, please use

.. code-block:: bash

   # Please run it outside of the k2-fsa/sherpa repo
   #
   pip uninstall k2-sherpa

For developers and advanced users
---------------------------------

You can also use the following commands to install `k2-fsa/sherpa`_.

The advantage is that you can have several versions of `k2-fsa/sherpa`_
in a single environment.

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

   cd /path/to/sherpa/build

   python3 -c "import sherpa; print(sherpa.__file__)"

   ./bin/sherpa-version

   ./bin/sherpa-offline --help
   ./bin/sherpa-online --help

   ./bin/sherpa-online-microphone --help
   ./bin/sherpa-offline-microphone --help

   ./bin/sherpa-offline-websocket-server --help
   ./bin/sherpa-offline-websocket-client --help

   ./bin/sherpa-online-websocket-server --help
   ./bin/sherpa-online-websocket-client --help
   ./bin/sherpa-online-websocket-client-microphone --help

Where to get help
-----------------

If you have any issues about the installation, please create an issue
at the following address:

  `<https://github.com/k2-fsa/sherpa/issues>`_
