.. _cpp_fronted_installation:

Installation
============

Before installing `sherpa`_, we assume you have installed:

- `PyTorch`_
- `k2`_
- `kaldifeat`_

You can use the following commands to install `sherpa`_:

.. code-block:: bash

   git clone http://github.com/k2-fsa/sherpa
   cd sherpa
   python3 setup.py bdist_wheel
   ls -lh dist
   pip install ./dist/k2_sherpa*.whl

.. caution::

   Please don't use ``python3 setup.py install``. Otherwise, you won't get
   `sherpa`_ related binaries installed, such as ``sherpa-offline`` and
   ``sherpa-online``.

To uninstall `sherpa`_, please use

.. code-block:: bash

   pip uninstall k2-sherpa

To test that you have installed `sherpa`_ successfully, you can run the
following commands:

.. code-block:: bash

   sherpa-version

   sherpa-offline --help
   sherpa-online --help
   sherpa-online-microphone --help

   sherpa-offline-websocket-server --help
   sherpa-offline-websocket-client --help

   sherpa-online-websocket-server --help
   sherpa-online-websocket-client --help
   sherpa-online-websocket-client-microphone --help

If you have any issues about the installation, please create an issue
at the following address:

  `<https://github.com/k2-fsa/sherpa/issues>`_

.. hint::

   If you have a `WeChat <https://www.wechat.com/>`_ account, you can scan
   the following QR code to join the WeChat group of next-gen Kaldi to get
   help.

   .. image:: pic/wechat-group-for-next-gen-kaldi.jpg
    :width: 200
    :align: center
    :alt: WeChat group of next-gen Kaldi


Installation for advanced users/developers
------------------------------------------

As an advanced user/developer, you can use the following method to
install `sherpa`_:


.. code-block:: bash

   git clone http://github.com/k2-fsa/sherpa
   cd sherpa
   mkdir build
   cd build
   cmake ..
   make -j

   export PATH=$PWD/bin:$PATH
   export PYTHONPATH=$PWD/lib:$PWD/../sherpa/python:$PYTHONPATH
