.. role:: strike

.. _cpp_installation:

Install from source (Linux/macOS/Windows)
=========================================

Install dependencies
--------------------

Install k2
^^^^^^^^^^

First, please refer to `<https://k2-fsa.github.io/k2/installation/index.html>`_
to install `k2`_.

.. hint::

   If you are using macOS, you can dowload pre-built wheels from
   `<https://k2-fsa.org/nightly/index.html>`_


Install other dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pip install -U kaldifeat kaldi_native_io

Install from source
-------------------

You can select ``one`` of the following methods to install ``sherpa``
from source.

Option 1
^^^^^^^^

.. code-block:: bash

   git clone https://github.com/k2-fsa/sherpa
   cd sherpa
   mkdir build
   cd build
   cmake -DCMAKE_BUILD_TYPE=Release ..
   make -j


After running the above commands, you will get two executables:
``./bin/sherpa`` and ``./bin/sherpa-version`` in the build directory.

You can use

.. code-block:: bash

   ./bin/sherpa --help

to view usage information.

``./bin/sherpa-version`` displays the information about the environment that
was used to build ``sherpa``.

Please read the section :ref:`cpp_non_streaming_asr` for more details.

Option 2
^^^^^^^^

.. code-block:: bash

   git clone https://github.com/k2-fsa/sherpa
   cd sherpa
   python3 setup.py bdist_wheel

   # It will generate a file in ./dist
   # For instance, if the file is ./dist/k2_sherpa-0.7.1-cp38-cp38-linux_x86_64.whl
   # You can use

   pip install ./dist/k2_sherpa-0.7.1-cp38-cp38-linux_x86_64.whl

   # If you want to uninstall it, use
   #
   #  pip uninstall k2-sherpa
   #


.. caution::

    The command to uninstall ``sherpa`` is ``pip uninstall k2-sherpa``,
    **NOT** :strike:`pip uninstall sherpa`

.. hint::

   If you use ``python3 setup.py install``, you won't find the executable
   ``sherpa`` and ``sherpa-version`` in your PATH.

To check that you have installed ``sherpa`` successfully, you can use:

.. code-block:: bash

      which sherpa
      which sherpa-version

      sherpa-version


      sherpa --help

