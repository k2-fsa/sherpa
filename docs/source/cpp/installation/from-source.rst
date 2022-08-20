.. _cpp_installation:

Install from source (Linux/macOS/Windows)
=========================================

Install dependencies
--------------------

First, please refer to `<https://k2-fsa.github.io/k2/installation/index.html>`_
to install `k2`_.


.. code-block:: bash

   pip install kaldifeat kaldi_native_io

Install from source
-------------------

.. code-block:: bash

   git clone https://github.com/k2-fsa/sherpa
   cd sherpa
   mkdir build
   cd build
   cmake -DCMAKE_BUILD_TYPE=Release ..
   make -j sherpa sherpa-version


After running the above commands, you will get two executables:
``./bin/sherpa`` and ``./bin/sherpa-version`` in the build directory.

You can use

.. code-block:: bash

   ./bin/sherpa --help

to view usage information.

``./bin/sherp-version`` displays the information about the environment that
was used to build ``sherpa``.

Please read the section :ref:`cpp_non_streaming_asr` for more details.
