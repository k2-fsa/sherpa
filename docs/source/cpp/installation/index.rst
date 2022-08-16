.. _cpp_installation:

Installation
============

Install dependencies
--------------------

First, please refer to `<https://k2-fsa.github.io/k2/installation/index.html>`_
to install `k2`_.


.. code-block:: bash

   pip install kaldifeat kaldi_native_io

Install C++ frontend of sherpa
------------------------------

.. code-block:: bash

   git clone https://github.com/k2-fsa/sherpa
   cd sherpa
   mkdir build
   cd build
   cmake -DCMAKE_BUILD_TYPE=Release ..
   make -j sherpa


After running the above commands, you will get a binary ``./bin/sherpa`` in the
build directory. You can use

.. code-block:: bash

   ./bin/sherpa --help

to view usage information.

Please read the section :ref:`cpp_non_streaming_asr` for more details.
