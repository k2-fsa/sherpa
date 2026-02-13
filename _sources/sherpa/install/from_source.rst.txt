.. _install_sherpa_from_source:

From source
===========

This section describe how to install ``k2-fsa/sherpa`` from source.


Install dependencies
--------------------

Before installing ``k2-fsa/sherpa`` from source, we have to install the following
dependencies.

  - `PyTorch`_
  - `k2`_
  - `kaldifeat`_

.. tabs::

   .. tab:: CPU

     Suppose that we select ``torch==2.0.1``. We can use the following
     commands to install the dependencies:

     .. tabs::

       .. tab:: Linux

        .. code-block:: bash

           pip install torch==2.0.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
           pip install k2==1.24.4.dev20231220+cpu.torch2.0.1 -f https://k2-fsa.github.io/k2/cpu.html
           pip install kaldifeat==1.25.3.dev20231221+cpu.torch2.0.1 -f https://csukuangfj.github.io/kaldifeat/cpu.html

       .. tab:: macOS

        .. code-block:: bash

           pip install torch==2.0.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
           pip install k2==1.24.4.dev20231220+cpu.torch2.0.1 -f https://k2-fsa.github.io/k2/cpu.html
           pip install kaldifeat==1.25.3.dev20231221+cpu.torch2.0.1 -f https://csukuangfj.github.io/kaldifeat/cpu.html

       .. tab:: Windows

        To be done.

   .. tab:: CUDA

     Suppose that we select ``torch==2.0.1+cu117``. We can use the following
     commands to install the dependencies:

      .. code-block:: bash

         pip install torch==2.0.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
         pip install k2==1.24.4.dev20231220+cuda11.7.torch2.0.1 -f https://k2-fsa.github.io/k2/cuda.html
         pip install kaldifeat==1.25.3.dev20231221+cuda11.7.torch2.0.1  -f https://csukuangfj.github.io/kaldifeat/cuda.html

     Next, please follow `<https://k2-fsa.github.io/k2/installation/cuda-cudnn.html>`_ to install CUDA toolkit.

Now we can start to build ``k2-fsa/sherpa`` from source.

For general users
-----------------

You can use the following commands to install `k2-fsa/sherpa`_:

.. code-block:: bash

   # Please make sure you have installed PyTorch, k2, and kaldifeat
   # before you continue
   #
   git clone http://github.com/k2-fsa/sherpa
   cd sherpa
   python3 -m pip install --verbose .

To uninstall `k2-fsa/sherpa`_, please use

.. code-block:: bash

   # Please run it outside of the k2-fsa/sherpa repo
   #
   pip uninstall k2-sherpa

Please see :ref:`check_sherpa_installation`.

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

   # For torch >= 2.0, please use
   #
   #  cmake -DCMAKE_CXX_STANDARD=17 ..
   #

   cmake ..
   make -j

   export PATH=$PWD/bin:$PATH
   export PYTHONPATH=$PWD/lib:$PWD/../sherpa/python:$PYTHONPATH

Please see :ref:`check_sherpa_installation`.

