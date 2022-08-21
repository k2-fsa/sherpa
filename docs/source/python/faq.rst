Frequently asked questions
==========================

Where to ask for help
---------------------
Please create a new issue at `<https://github.com/k2-fsa/sherpa/issues>`_.

.. hint::

   Please search for existing issues, if there are any, before creating
   a new one.

How to install a CPU version of sherpa
--------------------------------------

All you need to do is to install a CPU version of PyTorch before
installing ``sherpa``.

How to install sherpa with CUDA support
---------------------------------------

All you need to do is to install a CUDA version of PyTorch before
installing ``sherpa``

.. _fix cuDNN not found:

How to fix `Caffe2: Cannot find cuDNN library`
----------------------------------------------

This issue happens only when you have installed a CUDA version of PyTorch but
without installing cuDNN.

The fix is to install cuDNN.

If you have installed cuDNN and it still does not help, you can do

.. code-block::

  export SHERPA_CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -DCUDNN_LIBRARY_PATH=/path/to/cudnn/lib/libcudnn.so -DCUDNN_INCLUDE_PATH=/path/to/cudnn/include"

before running ``pip install --verbose k2-sherpa`` or ``python3 setup.py install``.

.. hint::

  The above command assumes that you have installed cuDNN to ``/path/to/cudnn``.

How to uninstall sherpa
-----------------------

.. code-block:: bash

   pip uninstall k2-sherpa
