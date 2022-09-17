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

Could not find PyTorch
----------------------

If you have the following error while installing ``sherpa``:

.. code-block::

  CMake Error at cmake/torch.cmake:14 (find_package):
    By not providing "FindTorch.cmake" in CMAKE_MODULE_PATH this project has
    asked CMake to find a package configuration file provided by "Torch", but
    CMake did not find one.

    Could not find a package configuration file provided by "Torch" with any of
    the following names:

      TorchConfig.cmake
      torch-config.cmake

    Add the installation prefix of "Torch" to CMAKE_PREFIX_PATH or set
    "Torch_DIR" to a directory containing one of the above files.  If "Torch"
    provides a separate development package or SDK, be sure it has been
    installed.
  Call Stack (most recent call first):
    CMakeLists.txt:120 (include)

The fix is to install ``PyTorch`` first and retry.

If it still does not work, please make sure you have used the same
(virtual) environment where ``PyTorch`` is installed to compile ``sherpa``.

.. hint::

   You can look for the path to the ``python3`` executable in the output of
   cmake to find out which environment ``cmake`` is using.


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

  The above command assumes that you have installed cuDNN to ``/path/to/cudnn``
  and you can find the following files:

    - ``/path/to/cudnn/lib/libcudnn.so``
    - ``/path/to/cudnn/include/cudnn.h``

.. hint::

   If you are using ``conda``, you can use:

    .. code-block:: bash

      conda install cudnn

   to install ``cudnn``. And possibly you don't need to set the above
   environment variable ``SHERPA_CMAKE_ARGS`` after you ran
   ``conda install cudnn``.

How to uninstall sherpa
-----------------------

.. code-block:: bash

   pip uninstall k2-sherpa
