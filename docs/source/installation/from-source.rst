Install from source
===================

Before installing ``sherpa``, you have to install `PyTorch <https://pytorch.org/>`_.
If you want to install a CUDA version of ``sherpa``, please install a CUDA version
of ``PyTorch``.

Supported operating systems, Python versions, PyTorch versions, and CUDA versions
are listed as follows:

    - |os_types|
    - |python_versions|
    - |pytorch_versions|
    - |cuda_versions|

.. |os_types| image:: ./pic/os-brightgreen.svg
  :alt: Supported operating systems

.. |python_versions| image:: ./pic/python_ge_3.7-blue.svg
  :alt: Supported python versions

.. |cuda_versions| image:: ./pic/cuda_ge_10.1-orange.svg
  :alt: Supported cuda versions

.. |pytorch_versions| image:: ./pic/pytorch_ge_1.6.0-blueviolet.svg
  :alt: Supported pytorch versions

.. HINT::

   If you install a CUDA version of PyTorch, please also install cuDNN.
   Otherwise, you will get CMake configuration errors later.

Assume you have installed ``PyTorch``. The steps to install ``sherpa`` from source
are given below.

.. code-block:: bash

  git clone https://github.com/k2-fsa/sherpa
  cd sherpa

  # Install the dependencies
  pip install -r ./requirements.txt

  # Install the C++ extension.
  # Use one of the following methods:
  #
  # (1)
  python3 setup.py install --verbose
  #
  # (2)
  # pip install --verbose k2-sherpa

  # To uninstall the C++ extension, use
  # pip uninstall k2-sherpa

.. note::

   You can use either ``python3 setup.py install --verbose``
   or ``pip install --verbose k2-sherpa`` to install the C++
   extension.

.. hint::

  Refer to :ref:`fix cuDNN not found` if you encouter this problem during
  installation.

To check that you have installed ``sherpa`` successfully, use

.. code-block:: bash

  python3 -c "import sherpa; print(sherpa.__version__)"

It should print the version of ``sherpa``.

**Congratulations!**

You have installed ``sherpa`` successfully. Let us start
to play with it.

The following shows you a `YouTube video <https://www.youtube.com/watch?v=z7HgaZv5W0U>`_,
demonstrating streaming ASR with ``sherpa``.

.. note::

   If you have no access to YouTube, please visit the following link from bilibili
   `<https://www.bilibili.com/video/BV1BU4y197bs>`_

..  youtube:: z7HgaZv5W0U
   :width: 120%

Read more to see how to use ``sherpa`` for streaming ASR and non-streaming ASR.

