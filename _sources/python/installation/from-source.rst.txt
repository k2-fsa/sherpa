.. _install_sherpa_from_source:

Install from source
===================

Before installing `sherpa`_, you have to install `PyTorch`_ and `k2`_.
If you want to install a CUDA version of `sherpa`_, please install a CUDA version
of `PyTorch`_ and `k2`_.

Supported operating systems, Python versions, `PyTorch`_ versions, and CUDA versions
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

Install `PyTorch`_
------------------

Please refer to `<https://pytorch.org/get-started/locally/>`_ to install PyTorch.

Install `k2`_
-------------

Please refer to `<https://k2-fsa.github.io/k2/installation/index.html>`_
to install `k2`_. You have to install ``k2 >= v1.16``.

Install `sherpa`_
-----------------

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

   ``python3 setup.py install`` always installs the latest version.

.. hint::

  Refer to :ref:`fix cuDNN not found` if you encouter this problem during
  installation.

.. caution::

   ``pip install -r ./requirements.txt`` won't install all dependencies
   of `sherpa`_. You have to install `PyTorch`_ and `k2`_ before you install
   `sherpa`_.

To check that you have installed `sherpa`_ successfully, run

.. code-block:: bash

  python3 -c "import sherpa; print(sherpa.__version__)"

It should print the version of `sherpa`_.

**Congratulations!**

You have installed `sherpa`_ successfully. Let us start
to play with it.

The following shows you a `YouTube video <https://www.youtube.com/watch?v=z7HgaZv5W0U>`_,
demonstrating streaming ASR with `sherpa`_.

.. note::

   If you have no access to YouTube, please visit the following link from bilibili
   `<https://www.bilibili.com/video/BV1BU4y197bs>`_

..  youtube:: z7HgaZv5W0U
   :width: 120%

Read more to see how to use `sherpa`_ for streaming ASR and non-streaming ASR.
