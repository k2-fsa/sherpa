conda for Linux
===============

.. note::

   We recommend creating a new virtual environment to install ``sherpa``.


CPU version
-----------

The command to install a CPU version of ``sherpa`` for Linux using ``conda`` is:

.. code-block:: bash

   conda install \
     -c k2-fsa \
     -c k2-fsa-sherpa \
     -c kaldifeat \
     -c kaldi_native_io \
     -c pytorch \
     cpuonly \
     k2 \
     sherpa \
     kaldifeat \
     kaldi_native_io \
     pytorch=1.12.0 \
     python=3.8

or the following command in one line:

.. code-block:: bash

   conda install -c k2-fsa -c k2-fsa-sherpa -c kaldifeat -c kaldi_native_io -c pytorch cpuonly k2 sherpa kaldifeat kaldi_native_io pytorch=1.12.0 python=3.8

.. note::

   You have to specify ``cpuonly`` to install a CPU version of ``sherpa``.

.. caution::

   It is of paramount importance that you specify the ``-c`` options while
   installing ``sherpa``. Otherwise, you will be SAD.

   You can switch the orders of different options for ``-c``, but you cannot
   omit them.

We provide pre-built conda packages for ``Python >= 3.7`` and ``PyTorch >= 1.6.0``.
Please consider installing ``sherpa`` from source if you have other requirements.

You can use:

.. code-block:: bash

   conda search -c k2-fsa-sherpa sherpa

to check all available ``sherpa`` packages for different combinations of
``Python`` and ``PyTorch``. A sample output of the above command is listed below:

.. code-block:: bash

  Loading channels: done
  # Name                       Version           Build  Channel
  sherpa                           0.6 cpu_py3.10_torch1.11.0  k2-fsa-sherpa
  sherpa                           0.6 cpu_py3.10_torch1.12.0  k2-fsa-sherpa
  sherpa                           0.6 cpu_py3.10_torch1.12.1  k2-fsa-sherpa
  sherpa                           0.6 cpu_py3.7_torch1.10.0  k2-fsa-sherpa
  sherpa                           0.6 cpu_py3.7_torch1.10.1  k2-fsa-sherpa
  sherpa                           0.6 cpu_py3.7_torch1.10.2  k2-fsa-sherpa
  sherpa                           0.6 cpu_py3.7_torch1.11.0  k2-fsa-sherpa
  sherpa                           0.6 cpu_py3.7_torch1.12.0  k2-fsa-sherpa
  sherpa                           0.6 cpu_py3.7_torch1.12.1  k2-fsa-sherpa
  sherpa                           0.6 cpu_py3.7_torch1.6.0  k2-fsa-sherpa

To check whether you have installed ``sherpa`` successfully, you can run:

.. code-block:: bash

   sherpa --help

which should show the usage information of ``sherpa``.

To display the information about the environment used to build ``sherpa``, you
can use:

.. code-block:: bash

   sherpa-version

Read :ref:`cpp_non_streaming_asr` to find more.

CUDA version
------------

To be done.

If you have any issues about installing ``sherpa``, please create an issue
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
