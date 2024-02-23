.. _install_sherpa_from_pre_compiled_wheels:

From pre-compiled wheels
========================

.. note::

   This method supports only Linux and macOS for now. If you want to
   use Windows, please refer to :ref:`install_sherpa_from_source`.

You can find a list of pre-compiled wheels at the following URLs:

  - CPU: `<https://k2-fsa.github.io/sherpa/cpu.html>`_
  - CUDA: `<https://k2-fsa.github.io/sherpa/cuda.html>`_

In the following, we demonstrate how to install ``k2-fsa/sherpa`` from
pre-compiled wheels.

Linux (CPU)
-----------

Suppose that we want to install the following wheel

.. code-block:: bash

   https://huggingface.co/csukuangfj/kaldifeat/resolve/main/ubuntu-cpu/k2_sherpa-1.3.dev20230725+cpu.torch2.0.1-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

we can use the following methods:

.. code-block:: bash

   # Before installing k2-fsa/sherpa, we have to install the following dependencies:
   #  torch, k2, and kaldifeat

   pip install torch==2.0.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
   pip install k2==1.24.4.dev20231220+cpu.torch2.0.1 -f https://k2-fsa.github.io/k2/cpu.html
   pip install kaldifeat==1.25.3.dev20231221+cpu.torch2.0.1 -f https://csukuangfj.github.io/kaldifeat/cpu.html

   # Now we can install k2-fsa/sherpa
   pip install k2_sherpa==1.3.dev20230725+cpu.torch2.0.1 -f https://k2-fsa.github.io/sherpa/cpu.html

Please see :ref:`check_sherpa_installation`.

macOS (CPU)
-----------

Suppose that we want to install the following wheel

.. code-block:: bash

   https://huggingface.co/csukuangfj/kaldifeat/resolve/main/macos/k2_sherpa-1.3.dev20230725+cpu.torch2.0.1-cp311-cp311-macosx_10_9_x86_64.whl

we can use the following methods:

.. code-block:: bash

   # Before installing k2-fsa/sherpa, we have to install the following dependencies:
   #  torch, k2, and kaldifeat

   pip install torch==2.0.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
   pip install k2==1.24.4.dev20231220+cpu.torch2.0.1 -f https://k2-fsa.github.io/k2/cpu.html
   pip install kaldifeat==1.25.3.dev20231221+cpu.torch2.0.1 -f https://csukuangfj.github.io/kaldifeat/cpu.html

   # Now we can install k2-fsa/sherpa
   pip install k2_sherpa==1.3.dev20230725+cpu.torch2.0.1 -f https://k2-fsa.github.io/sherpa/cpu.html

Please see :ref:`check_sherpa_installation`.

Linux (CUDA)
------------

Suppose that we want to install the following wheel

.. code-block:: bash

   https://huggingface.co/csukuangfj/kaldifeat/resolve/main/ubuntu-cuda/k2_sherpa-1.3.dev20230725+cuda11.7.torch2.0.1-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

we can use the following methods:

.. code-block:: bash

   # Before installing k2-fsa/sherpa, we have to install the following dependencies:
   #  torch, k2, and kaldifeat

   pip install torch==2.0.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
   pip install k2==1.24.4.dev20231220+cuda11.7.torch2.0.1 -f https://k2-fsa.github.io/k2/cuda.html
   pip install kaldifeat==1.25.3.dev20231221+cuda11.7.torch2.0.1  -f https://csukuangfj.github.io/kaldifeat/cuda.html


   # Now we can install k2-fsa/sherpa
   pip install k2_sherpa==1.3.dev20230725+cuda11.7.torch2.0.1 -f https://k2-fsa.github.io/sherpa/cuda.html

Please see :ref:`check_sherpa_installation`.
