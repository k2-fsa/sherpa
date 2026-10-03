Installation
============

We prepare a dockerfile based on official triton docker containers. The customized dockerfile intergrates `Triton-server`_, `Triton-client`_ and 
sherpa-related requirements into a single image. You need to install `Docker`_ first before starting installation.

.. hint::

  For your production environment, you could build triton manually to reduce the size of container.

Build Triton Image 
-------------------------------------

.. code-block:: bash

  git clone https://github.com/k2-fsa/sherpa
  cd sherpa/triton
  docker build . -f Dockerfile/Dockerfile.server -t sherpa_triton_server:latest

.. note::
   It may take a lot of time since we build k2 from source. If you only need to use greedy search scorer, you could comment k2-related lines. 

Launch a inference container
-----------------------------

.. code-block:: bash

  docker run --gpus all --name sherpa_server --net host --shm-size=1g -it sherpa_triton_server:latest

Now, you should enter into the container successfully.