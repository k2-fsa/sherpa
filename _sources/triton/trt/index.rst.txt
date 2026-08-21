TensorRT acceleration
=====================

This page explains how to use the TensorRT engine to accelerate inference for
K2 models.

Preparation
-----------

First, install TensorRT. We recommend using a Docker container for TRT. Run:

.. code-block:: bash

   docker run --gpus '"device=0"' -it --rm --net host -v $PWD/:/k2 nvcr.io/nvidia/tensorrt:22.12-py3


You can also see `the official TensorRT instructions <https://github.com/NVIDIA/TensorRT#build>`_
to build TRT on your machine.

.. note::

   Use TensorRT version ``8.5.3`` or newer.


If your TRT version is earlier than ``8.5.3``, download a newer TensorRT
release and run the following command inside the Docker container to use it:


.. code-block:: bash

   # inside the container
   bash tools/install.sh



Model export
------------

You have to prepare the ONNX model by referring
`here <https://github.com/k2-fsa/sherpa/tree/master/triton#prepare-pretrained-models>`_ to export your models into ONNX format.
Assume you have put your ONNX model in the ``$model_dir`` directory.
Then, just run the command:

.. code-block:: bash

   bash tools/build.sh $model_dir
   cp $model_dir/encoder.trt model_repo_offline_fast_beam_trt/encoder/1


The generated TRT model will be saved into ``$model_dir/encoder.trt``.
We also give an example of ``model_repo`` of TRT model. You can follow the same procedure as described
`here <https://github.com/k2-fsa/sherpa/tree/master/triton#deploy-on-triton-inference-server>`_ to deploy the pipeline using triton.


Benchmark for Conformer TRT encoder vs ONNX
-------------------------------------------

+-------+------------+-----------------+--------+
| Model | Batch size | Avg latency(ms) | QPS    |
+=======+============+=================+========+
| ONNX  | 1          | 7.44            | 134.48 |
+-------+------------+-----------------+--------+
|       | 8          | 14.92           | 536.09 |
+-------+------------+-----------------+--------+
|       | 16         | 22.84           | 700.67 |
+-------+------------+-----------------+--------+
|       | 32         | 41.62           | 768.84 |
+-------+------------+-----------------+--------+
|       | 64         | 80.48           | 795.27 |
+-------+------------+-----------------+--------+
|       | 128        | 171.97          | 744.32 |
+-------+------------+-----------------+--------+
| TRT   | 1          | 5.21834         | 193.93 |
+-------+------------+-----------------+--------+
|       | 8          | 11.7826         | 703.49 |
+-------+------------+-----------------+--------+
|       | 16         | 20.4444         | 815.79 |
+-------+------------+-----------------+--------+
|       | 32         | 37.583          | 893.56 |
+-------+------------+-----------------+--------+
|       | 64         | 69.8312         | 965.40 |
+-------+------------+-----------------+--------+
|       | 128        | 139.702         | 964.57 |
+-------+------------+-----------------+--------+
