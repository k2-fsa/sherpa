Perf Analyzer
=============

We can use perf_analyzer provided by Triton to test the performance of the service.

Generate Input Data from Audio Files
-------------------------------------

For offline ASR server:

.. code-block:: bash

  cd sherpa/triton/client
  # en
  python3 generate_perf_input.py --audio_file=test_wavs/1089-134686-0001.wav
  # zh
  python3 generate_perf_input.py --audio_file=test_wavs/zh/mid.wav


It will generate a ``offline_input.json`` file ``sherpa/triton/client``.

For streaming ASR server, you need to add a ``--streaming`` option:

.. code-block:: bash
   
   python3 generate_perf_input.py --audio_file=test_wavs/1089-134686-0001.wav --streaming

A ``online_input.json`` file would be generated.

Test Throughput using Perf Analyzer
------------------------------------

.. code-block:: bash

  # Offline ASR Test with grpc 
  perf_analyzer -m transducer -b 1 -a -p 20000 --concurrency-range 100:200:50 -i gRPC --input-data=offline_input.json  -u localhost:8001

  # Streaming ASR Test with grpc
  perf_analyzer -m transducer -b 1 -a -p 20000 --concurrency-range 100:200:50 -i gRPC --input-data=online_input.json  -u localhost:8001 --streaming


You could save the below results with a ``-f log.txt`` option.

+--------------+--------------------+--------------+---------------------------+---------------+-----------------------+-----------------------+------------------------+--------------+--------------+--------------+--------------+--------------+
| Concurrency  | Inferences/Second  | Client Send  | Network+Server Send/Recv  | Server Queue  | Server Compute Input  | Server Compute Infer  | Server Compute Output  | Client Recv  | p50 latency  | p90 latency  | p95 latency  | p99 latency  |
+==============+====================+==============+===========================+===============+=======================+=======================+========================+==============+==============+==============+==============+==============+
| 300          | 226.24             | 109          | 230434                    | 1             | 9314                  | 1068792               | 14512                  | 1            | 1254206      | 1616224      | 1958246      | 3551406      |
+--------------+--------------------+--------------+---------------------------+---------------+-----------------------+-----------------------+------------------------+--------------+--------------+--------------+--------------+--------------+


.. note::

   Please refer to
   `<https://github.com/triton-inference-server/server/blob/main/docs/user_guide/perf_analyzer.md>`_
   for advanced usuage.
