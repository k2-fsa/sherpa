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


.. literalinclude:: ./log/offline_perf.txt
   :caption: You could save the below results with a ``-f log.txt`` option.

.. note::

   Please refer to
   `<https://github.com/triton-inference-server/server/blob/main/docs/user_guide/perf_analyzer.md>`_
   for advanced usuage.
