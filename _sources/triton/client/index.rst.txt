Triton-client
==============

Send requests using client
--------------------------------------------------------------------
In the docker container, run the client script to do ASR inference:

.. code-block:: bash

  cd sherpa/triton/client
  # Test one audio using offline ASR
  python3 client.py --audio_file=./test_wavs/1089-134686-0001.wav --url=localhost:8001

  # Test one audio using streaming ASR
  python3 client.py --audio_file=./test_wavs/1089-134686-0001.wav --url=localhost:8001 --streaming


The above command sends a single audio ``1089-134686-0001.wav`` to the server and get the result. ``--url`` option specifies the IP and port of the server, 
in this example, we set the server and client on the same machine, therefore IP is ``localhost``, and we use port ``8001`` since it is the default port for gRPC in Triton. 

You can also test a bunch of audios together with the client. Just specify the path of ``wav.scp`` with ``--wavscp`` option, 
set the path of test set directory with ``--data_dir`` option, and set the path of ground-truth transcript file with ``--trans`` option, 
the client will infer all the audios in test set and calculate the WER upon the test set.

Decode manifests
------------------
You could also decode a whole dataset to benchmark metrics e.g. RTF, WER.

.. caution::
  Decode manifests in simulation streaming mode would be supported in the future.

.. code-block:: bash

    cd sherpa/triton/client

    # For aishell manifests:
    git lfs install
    git clone https://huggingface.co/csukuangfj/aishell-test-dev-manifests
    sudo mkdir -p /root/fangjun/open-source/icefall-aishell/egs/aishell/ASR/download/aishell
    tar xf ./aishell-test-dev-manifests/data_aishell.tar.gz -C /root/fangjun/open-source/icefall-aishell/egs/aishell/ASR/download/aishell/
    # dev set: ./aishell-test-dev-manifests/data/fbank/aishell_cuts_test.jsonl.gz 
    # test set: ./aishell-test-dev-manifests/data/fbank/aishell_cuts_test.jsonl.gz 

    python3 decode_manifest_triton.py \
       --server-addr localhost \
       --num-tasks 300 \
       --log-interval 20 \
       --model-name transducer \
       --manifest-filename ./aishell-test-dev-manifests/data/fbank/aishell_cuts_test.jsonl.gz \
       --compute-cer
