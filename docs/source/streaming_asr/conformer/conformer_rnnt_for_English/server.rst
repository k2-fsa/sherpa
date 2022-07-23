
.. _conformer_rnnt_server_english:

Server
======

.. hint::

   Please first refer to :ref:`installation` to install `sherpa`_
   before proceeding.

The server is responsible for accepting audio samples from the client,
decoding it, and sending the recognition results back to the client.


Usage
-----

.. code-block::

   cd /path/to/sherpa
   ./sherpa/bin/streaming_pruned_transducer_statelessX/streaming_server.py --help

shows the usage message.

You need two files to start the server:

  1. The neural network model, which is a torchscript file.
  2. The BPE model or the tokens file.

The above two files can be obtained after training your model
with `<https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/pruned_transducer_stateless4>`_.

If you don't want to train a model by yourself, you can try the
pretrained model: `<https://huggingface.co/pkufool/icefall_librispeech_streaming_pruned_transducer_stateless4_20220625>`_

.. hint::

   You can find pretrained models in ``RESULTS.md`` for all the recipes in
   `icefall <https://github.com/k2-fsa/icefall>`_.

   For instance, the pretrained models for the LibriSpeech dataset can be
   found at `<https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/RESULTS.md>`_.

The following shows you how to start the server with the above pretrained model.

.. code-block:: bash

    cd /path/to/sherpa

    git lfs install
    git clone https://huggingface.co/pkufool/icefall_librispeech_streaming_pruned_transducer_stateless4_20220625

    ./sherpa/bin/streaming_pruned_transducer_statelessX/streaming_server.py \
      --port 6006 \
      --max-batch-size 50 \
      --max-wait-ms 5 \
      --nn-pool-size 1 \
      --nn-model-filename ./icefall_librispeech_streaming_pruned_transducer_stateless4_20220625/exp/cpu_jit-epoch-25-avg-3.pt \
      --bpe-model-filename ./icefall_librispeech_streaming_pruned_transducer_stateless4_20220625/data/lang_bpe_500/bpe.model

That's it!

Now you can start the :ref:`conformer_rnnt_client_english`, record your voice in real-time,
and check the recognition results from the server.
