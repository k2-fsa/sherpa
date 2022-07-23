
.. _conformer_rnnt_server_chinese:

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
with `<https://github.com/k2-fsa/icefall/tree/master/egs/wenetspeech/ASR/pruned_transducer_stateless5>`_.

If you don't want to train a model by yourself, you can try the
pretrained model: `<https://huggingface.co/luomingshuang/icefall_asr_wenetspeech_pruned_transducer_stateless5_streaming>`_

.. hint::

   You can find pretrained models in ``RESULTS.md`` for all the recipes in
   `icefall <https://github.com/k2-fsa/icefall>`_.

   For instance, the pretrained models for the WenetSpeech dataset can be
   found at `<https://github.com/k2-fsa/icefall/blob/master/egs/wenetspeech/ASR/RESULTS.md>`_.

The following shows you how to start the server with the above pretrained model.

.. code-block:: bash

    cd /path/to/sherpa

    git lfs install
    git clone https://huggingface.co/luomingshuang/icefall_asr_wenetspeech_pruned_transducer_stateless5_streaming

    ./sherpa/bin/streaming_pruned_transducer_statelessX/streaming_server.py \
      --port 6006 \
      --max-batch-size 50 \
      --max-wait-ms 5 \
      --nn-pool-size 1 \
      --nn-model-filename ./icefall_asr_wenetspeech_pruned_transducer_stateless5_streaming/exp/cpu_jit_epoch_5_avg_1_torch.1.7.1.pt \
      --token-filename ./icefall_asr_wenetspeech_pruned_transducer_stateless5_streaming/data/lang_char/tokens.txt

That's it!

Now you can start the :ref:`conformer_rnnt_client_chinese`, record your voice in real-time,
and check the recognition results from the server.
