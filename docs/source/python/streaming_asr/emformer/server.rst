
.. _emformer_server:

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
   ./sherpa/bin/pruned_stateless_emformer_rnnt2/streaming_server.py --help

shows the usage message.

You need two files to start the server:

  1. The neural network model, which is a torchscript file.
  2. The BPE model.

The above two files can be obtained after training your model
with `<https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/pruned_stateless_emformer_rnnt2>`_.

If you don't want to train a model by yourself, you can try the
pretrained model: `<https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-stateless-emformer-rnnt2-2022-06-01>`_

.. hint::

   You can find pretrained models in ``RESULTS.md`` for all the recipes in
   `icefall <https://github.com/k2-fsa/icefall>`_.

   For instance, the pretrained models for the LibriSpeech dataset can be
   found at `<https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/RESULTS.md>`_.

The following shows you how to start the server with the above pretrained model.

.. code-block:: bash

    cd /path/to/sherpa

    git lfs install
    git clone https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-stateless-emformer-rnnt2-2022-06-01

    ./sherpa/bin/pruned_stateless_emformer_rnnt2/streaming_server.py \
      --port 6007 \
      --max-batch-size 50 \
      --max-wait-ms 5 \
      --nn-pool-size 1 \
      --nn-model-filename ./icefall-asr-librispeech-pruned-stateless-emformer-rnnt2-2022-06-01/exp/cpu_jit-epoch-39-avg-6-use-averaged-model-1.pt \
      --bpe-model-filename ./icefall-asr-librispeech-pruned-stateless-emformer-rnnt2-2022-06-01/data/lang_bpe_500/bpe.model

That's it!

Now you can start the :ref:`emformer_client`, record your voice in real-time,
and check the recognition results from the server.
