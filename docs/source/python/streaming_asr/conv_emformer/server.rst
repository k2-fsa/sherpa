
.. _conv_emformer_server:

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
   ./sherpa/bin/conv_emformer_transducer_stateless2/streaming_server.py --help

shows the usage message.

You need two files to start the server:

  1. The neural network model, which is a torchscript file.
  2. The BPE model.

The above two files can be obtained after training your model
with `<https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/conv_emformer_transducer_stateless2>`_.

If you don't want to train a model by yourself, you can try the
pretrained model: `<https://huggingface.co/Zengwei/icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05>`_

.. hint::

   You can find pretrained models in ``RESULTS.md`` for all the recipes in
   `icefall <https://github.com/k2-fsa/icefall>`_.

   For instance, the pretrained models for the LibriSpeech dataset can be
   found at `<https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/RESULTS.md>`_.

The following shows you how to start the server with the above pretrained model.

.. code-block:: bash

    cd /path/to/sherpa

    git lfs install
    git clone https://huggingface.co/Zengwei/icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05

    ./sherpa/bin/conv_emformer_transducer_stateless2/streaming_server.py \
      --endpoint.rule3.min-utterance-length 1000.0 \
      --port 6007 \
      --max-batch-size 50 \
      --max-wait-ms 5 \
      --nn-pool-size 1 \
      --nn-model-filename ./icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05/exp/cpu-jit-epoch-30-avg-10-torch-1.10.0.pt \
      --bpe-model-filename ./icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05/data/lang_bpe_500/bpe.model

That's it!

Now you can start the :ref:`conv_emformer_client`, record your voice in real-time,
and check the recognition results from the server.
