.. _lstm_server_english:

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
   ./sherpa/bin/lstm_transducer_stateless/streaming_server.py --help

shows the usage message.

You need the following files to start the server:

  1. The neural network model
  2. The BPE model ``bpe.model``.

The neural network model has three parts, the encoder, the decoder, and
the joiner, which are all exported using ``torch.jit.trace``.

The above two files can be obtained after training your model
with `<https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/lstm_transducer_stateless>`_.

If you don't want to train a model by yourself, you can try the
pretrained model: `<https://huggingface.co/Zengwei/icefall-asr-librispeech-lstm-transducer-stateless-2022-08-18>`_


.. hint::

   You can find pretrained models in ``RESULTS.md`` for all the recipes in
   `icefall <https://github.com/k2-fsa/icefall>`_.

   For instance, the pretrained models for the LibriSpeech dataset can be
   found at `<https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/RESULTS.md>`_.

The following shows you how to start the server with the above pretrained model.

.. code-block:: bash

    cd /path/to/sherpa

    git lfs install
    git clone https://huggingface.co/Zengwei/icefall-asr-librispeech-lstm-transducer-stateless-2022-08-18

    ./sherpa/bin/lstm_transducer_stateless/streaming_server.py \
      --endpoint.rule3.min-utterance-length 1000.0 \
      --port 6007 \
      --max-batch-size 50 \
      --max-wait-ms 5 \
      --nn-pool-size 1 \
      --nn-encoder-filename ./icefall-asr-librispeech-lstm-transducer-stateless-2022-08-18/exp/encoder_jit_trace.pt \
      --nn-decoder-filename ./icefall-asr-librispeech-lstm-transducer-stateless-2022-08-18/exp/decoder_jit_trace.pt \
      --nn-joiner-filename ./icefall-asr-librispeech-lstm-transducer-stateless-2022-08-18/exp/joiner_jit_trace.pt \
      --bpe-model-filename ./icefall-asr-librispeech-lstm-transducer-stateless-2022-08-18/data/lang_bpe_500/bpe.model

That's it!

Now you can start the :ref:`lstm_client_english`, record your voice in real-time,
and check the recognition results from the server.

.. hint::

   You can also try the following pretrained model trained using `GigaSpeech`_
   and `LibriSpeech`_ and has a lower WER than the above one:

   .. code-block:: bash

        git clone https://huggingface.co/csukuangfj/icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03

        nn_encoder_filename=./icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03/exp/encoder_jit_trace-iter-468000-avg-16.pt
        nn_decoder_filename=./icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03/exp/decoder_jit_trace-iter-468000-avg-16.pt
        nn_joiner_filename=./icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03/exp/joiner_jit_trace-iter-468000-avg-16.pt

        bpe_model_filename=./icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03/data/lang_bpe_500/bpe.model

        ./sherpa/bin/lstm_transducer_stateless/streaming_server.py \
          --endpoint.rule1.must-contain-nonsilence=false \
          --endpoint.rule1.min-trailing-silence=5.0 \
          --endpoint.rule2.min-trailing-silence=2.0 \
          --endpoint.rule3.min-utterance-length=50.0 \
          --port 6006 \
          --decoding-method greedy_search \
          --max-batch-size 50 \
          --max-wait-ms 5 \
          --nn-pool-size 1 \
          --max-active-connections 10 \
          --nn-encoder-filename $nn_encoder_filename \
          --nn-decoder-filename $nn_decoder_filename \
          --nn-joiner-filename $nn_joiner_filename \
          --bpe-model-filename $bpe_model_filename
