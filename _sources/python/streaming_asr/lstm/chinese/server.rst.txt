.. _lstm_server_chinese:

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
  2. The ``tokens.txt``.

The neural network model has three parts, the encoder, the decoder, and
the joiner, which are all exported using ``torch.jit.trace``.

The above files can be obtained after training your model
with `<https://github.com/k2-fsa/icefall/tree/master/egs/wenetspeech/ASR/lstm_transducer_stateless>`_.

If you don't want to train a model by yourself, you can try the
pretrained model: `<https://huggingface.co/csukuangfj/icefall-asr-wenetspeech-lstm-transducer-stateless-2022-09-19>`_

The following shows you how to start the server with the above pretrained model.

.. code-block:: bash

    cd /path/to/sherpa

    git lfs install
    git clone https://huggingface.co/csukuangfj/icefall-asr-wenetspeech-lstm-transducer-stateless-2022-09-19

    ./sherpa/bin/lstm_transducer_stateless/streaming_server.py \
      --endpoint.rule3.min-utterance-length 1000.0 \
      --port 6007 \
      --max-batch-size 50 \
      --max-wait-ms 5 \
      --nn-pool-size 1 \
      --nn-encoder-filename ./icefall-asr-wenetspeech-lstm-transducer-stateless-2022-09-19/exp/encoder_jit_trace-iter-420000-avg-10.pt \
      --nn-decoder-filename ./icefall-asr-wenetspeech-lstm-transducer-stateless-2022-09-19/exp/decoder_jit_trace-iter-420000-avg-10.pt \
      --nn-joiner-filename ./icefall-asr-wenetspeech-lstm-transducer-stateless-2022-09-19/exp/joiner_jit_trace-iter-420000-avg-10.pt \
      --token-filename ./icefall-asr-wenetspeech-lstm-transducer-stateless-2022-09-19/data/lang_char/tokens.txt

That's it!

Now you can start the :ref:`lstm_client_chinese`, record your voice in real-time,
and check the recognition results from the server.

.. warning::

   The above pretrained model has been trained only for 6 epochs. We will
   update it in the following days.
