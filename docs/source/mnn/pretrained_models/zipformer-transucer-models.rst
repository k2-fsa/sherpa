Zipformer-transducer-based Models
=================================

.. hint::

   Please refer to :ref:`install_sherpa_mnn` to install `sherpa-mnn`_
   before you read this section.

.. _sherpa_mnn_streaming_zipformer_small_bilingual_zh_en_2023_02_16:

meixu/sherpa-mnn-streaming-zipformer-small-bilingual-zh-en-2023-02-16 (Bilingual, Chinese + English)
----------------------------------------------------------------------------------------------------------

This model is converted from

`<https://huggingface.co/pfluo/k2fsa-zipformer-bilingual-zh-en-t>`_ (backup link:
`<https://huggingface.co/meixu/k2fsa-zipformer-bilingual-zh-en-t>`_)

which supports both Chinese and English. The model is contributed by the community
and is trained on tens of thousands of some internal dataset.

In the following, we describe how to download it and use it with `sherpa-mnn`_.

Download the model
~~~~~~~~~~~~~~~~~~


Please use the following commands to download it.

.. code-block:: bash

  cd /path/to/sherpa-mnn

  git clone https://huggingface.co/meixu/k2fsa-zipformer-bilingual-zh-en-t


Please check that the file sizes of the pre-trained models are correct.

Decode a single wave file
~~~~~~~~~~~~~~~~~~~~~~~~~

.. hint::

   It supports decoding only wave files with a single channel and the sampling rate
   should be 16 kHz.

.. code-block:: bash

  cd /path/to/sherpa-mnn

  for method in greedy_search modified_beam_search; do
    ./build/bin/sherpa-mnn \
      --tokens=k2fsa-zipformer-bilingual-zh-en-t/data/lang_char_bpe/tokens.txt \
      --encoder=k2fsa-zipformer-bilingual-zh-en-t/exp/encoder-epoch-99-avg-1.b1.mnn \
      --decoder=k2fsa-zipformer-bilingual-zh-en-t/exp/decoder-epoch-99-avg-1.b1.mnn \
      --joiner=k2fsa-zipformer-bilingual-zh-en-t/exp/joiner-epoch-99-avg-1.b1.mnn \
      --num-threads=2 \
      --decode-method=$method \
      k2fsa-zipformer-bilingual-zh-en-t/test_wavs/46.wav
  done

You should see the following output:

.. literalinclude:: ./code-zipformer/sherpa-mnn-streaming-zipformer-small-bilingual-zh-en-2023-02-16.txt

Real-time speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-mnn

  ./build/bin/sherpa-mnn-microphone \
    --tokens=k2fsa-zipformer-bilingual-zh-en-t/data/lang_char_bpe/tokens.txt \
    --encoder=k2fsa-zipformer-bilingual-zh-en-t/exp/encoder-epoch-99-avg-1.b1.mnn \
    --decoder=k2fsa-zipformer-bilingual-zh-en-t/exp/decoder-epoch-99-avg-1.b1.mnn \
    --joiner=k2fsa-zipformer-bilingual-zh-en-t/exp/joiner-epoch-99-avg-1.b1.mnn \
    --num-threads=2 \
    --decode-method=greedy_search

.. hint::

   If your system is Linux (including embedded Linux), you can also use
   :ref:`sherpa-mnn-alsa` to do real-time speech recognition with your
   microphone if ``sherpa-mnn-microphone`` does not work for you.

Decode a single wave file with a language model
~~~~~~~~~~~~~~~~~~~~~~~~~

.. hint::

   sherpa-mnn supports streaming decoding with a RNNLM, which could be trained
   and exported with icefall.

Please use the following commands to download a rnnlm.

.. code-block:: bash

  cd /path/to/sherpa-mnn

  git clone https://huggingface.co/meixu/streaming_lm_example_20231016

The decode command is as follows:

.. code-block:: bash

  cd /path/to/sherpa-mnn

  ./build/bin/sherpa-mnn \
    --tokens=k2fsa-zipformer-bilingual-zh-en-t/data/lang_char_bpe/tokens.txt \
    --encoder=k2fsa-zipformer-bilingual-zh-en-t/exp/encoder-epoch-99-avg-1.b1.mnn \
    --decoder=k2fsa-zipformer-bilingual-zh-en-t/exp/decoder-epoch-99-avg-1.b1.mnn \
    --joiner=k2fsa-zipformer-bilingual-zh-en-t/exp/joiner-epoch-99-avg-1.b1.mnn \
    --num-threads=2 \
    --lm-num-threads=2 \
    --decode-method=modified_beam_search_lm_shallow_fusion \
    --lm=streaming_lm_example_20231016/rnnlm/rnnlm_64_1.mnn \
    --lm-hidden-size=64 \
    --lm-num-layers=1 \
    k2fsa-zipformer-bilingual-zh-en-t/test_wavs/46.wav

You should see the following output:

.. literalinclude:: ./code-zipformer/sherpa-mnn-streaming-zipformer-small-bilingual-zh-en-2023-02-16-rnnlm.txt
