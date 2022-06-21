Conformer transducer based non-streaming ASR
============================================

This page describes how to use `sherpa`_ for non-streaming ASR based
on `Conformer`_ `transducer`_.

We use pre-trained models using the following datasets for demonstration:

  - `aishell`_
  - `LibriSpeech`_

`aishell`_ is a Chinese dataset and its pre-trained model uses
Chinese characters as modeling units; its vocabulary size is 4336.

`LibriSpeech`_ is an English dataset; its pre-trained model uses `BPE`_
as modeling units with vocabulary size 500.

For the demo of each dataset below, we describe the usage of the ``server``
as well as the ``client``. You can also find pre-trained models provided by us
in each demo so that you can play with it without any training.

.. toctree::
   :maxdepth: 2

   aishell
   librispeech

