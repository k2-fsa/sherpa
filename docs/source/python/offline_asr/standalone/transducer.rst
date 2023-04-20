Transducer
==========

In this section, we describe how to use pre-trained `transducer`_
models for offline (i.e., non-streaming) speech recognition.

.. hint::

  Please refer to :ref:`offline_transducer_pretrained_models` for a list of
  available pre-trained `transducer`_ models to download.

.. hint::

   We have a colab notebook for this section: |sherpa python offline transducer standalone recognizer colab notebook|

    .. |sherpa python offline transducer standalone recognizer colab notebook| image:: https://colab.research.google.com/assets/colab-badge.svg
     :target: https://github.com/k2-fsa/colab/blob/master/sherpa/sherpa_standalone_offline_transducer_speech_recognition.ipynb

   You can find the following in the above colab notebook:

    - how to setup the environment
    - how to download pre-trained models
    - how to use sherpa for speech recognition

   If you don't have access to Google, please find below the colab notebook
   in our GitHub repo. For instance, the above colab notebook can be found
   at `<https://github.com/k2-fsa/colab/blob/master/sherpa/sherpa_standalone_offline_transducer_speech_recognition.ipynb>`_

.. note::

   Please visit `<https://github.com/k2-fsa/colab>`_ for a list of available
   colab notebooks about the next-gen Kaldi project.

In the following, we use the pre-trained model
:ref:`icefall-asr-librispeech-pruned-transducer-stateless8-2022-12-02`
to demonstrate how to decode sound files.

.. caution::

   Make sure you have installed `sherpa`_ before you continue.

   Please refer to :ref:`install_sherpa_from_source` to install `sherpa`_
   from source.

Download the pre-trained model
------------------------------

Please refer to :ref:`icefall-asr-librispeech-pruned-transducer-stateless8-2022-12-02`
for detailed instructions.

For ease of reference, we duplicate the download commands below:

.. code-block:: bash

  cd /path/to/sherpa

  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/WeijiZhuang/icefall-asr-librispeech-pruned-transducer-stateless8-2022-12-02
  cd icefall-asr-librispeech-pruned-transducer-stateless8-2022-12-02
  git lfs pull --include "exp/cpu_jit-torch-1.10.pt"
  git lfs pull --include "data/lang_bpe_500/LG.pt"

In the following, we describe different decoding methods.

greedy search
-------------

.. code-block:: bash

    cd /path/to/sherpa

    python3 ./sherpa/bin/offline_transducer_asr.py \
      --nn-model ./icefall-asr-librispeech-pruned-transducer-stateless8-2022-12-02/exp/cpu_jit-torch-1.10.pt \
      --tokens ./icefall-asr-librispeech-pruned-transducer-stateless8-2022-12-02/data/lang_bpe_500/tokens.txt \
      --decoding-method greedy_search \
      --num-active-paths 4 \
      --use-gpu false \
      ./icefall-asr-librispeech-pruned-transducer-stateless8-2022-12-02/test_wavs/1089-134686-0001.wav \
      ./icefall-asr-librispeech-pruned-transducer-stateless8-2022-12-02/test_wavs/1221-135766-0001.wav \
      ./icefall-asr-librispeech-pruned-transducer-stateless8-2022-12-02/test_wavs/1221-135766-0002.wav

The output of the above command is given below:

.. literalinclude:: ./log/transducer-greedy-search.txt
   :caption: Output of greedy search

modified beam search
--------------------

.. code-block:: bash

    cd /path/to/sherpa

    python3 ./sherpa/bin/offline_transducer_asr.py \
      --nn-model ./icefall-asr-librispeech-pruned-transducer-stateless8-2022-12-02/exp/cpu_jit-torch-1.10.pt \
      --tokens ./icefall-asr-librispeech-pruned-transducer-stateless8-2022-12-02/data/lang_bpe_500/tokens.txt \
      --decoding-method modified_beam_search \
      --num-active-paths 4 \
      --use-gpu false \
      ./icefall-asr-librispeech-pruned-transducer-stateless8-2022-12-02/test_wavs/1089-134686-0001.wav \
      ./icefall-asr-librispeech-pruned-transducer-stateless8-2022-12-02/test_wavs/1221-135766-0001.wav \
      ./icefall-asr-librispeech-pruned-transducer-stateless8-2022-12-02/test_wavs/1221-135766-0002.wav

The output of the above command is given below:

.. literalinclude:: ./log/transducer-modified-beam-search.txt
   :caption: Output of modified beam search

fast beam search (without LG)
-----------------------------

.. code-block:: bash

    cd /path/to/sherpa

    python3 ./sherpa/bin/offline_transducer_asr.py \
      --nn-model ./icefall-asr-librispeech-pruned-transducer-stateless8-2022-12-02/exp/cpu_jit-torch-1.10.pt \
      --tokens ./icefall-asr-librispeech-pruned-transducer-stateless8-2022-12-02/data/lang_bpe_500/tokens.txt \
      --decoding-method fast_beam_search \
      --max-contexts 8 \
      --max-states 64 \
      --allow-partial true \
      --beam 4 \
      --use-gpu false \
      ./icefall-asr-librispeech-pruned-transducer-stateless8-2022-12-02/test_wavs/1089-134686-0001.wav \
      ./icefall-asr-librispeech-pruned-transducer-stateless8-2022-12-02/test_wavs/1221-135766-0001.wav \
      ./icefall-asr-librispeech-pruned-transducer-stateless8-2022-12-02/test_wavs/1221-135766-0002.wav

The output of the above command is given below:

.. literalinclude:: ./log/transducer-fast-beam-search.txt
   :caption: Output of fast beam search (without LG)

fast beam search (with LG)
--------------------------

.. code-block:: bash

    cd /path/to/sherpa

    python3 ./sherpa/bin/offline_transducer_asr.py \
      --nn-model ./icefall-asr-librispeech-pruned-transducer-stateless8-2022-12-02/exp/cpu_jit-torch-1.10.pt \
      --tokens ./icefall-asr-librispeech-pruned-transducer-stateless8-2022-12-02/data/lang_bpe_500/tokens.txt \
      --decoding-method fast_beam_search \
      --max-contexts 8 \
      --max-states 64 \
      --allow-partial true \
      --beam 4 \
      --LG ./icefall-asr-librispeech-pruned-transducer-stateless8-2022-12-02/data/lang_bpe_500/LG.pt \
      --ngram-lm-scale 0.01 \
      --use-gpu false \
      ./icefall-asr-librispeech-pruned-transducer-stateless8-2022-12-02/test_wavs/1089-134686-0001.wav \
      ./icefall-asr-librispeech-pruned-transducer-stateless8-2022-12-02/test_wavs/1221-135766-0001.wav \
      ./icefall-asr-librispeech-pruned-transducer-stateless8-2022-12-02/test_wavs/1221-135766-0002.wav

The output of the above command is given below:

.. literalinclude:: ./log/transducer-fast-beam-search-with-LG.txt
   :caption: Output of fast beam search (with LG)
