Pretrained model with GigaSpeech
================================

.. hint::

  We assume you have read :ref:`cpp_installation` and have compiled `sherpa`_.

Download the pretrained model
-----------------------------

.. code-block:: bash

   sudo apt-get install git-lfs
   git lfs install
   git clone https://huggingface.co/wgb14/icefall-asr-gigaspeech-pruned-transducer-stateless2

.. hint::

   You can find the training script by visiting
   `<https://github.com/k2-fsa/icefall/blob/master/egs/gigaspeech/ASR/RESULTS.md#gigaspeech-bpe-training-results-pruned-transducer-2>`_

   The torchscript model is exported using the script
   `<https://github.com/k2-fsa/icefall/blob/master/egs/gigaspeech/ASR/pruned_transducer_stateless2/export.py>`_

.. caution::

   You have to use `git lfs <https://git-lfs.github.com/>`_ to download/clone the repo.
   Otherwise, you will be SAD later.

After cloning the repo, you will find the following files:

.. code-block::

  icefall-asr-gigaspeech-pruned-transducer-stateless2/
  |-- README.md
  |-- data
  |   `-- lang_bpe_500
  |       `-- bpe.model
  |-- exp
  |   |-- cpu_jit-iter-3488000-avg-15.pt
  |   |-- cpu_jit-iter-3488000-avg-20.pt
  |   |-- pretrained-iter-3488000-avg-15.pt
  |   `-- pretrained-iter-3488000-avg-20.pt

- ``data/lang_bpe_500/bpe.model`` is the BPE model used in the training
- ``exp/cpu_jit-iter-3488000-avg-15.pt`` and ``exp/cpu_jit-iter-3488000-avg-20.pt``
  are two torchscript model exported using ``torch.jit.script()``. We can use
  any of them in the following tests.

.. note::

   We won't use ``pretrained-xxx.pt`` in sherpa.

Since the above repo does not contain test waves, we download some
test files from `<https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless5-2022-05-13>`_.
for testing.

.. code-block:: bash

   cd icefall-asr-gigaspeech-pruned-transducer-stateless2
   mkdir test_wavs
   cd test_wavs

   wget https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless5-2022-05-13/resolve/main/test_wavs/1089-134686-0001.wav

   wget https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless5-2022-05-13/resolve/main/test_wavs/1221-135766-0001.wav

   wget https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless5-2022-05-13/resolve/main/test_wavs/1221-135766-0002.wav

In the following, we show you how to use the downloaded model for speech
recognition.

Decode a single wave
--------------------

We assume you have placed the downloaded repo inside ``sherpa/build``.

.. code-block:: bash

    cd sherpa/build

    nn_model=./icefall-asr-gigaspeech-pruned-transducer-stateless2/exp/cpu_jit-iter-3488000-avg-15.pt
    bpe_model=./icefall-asr-gigaspeech-pruned-transducer-stateless2/data/lang_bpe_500/bpe.model
    wav1=./icefall-asr-gigaspeech-pruned-transducer-stateless2/test_wavs/1089-134686-0001.wav

    ./bin/sherpa \
      --decoding-method=greedy_search \
      --nn-model=$nn_model \
      --bpe-model=$bpe_model \
      $wav1

You will see the following output:

.. code-block::

  [I] /root/fangjun/open-source/sherpa/sherpa/csrc/parse_options.cc:495:int sherpa::ParseOptions::Read(int, const char* const*) 2022-08-16 22:18:03 ./bin/sherpa --decoding-method=greedy_search --nn-model=./icefall-asr-gigaspeech-pruned-transducer-stateless2/exp/cpu_jit-iter-3488000-avg-15.pt --bpe-model=./icefall-asr-gigaspeech-pruned-transducer-stateless2/data/lang_bpe_500/bpe.model ./icefall-asr-gigaspeech-pruned-transducer-stateless2/test_wavs/1089-134686-0001.wav

  [I] /root/fangjun/open-source/sherpa/sherpa/csrc/sherpa.cc:113:int main(int, char**) 2022-08-16 22:18:04
  --nn-model=./icefall-asr-gigaspeech-pruned-transducer-stateless2/exp/cpu_jit-iter-3488000-avg-15.pt
  --bpe-model=./icefall-asr-gigaspeech-pruned-transducer-stateless2/data/lang_bpe_500/bpe.model
  --decoding-method=greedy_search
  --use-gpu=false

  [I] /root/fangjun/open-source/sherpa/sherpa/csrc/sherpa.cc:262:int main(int, char**) 2022-08-16 22:18:05
  filename: ./icefall-asr-gigaspeech-pruned-transducer-stateless2/test_wavs/1089-134686-0001.wav
  result: AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE THE SQUALID QUARTER OF THE BROTHELS

.. hint::

   You can pass the option ``--use-gpu=true`` to use GPU for computation.
   Also, you can use ``--decoding-method=modified_beam_search`` to change
   the decoding method.

Decode multiple waves in parallel
---------------------------------

.. code-block:: bash

  cd sherpa/build

  nn_model=./icefall-asr-gigaspeech-pruned-transducer-stateless2/exp/cpu_jit-iter-3488000-avg-15.pt
  bpe_model=./icefall-asr-gigaspeech-pruned-transducer-stateless2/data/lang_bpe_500/bpe.model
  wav1=./icefall-asr-gigaspeech-pruned-transducer-stateless2/test_wavs/1089-134686-0001.wav
  wav2=./icefall-asr-gigaspeech-pruned-transducer-stateless2/test_wavs/1221-135766-0001.wav
  wav3=./icefall-asr-gigaspeech-pruned-transducer-stateless2/test_wavs/1221-135766-0002.wav

  ./bin/sherpa \
    --decoding-method=greedy_search \
    --nn-model=$nn_model \
    --bpe-model=$bpe_model \
    $wav1 \
    $wav2 \
    $wav3

You will see the following output:

.. code-block::

  [I] /root/fangjun/open-source/sherpa/sherpa/csrc/parse_options.cc:495:int sherpa::ParseOptions::Read(int, const char* const*) 2022-08-16 22:24:09 ./bin/sherpa --decoding-method=greedy_search --nn-model=./icefall-asr-gigaspeech-pruned-transducer-stateless2/exp/cpu_jit-iter-3488000-avg-15.pt --bpe-model=./icefall-asr-gigaspeech-pruned-transducer-stateless2/data/lang_bpe_500/bpe.model ./icefall-asr-gigaspeech-pruned-transducer-stateless2/test_wavs/1089-134686-0001.wav ./icefall-asr-gigaspeech-pruned-transducer-stateless2/test_wavs/1221-135766-0001.wav ./icefall-asr-gigaspeech-pruned-transducer-stateless2/test_wavs/1221-135766-0002.wav

  [I] /root/fangjun/open-source/sherpa/sherpa/csrc/sherpa.cc:113:int main(int, char**) 2022-08-16 22:24:10
  --nn-model=./icefall-asr-gigaspeech-pruned-transducer-stateless2/exp/cpu_jit-iter-3488000-avg-15.pt
  --bpe-model=./icefall-asr-gigaspeech-pruned-transducer-stateless2/data/lang_bpe_500/bpe.model
  --decoding-method=greedy_search
  --use-gpu=false

  [I] /root/fangjun/open-source/sherpa/sherpa/csrc/sherpa.cc:276:int main(int, char**) 2022-08-16 22:24:14
  filename: ./icefall-asr-gigaspeech-pruned-transducer-stateless2/test_wavs/1089-134686-0001.wav
  result: AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE THE SQUALID QUARTER OF THE BROTHELS

  filename: ./icefall-asr-gigaspeech-pruned-transducer-stateless2/test_wavs/1221-135766-0001.wav
  result: GOD AS A DIRECT CONSEQUENCE OF THE SIN WHICH MAN THUS PUNISHED HAD GIVEN HER A LOVELY CHILD WHOSE PLACE WAS ON THAT SAME DISHONORED BOSOM TO CONNECT HER PARENT FOR EVER WITH THE RACE AND DESCENT OF MORTALS AND TO BE FINALLY A BLESSED SOUL IN HEAVEN

  filename: ./icefall-asr-gigaspeech-pruned-transducer-stateless2/test_wavs/1221-135766-0002.wav
  result: YET THESE THOUGHTS AFFECTED HESTER PRYNNE LESS WITH HOPE THAN APPREHENSION

Decode wav.scp
--------------

If you have some experience with `Kaldi`_, you must know what ``wav.scp`` is.

We use the following code to generate ``wav.scp`` for our test data.

.. code-block:: bash

  cat > wav.scp <<EOF
  wav1 ./icefall-asr-gigaspeech-pruned-transducer-stateless2/test_wavs/1089-134686-0001.wav
  wav2 ./icefall-asr-gigaspeech-pruned-transducer-stateless2/test_wavs/1221-135766-0001.wav
  wav3 ./icefall-asr-gigaspeech-pruned-transducer-stateless2/test_wavs/1221-135766-0002.wav
  EOF

With the ``wav.scp`` ready, we can decode it with the following commands:

.. code-block:: bash

  nn_model=./icefall-asr-gigaspeech-pruned-transducer-stateless2/exp/cpu_jit-iter-3488000-avg-15.pt
  bpe_model=./icefall-asr-gigaspeech-pruned-transducer-stateless2/data/lang_bpe_500/bpe.model

  ./bin/sherpa \
    --decoding-method=greedy_search \
    --nn-model=$nn_model \
    --bpe-model=$bpe_model \
    --use-wav-scp=true \
    --batch-size=2 \
    scp:wav.scp \
    ark,scp,t:results.ark,results.scp

You will see the following output:

.. code-block:: bash

  [I] /root/fangjun/open-source/sherpa/sherpa/csrc/parse_options.cc:495:int sherpa::ParseOptions::Read(int, const char* const*) 2022-08-16 22:30:16 ./bin/sherpa --decoding-method=greedy_search --nn-model=./icefall-asr-gigaspeech-pruned-transducer-stateless2/exp/cpu_jit-iter-3488000-avg-15.pt --bpe-model=./icefall-asr-gigaspeech-pruned-transducer-stateless2/data/lang_bpe_500/bpe.model --use-wav-scp=true --batch-size=2 scp:wav.scp ark,scp,t:results.ark,results.scp

  [I] /root/fangjun/open-source/sherpa/sherpa/csrc/sherpa.cc:113:int main(int, char**) 2022-08-16 22:30:16
  --nn-model=./icefall-asr-gigaspeech-pruned-transducer-stateless2/exp/cpu_jit-iter-3488000-avg-15.pt
  --bpe-model=./icefall-asr-gigaspeech-pruned-transducer-stateless2/data/lang_bpe_500/bpe.model
  --decoding-method=greedy_search
  --use-gpu=false

We can view the recognition results using:

.. code-block:: bash

  $ cat results.ark

  wav1 AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE THE SQUALID QUARTER OF THE BROTHELS
  wav2 GOD AS A DIRECT CONSEQUENCE OF THE SIN WHICH MAN THUS PUNISHED HAD GIVEN HER A LOVELY CHILD WHOSE PLACE WAS ON THAT SAME DISHONORED BOSOM TO CONNECT HER PARENT FOR EVER WITH THE RACE AND DESCENT OF MORTALS AND TO BE FINALLY A BLESSED SOUL IN HEAVEN
  wav3 YET THESE THOUGHTS AFFECTED HESTER PRYNNE LESS WITH HOPE THAN APPREHENSION

Decode feats.scp
----------------

If you have precomputed feats, you can decode it with the following code:

.. code-block:: bash

  nn_model=./icefall-asr-gigaspeech-pruned-transducer-stateless2/exp/cpu_jit-iter-3488000-avg-15.pt
  bpe_model=./icefall-asr-gigaspeech-pruned-transducer-stateless2/data/lang_bpe_500/bpe.model

  ./bin/sherpa \
    --decoding-method=greedy_search \
    --nn-model=$nn_model \
    --bpe-model=$bpe_model \
    --use-feats-scp=true \
    --batch-size=2 \
    scp:feats.scp \
    ark,scp,t:results.ark,results.scp

.. caution:: bash

   ``feats.scp`` generated by kaldi's ``compute-fbank-feats`` are using
   unnormalized samples. That is, audio samples are in the range
   ``[-32768, 32767]``. However, models from `icefall`_ are trained with
   features using normalized samples, i.e., samples in the range ``[-1, 1]``.

   You cannot use ``feats.scp`` generated by Kaldi's ``compute-fbank-feats``
   to test models trained from icefall using normalized audio samples.
   Otherwise, you won't get good recognition results.

   It is perfectly OK to decode ``feats.scp`` from Kaldi using a model
   trained with features using unnormalized audio samples.
