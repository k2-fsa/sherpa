Pretrained model with GigaSpeech
================================

.. hint::

  We assume you have installed ``sherpa`` by following
  :ref:`cpp_fronted_installation` before you start this section.

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
  are two torchscript models exported using ``torch.jit.script()``. We can use
  any of them in the following tests.

.. note::

   We won't use ``pretrained-xxx.pt`` in sherpa.

Before we start, let us generate ``tokens.txt`` from the above ``bpe.model``:

.. code-block:: bash

  cd icefall-asr-gigaspeech-pruned-transducer-stateless2/data/lang_bpe_500
  wget https://raw.githubusercontent.com/k2-fsa/sherpa/master/scripts/bpe_model_to_tokens.py
  ./bpe_model_to_tokens.py ./bpe.model > tokens.txt


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

.. code-block:: bash

  nn_model=./icefall-asr-gigaspeech-pruned-transducer-stateless2/exp/cpu_jit-iter-3488000-avg-15.pt
  tokens=./icefall-asr-gigaspeech-pruned-transducer-stateless2/data/lang_bpe_500/tokens.txt

  wav1=./icefall-asr-gigaspeech-pruned-transducer-stateless2/test_wavs/1089-134686-0001.wav

  sherpa \
    --nn-model=$nn_model \
    --tokens=$tokens \
    --use-gpu=false \
    $wav1


You will see the following output:

.. code-block::

  [I] /usr/share/miniconda/envs/sherpa/conda-bld/sherpa_1661003501349/work/sherpa/csrc/parse_options.cc:495:int sherpa::ParseOptions::Read(int, const char* const*) 2022-08-20 22:35:42 sherpa --nn-model=./icefall-asr-gigaspeech-pruned-transducer-stateless2/exp/cpu_jit-iter-3488000-avg-15.pt --tokens=./icefall-asr-gigaspeech-pruned-transducer-stateless2/data/lang_bpe_500/tokens.txt --use-gpu=false ./icefall-asr-gigaspeech-pruned-transducer-stateless2/test_wavs/1089-134686-0001.wav

  [I] /usr/share/miniconda/envs/sherpa/conda-bld/sherpa_1661003501349/work/sherpa/csrc/sherpa.cc:126:int main(int, char**) 2022-08-20 22:35:42
  --nn-model=./icefall-asr-gigaspeech-pruned-transducer-stateless2/exp/cpu_jit-iter-3488000-avg-15.pt
  --tokens=./icefall-asr-gigaspeech-pruned-transducer-stateless2/data/lang_bpe_500/tokens.txt
  --decoding-method=greedy_search
  --use-gpu=false

  [I] /usr/share/miniconda/envs/sherpa/conda-bld/sherpa_1661003501349/work/sherpa/csrc/sherpa.cc:270:int main(int, char**) 2022-08-20 22:35:43
  filename: ./icefall-asr-gigaspeech-pruned-transducer-stateless2/test_wavs/1089-134686-0001.wav
  result:  AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE THE SQUALID QUARTER OF THE BROTHELS


.. hint::

   You can pass the option ``--use-gpu=true`` to use GPU for computation (Assume
   you have installed a CUDA version of ``sherpa``).

   Also, you can use ``--decoding-method=modified_beam_search`` to change
   the decoding method.

Decode multiple waves in parallel
---------------------------------

.. code-block:: bash

  nn_model=./icefall-asr-gigaspeech-pruned-transducer-stateless2/exp/cpu_jit-iter-3488000-avg-15.pt
  tokens=./icefall-asr-gigaspeech-pruned-transducer-stateless2/data/lang_bpe_500/tokens.txt

  wav1=./icefall-asr-gigaspeech-pruned-transducer-stateless2/test_wavs/1089-134686-0001.wav
  wav2=./icefall-asr-gigaspeech-pruned-transducer-stateless2/test_wavs/1221-135766-0001.wav
  wav3=./icefall-asr-gigaspeech-pruned-transducer-stateless2/test_wavs/1221-135766-0002.wav

  sherpa \
    --nn-model=$nn_model \
    --tokens=$tokens \
    --use-gpu=false \
    $wav1 \
    $wav2 \
    $wav3


You will see the following output:

.. code-block:: bash

  [I] /usr/share/miniconda/envs/sherpa/conda-bld/sherpa_1661003501349/work/sherpa/csrc/parse_options.cc:495:int sherpa::ParseOptions::Read(int, const char* const*) 2022-08-20 22:38:18 sherpa --nn-model=./icefall-asr-gigaspeech-pruned-transducer-stateless2/exp/cpu_jit-iter-3488000-avg-15.pt --tokens=./icefall-asr-gigaspeech-pruned-transducer-stateless2/data/lang_bpe_500/tokens.txt --use-gpu=false ./icefall-asr-gigaspeech-pruned-transducer-stateless2/test_wavs/1089-134686-0001.wav ./icefall-asr-gigaspeech-pruned-transducer-stateless2/test_wavs/1221-135766-0001.wav ./icefall-asr-gigaspeech-pruned-transducer-stateless2/test_wavs/1221-135766-0002.wav

  [I] /usr/share/miniconda/envs/sherpa/conda-bld/sherpa_1661003501349/work/sherpa/csrc/sherpa.cc:126:int main(int, char**) 2022-08-20 22:38:19
  --nn-model=./icefall-asr-gigaspeech-pruned-transducer-stateless2/exp/cpu_jit-iter-3488000-avg-15.pt
  --tokens=./icefall-asr-gigaspeech-pruned-transducer-stateless2/data/lang_bpe_500/tokens.txt
  --decoding-method=greedy_search
  --use-gpu=false

  [I] /usr/share/miniconda/envs/sherpa/conda-bld/sherpa_1661003501349/work/sherpa/csrc/sherpa.cc:284:int main(int, char**) 2022-08-20 22:38:23
  filename: ./icefall-asr-gigaspeech-pruned-transducer-stateless2/test_wavs/1089-134686-0001.wav
  result:  AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE THE SQUALID QUARTER OF THE BROTHELS

  filename: ./icefall-asr-gigaspeech-pruned-transducer-stateless2/test_wavs/1221-135766-0001.wav
  result:  GOD AS A DIRECT CONSEQUENCE OF THE SIN WHICH MAN THUS PUNISHED HAD GIVEN HER A LOVELY CHILD WHOSE PLACE WAS ON THAT SAME DISHONORED BOSOM TO CONNECT HER PARENT FOR EVER WITH THE RACE AND DESCENT OF MORTALS AND TO BE FINALLY A BLESSED SOUL IN HEAVEN

  filename: ./icefall-asr-gigaspeech-pruned-transducer-stateless2/test_wavs/1221-135766-0002.wav
  result:  YET THESE THOUGHTS AFFECTED HESTER PRYNNE LESS WITH HOPE THAN APPREHENSION



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
  tokens=./icefall-asr-gigaspeech-pruned-transducer-stateless2/data/lang_bpe_500/tokens.txt

  sherpa \
    --nn-model=$nn_model \
    --tokens=$tokens \
    --use-gpu=false \
    --use-wav-scp=true \
    scp:wav.scp \
    ark,scp,t:results.ark,results.scp

You will see the following output:

.. code-block:: bash

  [I] /usr/share/miniconda/envs/sherpa/conda-bld/sherpa_1661003501349/work/sherpa/csrc/parse_options.cc:495:int sherpa::ParseOptions::Read(int, const char* const*) 2022-08-20 22:40:36 sherpa --nn-model=./icefall-asr-gigaspeech-pruned-transducer-stateless2/exp/cpu_jit-iter-3488000-avg-15.pt --tokens=./icefall-asr-gigaspeech-pruned-transducer-stateless2/data/lang_bpe_500/tokens.txt --use-gpu=false --use-wav-scp=true scp:wav.scp ark,scp,t:results.ark,results.scp

  [I] /usr/share/miniconda/envs/sherpa/conda-bld/sherpa_1661003501349/work/sherpa/csrc/sherpa.cc:126:int main(int, char**) 2022-08-20 22:40:37
  --nn-model=./icefall-asr-gigaspeech-pruned-transducer-stateless2/exp/cpu_jit-iter-3488000-avg-15.pt
  --tokens=./icefall-asr-gigaspeech-pruned-transducer-stateless2/data/lang_bpe_500/tokens.txt
  --decoding-method=greedy_search
  --use-gpu=false

We can view the recognition results using:

.. code-block:: bash

  $ cat results.ark

  wav1 AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE THE SQUALID QUARTER OF THE BROTHELS
  wav2 GOD AS A DIRECT CONSEQUENCE OF THE SIN WHICH MAN THUS PUNISHED HAD GIVEN HER A LOVELY CHILD WHOSE PLACE WAS ON THAT SAME DISHONORED BOSOM TO CONNECT HER PARENT FOR EVER WITH THE RACE AND DESCENT OF MORTALS AND TO BE FINALLY A BLESSED SOUL IN HEAVEN
  wav3 YET THESE THOUGHTS AFFECTED HESTER PRYNNE LESS WITH HOPE THAN APPREHENSION

.. hint::

   You can pass the option ``--batch-size=20`` to control the batch size to be 20
   during decoding.

Decode feats.scp
----------------

If you have precomputed feats, you can decode it with the following code:

.. code-block:: bash

  nn_model=./icefall-asr-gigaspeech-pruned-transducer-stateless2/exp/cpu_jit-iter-3488000-avg-15.pt
  tokens=./icefall-asr-gigaspeech-pruned-transducer-stateless2/data/lang_bpe_500/tokens.txt

  sherpa \
    --nn-model=$nn_model \
    --tokens=$tokens \
    --use-gpu=false \
    --use-feats-scp=true \
    scp:feats.scp \
    ark,scp,t:results.ark,results.scp

.. hint::

   You can pass the option ``--batch-size=20`` to control the batch size to be 20
   during decoding.

.. caution::

   ``feats.scp`` generated by kaldi's ``compute-fbank-feats`` is using
   unnormalized samples. That is, audio samples are in the range
   ``[-32768, 32767]``. However, models from `icefall`_ are trained with
   features using normalized samples, i.e., samples in the range ``[-1, 1]``.

   You cannot use ``feats.scp`` generated by Kaldi's ``compute-fbank-feats``
   to test models trained from icefall using normalized audio samples.
   Otherwise, you won't get good recognition results.

   It is perfectly OK to decode ``feats.scp`` from Kaldi using a model
   trained with features using unnormalized audio samples.

.. note::

   We provide a script to generate ``feats.ark`` and ``feats.scp`` from
   ``wav.scp`` that can be used with models trained by icefall. Please see
   `<https://github.com/k2-fsa/sherpa/blob/master/.github/scripts/generate_feats_scp.py>`_
