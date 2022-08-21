Pretrained model with WenetSpeech
=================================

.. hint::

  We assume you have installed ``sherpa`` by following
  :ref:`cpp_fronted_installation` before you start this section.

Download the pretrained model
-----------------------------

.. code-block:: bash

   sudo apt-get install git-lfs
   git lfs install
   git clone https://huggingface.co/luomingshuang/icefall_asr_wenetspeech_pruned_transducer_stateless2

.. hint::

   You can find the training script by visiting
   `<https://github.com/k2-fsa/icefall/blob/master/egs/wenetspeech/ASR/RESULTS.md#wenetspeech-char-based-training-results-pruned-transducer-2>`_

   The torchscript model is exported using the script
   `<https://github.com/k2-fsa/icefall/blob/master/egs/wenetspeech/ASR/pruned_transducer_stateless2/export.py>`_

.. caution::

   You have to use `git lfs <https://git-lfs.github.com/>`_ to download/clone the repo.
   Otherwise, you will be SAD later.

After cloning the repo, you will find the following files:

.. code-block::

  icefall_asr_wenetspeech_pruned_transducer_stateless2/
  |-- data
  |   `-- lang_char
  |       |-- tokens.txt
  |-- exp
  |   |-- cpu_jit_epoch_10_avg_2_torch_1.11.0.pt
  |   |-- cpu_jit_epoch_10_avg_2_torch_1.7.1.pt
  `-- test_wavs
      |-- DEV_T0000000000.wav
      |-- DEV_T0000000001.wav
      |-- DEV_T0000000002.wav

We will use ``data/lang_char/tokens.txt`` and ``exp/cpu_jit_epoch_10_avg_2_torch_1.7.1.pt``
below.

Decode a single wave
--------------------

.. code-block:: bash

  nn_model=./icefall_asr_wenetspeech_pruned_transducer_stateless2/exp/cpu_jit_epoch_10_avg_2_torch_1.7.1.pt
  tokens=./icefall_asr_wenetspeech_pruned_transducer_stateless2/data/lang_char/tokens.txt

  wav1=./icefall_asr_wenetspeech_pruned_transducer_stateless2/test_wavs/DEV_T0000000000.wav

  sherpa \
    --nn-model=$nn_model \
    --tokens=$tokens \
    --use-gpu=false \
    --decoding-method=greedy_search \
    $wav1

You will see the following output:

.. code-block::

  [I] /usr/share/miniconda/envs/sherpa/conda-bld/sherpa_1661003501349/work/sherpa/csrc/parse_options.cc:495:int sherpa::ParseOptions::Read(int, const char* const*) 2022-08-20 23:06:09 sherpa --nn-model=./icefall_asr_wenetspeech_pruned_transducer_stateless2/exp/cpu_jit_epoch_10_avg_2_torch_1.7.1.pt --tokens=./icefall_asr_wenetspeech_pruned_transducer_stateless2/data/lang_char/tokens.txt --use-gpu=false --decoding-method=greedy_search ./icefall_asr_wenetspeech_pruned_transducer_stateless2/test_wavs/DEV_T0000000000.wav

  [I] /usr/share/miniconda/envs/sherpa/conda-bld/sherpa_1661003501349/work/sherpa/csrc/sherpa.cc:126:int main(int, char**) 2022-08-20 23:06:10
  --nn-model=./icefall_asr_wenetspeech_pruned_transducer_stateless2/exp/cpu_jit_epoch_10_avg_2_torch_1.7.1.pt
  --tokens=./icefall_asr_wenetspeech_pruned_transducer_stateless2/data/lang_char/tokens.txt
  --decoding-method=greedy_search
  --use-gpu=false

  [I] /usr/share/miniconda/envs/sherpa/conda-bld/sherpa_1661003501349/work/sherpa/csrc/sherpa.cc:270:int main(int, char**) 2022-08-20 23:06:11
  filename: ./icefall_asr_wenetspeech_pruned_transducer_stateless2/test_wavs/DEV_T0000000000.wav
  result: 对我做了介绍那么我想说的是呢大家如果对我的研究感兴趣呢

.. hint::

   You can pass the option ``--use-gpu=true`` to use GPU for computation (Assume
   you have installed a CUDA version of ``sherpa``).

   Also, you can use ``--decoding-method=modified_beam_search`` to change
   the decoding method.

Decode multiple waves in parallel
---------------------------------

.. code-block:: bash

  nn_model=./icefall_asr_wenetspeech_pruned_transducer_stateless2/exp/cpu_jit_epoch_10_avg_2_torch_1.7.1.pt
  tokens=./icefall_asr_wenetspeech_pruned_transducer_stateless2/data/lang_char/tokens.txt

  wav1=./icefall_asr_wenetspeech_pruned_transducer_stateless2/test_wavs/DEV_T0000000000.wav
  wav2=./icefall_asr_wenetspeech_pruned_transducer_stateless2/test_wavs/DEV_T0000000001.wav
  wav3=./icefall_asr_wenetspeech_pruned_transducer_stateless2/test_wavs/DEV_T0000000002.wav

  sherpa \
    --nn-model=$nn_model \
    --tokens=$tokens \
    --use-gpu=false \
    --decoding-method=greedy_search \
    $wav1 \
    $wav2 \
    $wav3

You will see the following output:

.. code-block:: bash

  [I] /usr/share/miniconda/envs/sherpa/conda-bld/sherpa_1661003501349/work/sherpa/csrc/parse_options.cc:495:int sherpa::ParseOptions::Read(int, const char* const*) 2022-08-20 23:07:05 sherpa --nn-model=./icefall_asr_wenetspeech_pruned_transducer_stateless2/exp/cpu_jit_epoch_10_avg_2_torch_1.7.1.pt --tokens=./icefall_asr_wenetspeech_pruned_transducer_stateless2/data/lang_char/tokens.txt --use-gpu=false --decoding-method=greedy_search ./icefall_asr_wenetspeech_pruned_transducer_stateless2/test_wavs/DEV_T0000000000.wav ./icefall_asr_wenetspeech_pruned_transducer_stateless2/test_wavs/DEV_T0000000001.wav ./icefall_asr_wenetspeech_pruned_transducer_stateless2/test_wavs/DEV_T0000000002.wav

  [I] /usr/share/miniconda/envs/sherpa/conda-bld/sherpa_1661003501349/work/sherpa/csrc/sherpa.cc:126:int main(int, char**) 2022-08-20 23:07:06
  --nn-model=./icefall_asr_wenetspeech_pruned_transducer_stateless2/exp/cpu_jit_epoch_10_avg_2_torch_1.7.1.pt
  --tokens=./icefall_asr_wenetspeech_pruned_transducer_stateless2/data/lang_char/tokens.txt
  --decoding-method=greedy_search
  --use-gpu=false

  [I] /usr/share/miniconda/envs/sherpa/conda-bld/sherpa_1661003501349/work/sherpa/csrc/sherpa.cc:284:int main(int, char**) 2022-08-20 23:07:07
  filename: ./icefall_asr_wenetspeech_pruned_transducer_stateless2/test_wavs/DEV_T0000000000.wav
  result: 对我做了介绍那么我想说的是呢大家如果对我的研究感兴趣呢

  filename: ./icefall_asr_wenetspeech_pruned_transducer_stateless2/test_wavs/DEV_T0000000001.wav
  result: 重点想谈三个问题首先呢就是这一轮全球金融动荡的表现

  filename: ./icefall_asr_wenetspeech_pruned_transducer_stateless2/test_wavs/DEV_T0000000002.wav
  result: 深入地分析这一次全球金融动荡背后的根源

Decode wav.scp
--------------

If you have some experience with `Kaldi`_, you must know what ``wav.scp`` is.

We use the following code to generate ``wav.scp`` for our test data.

.. code-block:: bash

  cat > wav2.scp <<EOF
  wav0 ./icefall_asr_wenetspeech_pruned_transducer_stateless2/test_wavs/DEV_T0000000000.wav
  wav1 ./icefall_asr_wenetspeech_pruned_transducer_stateless2/test_wavs/DEV_T0000000001.wav
  wav2 ./icefall_asr_wenetspeech_pruned_transducer_stateless2/test_wavs/DEV_T0000000002.wav
  EOF

With the ``wav.scp`` ready, we can decode it with the following commands:

.. code-block:: bash

  nn_model=./icefall_asr_wenetspeech_pruned_transducer_stateless2/exp/cpu_jit_epoch_10_avg_2_torch_1.7.1.pt
  tokens=./icefall_asr_wenetspeech_pruned_transducer_stateless2/data/lang_char/tokens.txt

  sherpa \
    --nn-model=$nn_model \
    --tokens=$tokens \
    --use-gpu=false \
    --decoding-method=greedy_search \
    --use-wav-scp=true \
    scp:wav.scp \
    ark,scp,t:results.ark,results.scp

You will see the following output:

.. code-block:: bash

  [I] /usr/share/miniconda/envs/sherpa/conda-bld/sherpa_1661003501349/work/sherpa/csrc/parse_options.cc:495:int sherpa::ParseOptions::Read(int, const char* const*) 2022-08-20 23:10:01 sherpa --nn-model=./icefall_asr_wenetspeech_pruned_transducer_stateless2/exp/cpu_jit_epoch_10_avg_2_torch_1.7.1.pt --tokens=./icefall_asr_wenetspeech_pruned_transducer_stateless2/data/lang_char/tokens.txt --use-gpu=false --decoding-method=greedy_search --use-wav-scp=true scp:wav.scp ark,scp,t:results.ark,results.scp

  [I] /usr/share/miniconda/envs/sherpa/conda-bld/sherpa_1661003501349/work/sherpa/csrc/sherpa.cc:126:int main(int, char**) 2022-08-20 23:10:02
  --nn-model=./icefall_asr_wenetspeech_pruned_transducer_stateless2/exp/cpu_jit_epoch_10_avg_2_torch_1.7.1.pt
  --tokens=./icefall_asr_wenetspeech_pruned_transducer_stateless2/data/lang_char/tokens.txt
  --decoding-method=greedy_search
  --use-gpu=false

We can view the recognition results using:

.. code-block:: bash

  $ cat results.ark

  wav0 对我做了介绍那么我想说的是呢大家如果对我的研究感兴趣呢
  wav1 重点想谈三个问题首先呢就是这一轮全球金融动荡的表现
  wav2 深入地分析这一次全球金融动荡背后的根源

.. hint::

   You can pass the option ``--batch-size=20`` to control the batch size to be 20
   during decoding.

Decode feats.scp
----------------

If you have precomputed feats, you can decode it with the following code:

.. code-block:: bash

  nn_model=./icefall_asr_wenetspeech_pruned_transducer_stateless2/exp/cpu_jit_epoch_10_avg_2_torch_1.7.1.pt
  tokens=./icefall_asr_wenetspeech_pruned_transducer_stateless2/data/lang_char/tokens.txt

  sherpa \
    --nn-model=$nn_model \
    --tokens=$tokens \
    --use-gpu=false \
    --decoding-method=greedy_search \
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
