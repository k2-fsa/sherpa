Matcha
======

This page lists pre-trained models using `Matcha-TTS <https://arxiv.org/abs/2309.03199>`_.

.. caution::

   Models are from `icefall <https://github.com/k2-fsa/icefall>`_.

   We don't support models from  `<https://github.com/shivammehta25/Matcha-TTS>`_.

.. _matcha-icefall-en_US-ljspeech:

matcha-icefall-en_US-ljspeech (American English, 1 female speaker)
------------------------------------------------------------------

This model is trained using

  `<https://github.com/k2-fsa/icefall/tree/master/egs/ljspeech/TTS#matcha>`_

The dataset used to train the model is from

  `<https://keithito.com/LJ-Speech-Dataset//>`_.

In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-en_US-ljspeech.tar.bz2
  tar xvf matcha-icefall-en_US-ljspeech.tar.bz2
  rm matcha-icefall-en_US-ljspeech.tar.bz2

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos-22khz-univ.onnx

.. caution::

   Remember to also download the vocoder model. We use `vocos-22khz-univ.onnx <https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos-22khz-univ.onnx>`_ in the example.
   You can also select `hifigan_v1 <https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/hifigan_v1.onnx>`_,
   `hifigan_v2 <https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/hifigan_v2.onnx>`_, or
   `hifigan_v3 <https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/hifigan_v3.onnx>`_.

Please check that the file sizes of the pre-trained models are correct. See
the file sizes of ``*.onnx`` files below.

.. code-block:: bash

  ls -lh matcha-icefall-en_US-ljspeech/
  total 144856
  -rw-r--r--    1 fangjun  staff   251B Jan  2 11:05 README.md
  drwxr-xr-x  122 fangjun  staff   3.8K Nov 28  2023 espeak-ng-data
  -rw-r--r--@   1 fangjun  staff    71M Jan  2 04:04 model-steps-3.onnx
  -rw-r--r--    1 fangjun  staff   954B Jan  2 11:05 tokens.txt

  ls -lh vocos-22khz-univ.onnx
  -rw-r--r--  1 fangjun  staff    51M Mar 17 15:28 vocos-22khz-univ.onnx

Generate speech with executables compiled from C++
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-offline-tts \
    --matcha-acoustic-model=./matcha-icefall-en_US-ljspeech/model-steps-3.onnx \
    --matcha-vocoder=./vocos-22khz-univ.onnx \
    --matcha-tokens=./matcha-icefall-en_US-ljspeech/tokens.txt \
    --matcha-data-dir=./matcha-icefall-en_US-ljspeech/espeak-ng-data \
    --num-threads=2 \
    --output-filename=./matcha-ljspeech-0.wav \
    --debug=1 \
    "Today as always, men fall into two groups: slaves and free men. Whoever does not have two-thirds of his day for himself, is a slave, whatever he may be: a statesman, a businessman, an official, or a scholar."

After running, it will generate a file ``matcha-ljspeech-0.wav`` in the
current directory.

.. code-block:: bash

  soxi ./matcha-ljspeech-0.wav

  Input File     : './matcha-ljspeech-0.wav'
  Channels       : 1
  Sample Rate    : 22050
  Precision      : 16-bit
  Duration       : 00:00:15.06 = 332032 samples ~ 1129.36 CDDA sectors
  File Size      : 664k
  Bit Rate       : 353k
  Sample Encoding: 16-bit Signed Integer PCM

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Text</th>
    </tr>
    <tr>
      <td>matcha-ljspeech-0.wav</td>
      <td>
       <audio title="Generated ./matcha-ljspeech-0.wav" controls="controls">
             <source src="/sherpa/_static/matcha-icefall-en_US-ljspeech/matcha-ljspeech-0.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
    "Today as always, men fall into two groups: slaves and free men. Whoever does not have two-thirds of his day for himself, is a slave, whatever he may be: a statesman, a businessman, an official, or a scholar."
      </td>
    </tr>
  </table>

Generate speech with Python script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  python3 ./python-api-examples/offline-tts.py \
    --matcha-acoustic-model=./matcha-icefall-en_US-ljspeech/model-steps-3.onnx \
    --matcha-vocoder=./vocos-22khz-univ.onnx \
    --matcha-tokens=./matcha-icefall-en_US-ljspeech/tokens.txt \
    --matcha-data-dir=./matcha-icefall-en_US-ljspeech/espeak-ng-data \
    --num-threads=2 \
    --output-filename=./matcha-ljspeech-1.wav \
    --debug=1 \
    "Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone."

.. code-block::

  soxi ./matcha-ljspeech-1.wav

  Input File     : './matcha-ljspeech-1.wav'
  Channels       : 1
  Sample Rate    : 22050
  Precision      : 16-bit
  Duration       : 00:00:07.92 = 174592 samples ~ 593.85 CDDA sectors
  File Size      : 349k
  Bit Rate       : 353k
  Sample Encoding: 16-bit Signed Integer PCM

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Text</th>
    </tr>
    <tr>
      <td>matcha-ljspeech-1.wav</td>
      <td>
       <audio title="Generated ./matcha-ljspeech-1.wav" controls="controls">
             <source src="/sherpa/_static/matcha-icefall-en_US-ljspeech/matcha-ljspeech-1.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
    "Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone."
      </td>
    </tr>
  </table>

RTF on Raspberry Pi 4 Model B Rev 1.5
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We use the following command to test the RTF of this model on Raspberry Pi 4 Model B Rev 1.5:

.. code-block:: bash


   for t in 1 2 3 4; do
    build/bin/sherpa-onnx-offline-tts \
      --num-threads=$t \
      --matcha-acoustic-model=./matcha-icefall-en_US-ljspeech/model-steps-3.onnx \
      --matcha-vocoder=./vocos-22khz-univ.onnx \
      --matcha-tokens=./matcha-icefall-en_US-ljspeech/tokens.txt \
      --matcha-data-dir=./matcha-icefall-en_US-ljspeech/espeak-ng-data \
      --output-filename=./matcha-ljspeech-1.wav \
      "Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone."
   done

The results are given below:

  +-------------+-------+-------+-------+-------+
  | num_threads | 1     | 2     | 3     | 4     |
  +=============+=======+=======+=======+=======+
  | RTF         | 0.941 | 0.561 | 0.451 | 0.411 |
  +-------------+-------+-------+-------+-------+


.. _matcha-icefall-zh-baker:

matcha-icefall-zh-baker (Chinese, 1 female speaker)
---------------------------------------------------

This model is trained using

  `<https://github.com/k2-fsa/icefall/tree/master/egs/baker_zh/TTS#matcha>`_

The dataset used to train the model is from

  `<https://en.data-baker.com/datasets/freeDatasets/>`_.

.. caution::

   The dataset is for ``non-commercial`` use only.

In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-zh-baker.tar.bz2
  tar xvf matcha-icefall-zh-baker.tar.bz2
  rm matcha-icefall-zh-baker.tar.bz2

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos-22khz-univ.onnx

.. caution::

   Remember to also download the vocoder model. We use `vocos-22khz-univ.onnx <https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos-22khz-univ.onnx>`_ in the example.
   You can also select `hifigan_v1 <https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/hifigan_v1.onnx>`_,
   `hifigan_v2 <https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/hifigan_v2.onnx>`_, or
   `hifigan_v3 <https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/hifigan_v3.onnx>`_.

Please check that the file sizes of the pre-trained models are correct. See
the file sizes of ``*.onnx`` files below.

.. code-block:: bash

  ls -lh matcha-icefall-zh-baker/
  total 167344
  -rw-r--r--  1 fangjun  staff   370B Dec 31 14:51 README.md
  -rw-r--r--  1 fangjun  staff    58K Dec 31 14:51 date.fst
  drwxr-xr-x  9 fangjun  staff   288B Apr 19  2024 dict
  -rw-r--r--  1 fangjun  staff   1.3M Dec 31 14:51 lexicon.txt
  -rw-r--r--  1 fangjun  staff    72M Dec 31 14:51 model-steps-3.onnx
  -rw-r--r--  1 fangjun  staff    63K Dec 31 14:51 number.fst
  -rw-r--r--  1 fangjun  staff    87K Dec 31 14:51 phone.fst
  -rw-r--r--  1 fangjun  staff    19K Dec 31 14:51 tokens.txt

  ls -lh vocos-22khz-univ.onnx
  -rw-r--r--  1 fangjun  staff    51M Mar 17 15:28 vocos-22khz-univ.onnx

Generate speech with executables compiled from C++
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-offline-tts \
    --matcha-acoustic-model=./matcha-icefall-zh-baker/model-steps-3.onnx \
    --matcha-vocoder=./vocos-22khz-univ.onnx \
    --matcha-lexicon=./matcha-icefall-zh-baker/lexicon.txt \
    --matcha-tokens=./matcha-icefall-zh-baker/tokens.txt \
    --matcha-dict-dir=./matcha-icefall-zh-baker/dict \
    --num-threads=2 \
    --output-filename=./matcha-baker-0.wav \
    --debug=1 \
    "当夜幕降临，星光点点，伴随着微风拂面，我在静谧中感受着时光的流转，思念如涟漪荡漾，梦境如画卷展开，我与自然融为一体，沉静在这片宁静的美丽之中，感受着生命的奇迹与温柔."

  ./build/bin/sherpa-onnx-offline-tts \
     --matcha-acoustic-model=./matcha-icefall-zh-baker/model-steps-3.onnx \
     --matcha-vocoder=./vocos-22khz-univ.onnx \
     --matcha-lexicon=./matcha-icefall-zh-baker/lexicon.txt \
     --matcha-tokens=./matcha-icefall-zh-baker/tokens.txt \
     --tts-rule-fsts=./matcha-icefall-zh-baker/phone.fst,./matcha-icefall-zh-baker/date.fst,./matcha-icefall-zh-baker/number.fst \
     --matcha-dict-dir=./matcha-icefall-zh-baker/dict \
     --output-filename=./matcha-baker-1.wav \
     "某某银行的副行长和一些行政领导表示，他们去过长江和长白山; 经济不断增长。2024年12月31号，拨打110或者18920240511。123456块钱。"

After running, it will generate two files, ``matcha-baker-0.wav`` and
``matcha-baker-1.wav``, in the current directory.

.. code-block:: bash

  soxi matcha-baker-*.wav

  Input File     : 'matcha-baker-0.wav'
  Channels       : 1
  Sample Rate    : 22050
  Precision      : 16-bit
  Duration       : 00:00:22.65 = 499456 samples ~ 1698.83 CDDA sectors
  File Size      : 999k
  Bit Rate       : 353k
  Sample Encoding: 16-bit Signed Integer PCM


  Input File     : 'matcha-baker-1.wav'
  Channels       : 1
  Sample Rate    : 22050
  Precision      : 16-bit
  Duration       : 00:00:22.65 = 499456 samples ~ 1698.83 CDDA sectors
  File Size      : 999k
  Bit Rate       : 353k
  Sample Encoding: 16-bit Signed Integer PCM

  Total Duration of 2 files: 00:00:45.30

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Text</th>
    </tr>
    <tr>
      <td>matcha-baker-0.wav</td>
      <td>
       <audio title="Generated ./matcha-baker-0.wav" controls="controls">
             <source src="/sherpa/_static/matcha-icefall-baker-zh/matcha-baker-0.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
    "当夜幕降临，星光点点，伴随着微风拂面，我在静谧中感受着时光的流转，思念如涟漪荡漾，梦境如画卷展开，我与自然融为一体，沉静在这片宁静的美丽之中，感受着生命的奇迹与温柔."
      </td>
    </tr>

    <tr>
      <td>matcha-baker-1.wav</td>
      <td>
       <audio title="Generated ./matcha-baker-1.wav" controls="controls">
             <source src="/sherpa/_static/matcha-icefall-baker-zh/matcha-baker-1.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
     "某某银行的副行长和一些行政领导表示，他们去过长江和长白山; 经济不断增长。2024年12月31号，拨打110或者18920240511。123456块钱。"
      </td>
    </tr>
  </table>

Generate speech with Python script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  python3 ./python-api-examples/offline-tts.py \
   --matcha-acoustic-model=./matcha-icefall-zh-baker/model-steps-3.onnx \
   --matcha-vocoder=./vocos-22khz-univ.onnx \
   --matcha-lexicon=./matcha-icefall-zh-baker/lexicon.txt \
   --matcha-tokens=./matcha-icefall-zh-baker/tokens.txt \
   --tts-rule-fsts=./matcha-icefall-zh-baker/phone.fst,./matcha-icefall-zh-baker/date.fst,./matcha-icefall-zh-baker/number.fst \
   --matcha-dict-dir=./matcha-icefall-zh-baker/dict \
   --output-filename=./matcha-baker-2.wav \
   --debug=1 \
   "三百六十行，行行出状元。你行的！明天就是 2025年1月1号啦！银行卡被卡住了，你帮个忙，行不行？"

After running, it will generate a file ``matcha-baker-zh-2.wav`` in the current directory.

.. code-block:: bash

  soxi matcha-baker-2.wav

  Input File     : 'matcha-baker-2.wav'
  Channels       : 1
  Sample Rate    : 22050
  Precision      : 16-bit
  Duration       : 00:00:12.71 = 280320 samples ~ 953.469 CDDA sectors
  File Size      : 561k
  Bit Rate       : 353k
  Sample Encoding: 16-bit Signed Integer PCM

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Text</th>
    </tr>
    <tr>
      <td>matcha-baker-2.wav</td>
      <td>
       <audio title="Generated ./matcha-baker-2.wav" controls="controls">
             <source src="/sherpa/_static/matcha-icefall-baker-zh/matcha-baker-2.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
        "三百六十行，行行出状元。你行的！明天就是 2025年1月1号啦！银行卡被卡住了，你帮个忙，行不行？"
      </td>
    </tr>
  </table>

RTF on Raspberry Pi 4 Model B Rev 1.5
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We use the following command to test the RTF of this model on Raspberry Pi 4 Model B Rev 1.5:

.. code-block:: bash


   for t in 1 2 3 4; do
    build/bin/sherpa-onnx-offline-tts \
      --num-threads=$t \
      --matcha-acoustic-model=./matcha-icefall-zh-baker/model-steps-3.onnx \
      --matcha-vocoder=./vocos-22khz-univ.onnx \
      --matcha-lexicon=./matcha-icefall-zh-baker/lexicon.txt \
      --matcha-tokens=./matcha-icefall-zh-baker/tokens.txt \
      --matcha-dict-dir=./matcha-icefall-zh-baker/dict \
      --output-filename=./matcha-baker-0.wav \
      "当夜幕降临，星光点点，伴随着微风拂面，我在静谧中感受着时光的流转，思念如涟漪荡漾，梦境如画卷展开，我与自然融为一体，沉静在这片宁静的美丽之中，感受 着生命的奇迹与温柔."
   done

The results are given below:

  +-------------+-------+-------+-------+-------+
  | num_threads | 1     | 2     | 3     | 4     |
  +=============+=======+=======+=======+=======+
  | RTF         | 0.892 | 0.536 | 0.432 | 0.391 |
  +-------------+-------+-------+-------+-------+

