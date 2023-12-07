vits
====

This page lists pre-trained `vits`_ models.

All models in a single table
-----------------------------

The following table summarizes the information of all models in this page.

.. note::

   Since there are more than ``100`` pre-trained models for over ``40`` languages,
   we don't list all of them on this page. Please find them at
   `<https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models>`_.

   You can try all the models at the following huggingface space.
   `<https://huggingface.co/spaces/k2-fsa/text-to-speech>`_.


.. hint::

   You can find Android APKs for each model at the following page

   `<https://k2-fsa.github.io/sherpa/onnx/tts/apk.html>`_

.. list-table::

 * - Model
   - Language
   - # Speakers
   - Dataset
   - Model filesize (MB)
   - Sample rate (Hz)
 * - :ref:`vits-model-aishell3`
   - Chinese
   - 174
   - `aishell3`_
   - 116
   - 8000
 * - :ref:`vits-zh-hf-fanchen-C`
   - Chinese
   - 1
   - N/A
   - 116
   - 16000
 * - :ref:`vits-zh-hf-fanchen-wnj`
   - Chinese
   - 1
   - N/A
   - 116
   - 16000
 * - :ref:`vits-zh-hf-theresa`
   - Chinese
   - 804
   - N/A
   - 117
   - 22050
 * - :ref:`vits-zh-hf-eula`
   - Chinese
   - 804
   - N/A
   - 117
   - 22050
 * - :ref:`vits-model-vits-ljspeech`
   - English (US)
   - 1 (Female)
   - `LJ Speech`_
   - 109
   - 22050
 * - :ref:`vits-model-vits-vctk`
   - English
   - 109
   - `VCTK`_
   - 116
   - 22050
 * - :ref:`vits-model-en_US-lessac-medium`
   - English (US)
   - 1 (Male)
   - `lessac_blizzard2013`_
   - 61
   - 22050

.. _vits-model-vits-ljspeech:

ljspeech (English, single-speaker)
----------------------------------

This model is converted from `pretrained_ljspeech.pth <https://drive.google.com/file/d/1q86w74Ygw2hNzYP9cWkeClGT5X25PvBT/view?usp=drive_link>`_,
which is trained by the `vits`_ author `Jaehyeon Kim <https://github.com/jaywalnut310>`_ on
the `LJ Speech`_ dataset. It supports only English and is a single-speaker model.

.. note::

   If you are interested in how the model is converted, please see
   `<https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/vits/export-onnx-ljs.py>`_

In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/vits-ljs
  cd vits-ljs
  git lfs pull --include "*.onnx"

Please check that the file sizes of the pre-trained models are correct. See
the file sizes of ``*.onnx`` files below.

.. code-block:: bash

  vits-ljs fangjun$ ls -lh *.onnx
  -rw-r--r--  1 fangjun  staff    36M Oct 16 15:16 vits-ljs.int8.onnx
  -rw-r--r--  1 fangjun  staff   109M Oct 16 15:16 vits-ljs.onnx

Generate speech with executable compiled from C++
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-offline-tts \
    --vits-model=./vits-ljs/vits-ljs.onnx \
    --vits-lexicon=./vits-ljs/lexicon.txt \
    --vits-tokens=./vits-ljs/tokens.txt \
    --output-filename=./liliana.wav \
    'liliana, the most beautiful and lovely assistant of our team!'

After running, it will generate a file ``liliana.wav`` in the current directory.

.. code-block:: bash

  soxi ./liliana.wav

  Input File     : './liliana.wav'
  Channels       : 1
  Sample Rate    : 22050
  Precision      : 16-bit
  Duration       : 00:00:04.39 = 96768 samples ~ 329.143 CDDA sectors
  File Size      : 194k
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
      <td>liliana.wav</td>
      <td>
       <audio title="Generated ./liliana.wav" controls="controls">
             <source src="/sherpa/_static/vits-ljs/liliana.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
        liliana, the most beautiful and lovely assistant of our team!
      </td>
    </tr>
  </table>

Generate speech with Python script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cd /path/to/sherpa-onnx

   python3 ./python-api-examples/offline-tts.py \
    --vits-model=./vits-ljs/vits-ljs.onnx \
    --vits-lexicon=./vits-ljs/lexicon.txt \
    --vits-tokens=./vits-ljs/tokens.txt \
    --output-filename=./armstrong.wav \
    "That's one small step for a man, a giant leap for mankind."

After running, it will generate a file ``armstrong.wav`` in the current directory.

.. code-block:: bash

  soxi ./armstrong.wav

  Input File     : './armstrong.wav'
  Channels       : 1
  Sample Rate    : 22050
  Precision      : 16-bit
  Duration       : 00:00:04.81 = 105984 samples ~ 360.49 CDDA sectors
  File Size      : 212k
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
      <td>armstrong.wav</td>
      <td>
       <audio title="Generated ./armstrong.wav" controls="controls">
             <source src="/sherpa/_static/vits-ljs/armstrong.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
        That's one small step for a man, a giant leap for mankind.
      </td>
    </tr>
  </table>

.. _vits-model-vits-vctk:

VCTK (English, multi-speaker, 109 speakers)
-------------------------------------------

This model is converted from `pretrained_vctk.pth <https://drive.google.com/file/d/11aHOlhnxzjpdWDpsz1vFDCzbeEfoIxru/view?usp=drive_link>`_,
which is trained by the `vits`_ author `Jaehyeon Kim <https://github.com/jaywalnut310>`_ on
the `VCTK`_ dataset. It supports only English and is a multi-speaker model. It contains
109 speakers.

.. note::

   If you are interested in how the model is converted, please see
   `<https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/vits/export-onnx-vctk.py>`_

In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/vits-vctk
  cd vits-ctk
  git lfs pull --include "*.onnx"

Please check that the file sizes of the pre-trained models are correct. See
the file sizes of ``*.onnx`` files below.

.. code-block:: bash

  vits-vctk fangjun$ ls -lh *.onnx
  -rw-r--r--  1 fangjun  staff    37M Oct 16 10:57 vits-vctk.int8.onnx
  -rw-r--r--  1 fangjun  staff   116M Oct 16 10:57 vits-vctk.onnx

Generate speech with executable compiled from C++
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since there are 109 speakers available, we can choose a speaker from 0 to 198.
The default speaker ID is 0.

We use speaker ID 0, 10, and 108 below to generate audio for the same text.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-offline-tts \
    --vits-model=./vits-vctk/vits-vctk.onnx \
    --vits-lexicon=./vits-vctk/lexicon.txt \
    --vits-tokens=./vits-vctk/tokens.txt \
    --sid=0 \
    --output-filename=./kennedy-0.wav \
    'Ask not what your country can do for you; ask what you can do for your country.'

  ./build/bin/sherpa-onnx-offline-tts \
    --vits-model=./vits-vctk/vits-vctk.onnx \
    --vits-lexicon=./vits-vctk/lexicon.txt \
    --vits-tokens=./vits-vctk/tokens.txt \
    --sid=10 \
    --output-filename=./kennedy-10.wav \
    'Ask not what your country can do for you; ask what you can do for your country.'

  ./build/bin/sherpa-onnx-offline-tts \
    --vits-model=./vits-vctk/vits-vctk.onnx \
    --vits-lexicon=./vits-vctk/lexicon.txt \
    --vits-tokens=./vits-vctk/tokens.txt \
    --sid=108 \
    --output-filename=./kennedy-108.wav \
    'Ask not what your country can do for you; ask what you can do for your country.'

It will generate 3 files: ``kennedy-0.wav``, ``kennedy-10.wav``, and ``kennedy-108.wav``.

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Text</th>
    </tr>
    <tr>
      <td>kennedy-0.wav</td>
      <td>
       <audio title="Generated ./kennedy-0.wav" controls="controls">
             <source src="/sherpa/_static/vits-vctk/kennedy-0.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
        Ask not what your country can do for you; ask what you can do for your country.
      </td>
    </tr>
    <tr>
      <td>kennedy-10.wav</td>
      <td>
       <audio title="Generated ./kennedy-10.wav" controls="controls">
             <source src="/sherpa/_static/vits-vctk/kennedy-10.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
        Ask not what your country can do for you; ask what you can do for your country.
      </td>
    </tr>
    <tr>
      <td>kennedy-108.wav</td>
      <td>
       <audio title="Generated ./kennedy-108.wav" controls="controls">
             <source src="/sherpa/_static/vits-vctk/kennedy-108.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
        Ask not what your country can do for you; ask what you can do for your country.
      </td>
    </tr>
  </table>

Generate speech with Python script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We use speaker ID 30, 66, and 99 below to generate audio for different transcripts.

.. code-block:: bash

   cd /path/to/sherpa-onnx

   python3 ./python-api-examples/offline-tts.py \
    --vits-model=./vits-vctk/vits-vctk.onnx \
    --vits-lexicon=./vits-vctk/lexicon.txt \
    --vits-tokens=./vits-vctk/tokens.txt \
    --sid=30 \
    --output-filename=./einstein-30.wav \
    "Life is like riding a bicycle. To keep your balance, you must keep moving."

   python3 ./python-api-examples/offline-tts.py \
    --vits-model=./vits-vctk/vits-vctk.onnx \
    --vits-lexicon=./vits-vctk/lexicon.txt \
    --vits-tokens=./vits-vctk/tokens.txt \
    --sid=66 \
    --output-filename=./franklin-66.wav \
    "Three can keep a secret, if two of them are dead."

   python3 ./python-api-examples/offline-tts.py \
    --vits-model=./vits-vctk/vits-vctk.onnx \
    --vits-lexicon=./vits-vctk/lexicon.txt \
    --vits-tokens=./vits-vctk/tokens.txt \
    --sid=99 \
    --output-filename=./martin-99.wav \
    "Darkness cannot drive out darkness: only light can do that. Hate cannot drive out hate: only love can do that"

It will generate 3 files: ``einstein-30.wav``, ``franklin-66.wav``, and ``martin-99.wav``.

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Text</th>
    </tr>
    <tr>
      <td>einstein-30.wav</td>
      <td>
       <audio title="Generated ./einstein-30.wav" controls="controls">
             <source src="/sherpa/_static/vits-vctk/einstein-30.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
        Life is like riding a bicycle. To keep your balance, you must keep moving.
      </td>
    </tr>
    <tr>
      <td>franklin-66.wav</td>
      <td>
       <audio title="Generated ./franklin-66.wav" controls="controls">
             <source src="/sherpa/_static/vits-vctk/franklin-66.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
        Three can keep a secret, if two of them are dead.
      </td>
    </tr>
    <tr>
      <td>martin-99.wav</td>
      <td>
       <audio title="Generated ./martin-99.wav" controls="controls">
             <source src="/sherpa/_static/vits-vctk/martin-99.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
        Darkness cannot drive out darkness: only light can do that. Hate cannot drive out hate: only love can do that
      </td>
    </tr>
  </table>

.. _vits-model-aishell3:

aishell3 (Chinese, multi-speaker, 174 speakers)
-----------------------------------------------

This model is converted from `<https://huggingface.co/jackyqs/vits-aishell3-175-chinese>`_,
which is trained on the `aishell3`_ dataset. It supports only Chinese and it's a multi-speaker model.
It contains 174 speakers.

.. note::

   If you are interested in how the model is converted, please see
   `<https://github.com/csukuangfj/vits_chinese/blob/master/export_onnx_aishell3.py>`_

In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/vits-zh-aishell3
  cd vits-zh-aishell3
  git lfs pull --include "*.onnx"

Please check that the file sizes of the pre-trained models are correct. See
the file sizes of ``*.onnx`` files below.

.. code-block:: bash

  vits-zh-aishell3 fangjun$ ls -lh *.onnx
  -rw-r--r--  1 fangjun  staff    37M Oct 18 11:01 vits-aishell3.int8.onnx
  -rw-r--r--  1 fangjun  staff   116M Oct 18 11:01 vits-aishell3.onnx

Generate speech with executable compiled from C++
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since there are 174 speakers available, we can choose a speaker from 0 to 173.
The default speaker ID is 0.

We use speaker ID 10, 33, and 99 below to generate audio for the same text.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-offline-tts \
    --vits-model=./vits-zh-aishell3/vits-aishell3.onnx \
    --vits-lexicon=./vits-zh-aishell3/lexicon.txt \
    --vits-tokens=./vits-zh-aishell3/tokens.txt \
    --sid=10 \
    --output-filename=./liliana-10.wav \
    "林美丽最美丽、最漂亮、最可爱！"

  ./build/bin/sherpa-onnx-offline-tts \
    --vits-model=./vits-zh-aishell3/vits-aishell3.onnx \
    --vits-lexicon=./vits-zh-aishell3/lexicon.txt \
    --vits-tokens=./vits-zh-aishell3/tokens.txt \
    --sid=33 \
    --output-filename=./liliana-33.wav \
    "林美丽最美丽、最漂亮、最可爱！"

  ./build/bin/sherpa-onnx-offline-tts \
    --vits-model=./vits-zh-aishell3/vits-aishell3.onnx \
    --vits-lexicon=./vits-zh-aishell3/lexicon.txt \
    --vits-tokens=./vits-zh-aishell3/tokens.txt \
    --sid=99 \
    --output-filename=./liliana-99.wav \
    "林美丽最美丽、最漂亮、最可爱！"

It will generate 3 files: ``liliana-10.wav``, ``liliana-33.wav``, and ``liliana-99.wav``.

We also support rule-based text normalization, which is implemented with `OpenFst`_.
Currently, only number normalization is supported.

.. hint::

   We will support other normalization rules later.

The following is an example:

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline-tts \
    --vits-model=./vits-zh-aishell3/vits-aishell3.onnx \
    --vits-lexicon=./vits-zh-aishell3/lexicon.txt \
    --vits-tokens=./vits-zh-aishell3/tokens.txt \
    --tts-rule-fsts=./vits-zh-aishell3/rule.fst \
    --sid=66 \
    --output-filename=./rule-66.wav \
    "35年前，他于长沙出生, 在长白山长大。9年前他当上了银行的领导，主管行政。1天前莅临我行指导工作。"

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Text</th>
    </tr>
    <tr>
      <td>liliana-10.wav</td>
      <td>
       <audio title="Generated ./liliana-10.wav" controls="controls">
             <source src="/sherpa/_static/vits-zh-aishell3/liliana-10.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
        林美丽最美丽、最漂亮、最可爱！
      </td>
    </tr>
    <tr>
      <td>liliana-33.wav</td>
      <td>
       <audio title="Generated ./liliana-33.wav" controls="controls">
             <source src="/sherpa/_static/vits-zh-aishell3/liliana-33.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
        林美丽最美丽、最漂亮、最可爱！
      </td>
    </tr>
    <tr>
      <td>liliana-99.wav</td>
      <td>
       <audio title="Generated ./liliana-99.wav" controls="controls">
             <source src="/sherpa/_static/vits-zh-aishell3/liliana-99.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
        林美丽最美丽、最漂亮、最可爱！
      </td>
    </tr>
    <tr>
      <td>rule-66.wav</td>
      <td>
       <audio title="Generated ./rle66-99.wav" controls="controls">
             <source src="/sherpa/_static/vits-zh-aishell3/rule-66.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
        35年前，他于长沙出生, 在长白山长大。9年前他当上了银行的领导，主管行政。1天前莅临我行指导工作。
      </td>
    </tr>
  </table>

Generate speech with Python script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We use speaker ID 21, 41, and 45 below to generate audio for different transcripts.

.. code-block:: bash

   cd /path/to/sherpa-onnx

   python3 ./python-api-examples/offline-tts.py \
    --vits-model=./vits-zh-aishell3/vits-aishell3.onnx \
    --vits-lexicon=./vits-zh-aishell3/lexicon.txt \
    --vits-tokens=./vits-zh-aishell3/tokens.txt \
    --sid=21 \
    --output-filename=./liubei-21.wav \
    "勿以恶小而为之，勿以善小而不为。惟贤惟德，能服于人。"

   python3 ./python-api-examples/offline-tts.py \
    --vits-model=./vits-zh-aishell3/vits-aishell3.onnx \
    --vits-lexicon=./vits-zh-aishell3/lexicon.txt \
    --vits-tokens=./vits-zh-aishell3/tokens.txt \
    --sid=41 \
    --output-filename=./demokelite-41.wav \
    "要留心，即使当你独自一人时，也不要说坏话或做坏事，而要学得在你自己面前比在别人面前更知耻。"

   python3 ./python-api-examples/offline-tts.py \
    --vits-model=./vits-zh-aishell3/vits-aishell3.onnx \
    --vits-lexicon=./vits-zh-aishell3/lexicon.txt \
    --vits-tokens=./vits-zh-aishell3/tokens.txt \
    --sid=45 \
    --output-filename=./zhugeliang-45.wav \
    "夫君子之行，静以修身，俭以养德，非淡泊无以明志，非宁静无以致远。"


It will generate 3 files: ``liubei-21.wav``, ``demokelite-41.wav``, and ``zhugeliang-45.wav``.

The Python script also supports rule-based text normalization.

.. code-block:: bash

   python3 ./python-api-examples/offline-tts.py \
    --vits-model=./vits-zh-aishell3/vits-aishell3.onnx \
    --vits-lexicon=./vits-zh-aishell3/lexicon.txt \
    --vits-tokens=./vits-zh-aishell3/tokens.txt \
    --tts-rule-fsts=./vits-zh-aishell3/rule.fst \
    --sid=103 \
    --output-filename=./rule-103.wav \
    "根据第7次全国人口普查结果表明，我国总人口有1443497378人。普查登记的大陆31个省、自治区、直辖市和现役军人的人口共1411778724人。"

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Text</th>
    </tr>
    <tr>
      <td>liube-21.wav</td>
      <td>
       <audio title="Generated ./liubei-21.wav" controls="controls">
             <source src="/sherpa/_static/vits-zh-aishell3/liubei-21.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
        勿以恶小而为之，勿以善小而不为。惟贤惟德，能服于人。
      </td>
    </tr>
    <tr>
      <td>demokelite-41.wav</td>
      <td>
       <audio title="Generated ./demokelite-41.wav" controls="controls">
             <source src="/sherpa/_static/vits-zh-aishell3/demokelite-41.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
        要留心，即使当你独自一人时，也不要说坏话或做坏事，而要学得在你自己面前比在别人面前更知耻。
      </td>
    </tr>
    <tr>
      <td>zhugeliang-45.wav</td>
      <td>
       <audio title="Generated ./zhugeliang-45.wav" controls="controls">
             <source src="/sherpa/_static/vits-zh-aishell3/zhugeliang-45.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
        夫君子之行，静以修身，俭以养德，非淡泊无以明志，非宁静无以致远。
      </td>
    </tr>
    <tr>
      <td>rule-103.wav</td>
      <td>
       <audio title="Generated ./rule-103.wav" controls="controls">
             <source src="/sherpa/_static/vits-zh-aishell3/rule-103.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
        根据第7次全国人口普查结果表明，我国总人口有1443497378人。普查登记的大陆31个省、自治区、直辖市和现役军人的人口共1411778724人。
      </td>
    </tr>
  </table>

.. _vits-zh-hf-fanchen-C:

csukuangfj/vits-zh-hf-fanchen-C (Chinese, 1 female)
---------------------------------------------------

You can download the model at
`<https://huggingface.co/csukuangfj/vits-zh-hf-fanchen-C/tree/main>`_

.. hint::

   This model is converted from
   `<https://huggingface.co/spaces/lkz99/tts_model/tree/main/zh>`_

.. code-block:: bash

    # information about model files

    ls -lh vits-zh-hf-fanchen-C

    total 117M
    -rw-r--r-- 1 root root 346K Nov  7 14:57 lexicon.txt
    -rw-r--r-- 1 root root  63K Nov  7 14:57 rule.fst
    -rw-r--r-- 1 root root  331 Nov  7 14:57 tokens.txt
    -rw-r--r-- 1 root root 116M Nov  7 10:55 vits-zh-hf-fanchen-C.onnx

**usage**:

.. code-block:: bash

  sherpa-onnx-offline-tts \
    --vits-model=./vits-zh-hf-fanchen-C/vits-zh-hf-fanchen-C.onnx \
    --vits-lexicon=./vits-zh-hf-fanchen-C/lexicon.txt \
    --vits-tokens=./vits-zh-hf-fanchen-C/tokens.txt \
    --vits-length-scale=0.5 \
    --output-filename="./value-2x.wav" \
    "小米的核心价值观是什么？答案是真诚热爱！"

  sherpa-onnx-offline-tts \
    --vits-model=./vits-zh-hf-fanchen-C/vits-zh-hf-fanchen-C.onnx \
    --vits-lexicon=./vits-zh-hf-fanchen-C/lexicon.txt \
    --vits-tokens=./vits-zh-hf-fanchen-C/tokens.txt \
    --vits-length-scale=1.0 \
    --tts-rule-fsts=./vits-zh-hf-fanchen-C/rule.fst \
    --output-filename="./numbers.wav" \
    "小米有13岁了"

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Text</th>
    </tr>
    <tr>
      <td>value-2x.wav</td>
      <td>
       <audio title="Generated ./value-2x.wav" controls="controls">
             <source src="/sherpa/_static/vits-zh-hf-fanchen-C/value-2x.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
        小米的核心价值观是什么？答案是真诚热爱！
      </td>
    </tr>
    <tr>
      <td>numbers.wav</td>
      <td>
       <audio title="Generated ./numbers.wav" controls="controls">
             <source src="/sherpa/_static/vits-zh-hf-fanchen-C/numbers.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
        小米有13岁了
      </td>
    </tr>
  </table>

.. _vits-zh-hf-fanchen-wnj:

csukuangfj/vits-zh-hf-fanchen-wnj (Chinese, 1 male)
---------------------------------------------------

You can download the model at
`<https://huggingface.co/csukuangfj/vits-zh-hf-fanchen-wnj/tree/main>`_

.. hint::

   This model is converted from
   `<https://huggingface.co/spaces/lkz99/tts_model/blob/main/G_wnj_latest.pth>`_

.. code-block:: bash

    # information about model files

    ls -lh vits-zh-hf-fanchen-wnj

    total 116M
    -rw-r--r-- 1 root root 346K Nov  7 15:12 lexicon.txt
    -rw-r--r-- 1 root root  63K Nov  7 15:12 rule.fst
    -rw-r--r-- 1 root root  331 Nov  7 15:12 tokens.txt
    -rw-r--r-- 1 root root 116M Nov  7 06:34 vits-zh-hf-fanchen-wnj.onnx

**usage**:

.. code-block:: bash

  sherpa-onnx-offline-tts \
    --vits-model=./vits-zh-hf-fanchen-wnj/vits-zh-hf-fanchen-wnj.onnx \
    --vits-lexicon=./vits-zh-hf-fanchen-wnj/lexicon.txt \
    --vits-tokens=./vits-zh-hf-fanchen-wnj/tokens.txt \
    --output-filename="./kuayue.wav" \
    "升级人车家全生态，小米迎跨越时刻。"

  sherpa-onnx-offline-tts \
    --vits-model=./vits-zh-hf-fanchen-wnj/vits-zh-hf-fanchen-wnj.onnx \
    --vits-lexicon=./vits-zh-hf-fanchen-wnj/lexicon.txt \
    --vits-tokens=./vits-zh-hf-fanchen-wnj/tokens.txt \
    --tts-rule-fsts=./vits-zh-hf-fanchen-wnj/rule.fst \
    --output-filename="./os.wav" \
    "这一全新操作系统，是小米13年来技术积淀的结晶。"

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Text</th>
    </tr>
    <tr>
      <td>kuayue.wav</td>
      <td>
       <audio title="Generated ./kuayue.wav" controls="controls">
             <source src="/sherpa/_static/vits-zh-hf-fanchen-wnj/kuayue.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
        升级人车家全生态，小米迎跨越时刻。
      </td>
    </tr>
    <tr>
      <td>os.wav</td>
      <td>
       <audio title="Generated ./os.wav" controls="controls">
             <source src="/sherpa/_static/vits-zh-hf-fanchen-wnj/os.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
        这一全新操作系统，是小米13年来技术积淀的结晶。
      </td>
    </tr>
  </table>

.. _vits-zh-hf-theresa:

csukuangfj/vits-zh-hf-theresa (Chinese, 804 speakers)
-----------------------------------------------------

You can download the model at
`<https://huggingface.co/csukuangfj/vits-zh-hf-theresa/tree/main>`_

.. hint::

   This model is converted from
   `<https://huggingface.co/spaces/zomehwh/vits-models-genshin-bh3/tree/main/pretrained_models/theresa>`_

.. code-block:: bash

    # information about model files

    ls -lh vits-zh-hf-theresa

    total 117M
    -rw-r--r-- 1 root root 384K Nov  7 15:26 lexicon.txt
    -rw-r--r-- 1 root root  63K Nov  7 15:26 rule.fst
    -rw-r--r-- 1 root root 117M Nov  5 17:04 theresa.onnx
    -rw-r--r-- 1 root root  268 Nov  7 15:26 tokens.txt

**usage**:

.. code-block:: bash

  sherpa-onnx-offline-tts \
    --vits-model=./vits-zh-hf-theresa/theresa.onnx \
    --vits-lexicon=./vits-zh-hf-theresa/lexicon.txt \
    --vits-tokens=./vits-zh-hf-theresa/tokens.txt \
    --sid=0 \
    --output-filename="./reai-0.wav" \
    "真诚就是不欺人也不自欺。热爱就是全心投入并享受其中。"

  sherpa-onnx-offline-tts \
    --vits-model=./vits-zh-hf-theresa/theresa.onnx \
    --vits-lexicon=./vits-zh-hf-theresa/lexicon.txt \
    --vits-tokens=./vits-zh-hf-theresa/tokens.txt \
    --tts-rule-fsts=./vits-zh-hf-theresa/rule.fst \
    --debug=1 \
    --sid=88 \
    --output-filename="./mi14-88.wav" \
    "小米14一周销量破1000000!"

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Text</th>
    </tr>
    <tr>
      <td>reai-0.wav</td>
      <td>
       <audio title="Generated ./reai-0.wav" controls="controls">
             <source src="/sherpa/_static/vits-zh-hf-theresa/reai-0.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
        真诚就是不欺人也不自欺。热爱就是全心投入并享受其中。
      </td>
    </tr>
    <tr>
      <td>m14-88.wav</td>
      <td>
       <audio title="Generated ./mi14-88.wav" controls="controls">
             <source src="/sherpa/_static/vits-zh-hf-theresa/mi14-88.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
        小米14一周销量破1000000!
      </td>
    </tr>
  </table>

.. _vits-zh-hf-eula:

csukuangfj/vits-zh-hf-eula (Chinese, 804 speakers)
--------------------------------------------------

You can download the model at
`<https://huggingface.co/csukuangfj/vits-zh-hf-eula/tree/main>`_

.. hint::

   This model is converted from
   `<https://huggingface.co/spaces/zomehwh/vits-models-genshin-bh3/tree/main/pretrained_models/eula>`_

.. code-block:: bash

    # information about model files

    ls -lh vits-zh-hf-eula

    -rw-r--r-- 1 root root 117M Nov  5 17:00 eula.onnx
    -rw-r--r-- 1 root root 384K Nov  7 15:42 lexicon.txt
    -rw-r--r-- 1 root root  63K Nov  7 15:42 rule.fst
    -rw-r--r-- 1 root root  268 Nov  7 15:42 tokens.txt

**usage**:

.. code-block:: bash

  sherpa-onnx-offline-tts \
    --vits-model=./vits-zh-hf-eula/eula.onnx \
    --vits-lexicon=./vits-zh-hf-eula/lexicon.txt \
    --vits-tokens=./vits-zh-hf-eula/tokens.txt \
    --debug=1 \
    --sid=666 \
    --output-filename="./news-666.wav" \
    "小米在今天上午举办的核心干部大会上，公布了新十年的奋斗目标和科技战略，并发布了小米价值观的八条诠释。"

  sherpa-onnx-offline-tts \
    --vits-model=./vits-zh-hf-eula/eula.onnx \
    --vits-lexicon=./vits-zh-hf-eula/lexicon.txt \
    --vits-tokens=./vits-zh-hf-eula/tokens.txt \
    --tts-rule-fsts=./vits-zh-hf-eula/rule.fst \
    --sid=99 \
    --output-filename="./news-99.wav" \
    "9月25日消息，雷军今日在微博发文称"

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Text</th>
    </tr>
    <tr>
      <td>news-666.wav</td>
      <td>
       <audio title="Generated ./news-666.wav" controls="controls">
             <source src="/sherpa/_static/vits-zh-hf-eula/news-666.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
        小米在今天上午举办的核心干部大会上，公布了新十年的奋斗目标和科技战略，并发布了小米价值观的八条诠释。
      </td>
    </tr>
    <tr>
      <td>news-99.wav</td>
      <td>
       <audio title="Generated ./news-99.wav" controls="controls">
             <source src="/sherpa/_static/vits-zh-hf-eula/news-99.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
        9月25日消息，雷军今日在微博发文称
      </td>
    </tr>
  </table>



.. _vits-model-en_US-lessac-medium:

en_US-lessac-medium (English, single-speaker)
---------------------------------------------

This model is converted from `<https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/lessac/medium>`_.

The dataset used to train the model is `lessac_blizzard2013`_.

.. hint::

  The model is from `piper`_.

In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-lessac-medium.tar.bz2
  tar xf vits-piper-en_US-lessac-medium.tar.bz2

.. hint::

   You can find a lot of pre-trained models for over 40 languages at
   `<https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models>`.

Generate speech with executable compiled from C++
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-offline-tts \
    --vits-model=./vits-piper-en_US-lessac-medium/en_US-lessac-medium.onnx \
    --vits-data-dir=./vits-piper-en_US-lessac-medium/espeak-ng-data \
    --vits-tokens=./vits-piper-en_US-lessac-medium/tokens.txt \
    --output-filename=./liliana-piper-en_US-lessac-medium.wav \
    'liliana, the most beautiful and lovely assistant of our team!'

.. hint::

   You can also use

    .. code-block:: bash

      cd /path/to/sherpa-onnx

      ./build/bin/sherpa-onnx-offline-tts-play \
        --vits-model=./vits-piper-en_US-lessac-medium/en_US-lessac-medium.onnx \
        --vits-data-dir=./vits-piper-en_US-lessac-medium/espeak-ng-data \
        --vits-tokens=./vits-piper-en_US-lessac-medium/tokens.txt \
        --output-filename=./liliana-piper-en_US-lessac-medium.wav \
        'liliana, the most beautiful and lovely assistant of our team!'

    which will play the audio as it is generating.


After running, it will generate a file ``liliana-piper.wav`` in the current directory.

.. code-block:: bash

   soxi ./liliana-piper-en_US-lessac-medium.wav

   Input File     : './liliana-piper-en_US-lessac-medium.wav'
   Channels       : 1
   Sample Rate    : 22050
   Precision      : 16-bit
   Duration       : 00:00:03.48 = 76800 samples ~ 261.224 CDDA sectors
   File Size      : 154k
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
      <td>liliana-piper-en_US-lessac-medium.wav</td>
      <td>
       <audio title="Generated ./liliana-piper-en_US-lessac-medium.wav" controls="controls">
             <source src="/sherpa/_static/vits-piper/liliana-piper-en_US-lessac-medium.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
        liliana, the most beautiful and lovely assistant of our team!
      </td>
    </tr>
  </table>

Generate speech with Python script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cd /path/to/sherpa-onnx

   python3 ./python-api-examples/offline-tts.py \
    --vits-model=./vits-piper-en_US-lessac-medium/en_US-lessac-medium.onnx \
    --vits-data-dir=./vits-piper-en_US-lessac-medium/espeak-ng-data \
    --vits-tokens=./vits-piper-en_US-lessac-medium/tokens.txt \
    --output-filename=./armstrong-piper-en_US-lessac-medium.wav \
    "That's one small step for a man, a giant leap for mankind."

.. hint::

   You can also use

    .. code-block:: bash

      cd /path/to/sherpa-onnx

      python3 ./python-api-examples/offline-tts-play.py \
        --vits-model=./vits-piper-en_US-lessac-medium/en_US-lessac-medium.onnx \
        --vits-data-dir=./vits-piper-en_US-lessac-medium/espeak-ng-data \
        --vits-tokens=./vits-piper-en_US-lessac-medium/tokens.txt \
        --output-filename=./armstrong-piper-en_US-lessac-medium.wav \
        "That's one small step for a man, a giant leap for mankind."

    which will play the audio as it is generating.

After running, it will generate a file ``armstrong-piper-en_US-lessac-medium.wav`` in the current directory.

.. code-block:: bash

   soxi ./armstrong-piper-en_US-lessac-medium.wav

   Input File     : './armstrong-piper-en_US-lessac-medium.wav'
   Channels       : 1
   Sample Rate    : 22050
   Precision      : 16-bit
   Duration       : 00:00:03.74 = 82432 samples ~ 280.381 CDDA sectors
   File Size      : 165k
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
      <td>armstrong-piper-en_US-lessac-medium.wav</td>
      <td>
       <audio title="Generated ./armstrong-piper-en_US-lessac-medium.wav" controls="controls">
             <source src="/sherpa/_static/vits-piper/armstrong-piper-en_US-lessac-medium.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
        That's one small step for a man, a giant leap for mankind.
      </td>
    </tr>
  </table>
