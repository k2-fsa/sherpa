vits
====

This page lists pre-trained `vits`_ models.

ljspeech (English, single-speaker)
----------------------------------

This model is converted from `pretrained_ljspeech.pth <https://drive.google.com/file/d/1q86w74Ygw2hNzYP9cWkeClGT5X25PvBT/view?usp=drive_link>`_,
which is trained by the `vits`_ author `Jaehyeon Kim <https://github.com/jaywalnut310>`_ on
the `ljspeech`_ dataset. It supports only English and is a single-speaker model.

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
  git lfs pull --include ".*onnx"

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
  git lfs pull --include ".*onnx"

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

aishell3 (Chinese, multi-speaker, 174 speakers)
-----------------------------------------------

This model is converted from `<https://huggingface.co/jackyqs/vits-aishell3-175-chinese>`_,
which is trained on the aishell3 dataset. It supports only Chinese and it's a multi-speaker model.
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
  git lfs pull --include ".*onnx"

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
  </table>
