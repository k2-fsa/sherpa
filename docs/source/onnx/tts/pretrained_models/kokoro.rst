Kokoro
======

This page lists pre-trained models from `<https://huggingface.co/hexgrad/Kokoro-82M>`_.

.. _kokoro-en-v0_19:

kokoro-en-v0_19 (English, 11 speakers)
--------------------------------------

This model contains 11 speakers. The ONNX model is from
`<https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files>`_

The script for adding meta data to the ONNX model can be found at
`<https://github.com/k2-fsa/sherpa-onnx/tree/master/scripts/kokoro>`_

In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

  cd /path/to/sherpa-onnx
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-en-v0_19.tar.bz2
  tar xf kokoro-en-v0_19.tar.bz2
  rm kokoro-en-v0_19.tar.bz2

Please check that the file sizes of the pre-trained models are correct. See
the file sizes of ``*.onnx`` files below.

.. code-block::

  ls -lh kokoro-en-v0_19/

  total 686208
  -rw-r--r--    1 fangjun  staff    11K Jan 15 16:23 LICENSE
  -rw-r--r--    1 fangjun  staff   235B Jan 15 16:25 README.md
  drwxr-xr-x  122 fangjun  staff   3.8K Nov 28  2023 espeak-ng-data
  -rw-r--r--    1 fangjun  staff   330M Jan 15 16:25 model.onnx
  -rw-r--r--    1 fangjun  staff   1.1K Jan 15 16:25 tokens.txt
  -rw-r--r--    1 fangjun  staff   5.5M Jan 15 16:25 voices.bin

Map between speaker ID and speaker name
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The model contains 11 speakers and we use integer IDs ``0-10`` to represent.
each speaker.

The map is given below:

.. list-table::

 * - Speaker ID
   - 0
   - 1
   - 2
   - 3
   - 4
   - 5
   - 6
   - 7
   - 8
   - 9
   - 10
 * - Speaker Name
   - af
   - af_bella
   - af_nicole
   - af_sarah
   - af_sky
   - am_adam
   - am_michael
   - bf_emma
   - bf_isabella
   - bm_george
   - bm_lewis

.. raw:: html

  <table>
    <tr>
      <th>ID</th>
      <th>name</th>
      <th>Test wave</th>
    </tr>

    <tr>
      <td>0</td>
      <td>af</td>
      <td>
       <audio title="./0-af.wav" controls="controls">
             <source src="/sherpa/_static/kokoro-en-v0_19/sid/0-af.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
    </tr>

    <tr>
      <td>1</td>
      <td>af_bella</td>
      <td>
       <audio title="./1-af_bella.wav" controls="controls">
             <source src="/sherpa/_static/kokoro-en-v0_19/sid/1-af_bella.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
    </tr>

    <tr>
      <td>2</td>
      <td>af_nicole</td>
      <td>
       <audio title="./2-af_nicole.wav" controls="controls">
             <source src="/sherpa/_static/kokoro-en-v0_19/sid/2-af_nicole.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
    </tr>

    <tr>
      <td>3</td>
      <td>af_sarah</td>
      <td>
       <audio title="./3-af_sarah.wav" controls="controls">
             <source src="/sherpa/_static/kokoro-en-v0_19/sid/3-af_sarah.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
    </tr>

    <tr>
      <td>4</td>
      <td>af_sky</td>
      <td>
       <audio title="./4-af_sky.wav" controls="controls">
             <source src="/sherpa/_static/kokoro-en-v0_19/sid/4-af_sky.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
    </tr>

    <tr>
      <td>5</td>
      <td>am_adam</td>
      <td>
       <audio title="./5-am_adam.wav" controls="controls">
             <source src="/sherpa/_static/kokoro-en-v0_19/sid/5-am_adam.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
    </tr>

    <tr>
      <td>6</td>
      <td>am_michael</td>
      <td>
       <audio title="./6-am_michael.wav" controls="controls">
             <source src="/sherpa/_static/kokoro-en-v0_19/sid/6-am_michael.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
    </tr>

    <tr>
      <td>7</td>
      <td>bf_emma</td>
      <td>
       <audio title="./7-bf_emma.wav" controls="controls">
             <source src="/sherpa/_static/kokoro-en-v0_19/sid/7-bf_emma.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
    </tr>

    <tr>
      <td>8</td>
      <td>bf_isabella</td>
      <td>
       <audio title="./8-bf_isabella.wav" controls="controls">
             <source src="/sherpa/_static/kokoro-en-v0_19/sid/8-bf_isabella.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
    </tr>

    <tr>
      <td>9</td>
      <td>bm_george</td>
      <td>
       <audio title="./9-bm_george.wav" controls="controls">
             <source src="/sherpa/_static/kokoro-en-v0_19/sid/9-bm_george.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
    </tr>

    <tr>
      <td>10</td>
      <td>bm_lewis</td>
      <td>
       <audio title="./10-bm_lewis.wav" controls="controls">
             <source src="/sherpa/_static/kokoro-en-v0_19/sid/10-bm_lewis.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
    </tr>

  </table>

Generate speech with executables compiled from C++
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-offline-tts \
    --kokoro-model=./kokoro-en-v0_19/model.onnx \
    --kokoro-voices=./kokoro-en-v0_19/voices.bin \
    --kokoro-tokens=./kokoro-en-v0_19/tokens.txt \
    --kokoro-data-dir=./kokoro-en-v0_19/espeak-ng-data \
    --num-threads=2 \
    --sid=10 \
    --output-filename="./10-bm_lewis.wav" \
    "Today as always, men fall into two groups: slaves and free men. Whoever does not have two-thirds of his day for himself, is a slave, whatever he may be, a statesman, a businessman, an official, or a scholar."

After running, it will generate a file ``10-bm_lewis`` in the
current directory.

.. code-block:: bash

  soxi ./10-bm_lewis.wav

  Input File     : './10-bm_lewis.wav'
  Channels       : 1
  Sample Rate    : 24000
  Precision      : 16-bit
  Duration       : 00:00:15.80 = 379200 samples ~ 1185 CDDA sectors
  File Size      : 758k
  Bit Rate       : 384k
  Sample Encoding: 16-bit Signed Integer PCM

.. hint::

   Sample rate of this model is fixed to ``24000 Hz``.

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Text</th>
    </tr>
    <tr>
      <td>10-bm_lewis.wav</td>
      <td>
       <audio title="Generated ./10-bm_lewis.wav" controls="controls">
             <source src="/sherpa/_static/kokoro-en-v0_19/10-bm_lewis.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
    "Today as always, men fall into two groups: slaves and free men. Whoever does not have two-thirds of his day for himself, is a slave, whatever he may be, a statesman, a businessman, an official, or a scholar."
      </td>
    </tr>
  </table>

Generate speech with Python script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  python3 ./python-api-examples/offline-tts.py \
    --kokoro-model=./kokoro-en-v0_19/model.onnx \
    --kokoro-voices=./kokoro-en-v0_19/voices.bin \
    --kokoro-tokens=./kokoro-en-v0_19/tokens.txt \
    --kokoro-data-dir=./kokoro-en-v0_19/espeak-ng-data \
    --num-threads=2 \
    --sid=2 \
    --output-filename=./2-af_nicole.wav \
    "Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone."

.. code-block:: bash

  soxi ./2-af_nicole.wav

  Input File     : './2-af_nicole.wav'
  Channels       : 1
  Sample Rate    : 24000
  Precision      : 16-bit
  Duration       : 00:00:11.45 = 274800 samples ~ 858.75 CDDA sectors
  File Size      : 550k
  Bit Rate       : 384k
  Sample Encoding: 16-bit Signed Integer PCM

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Text</th>
    </tr>
    <tr>
      <td>2-af_nicole.wav</td>
      <td>
       <audio title="Generated ./2-af_nicole.wav" controls="controls">
             <source src="/sherpa/_static/kokoro-en-v0_19/2-af_nicole.wav" type="audio/wav">
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
      --kokoro-model=./kokoro-en-v0_19/model.onnx \
      --kokoro-voices=./kokoro-en-v0_19/voices.bin \
      --kokoro-tokens=./kokoro-en-v0_19/tokens.txt \
      --kokoro-data-dir=./kokoro-en-v0_19/espeak-ng-data \
      --sid=2 \
      --output-filename=./2-af_nicole.wav \
      "Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone."
   done

The results are given below:

  +-------------+-------+-------+-------+-------+
  | num_threads | 1     | 2     | 3     | 4     |
  +=============+=======+=======+=======+=======+
  | RTF         | 6.629 | 3.870 | 2.999 | 2.774 |
  +-------------+-------+-------+-------+-------+
