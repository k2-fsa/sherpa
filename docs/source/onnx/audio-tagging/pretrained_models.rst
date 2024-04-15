Pre-trained models
==================

This section lists pre-trained models for audio tagging.

You can find all models at the following URL:

  `<https://github.com/k2-fsa/sherpa-onnx/releases/tag/audio-tagging-models>`_

sherpa-onnx-zipformer-small-audio-tagging-2024-04-15
----------------------------------------------------

This model is trained by `<https://github.com/k2-fsa/icefall/pull/1421>`_
using the dataset `audioset`_.

In the following, we describe how to download and use it with `sherpa-onnx`_.

Download the model
^^^^^^^^^^^^^^^^^^

Please use the following commands to download it::

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/audio-tagging-models/sherpa-onnx-zipformer-small-audio-tagging-2024-04-15.tar.bz2

  # For Chinese users, you can aso use the following mirror:
  wget https://hub.nuaa.cf/k2-fsa/sherpa-onnx/releases/download/audio-tagging-models/sherpa-onnx-zipformer-small-audio-tagging-2024-04-15.tar.bz2

  tar xvf sherpa-onnx-zipformer-small-audio-tagging-2024-04-15.tar.bz2
  rm sherpa-onnx-zipformer-small-audio-tagging-2024-04-15.tar.bz2

You will find the following files after unzipping::

  -rw-r--r--   1 fangjun  staff   243B Apr 15 16:14 README.md
  -rw-r--r--   1 fangjun  staff    14K Apr 15 16:14 class_labels_indices.csv
  -rw-r--r--   1 fangjun  staff    26M Apr 15 16:14 model.int8.onnx
  -rw-r--r--   1 fangjun  staff    88M Apr 15 16:14 model.onnx
  drwxr-xr-x  15 fangjun  staff   480B Apr 15 16:14 test_wavs

C++ binary examples
^^^^^^^^^^^^^^^^^^^

.. hint::

   You can find the binary executable file ``sherpa-onnx-offline-audio-tagging``
   after installing `sherpa-onnx`_ either from source or using ``pip install sherpa-onnx``_.

Cat
:::

For the following test wave,

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Wave</th>
    </tr>
    <tr>
      <td>1.wav</td>
      <td>
       <audio title="./1.wav" controls="controls">
             <source src="/sherpa/_static/audio-tagging/zipformer-small/1.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
    </tr>
  </table>

the command::

  ./bin/sherpa-onnx-offline-audio-tagging \
    --zipformer-model=./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/model.int8.onnx \
    --labels=./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/class_labels_indices.csv \
    ./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/test_wavs/1.wav

prints the following::

  0: AudioEvent(name="Animal", index=72, prob=0.947886)
  1: AudioEvent(name="Cat", index=81, prob=0.938876)
  2: AudioEvent(name="Domestic animals, pets", index=73, prob=0.931975)
  3: AudioEvent(name="Caterwaul", index=85, prob=0.178876)
  4: AudioEvent(name="Meow", index=83, prob=0.176177)
  Num threads: 1
  Wave duration: 10.000
  Elapsed seconds: 0.297 s
  Real time factor (RTF): 0.297 / 10.000 = 0.030

.. hint::

   By default, it outputs the top 5 events. The first event has the
   largest probability.

Whistle
:::::::

For the following test wave,

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Wave</th>
    </tr>
    <tr>
      <td>2.wav</td>
      <td>
       <audio title="./2.wav" controls="controls">
             <source src="/sherpa/_static/audio-tagging/zipformer-small/2.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
    </tr>
  </table>

the command::

  ./bin/sherpa-onnx-offline-audio-tagging \
    --zipformer-model=./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/model.int8.onnx \
    --labels=./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/class_labels_indices.csv \
    ./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/test_wavs/2.wav

prints the following::

  0: AudioEvent(name="Whistling", index=40, prob=0.804928)
  1: AudioEvent(name="Music", index=137, prob=0.27548)
  2: AudioEvent(name="Piano", index=153, prob=0.135418)
  3: AudioEvent(name="Keyboard (musical)", index=152, prob=0.0580414)
  4: AudioEvent(name="Musical instrument", index=138, prob=0.0400399)
  Num threads: 1
  Wave duration: 10.000
  Elapsed seconds: 0.289 s
  Real time factor (RTF): 0.289 / 10.000 = 0.029

Music
:::::

For the following test wave,

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Wave</th>
    </tr>
    <tr>
      <td>3.wav</td>
      <td>
       <audio title="./3.wav" controls="controls">
             <source src="/sherpa/_static/audio-tagging/zipformer-small/3.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
    </tr>
  </table>

the command::

  ./bin/sherpa-onnx-offline-audio-tagging \
    --zipformer-model=./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/model.int8.onnx \
    --labels=./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/class_labels_indices.csv \
    ./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/test_wavs/3.wav

prints the following::

  0: AudioEvent(name="Music", index=137, prob=0.79673)
  1: AudioEvent(name="A capella", index=255, prob=0.765521)
  2: AudioEvent(name="Singing", index=27, prob=0.473899)
  3: AudioEvent(name="Vocal music", index=254, prob=0.459337)
  4: AudioEvent(name="Choir", index=28, prob=0.458174)
  Num threads: 1
  Wave duration: 10.000
  Elapsed seconds: 0.279 s
  Real time factor (RTF): 0.279 / 10.000 = 0.028

Laughter
::::::::

For the following test wave,

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Wave</th>
    </tr>
    <tr>
      <td>4.wav</td>
      <td>
       <audio title="./4.wav" controls="controls">
             <source src="/sherpa/_static/audio-tagging/zipformer-small/4.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
    </tr>
  </table>

the command::

  ./bin/sherpa-onnx-offline-audio-tagging \
    --zipformer-model=./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/model.int8.onnx \
    --labels=./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/class_labels_indices.csv \
    ./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/test_wavs/4.wav

prints the following::

  0: AudioEvent(name="Laughter", index=16, prob=0.929239)
  1: AudioEvent(name="Snicker", index=19, prob=0.321969)
  2: AudioEvent(name="Giggle", index=18, prob=0.149667)
  3: AudioEvent(name="Inside, small room", index=506, prob=0.119332)
  4: AudioEvent(name="Belly laugh", index=20, prob=0.100728)
  Num threads: 1
  Wave duration: 10.000
  Elapsed seconds: 0.314 s
  Real time factor (RTF): 0.314 / 10.000 = 0.031

Finger snapping
:::::::::::::::

For the following test wave,

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Wave</th>
    </tr>
    <tr>
      <td>5.wav</td>
      <td>
       <audio title="./5.wav" controls="controls">
             <source src="/sherpa/_static/audio-tagging/zipformer-small/5.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
    </tr>
  </table>

the command::

  ./bin/sherpa-onnx-offline-audio-tagging \
    --zipformer-model=./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/model.int8.onnx \
    --labels=./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/class_labels_indices.csv \
    ./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/test_wavs/5.wav

prints the following::

  0: AudioEvent(name="Finger snapping", index=62, prob=0.690543)
  1: AudioEvent(name="Slap, smack", index=467, prob=0.452133)
  2: AudioEvent(name="Clapping", index=63, prob=0.179213)
  3: AudioEvent(name="Sound effect", index=504, prob=0.101151)
  4: AudioEvent(name="Whack, thwack", index=468, prob=0.0294559)
  Num threads: 1
  Wave duration: 8.284
  Elapsed seconds: 0.225 s
  Real time factor (RTF): 0.225 / 8.284 = 0.027

Baby cry
::::::::

For the following test wave,

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Wave</th>
    </tr>
    <tr>
      <td>6.wav</td>
      <td>
       <audio title="./6.wav" controls="controls">
             <source src="/sherpa/_static/audio-tagging/zipformer-small/6.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
    </tr>
  </table>

the command::

  ./bin/sherpa-onnx-offline-audio-tagging \
    --zipformer-model=./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/model.int8.onnx \
    --labels=./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/class_labels_indices.csv \
    ./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/test_wavs/6.wav

prints the following::

  0: AudioEvent(name="Baby cry, infant cry", index=23, prob=0.912273)
  1: AudioEvent(name="Crying, sobbing", index=22, prob=0.670927)
  2: AudioEvent(name="Whimper", index=24, prob=0.187221)
  3: AudioEvent(name="Inside, small room", index=506, prob=0.0314955)
  4: AudioEvent(name="Sound effect", index=504, prob=0.0118726)
  Num threads: 1
  Wave duration: 8.719
  Elapsed seconds: 0.232 s
  Real time factor (RTF): 0.232 / 8.719 = 0.027

Smoke alarm
:::::::::::

For the following test wave,

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Wave</th>
    </tr>
    <tr>
      <td>7.wav</td>
      <td>
       <audio title="./7.wav" controls="controls">
             <source src="/sherpa/_static/audio-tagging/zipformer-small/7.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
    </tr>
  </table>

the command::

  ./bin/sherpa-onnx-offline-audio-tagging \
    --zipformer-model=./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/model.int8.onnx \
    --labels=./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/class_labels_indices.csv \
    ./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/test_wavs/7.wav

prints the following::

  0: AudioEvent(name="Smoke detector, smoke alarm", index=399, prob=0.781478)
  1: AudioEvent(name="Beep, bleep", index=481, prob=0.641056)
  2: AudioEvent(name="Buzzer", index=398, prob=0.218576)
  3: AudioEvent(name="Fire alarm", index=400, prob=0.140145)
  4: AudioEvent(name="Alarm", index=388, prob=0.012525)
  Num threads: 1
  Wave duration: 2.819
  Elapsed seconds: 0.080 s
  Real time factor (RTF): 0.080 / 2.819 = 0.028

Siren
:::::

For the following test wave,

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Wave</th>
    </tr>
    <tr>
      <td>8.wav</td>
      <td>
       <audio title="./8.wav" controls="controls">
             <source src="/sherpa/_static/audio-tagging/zipformer-small/8.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
    </tr>
  </table>

the command::

  ./bin/sherpa-onnx-offline-audio-tagging \
    --zipformer-model=./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/model.int8.onnx \
    --labels=./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/class_labels_indices.csv \
    ./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/test_wavs/8.wav

prints the following::

  0: AudioEvent(name="Siren", index=396, prob=0.877108)
  1: AudioEvent(name="Civil defense siren", index=397, prob=0.732789)
  2: AudioEvent(name="Vehicle", index=300, prob=0.0113797)
  3: AudioEvent(name="Inside, small room", index=506, prob=0.00537381)
  4: AudioEvent(name="Outside, urban or manmade", index=509, prob=0.00261939)
  Num threads: 1
  Wave duration: 7.721
  Elapsed seconds: 0.220 s
  Real time factor (RTF): 0.220 / 7.721 = 0.028

Stream water
::::::::::::

For the following test wave,

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Wave</th>
    </tr>
    <tr>
      <td>10.wav</td>
      <td>
       <audio title="./10.wav" controls="controls">
             <source src="/sherpa/_static/audio-tagging/zipformer-small/10.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
    </tr>
  </table>

the command::

  ./bin/sherpa-onnx-offline-audio-tagging \
    --zipformer-model=./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/model.int8.onnx \
    --labels=./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/class_labels_indices.csv \
    ./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/test_wavs/10.wav

prints the following::

  0: AudioEvent(name="Stream", index=292, prob=0.247785)
  1: AudioEvent(name="Water", index=288, prob=0.231587)
  2: AudioEvent(name="Gurgling", index=297, prob=0.170981)
  3: AudioEvent(name="Trickle, dribble", index=450, prob=0.108859)
  4: AudioEvent(name="Liquid", index=444, prob=0.0693812)
  Num threads: 1
  Wave duration: 7.837
  Elapsed seconds: 0.212 s
  Real time factor (RTF): 0.212 / 7.837 = 0.027

Meow
::::

For the following test wave,

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Wave</th>
    </tr>
    <tr>
      <td>11.wav</td>
      <td>
       <audio title="./11.wav" controls="controls">
             <source src="/sherpa/_static/audio-tagging/zipformer-small/11.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
    </tr>
  </table>

the command::

  ./bin/sherpa-onnx-offline-audio-tagging \
    --zipformer-model=./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/model.int8.onnx \
    --labels=./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/class_labels_indices.csv \
    ./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/test_wavs/11.wav

prints the following::

  0: AudioEvent(name="Meow", index=83, prob=0.814944)
  1: AudioEvent(name="Cat", index=81, prob=0.698858)
  2: AudioEvent(name="Domestic animals, pets", index=73, prob=0.564516)
  3: AudioEvent(name="Animal", index=72, prob=0.535303)
  4: AudioEvent(name="Music", index=137, prob=0.105332)
  Num threads: 1
  Wave duration: 11.483
  Elapsed seconds: 0.361 s
  Real time factor (RTF): 0.361 / 11.483 = 0.031

Dog bark
::::::::

For the following test wave,

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Wave</th>
    </tr>
    <tr>
      <td>12.wav</td>
      <td>
       <audio title="./12.wav" controls="controls">
             <source src="/sherpa/_static/audio-tagging/zipformer-small/12.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
    </tr>
  </table>

the command::

  ./bin/sherpa-onnx-offline-audio-tagging \
    --zipformer-model=./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/model.int8.onnx \
    --labels=./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/class_labels_indices.csv \
    ./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/test_wavs/12.wav

prints the following::

  0: AudioEvent(name="Animal", index=72, prob=0.688237)
  1: AudioEvent(name="Dog", index=74, prob=0.637803)
  2: AudioEvent(name="Bark", index=75, prob=0.608597)
  3: AudioEvent(name="Bow-wow", index=78, prob=0.515501)
  4: AudioEvent(name="Domestic animals, pets", index=73, prob=0.495074)
  Num threads: 1
  Wave duration: 8.974
  Elapsed seconds: 0.261 s
  Real time factor (RTF): 0.261 / 8.974 = 0.029

Oink (pig)
::::::::::

For the following test wave,

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Wave</th>
    </tr>
    <tr>
      <td>13.wav</td>
      <td>
       <audio title="./13.wav" controls="controls">
             <source src="/sherpa/_static/audio-tagging/zipformer-small/13.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
    </tr>
  </table>

the command::

  ./bin/sherpa-onnx-offline-audio-tagging \
    --zipformer-model=./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/model.int8.onnx \
    --labels=./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/class_labels_indices.csv \
    ./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/test_wavs/13.wav

prints the following::

  0: AudioEvent(name="Oink", index=94, prob=0.888416)
  1: AudioEvent(name="Pig", index=93, prob=0.164295)
  2: AudioEvent(name="Animal", index=72, prob=0.160802)
  3: AudioEvent(name="Speech", index=0, prob=0.0276513)
  4: AudioEvent(name="Snort", index=46, prob=0.0201952)
  Num threads: 1
  Wave duration: 9.067
  Elapsed seconds: 0.261 s
  Real time factor (RTF): 0.261 / 9.067 = 0.029

Python API examples
^^^^^^^^^^^^^^^^^^^

Please see

  `<https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/audio-tagging-from-a-file.py>`_

Huggingface space
^^^^^^^^^^^^^^^^^

You can try audio tagging with `sherpa-onnx`_ from within you browser by visiting the following URL:

  `<https://huggingface.co/spaces/k2-fsa/audio-tagging>`_

.. note::

   For Chinese users, please use

    `<https://hf-mirror.com/spaces/k2-fsa/audio-tagging>`_
