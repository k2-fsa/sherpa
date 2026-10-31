Source separation models
========================

This page lists the source separation models supported in `sherpa-onnx`_.

We only describe ``some`` of the models. You can find ``ALL`` models from
the following address:

  `<https://github.com/k2-fsa/sherpa-onnx/releases/tag/source-separation-models>`_

Spleeter
--------

It is from `<https://github.com/deezer/spleeter>`_.

We only support the ``2-stem`` model at present.

.. hint::

   For those who want to learn how to convert the PyTorch checkpoint to
   the model supported in `sherpa-onnx`_, please see the scripts in following
   address:

    `<https://github.com/k2-fsa/sherpa-onnx/tree/master/scripts/spleeter>`_

There variants of the ``2-stem`` models are given below:


.. list-table::

 * - Model
   - Comment
 * - `sherpa-onnx-spleeter-2stems.tar.bz2 <https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/sherpa-onnx-spleeter-2stems.tar.bz2>`_
   - No quantization
 * - `sherpa-onnx-spleeter-2stems-int8.tar.bz2 <https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/sherpa-onnx-spleeter-2stems-int8.tar.bz2>`_
   - ``int8`` quantization
 * - `sherpa-onnx-spleeter-2stems-fp16.tar.bz2 <https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/sherpa-onnx-spleeter-2stems-fp16.tar.bz2>`_
   - ``fp16`` quantization

We describe how to use the ``fp16`` quantized model. Steps below are also applicable to other variants.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it:

.. code-block:: bash

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/sherpa-onnx-spleeter-2stems-fp16.tar.bz2
   tar xvf sherpa-onnx-spleeter-2stems-fp16.tar.bz2
   rm sherpa-onnx-spleeter-2stems-fp16.tar.bz2

   ls -lh sherpa-onnx-spleeter-2stems-fp16

You should see the following output:

.. code-block:: bash

  $ ls -lh sherpa-onnx-spleeter-2stems-fp16/

  total 76880
  -rw-r--r--  1 fangjun  staff    19M May 23 15:27 accompaniment.fp16.onnx
  -rw-r--r--  1 fangjun  staff    19M May 23 15:27 vocals.fp16.onnx

Download test files
~~~~~~~~~~~~~~~~~~~

We use the following two test wave files:

.. code-block:: bash

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/qi-feng-le-zh.wav

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/audio_example.wav

.. code-block:: bash

  ls -lh audio_example.wav qi-feng-le-zh.wav

  -rw-r--r--@ 1 fangjun  staff   1.8M May 23 15:59 audio_example.wav
  -rw-r--r--@ 1 fangjun  staff   4.4M May 23 22:06 qi-feng-le-zh.wav

.. hint::

   To make things easier, we support only ``*.wav`` files. If you have other formats, e.g.,
   ``*.mp3``, ``*.mp4``, or ``*.mov``, you can use

    .. code-block:: bash

      ffmpeg -i your.mp3 -vn -acodec pcm_s16le -ar 44100 -ac 2 your.wav
      ffmpeg -i your.mp4 -vn -acodec pcm_s16le -ar 44100 -ac 2 your.wav
      ffmpeg -i your.mov -vn -acodec pcm_s16le -ar 44100 -ac 2 your.wav

   to convert them to ``*.wav`` files.

The downloaded test files are given below.

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
    </tr>
    <tr>
      <td>qi-feng-le-zh.wav</td>
      <td>
       <audio title="qi-feng-le-zh.wav" controls="controls">
             <source src="/sherpa/_static/source-separation/qi-feng-le-zh.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
    </tr>

    <tr>
      <td>audio_example.wav</td>
      <td>
       <audio title="audio_example.wav" controls="controls">
             <source src="/sherpa/_static/source-separation/audio_example.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
    </tr>
  </table>

Example 1/2 with qi-feng-le-zh.wav
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline-source-separation \
    --spleeter-vocals=sherpa-onnx-spleeter-2stems-fp16/vocals.fp16.onnx \
    --spleeter-accompaniment=sherpa-onnx-spleeter-2stems-fp16/accompaniment.fp16.onnx \
    --num-threads=1 \
    --input-wav=./qi-feng-le-zh.wav \
    --output-vocals-wav=spleeter_qi_feng_le_vocals.wav \
    --output-accompaniment-wav=spleeter_qi_feng_le_non_vocals.wav

Output logs are given below::

  OfflineSourceSeparationConfig(model=OfflineSourceSeparationModelConfig(spleeter=OfflineSourceSeparationSpleeterModelConfig(vocals="sherpa-onnx-spleeter-2stems-fp16/vocals.fp16.onnx", accompaniment="sherpa-onnx-spleeter-2stems-fp16/accompaniment.fp16.onnx"), uvr=OfflineSourceSeparationUvrModelConfig(model=""), num_threads=1, debug=False, provider="cpu"))
  Started
  Done
  Saved to write to 'spleeter_qi_feng_le_vocals.wav' and 'spleeter_qi_feng_le_non_vocals.wav'
  num threads: 1
  Elapsed seconds: 2.052 s
  Real time factor (RTF): 2.052 / 26.102 = 0.079

.. hint::

   Pay special attention to its ``RTF``. It is super fast, on CPU, with only 1 thread!

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
    </tr>
    <tr>
      <td>qi-feng-le-zh.wav</td>
      <td>
       <audio title="qi-feng-le-zh.wav" controls="controls">
             <source src="/sherpa/_static/source-separation/qi-feng-le-zh.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
    </tr>

    <tr>
      <td>spleeter_qi_feng_le_<b style="color:red;">vocals</b>.wav</td>
      <td>
       <audio title="spleeter_qi_feng_le_vocals.wav" controls="controls">
             <source src="/sherpa/_static/source-separation/spleeter_qi_feng_le_vocals.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
    </tr>

    <tr>
      <td>spleeter_qi_feng_le_<b style="color:red;">non_vocals</b>.wav</td>
      <td>
       <audio title="spleeter_qi_feng_le_non_vocals.wav" controls="controls">
             <source src="/sherpa/_static/source-separation/spleeter_qi_feng_le_non_vocals.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
    </tr>

  </table>

Example 2/2 with audio_example.wav
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline-source-separation \
    --spleeter-vocals=sherpa-onnx-spleeter-2stems-fp16/vocals.fp16.onnx \
    --spleeter-accompaniment=sherpa-onnx-spleeter-2stems-fp16/accompaniment.fp16.onnx \
    --num-threads=1 \
    --input-wav=./audio_example.wav \
    --output-vocals-wav=spleeter_audio_example_vocals.wav \
    --output-accompaniment-wav=spleeter_audio_example_non_vocals.wav

Output logs are given below::

  /Users/fangjun/open-source/sherpa-onnx/sherpa-onnx/csrc/parse-options.cc:Read:372 ./build/bin/sherpa-onnx-offline-source-separation --spleeter-vocals=sherpa-onnx-spleeter-2stems-fp16/vocals.fp16.onnx --spleeter-accompaniment=sherpa-onnx-spleeter-2stems-fp16/accompaniment.fp16.onnx --num-threads=1 --input-wav=./audio_example.wav --output-vocals-wav=spleeter_audio_example_vocals.wav --output-accompaniment-wav=spleeter_audio_example_non_vocals.wav

  OfflineSourceSeparationConfig(model=OfflineSourceSeparationModelConfig(spleeter=OfflineSourceSeparationSpleeterModelConfig(vocals="sherpa-onnx-spleeter-2stems-fp16/vocals.fp16.onnx", accompaniment="sherpa-onnx-spleeter-2stems-fp16/accompaniment.fp16.onnx"), uvr=OfflineSourceSeparationUvrModelConfig(model=""), num_threads=1, debug=False, provider="cpu"))
  Started
  Done
  Saved to write to 'spleeter_audio_example_vocals.wav' and 'spleeter_audio_example_non_vocals.wav'
  num threads: 1
  Elapsed seconds: 0.787 s
  Real time factor (RTF): 0.787 / 10.919 = 0.072

.. hint::

   Pay special attention to its ``RTF``. It is super fast, on CPU, with only 1 thread!

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
    </tr>
    <tr>
      <td>audio_example.wav</td>
      <td>
       <audio title="audio_example.wav" controls="controls">
             <source src="/sherpa/_static/source-separation/audio_example.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
    </tr>

    <tr>
      <td>spleeter_audio_example_<b style="color:red;">vocals</b>.wav</td>
      <td>
       <audio title="spleeter_audio_example_vocals.wav" controls="controls">
             <source src="/sherpa/_static/source-separation/spleeter_audio_example_vocals.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
    </tr>

    <tr>
      <td>spleeter_audio_example_<b style="color:red;">non_vocals</b>.wav</td>
      <td>
       <audio title="spleeter_audio_example_non_vocals.wav" controls="controls">
             <source src="/sherpa/_static/source-separation/spleeter_audio_example_non_vocals.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
    </tr>

  </table>

RTF on RK3588
~~~~~~~~~~~~~

We use the following code to test the RTF of `Spleeter` on RK3588
with Cortex ``A76`` CPU.

.. code-block:: bash

  # 1 thread
  taskset 0x80  ./build/bin/sherpa-onnx-offline-source-separation \
    --num-threads=1 \
    --spleeter-vocals=sherpa-onnx-spleeter-2stems-fp16/vocals.fp16.onnx \
    --spleeter-accompaniment=sherpa-onnx-spleeter-2stems-fp16/accompaniment.fp16.onnx \
    --input-wav=./qi-feng-le-zh.wav \
    --output-vocals-wav=spleeter_qi_feng_le_vocals.wav \
    --output-accompaniment-wav=spleeter_qi_feng_le_non_vocals.wav

  # 2 threads
  taskset 0xc0  ./build/bin/sherpa-onnx-offline-source-separation \
    --num-threads=2 \
    --spleeter-vocals=sherpa-onnx-spleeter-2stems-fp16/vocals.fp16.onnx \
    --spleeter-accompaniment=sherpa-onnx-spleeter-2stems-fp16/accompaniment.fp16.onnx \
    --input-wav=./qi-feng-le-zh.wav \
    --output-vocals-wav=spleeter_qi_feng_le_vocals.wav \
    --output-accompaniment-wav=spleeter_qi_feng_le_non_vocals.wav

  # 3 threads
  taskset 0xe0  ./build/bin/sherpa-onnx-offline-source-separation \
    --num-threads=3 \
    --spleeter-vocals=sherpa-onnx-spleeter-2stems-fp16/vocals.fp16.onnx \
    --spleeter-accompaniment=sherpa-onnx-spleeter-2stems-fp16/accompaniment.fp16.onnx \
    --input-wav=./qi-feng-le-zh.wav \
    --output-vocals-wav=spleeter_qi_feng_le_vocals.wav \
    --output-accompaniment-wav=spleeter_qi_feng_le_non_vocals.wav

  # 4 threads
  taskset 0xf0  ./build/bin/sherpa-onnx-offline-source-separation \
    --num-threads=4 \
    --spleeter-vocals=sherpa-onnx-spleeter-2stems-fp16/vocals.fp16.onnx \
    --spleeter-accompaniment=sherpa-onnx-spleeter-2stems-fp16/accompaniment.fp16.onnx \
    --input-wav=./qi-feng-le-zh.wav \
    --output-vocals-wav=spleeter_qi_feng_le_vocals.wav \
    --output-accompaniment-wav=spleeter_qi_feng_le_non_vocals.wav

The results are given below:

  +------------------------+-------+-------+-------+-------+
  | num_threads            | 1     | 2     | 3     | 4     |
  +========================+=======+=======+=======+=======+
  | RTF on Cortex A76 CPU  | 0.258 | 0.176 | 0.138 | 0.127 |
  +------------------------+-------+-------+-------+-------+

Python example
~~~~~~~~~~~~~~

Please see

  `<https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/offline-source-separation-spleeter.py>`_

UVR
---

It is from `<https://github.com/TRvlvr/model_repo/releases/tag/all_public_uvr_models>`_.

.. hint::

   For those who want to learn how to add meta data to the original ONNX models,
   please see the scripts in following address:

    `<https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/uvr_mdx/>`_

.. warning::

   Please download  ``UVR`` models from `<https://github.com/k2-fsa/sherpa-onnx/releases/tag/source-separation-models>`_

   Please ``don't`` download ``UVR`` models from `<https://github.com/TRvlvr/model_repo/releases/tag/all_public_uvr_models>`_


We support the following ``UVR`` models for source separation.

.. list-table::

 * - Model
   - File size (MB)
 * - `UVR-MDX-NET-Inst_1.onnx <https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/UVR-MDX-NET-Inst_1.onnx>`_
   - 63.7
 * - `UVR-MDX-NET-Inst_2.onnx <https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/UVR-MDX-NET-Inst_2.onnx>`_
   - 63.7
 * - `UVR-MDX-NET-Inst_3.onnx <https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/UVR-MDX-NET-Inst_3.onnx>`_
   - 63.7
 * - `UVR-MDX-NET-Inst_HQ_1.onnx <https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/UVR-MDX-NET-Inst_HQ_1.onnx>`_
   - 63.7
 * - `UVR-MDX-NET-Inst_HQ_2.onnx <https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/UVR-MDX-NET-Inst_HQ_2.onnx>`_
   - 63.7
 * - `UVR-MDX-NET-Inst_HQ_3.onnx <https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/UVR-MDX-NET-Inst_HQ_3.onnx>`_
   - 63.7
 * - `UVR-MDX-NET-Inst_HQ_4.onnx <https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/UVR-MDX-NET-Inst_HQ_4.onnx>`_
   - 56.3
 * - `UVR-MDX-NET-Inst_HQ_5.onnx <https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/UVR-MDX-NET-Inst_HQ_5.onnx>`_
   - 56.3
 * - `UVR-MDX-NET-Inst_Main.onnx <https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/UVR-MDX-NET-Inst_Main.onnx>`_
   - 50.3
 * - `UVR-MDX-NET-Voc_FT.onnx <https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/UVR-MDX-NET-Voc_FT.onnx>`_
   - 63.7
 * - `UVR-MDX-NET_Crowd_HQ_1.onnx <https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/UVR-MDX-NET_Crowd_HQ_1.onnx>`_
   - 56.3
 * - `UVR_MDXNET_1_9703.onnx <https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/UVR_MDXNET_1_9703.onnx>`_
   - 28.3
 * - `UVR_MDXNET_2_9682.onnx <https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/UVR_MDXNET_2_9682.onnx>`_
   - 28.3
 * - `UVR_MDXNET_3_9662.onnx <https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/UVR_MDXNET_3_9662.onnx>`_
   - 28.3
 * - `UVR_MDXNET_9482.onnx <https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/UVR_MDXNET_9482.onnx>`_
   - 28.3
 * - `UVR_MDXNET_KARA.onnx <https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/UVR_MDXNET_KARA.onnx>`_
   - 28.3
 * - `UVR_MDXNET_KARA_2.onnx <https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/UVR_MDXNET_KARA_2.onnx>`_
   - 50.3
 * - `UVR_MDXNET_Main.onnx <https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/UVR_MDXNET_Main.onnx>`_
   - 63.7


In the following, we show how to use the model
`UVR_MDXNET_9482.onnx <https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/UVR_MDXNET_9482.onnx>`_

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it:

.. code-block:: bash

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/UVR_MDXNET_9482.onnx

   ls -lh UVR_MDXNET_9482.onnx

You should see the following output:

.. code-block:: bash

  ls -lh UVR_MDXNET_9482.onnx

  -rw-r--r--  1 fangjun  staff    28M May 31 13:33 UVR_MDXNET_9482.onnx

Download test files
~~~~~~~~~~~~~~~~~~~

We use the following two test wave files:

.. code-block:: bash

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/qi-feng-le-zh.wav

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/audio_example.wav

.. code-block:: bash

  ls -lh audio_example.wav qi-feng-le-zh.wav

  -rw-r--r--@ 1 fangjun  staff   1.8M May 23 15:59 audio_example.wav
  -rw-r--r--@ 1 fangjun  staff   4.4M May 23 22:06 qi-feng-le-zh.wav

.. hint::

   To make things easier, we support only ``*.wav`` files. If you have other formats, e.g.,
   ``*.mp3``, ``*.mp4``, or ``*.mov``, you can use

    .. code-block:: bash

      ffmpeg -i your.mp3 -vn -acodec pcm_s16le -ar 44100 -ac 2 your.wav
      ffmpeg -i your.mp4 -vn -acodec pcm_s16le -ar 44100 -ac 2 your.wav
      ffmpeg -i your.mov -vn -acodec pcm_s16le -ar 44100 -ac 2 your.wav

   to convert them to ``*.wav`` files.

The downloaded test files are given below.

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
    </tr>
    <tr>
      <td>qi-feng-le-zh.wav</td>
      <td>
       <audio title="qi-feng-le-zh.wav" controls="controls">
             <source src="/sherpa/_static/source-separation/qi-feng-le-zh.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
    </tr>

    <tr>
      <td>audio_example.wav</td>
      <td>
       <audio title="audio_example.wav" controls="controls">
             <source src="/sherpa/_static/source-separation/audio_example.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
    </tr>
  </table>



Example 1/2 with qi-feng-le-zh.wav
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline-source-separation \
    --num-threads=1 \
    --uvr-model=./UVR_MDXNET_9482.onnx \
    --input-wav=./qi-feng-le-zh.wav \
    --output-vocals-wav=uvr_qi_feng_le_vocals.wav \
    --output-accompaniment-wav=uvr_qi_feng_le_non_vocals.wav

Output logs are given below::

  /Users/fangjun/open-source/sherpa-onnx/sherpa-onnx/csrc/parse-options.cc:Read:372 ./build/bin/sherpa-onnx-offline-source-separation --num-threads=1 --uvr-model=./UVR_MDXNET_9482.onnx --input-wav=./qi-feng-le-zh.wav --output-vocals-wav=uvr_qi_feng_le_vocals.wav --output-accompaniment-wav=uvr_qi_feng_le_non_vocals.wav

  OfflineSourceSeparationConfig(model=OfflineSourceSeparationModelConfig(spleeter=OfflineSourceSeparationSpleeterModelConfig(vocals="", accompaniment=""), uvr=OfflineSourceSeparationUvrModelConfig(model="./UVR_MDXNET_9482.onnx"), num_threads=1, debug=False, provider="cpu"))
  Started
  Done
  Saved to write to 'uvr_qi_feng_le_vocals.wav' and 'uvr_qi_feng_le_non_vocals.wav'
  num threads: 1
  Elapsed seconds: 19.110 s
  Real time factor (RTF): 19.110 / 26.102 = 0.732

.. hint::

   It is ``10x`` slower than ``Spleeter``! Also, we have selected a small model.
   If you select a model with more parameters, it is even slower.

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
    </tr>
    <tr>
      <td>qi-feng-le-zh.wav</td>
      <td>
       <audio title="qi-feng-le-zh.wav" controls="controls">
             <source src="/sherpa/_static/source-separation/qi-feng-le-zh.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
    </tr>

    <tr>
      <td>uvr_qi_feng_le_<b style="color:red;">vocals</b>.wav</td>
      <td>
       <audio title="uvr_qi_feng_le_vocals.wav" controls="controls">
             <source src="/sherpa/_static/source-separation/uvr_qi_feng_le_vocals.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
    </tr>

    <tr>
      <td>uvr_qi_feng_le_<b style="color:red;">non_vocals</b>.wav</td>
      <td>
       <audio title="uvr_qi_feng_le_non_vocals.wav" controls="controls">
             <source src="/sherpa/_static/source-separation/uvr_qi_feng_le_non_vocals.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
    </tr>

  </table>

Example 2/2 with audio_example.wav
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline-source-separation \
    --num-threads=1 \
    --uvr-model=./UVR_MDXNET_9482.onnx \
    --input-wav=./audio_example.wav \
    --output-vocals-wav=uvr_audio_example_vocals.wav \
    --output-accompaniment-wav=uvr_audio_example_non_vocals.wav

Output logs are given below::

  /Users/fangjun/open-source/sherpa-onnx/sherpa-onnx/csrc/parse-options.cc:Read:372 ./build/bin/sherpa-onnx-offline-source-separation --num-threads=1 --uvr-model=./UVR_MDXNET_9482.onnx --input-wav=./audio_example.wav --output-vocals-wav=uvr_audio_example_vocals.wav --output-accompaniment-wav=uvr_audio_example_non_vocals.wav

  OfflineSourceSeparationConfig(model=OfflineSourceSeparationModelConfig(spleeter=OfflineSourceSeparationSpleeterModelConfig(vocals="", accompaniment=""), uvr=OfflineSourceSeparationUvrModelConfig(model="./UVR_MDXNET_9482.onnx"), num_threads=1, debug=False, provider="cpu"))
  Started
  Done
  Saved to write to 'uvr_audio_example_vocals.wav' and 'uvr_audio_example_non_vocals.wav'
  num threads: 1
  Elapsed seconds: 6.420 s
  Real time factor (RTF): 6.420 / 10.919 = 0.588


.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
    </tr>
    <tr>
      <td>audio_example.wav</td>
      <td>
       <audio title="audio_example.wav" controls="controls">
             <source src="/sherpa/_static/source-separation/audio_example.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
    </tr>

    <tr>
      <td>uvr_audio_example_<b style="color:red;">vocals</b>.wav</td>
      <td>
       <audio title="uvr_audio_example_vocals.wav" controls="controls">
             <source src="/sherpa/_static/source-separation/uvr_audio_example_vocals.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
    </tr>

    <tr>
      <td>uvr_audio_example_<b style="color:red;">non_vocals</b>.wav</td>
      <td>
       <audio title="uvr_audio_example_non_vocals.wav" controls="controls">
             <source src="/sherpa/_static/source-separation/uvr_audio_example_non_vocals.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
    </tr>

  </table>

Python example
~~~~~~~~~~~~~~

Please see

  `<https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/offline-source-separation-uvr.py>`_
