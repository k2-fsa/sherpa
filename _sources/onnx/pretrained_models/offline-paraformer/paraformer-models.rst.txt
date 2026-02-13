Paraformer models
=================

.. hint::

   Please refer to :ref:`install_sherpa_onnx` to install `sherpa-onnx`_
   before you read this section.

.. _sherpa-onnx-paraformer-zh-int8-2025-10-07:

sherpa-onnx-paraformer-zh-int8-2025-10-07 (四川话、重庆话、川渝方言)
----------------------------------------------------------------------------

This model is converted from

  `<https://huggingface.co/ASLP-lab/WSChuan-ASR/tree/main/Paraformer-large-Chuan>`_

It is fine-tuned on :ref:`sherpa_onnx_offline_paraformer_zh_2023_03_28_chinese` with 10k hours
of ``Sichuanese`` (川渝方言) data.

In the following, we describe how to use it.

Huggingface space
^^^^^^^^^^^^^^^^^

You can visit

  `<https://huggingface.co/spaces/k2-fsa/automatic-speech-recognition>`_

to try this model in your browser.

.. hint::

   You need to first select the language ``四川话``
   and then select the model  ``csukuangfj/sherpa-onnx-paraformer-zh-int8-2025-10-07``.


Android APKs
^^^^^^^^^^^^

Real-time speech recognition Android APKs can be found at

  `<https://k2-fsa.github.io/sherpa/onnx/android/apk-simulate-streaming-asr.html>`_

Please always download the latest version.

.. hint::

   Please search for ``paraformer_四川话.apk`` in the above page, e.g.,
   ``sherpa-onnx-1.12.15-arm64-v8a-simulated_streaming_asr-zh-paraformer_四川话.apk``.

.. hint::

   For Chinese users, you can also visit `<https://k2-fsa.github.io/sherpa/onnx/android/apk-simulate-streaming-asr-cn.html>`_

Download
^^^^^^^^

Please use the following commands to download it::

  cd /path/to/sherpa-onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-int8-2025-10-07.tar.bz2
  tar xvf sherpa-onnx-paraformer-zh-int8-2025-10-07.tar.bz2
  rm sherpa-onnx-paraformer-zh-int8-2025-10-07.tar.bz2

After downloading, you should find the following files::

  ls -lh sherpa-onnx-paraformer-zh-int8-2025-10-07

  total 491872
  -rw-r--r--@  1 fangjun  staff   227M  7 Oct 20:19 model.int8.onnx
  -rw-r--r--@  1 fangjun  staff   337B  7 Oct 20:19 README.md
  drwxr-xr-x@ 18 fangjun  staff   576B  7 Oct 20:19 test_wavs
  -rw-r--r--@  1 fangjun  staff    74K  7 Oct 20:19 tokens.txt

.. code-block::

  ls sherpa-onnx-paraformer-zh-int8-2025-10-07/test_wavs/

  1.wav  10.wav 11.wav 12.wav 13.wav 14.wav 15.wav 16.wav 2.wav  3.wav  4.wav  5.wav  6.wav  7.wav  8.wav  9.wav

In the following, we show how to decode the files ``sherpa-onnx-paraformer-zh-int8-2025-10-07/test_wavs/*.wav``.

1.wav
:::::

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Ground truth</th>
    </tr>
    <tr>
      <td>1.wav</td>
      <td>
       <audio title="1.wav" controls="controls">
             <source src="/sherpa/_static/sherpa-onnx-paraformer-zh-int8-2025-10-07/1.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
      来，哥哥再给你唱首歌。好儿，哎呦，把伴奏给我放起来，放就放嘛，还要躲人家钩子。
      </td>
    </tr>
  </table>

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-paraformer-zh-int8-2025-10-07/tokens.txt \
    --paraformer=./sherpa-onnx-paraformer-zh-int8-2025-10-07/model.int8.onnx \
    --num-threads=1 \
    sherpa-onnx-paraformer-zh-int8-2025-10-07/test_wavs/1.wav

.. literalinclude:: ./code-paraformer/chuan-1.txt

2.wav
:::::

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Ground truth</th>
    </tr>
    <tr>
      <td>2.wav</td>
      <td>
       <audio title="2.wav" controls="controls">
             <source src="/sherpa/_static/sherpa-onnx-paraformer-zh-int8-2025-10-07/2.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
      对不起，只有二娃才能让我真正体会作为女人的快乐。
      </td>
    </tr>
  </table>

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-paraformer-zh-int8-2025-10-07/tokens.txt \
    --paraformer=./sherpa-onnx-paraformer-zh-int8-2025-10-07/model.int8.onnx \
    --num-threads=1 \
    sherpa-onnx-paraformer-zh-int8-2025-10-07/test_wavs/2.wav

.. literalinclude:: ./code-paraformer/chuan-2.txt

3.wav
:::::

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Ground truth</th>
    </tr>
    <tr>
      <td>3.wav</td>
      <td>
       <audio title="3.wav" controls="controls">
             <source src="/sherpa/_static/sherpa-onnx-paraformer-zh-int8-2025-10-07/3.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
      我想去逛街，欢迎进入直播间，晚上好，那我的名字是怎么说的呢？
      </td>
    </tr>
  </table>

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-paraformer-zh-int8-2025-10-07/tokens.txt \
    --paraformer=./sherpa-onnx-paraformer-zh-int8-2025-10-07/model.int8.onnx \
    --num-threads=1 \
    sherpa-onnx-paraformer-zh-int8-2025-10-07/test_wavs/3.wav

.. literalinclude:: ./code-paraformer/chuan-3.txt

4.wav
:::::

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Ground truth</th>
    </tr>
    <tr>
      <td>4.wav</td>
      <td>
       <audio title="4.wav" controls="controls">
             <source src="/sherpa/_static/sherpa-onnx-paraformer-zh-int8-2025-10-07/4.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
      梦见的就是你，不行啊，有四川话根本唱不起来，根本唱不起来呀！
      </td>
    </tr>
  </table>

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-paraformer-zh-int8-2025-10-07/tokens.txt \
    --paraformer=./sherpa-onnx-paraformer-zh-int8-2025-10-07/model.int8.onnx \
    --num-threads=1 \
    sherpa-onnx-paraformer-zh-int8-2025-10-07/test_wavs/4.wav

.. literalinclude:: ./code-paraformer/chuan-4.txt

5.wav
:::::

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Ground truth</th>
    </tr>
    <tr>
      <td>5.wav</td>
      <td>
       <audio title="5.wav" controls="controls">
             <source src="/sherpa/_static/sherpa-onnx-paraformer-zh-int8-2025-10-07/5.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
      就临走那天挑了个飘了一下嗨呀，弟弟灵魂儿就飞上九霄云，就飘着一下魂都飞了，对不对？
      </td>
    </tr>
  </table>

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-paraformer-zh-int8-2025-10-07/tokens.txt \
    --paraformer=./sherpa-onnx-paraformer-zh-int8-2025-10-07/model.int8.onnx \
    --num-threads=1 \
    sherpa-onnx-paraformer-zh-int8-2025-10-07/test_wavs/5.wav

.. literalinclude:: ./code-paraformer/chuan-5.txt

6.wav
:::::

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Ground truth</th>
    </tr>
    <tr>
      <td>6.wav</td>
      <td>
       <audio title="6.wav" controls="controls">
             <source src="/sherpa/_static/sherpa-onnx-paraformer-zh-int8-2025-10-07/6.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
      是不是给人感觉后头是青花亮色的然后说话是很平和的眼神是不慌乱的不散的。
      </td>
    </tr>
  </table>

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-paraformer-zh-int8-2025-10-07/tokens.txt \
    --paraformer=./sherpa-onnx-paraformer-zh-int8-2025-10-07/model.int8.onnx \
    --num-threads=1 \
    sherpa-onnx-paraformer-zh-int8-2025-10-07/test_wavs/6.wav

.. literalinclude:: ./code-paraformer/chuan-6.txt

7.wav
:::::

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Ground truth</th>
    </tr>
    <tr>
      <td>7.wav</td>
      <td>
       <audio title="7.wav" controls="controls">
             <source src="/sherpa/_static/sherpa-onnx-paraformer-zh-int8-2025-10-07/7.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
      他坐在椅子上，挺直起腰杆，脸上展现出灿烂的笑容。
      </td>
    </tr>
  </table>

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-paraformer-zh-int8-2025-10-07/tokens.txt \
    --paraformer=./sherpa-onnx-paraformer-zh-int8-2025-10-07/model.int8.onnx \
    --num-threads=1 \
    sherpa-onnx-paraformer-zh-int8-2025-10-07/test_wavs/7.wav

.. literalinclude:: ./code-paraformer/chuan-7.txt

8.wav
:::::

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Ground truth</th>
    </tr>
    <tr>
      <td>8.wav</td>
      <td>
       <audio title="8.wav" controls="controls">
             <source src="/sherpa/_static/sherpa-onnx-paraformer-zh-int8-2025-10-07/8.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
      唤起路由无限的感慨，使他，更加痛恨官场的欺诈污浊。
      </td>
    </tr>
  </table>

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-paraformer-zh-int8-2025-10-07/tokens.txt \
    --paraformer=./sherpa-onnx-paraformer-zh-int8-2025-10-07/model.int8.onnx \
    --num-threads=1 \
    sherpa-onnx-paraformer-zh-int8-2025-10-07/test_wavs/8.wav

.. literalinclude:: ./code-paraformer/chuan-8.txt

9.wav
:::::

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Ground truth</th>
    </tr>
    <tr>
      <td>9.wav</td>
      <td>
       <audio title="9.wav" controls="controls">
             <source src="/sherpa/_static/sherpa-onnx-paraformer-zh-int8-2025-10-07/9.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
      看面貌约五十左右却自称活了两百多岁，在清顺治时出家当过和尚，还有杜蝶为证。
      </td>
    </tr>
  </table>

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-paraformer-zh-int8-2025-10-07/tokens.txt \
    --paraformer=./sherpa-onnx-paraformer-zh-int8-2025-10-07/model.int8.onnx \
    --num-threads=1 \
    sherpa-onnx-paraformer-zh-int8-2025-10-07/test_wavs/9.wav

.. literalinclude:: ./code-paraformer/chuan-9.txt

10.wav
:::::

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Ground truth</th>
    </tr>
    <tr>
      <td>10.wav</td>
      <td>
       <audio title="10.wav" controls="controls">
             <source src="/sherpa/_static/sherpa-onnx-paraformer-zh-int8-2025-10-07/10.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
      其言曰，士大夫以其见闻之广反各有所偏，自有负担杀者有负良骑者。
      </td>
    </tr>
  </table>

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-paraformer-zh-int8-2025-10-07/tokens.txt \
    --paraformer=./sherpa-onnx-paraformer-zh-int8-2025-10-07/model.int8.onnx \
    --num-threads=1 \
    sherpa-onnx-paraformer-zh-int8-2025-10-07/test_wavs/10.wav

.. literalinclude:: ./code-paraformer/chuan-10.txt

11.wav
:::::

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Ground truth</th>
    </tr>
    <tr>
      <td>11.wav</td>
      <td>
       <audio title="11.wav" controls="controls">
             <source src="/sherpa/_static/sherpa-onnx-paraformer-zh-int8-2025-10-07/11.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
      据说有网友坐飞机的时候呢，广播全程播报。
      </td>
    </tr>
  </table>

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-paraformer-zh-int8-2025-10-07/tokens.txt \
    --paraformer=./sherpa-onnx-paraformer-zh-int8-2025-10-07/model.int8.onnx \
    --num-threads=1 \
    sherpa-onnx-paraformer-zh-int8-2025-10-07/test_wavs/11.wav

.. literalinclude:: ./code-paraformer/chuan-11.txt

12.wav
:::::

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Ground truth</th>
    </tr>
    <tr>
      <td>12.wav</td>
      <td>
       <audio title="12.wav" controls="controls">
             <source src="/sherpa/_static/sherpa-onnx-paraformer-zh-int8-2025-10-07/12.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
      将溃疡两周以上都应该及时就医，据了解啊小云平时呢都喜欢吃比较烫的饭菜，也喜欢吃麻辣烫火锅之类的高温食物。
      </td>
    </tr>
  </table>

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-paraformer-zh-int8-2025-10-07/tokens.txt \
    --paraformer=./sherpa-onnx-paraformer-zh-int8-2025-10-07/model.int8.onnx \
    --num-threads=1 \
    sherpa-onnx-paraformer-zh-int8-2025-10-07/test_wavs/12.wav

.. literalinclude:: ./code-paraformer/chuan-12.txt

13.wav
:::::

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Ground truth</th>
    </tr>
    <tr>
      <td>13.wav</td>
      <td>
       <audio title="13.wav" controls="controls">
             <source src="/sherpa/_static/sherpa-onnx-paraformer-zh-int8-2025-10-07/13.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
      绝佳好位置好像我被看到了，就问你敢不敢进来吧你，一套带走猪脚亮。
      </td>
    </tr>
  </table>

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-paraformer-zh-int8-2025-10-07/tokens.txt \
    --paraformer=./sherpa-onnx-paraformer-zh-int8-2025-10-07/model.int8.onnx \
    --num-threads=1 \
    sherpa-onnx-paraformer-zh-int8-2025-10-07/test_wavs/13.wav

.. literalinclude:: ./code-paraformer/chuan-13.txt

14.wav
:::::

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Ground truth</th>
    </tr>
    <tr>
      <td>14.wav</td>
      <td>
       <audio title="14.wav" controls="controls">
             <source src="/sherpa/_static/sherpa-onnx-paraformer-zh-int8-2025-10-07/14.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
      两岸猿声啼不住，有家难回车里住。
      </td>
    </tr>
  </table>

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-paraformer-zh-int8-2025-10-07/tokens.txt \
    --paraformer=./sherpa-onnx-paraformer-zh-int8-2025-10-07/model.int8.onnx \
    --num-threads=1 \
    sherpa-onnx-paraformer-zh-int8-2025-10-07/test_wavs/14.wav

.. literalinclude:: ./code-paraformer/chuan-14.txt

15.wav
:::::

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Ground truth</th>
    </tr>
    <tr>
      <td>15.wav</td>
      <td>
       <audio title="15.wav" controls="controls">
             <source src="/sherpa/_static/sherpa-onnx-paraformer-zh-int8-2025-10-07/15.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
      杨大人一律就退还会再要求，以关注货币，来补助这个差额，天宝年间杨胜坚转任。
      </td>
    </tr>
  </table>

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-paraformer-zh-int8-2025-10-07/tokens.txt \
    --paraformer=./sherpa-onnx-paraformer-zh-int8-2025-10-07/model.int8.onnx \
    --num-threads=1 \
    sherpa-onnx-paraformer-zh-int8-2025-10-07/test_wavs/15.wav

.. literalinclude:: ./code-paraformer/chuan-15.txt

16.wav
:::::

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Ground truth</th>
    </tr>
    <tr>
      <td>16.wav</td>
      <td>
       <audio title="16.wav" controls="controls">
             <source src="/sherpa/_static/sherpa-onnx-paraformer-zh-int8-2025-10-07/16.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
      做钱的速度还快，这真的是，一个经济爆发式增长的时代。
      </td>
    </tr>
  </table>

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-paraformer-zh-int8-2025-10-07/tokens.txt \
    --paraformer=./sherpa-onnx-paraformer-zh-int8-2025-10-07/model.int8.onnx \
    --num-threads=1 \
    sherpa-onnx-paraformer-zh-int8-2025-10-07/test_wavs/16.wav

.. literalinclude:: ./code-paraformer/chuan-16.txt

Speech recognition from a microphone
:::::::::::::::::::::::::::::::::::::

.. code-block:: bash

  ./build/bin/sherpa-onnx-microphone-offline \
    --tokens=./sherpa-onnx-paraformer-zh-int8-2025-10-07/tokens.txt \
    --paraformer=./sherpa-onnx-paraformer-zh-int8-2025-10-07/model.int8.onnx

Speech recognition from a microphone with VAD
::::::::::::::::::::::::::::::::::::::::::::::

.. code-block:: bash

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

  ./build/bin/sherpa-onnx-vad-microphone-offline-asr \
    --silero-vad-model=./silero_vad.onnx \
    --tokens=./sherpa-onnx-paraformer-zh-int8-2025-10-07/tokens.txt \
    --paraformer=./sherpa-onnx-paraformer-zh-int8-2025-10-07/model.int8.onnx

Real-time speech recognition from a microphone with VAD
:::::::::::::::::::::::::::::::::::::::::::::::::::::::

.. code-block:: bash

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

  ./build/bin/sherpa-onnx-vad-microphone-simulated-streaming-asr \
    --silero-vad-model=./silero_vad.onnx \
    --tokens=./sherpa-onnx-paraformer-zh-int8-2025-10-07/tokens.txt \
    --paraformer=./sherpa-onnx-paraformer-zh-int8-2025-10-07/model.int8.onnx

.. _sherpa_onnx_offline_paraformer_trilingual_zh_cantonese_en:

csukuangfj/sherpa-onnx-paraformer-trilingual-zh-cantonese-en (Chinese + English + Cantonese 粤语)
-------------------------------------------------------------------------------------------------

.. note::

   This model does not support timestamps. It is a trilingual model, supporting
   both Chinese and English. (支持普通话、``粤语``、河南话、天津话、四川话等方言)

This model is converted from

`<https://www.modelscope.cn/models/dengcunqin/speech_seaco_paraformer_large_asr_nat-zh-cantonese-en-16k-common-vocab11666-pytorch/summary>`_

In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

  cd /path/to/sherpa-onnx
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-trilingual-zh-cantonese-en.tar.bz2

  tar xvf sherpa-onnx-paraformer-trilingual-zh-cantonese-en.tar.bz2

Please check that the file sizes of the pre-trained models are correct. See
the file sizes of ``*.onnx`` files below.

.. code-block:: bash

  sherpa-onnx-paraformer-trilingual-zh-cantonese-en$ ls -lh *.onnx

  -rw-r--r-- 1 1001 127 234M Mar 10 02:12 model.int8.onnx
  -rw-r--r-- 1 1001 127 831M Mar 10 02:12 model.onnx

Decode wave files
~~~~~~~~~~~~~~~~~

.. hint::

   It supports decoding only wave files of a single channel with 16-bit
   encoded samples, while the sampling rate does not need to be 16 kHz.

fp32
^^^^

The following code shows how to use ``fp32`` models to decode wave files:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-paraformer-trilingual-zh-cantonese-en/tokens.txt \
    --paraformer=./sherpa-onnx-paraformer-trilingual-zh-cantonese-en/model.onnx \
    ./sherpa-onnx-paraformer-trilingual-zh-cantonese-en/test_wavs/1.wav \
    ./sherpa-onnx-paraformer-trilingual-zh-cantonese-en/test_wavs/2.wav \
    ./sherpa-onnx-paraformer-trilingual-zh-cantonese-en/test_wavs/3-sichuan.wav \
    ./sherpa-onnx-paraformer-trilingual-zh-cantonese-en/test_wavs/4-tianjin.wav \
    ./sherpa-onnx-paraformer-trilingual-zh-cantonese-en/test_wavs/5-henan.wav \
    ./sherpa-onnx-paraformer-trilingual-zh-cantonese-en/test_wavs/6-zh-en.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

You should see the following output:

.. literalinclude:: ./code-paraformer/sherpa-onnx-paraformer-trilingual-zh-cantonese-en.txt

int8
^^^^

The following code shows how to use ``int8`` models to decode wave files:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-paraformer-trilingual-zh-cantonese-en/tokens.txt \
    --paraformer=./sherpa-onnx-paraformer-trilingual-zh-cantonese-en/model.int8.onnx \
    ./sherpa-onnx-paraformer-trilingual-zh-cantonese-en/test_wavs/1.wav \
    ./sherpa-onnx-paraformer-trilingual-zh-cantonese-en/test_wavs/2.wav \
    ./sherpa-onnx-paraformer-trilingual-zh-cantonese-en/test_wavs/3-sichuan.wav \
    ./sherpa-onnx-paraformer-trilingual-zh-cantonese-en/test_wavs/4-tianjin.wav \
    ./sherpa-onnx-paraformer-trilingual-zh-cantonese-en/test_wavs/5-henan.wav \
    ./sherpa-onnx-paraformer-trilingual-zh-cantonese-en/test_wavs/6-zh-en.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

You should see the following output:

.. literalinclude:: ./code-paraformer/sherpa-onnx-paraformer-trilingual-zh-cantonese-en-int8.txt

Speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-microphone-offline \
    --tokens=./sherpa-onnx-paraformer-trilingual-zh-cantonese-en/tokens.txt \
    --paraformer=./sherpa-onnx-paraformer-trilingual-zh-cantonese-en/model.int8.onnx

.. _sherpa_onnx_offline_paraformer_en_2024_03_09_english:

csukuangfj/sherpa-onnx-paraformer-en-2024-03-09 (English)
---------------------------------------------------------

.. note::

   This model does not support timestamps. It supports only English.

This model is converted from

`<https://www.modelscope.cn/models/iic/speech_paraformer_asr-en-16k-vocab4199-pytorch/summary>`_

In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

  cd /path/to/sherpa-onnx
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-en-2024-03-09.tar.bz2

  tar xvf sherpa-onnx-paraformer-en-2024-03-09.tar.bz2

Please check that the file sizes of the pre-trained models are correct. See
the file sizes of ``*.onnx`` files below.

.. code-block:: bash

  sherpa-onnx-paraformer-en-2024-03-09$ ls -lh *.onnx

  -rw-r--r-- 1 1001 127 220M Mar 10 02:12 model.int8.onnx
  -rw-r--r-- 1 1001 127 817M Mar 10 02:12 model.onnx

Decode wave files
~~~~~~~~~~~~~~~~~

.. hint::

   It supports decoding only wave files of a single channel with 16-bit
   encoded samples, while the sampling rate does not need to be 16 kHz.

fp32
^^^^

The following code shows how to use ``fp32`` models to decode wave files:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-paraformer-en-2024-03-09/tokens.txt \
    --paraformer=./sherpa-onnx-paraformer-en-2024-03-09/model.onnx \
    ./sherpa-onnx-paraformer-en-2024-03-09/test_wavs/0.wav \
    ./sherpa-onnx-paraformer-en-2024-03-09/test_wavs/1.wav \
    ./sherpa-onnx-paraformer-en-2024-03-09/test_wavs/8k.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

You should see the following output:

.. literalinclude:: ./code-paraformer/sherpa-onnx-paraformer-en-2024-03-09.txt

int8
^^^^

The following code shows how to use ``int8`` models to decode wave files:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-paraformer-en-2024-03-09/tokens.txt \
    --paraformer=./sherpa-onnx-paraformer-en-2024-03-09/model.int8.onnx \
    ./sherpa-onnx-paraformer-en-2024-03-09/test_wavs/0.wav \
    ./sherpa-onnx-paraformer-en-2024-03-09/test_wavs/1.wav \
    ./sherpa-onnx-paraformer-en-2024-03-09/test_wavs/8k.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

You should see the following output:

.. literalinclude:: ./code-paraformer/sherpa-onnx-paraformer-en-2024-03-09-int8.txt

Speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-microphone-offline \
    --tokens=./sherpa-onnx-paraformer-en-2024-03-09/tokens.txt \
    --paraformer=./sherpa-onnx-paraformer-en-2024-03-09/model.int8.onnx

.. _sherpa_onnx_offline_paraformer_zh_small_2024_03_09_chinese_english:

csukuangfj/sherpa-onnx-paraformer-zh-small-2024-03-09 (Chinese + English)
-------------------------------------------------------------------------

.. note::

   This model does not support timestamps. It is a bilingual model, supporting
   both Chinese and English. (支持普通话、河南话、天津话、四川话等方言)

This model is converted from

`<https://www.modelscope.cn/models/crazyant/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-onnx/summary>`_

In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

  cd /path/to/sherpa-onnx
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-small-2024-03-09.tar.bz2

  tar xvf sherpa-onnx-paraformer-zh-small-2024-03-09.tar.bz2

Please check that the file sizes of the pre-trained models are correct. See
the file sizes of ``*.onnx`` files below.

.. code-block:: bash

  sherpa-onnx-paraformer-zh-small-2024-03-09$ ls -lh *.onnx

  -rw-r--r-- 1 1001 127 79M Mar 10 00:48 model.int8.onnx

Decode wave files
~~~~~~~~~~~~~~~~~

.. hint::

   It supports decoding only wave files of a single channel with 16-bit
   encoded samples, while the sampling rate does not need to be 16 kHz.

int8
^^^^

The following code shows how to use ``int8`` models to decode wave files:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-paraformer-zh-small-2024-03-09/tokens.txt \
    --paraformer=./sherpa-onnx-paraformer-zh-small-2024-03-09/model.int8.onnx \
    ./sherpa-onnx-paraformer-zh-small-2024-03-09/test_wavs/0.wav \
    ./sherpa-onnx-paraformer-zh-small-2024-03-09/test_wavs/1.wav \
    ./sherpa-onnx-paraformer-zh-small-2024-03-09/test_wavs/8k.wav \
    ./sherpa-onnx-paraformer-zh-small-2024-03-09/test_wavs/2-zh-en.wav \
    ./sherpa-onnx-paraformer-zh-small-2024-03-09/test_wavs/3-sichuan.wav \
    ./sherpa-onnx-paraformer-zh-small-2024-03-09/test_wavs/4-tianjin.wav \
    ./sherpa-onnx-paraformer-zh-small-2024-03-09/test_wavs/5-henan.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

You should see the following output:

.. literalinclude:: ./code-paraformer/sherpa-onnx-paraformer-zh-small-2024-03-09-int8.txt

Speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-microphone-offline \
    --tokens=./sherpa-onnx-paraformer-zh-small-2024-03-09/tokens.txt \
    --paraformer=./sherpa-onnx-paraformer-zh-small-2024-03-09/model.int8.onnx

.. _sherpa_onnx_offline_paraformer_zh_2024_03_09_chinese_english:

csukuangfj/sherpa-onnx-paraformer-zh-2024-03-09 (Chinese + English)
-------------------------------------------------------------------------

.. note::

   This model does not support timestamps. It is a bilingual model, supporting
   both Chinese and English. (支持普通话、河南话、天津话、四川话等方言)

This model is converted from

`<https://www.modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1/summary>`_

In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

  cd /path/to/sherpa-onnx
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2024-03-09.tar.bz2

  tar xvf sherpa-onnx-paraformer-zh-2024-03-09.tar.bz2

Please check that the file sizes of the pre-trained models are correct. See
the file sizes of ``*.onnx`` files below.

.. code-block:: bash

  sherpa-onnx-paraformer-zh-2024-03-09$ ls -lh *.onnx

  -rw-r--r-- 1 1001 127 217M Mar 10 02:22 model.int8.onnx
  -rw-r--r-- 1 1001 127 785M Mar 10 02:22 model.onnx

Decode wave files
~~~~~~~~~~~~~~~~~

.. hint::

   It supports decoding only wave files of a single channel with 16-bit
   encoded samples, while the sampling rate does not need to be 16 kHz.

fp32
^^^^

The following code shows how to use ``fp32`` models to decode wave files:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-paraformer-zh-2024-03-09/tokens.txt \
    --paraformer=./sherpa-onnx-paraformer-zh-2024-03-09/model.onnx \
    ./sherpa-onnx-paraformer-zh-2024-03-09/test_wavs/0.wav \
    ./sherpa-onnx-paraformer-zh-2024-03-09/test_wavs/1.wav \
    ./sherpa-onnx-paraformer-zh-2024-03-09/test_wavs/8k.wav \
    ./sherpa-onnx-paraformer-zh-2024-03-09/test_wavs/2-zh-en.wav \
    ./sherpa-onnx-paraformer-zh-2024-03-09/test_wavs/3-sichuan.wav \
    ./sherpa-onnx-paraformer-zh-2024-03-09/test_wavs/4-tianjin.wav \
    ./sherpa-onnx-paraformer-zh-2024-03-09/test_wavs/5-henan.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

You should see the following output:

.. literalinclude:: ./code-paraformer/sherpa-onnx-paraformer-zh-2024-03-09.txt

int8
^^^^

The following code shows how to use ``int8`` models to decode wave files:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-paraformer-zh-2024-03-09/tokens.txt \
    --paraformer=./sherpa-onnx-paraformer-zh-2024-03-09/model.int8.onnx \
    ./sherpa-onnx-paraformer-zh-2024-03-09/test_wavs/0.wav \
    ./sherpa-onnx-paraformer-zh-2024-03-09/test_wavs/1.wav \
    ./sherpa-onnx-paraformer-zh-2024-03-09/test_wavs/8k.wav \
    ./sherpa-onnx-paraformer-zh-2024-03-09/test_wavs/2-zh-en.wav \
    ./sherpa-onnx-paraformer-zh-2024-03-09/test_wavs/3-sichuan.wav \
    ./sherpa-onnx-paraformer-zh-2024-03-09/test_wavs/4-tianjin.wav \
    ./sherpa-onnx-paraformer-zh-2024-03-09/test_wavs/5-henan.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

You should see the following output:

.. literalinclude:: ./code-paraformer/sherpa-onnx-paraformer-zh-2024-03-09.txt

Speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-microphone-offline \
    --tokens=./sherpa-onnx-paraformer-zh-2024-03-09/tokens.txt \
    --paraformer=./sherpa-onnx-paraformer-zh-2024-03-09/model.int8.onnx


.. _sherpa_onnx_offline_paraformer_zh_2023_03_28_chinese:

csukuangfj/sherpa-onnx-paraformer-zh-2023-03-28 (Chinese + English)
-------------------------------------------------------------------

.. note::

   This model does not support timestamps. It is a bilingual model, supporting
   both Chinese and English. (支持普通话、河南话、天津话、四川话等方言)


This model is converted from

`<https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch>`_

The code for converting can be found at

`<https://huggingface.co/csukuangfj/paraformer-onnxruntime-python-example/tree/main>`_


In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

  cd /path/to/sherpa-onnx
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2023-03-28.tar.bz2

  tar xvf sherpa-onnx-paraformer-zh-2023-03-28.tar.bz2

Please check that the file sizes of the pre-trained models are correct. See
the file sizes of ``*.onnx`` files below.

.. code-block:: bash

  sherpa-onnx-paraformer-zh-2023-03-28$ ls -lh *.onnx
  -rw-r--r-- 1 kuangfangjun root 214M Apr  1 07:28 model.int8.onnx
  -rw-r--r-- 1 kuangfangjun root 824M Apr  1 07:28 model.onnx

Decode wave files
~~~~~~~~~~~~~~~~~

.. hint::

   It supports decoding only wave files of a single channel with 16-bit
   encoded samples, while the sampling rate does not need to be 16 kHz.

fp32
^^^^

The following code shows how to use ``fp32`` models to decode wave files:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-paraformer-zh-2023-03-28/tokens.txt \
    --paraformer=./sherpa-onnx-paraformer-zh-2023-03-28/model.onnx \
    ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/0.wav \
    ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/1.wav \
    ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/2.wav \
    ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/3-sichuan.wav \
    ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/4-tianjin.wav \
    ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/5-henan.wav \
    ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/6-zh-en.wav \
    ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/8k.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

You should see the following output:

.. literalinclude:: ./code-paraformer/sherpa-onnx-paraformer-zh-2023-03-28.txt

int8
^^^^

The following code shows how to use ``int8`` models to decode wave files:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-paraformer-zh-2023-03-28/tokens.txt \
    --paraformer=./sherpa-onnx-paraformer-zh-2023-03-28/model.int8.onnx \
    ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/0.wav \
    ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/1.wav \
    ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/2.wav \
    ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/3-sichuan.wav \
    ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/4-tianjin.wav \
    ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/5-henan.wav \
    ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/6-zh-en.wav \
    ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/8k.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

You should see the following output:

.. literalinclude:: ./code-paraformer/sherpa-onnx-paraformer-zh-2023-03-28-int8.txt

Speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-microphone-offline \
    --tokens=./sherpa-onnx-paraformer-zh-2023-03-28/tokens.txt \
    --paraformer=./sherpa-onnx-paraformer-zh-2023-03-28/model.int8.onnx

.. _sherpa_onnx_offline_paraformer_zh_2023_09_14_chinese:

csukuangfj/sherpa-onnx-paraformer-zh-2023-09-14 (Chinese + English))
---------------------------------------------------------------------

.. note::

   This model supports timestamps. It is a bilingual model, supporting
   both Chinese and English. (支持普通话、河南话、天津话、四川话等方言)


This model is converted from

`<https://www.modelscope.cn/models/iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx/summary>`_

In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2

  tar xvf sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2

Please check that the file sizes of the pre-trained models are correct. See
the file sizes of ``*.onnx`` files below.

.. code-block:: bash

  sherpa-onnx-paraformer-zh-2023-09-14$ ls -lh *.onnx
  -rw-r--r--  1 fangjun  staff   232M Sep 14 13:46 model.int8.onnx

Decode wave files
~~~~~~~~~~~~~~~~~

.. hint::

   It supports decoding only wave files of a single channel with 16-bit
   encoded samples, while the sampling rate does not need to be 16 kHz.

int8
^^^^

The following code shows how to use ``int8`` models to decode wave files:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-paraformer-zh-2023-09-14/tokens.txt \
    --paraformer=./sherpa-onnx-paraformer-zh-2023-09-14/model.int8.onnx \
    --model-type=paraformer \
    ./sherpa-onnx-paraformer-zh-2023-09-14/test_wavs/0.wav \
    ./sherpa-onnx-paraformer-zh-2023-09-14/test_wavs/1.wav \
    ./sherpa-onnx-paraformer-zh-2023-09-14/test_wavs/2.wav \
    ./sherpa-onnx-paraformer-zh-2023-09-14/test_wavs/3-sichuan.wav \
    ./sherpa-onnx-paraformer-zh-2023-09-14/test_wavs/4-tianjin.wav \
    ./sherpa-onnx-paraformer-zh-2023-09-14/test_wavs/5-henan.wav \
    ./sherpa-onnx-paraformer-zh-2023-09-14/test_wavs/6-zh-en.wav \
    ./sherpa-onnx-paraformer-zh-2023-09-14/test_wavs/8k.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

You should see the following output:

.. literalinclude:: ./code-paraformer/sherpa-onnx-paraformer-zh-2023-09-14-int8.txt

Speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-microphone-offline \
    --tokens=./sherpa-onnx-paraformer-zh-2023-09-14/tokens.txt \
    --paraformer=./sherpa-onnx-paraformer-zh-2023-09-14/model.int8.onnx \
    --model-type=paraformer
