Pre-trained Models
==================

This page describes how to download pre-trained `SenseVoice`_ models.

.. _sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2024-07-17:

sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2024-07-17 (Chinese, English, Japanese, Korean, Cantonese, 中英日韩粤语)
------------------------------------------------------------------------------------------------------------------------

This model is converted from `<https://www.modelscope.cn/models/iic/SenseVoiceSmall>`_
using the script `export-ncnn.py <https://github.com/k2-fsa/sherpa-ncnn/blob/master/scripts/sense-voice/export-ncnn.py>`_.

It supports the following 5 languages:

  - Chinese (Mandarin, 普通话)
  - Cantonese (粤语, 广东话)
  - English
  - Japanese
  - Korean

In the following, we describe how to use it.

.. hint::

   For ``RKNN`` users, please refer to :ref:`sherpa-onnx-rk3588-20-seconds-sense-voice-zh-en-ja-ko-yue-2024-07-17`.

   For ``onnxruntime`` users, please refer to :ref:`sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17`.

Download
^^^^^^^^

Please use the following commands to download it::

  cd /path/to/sherpa-ncnn

  wget https://github.com/k2-fsa/sherpa-ncnn/releases/download/asr-models/sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
  tar xvf sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
  rm sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2

.. code-block:: bash

  ls -lh  sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2024-07-17/

  total 907400
  -rw-r--r--  1 fangjun  staff    71B Sep 13 19:17 LICENSE
  -rw-r--r--  1 fangjun  staff   104B Sep 13 19:17 README.md
  -rw-r--r--  1 fangjun  staff   443M Sep 13 19:17 model.ncnn.bin
  -rw-r--r--  1 fangjun  staff   162K Sep 13 19:17 model.ncnn.param
  drwxr-xr-x  7 fangjun  staff   224B Sep 13 19:17 test_wavs
  -rw-r--r--  1 fangjun  staff   308K Sep 13 19:17 tokens.txt

Decode a file
^^^^^^^^^^^^^

Without inverse text normalization
::::::::::::::::::::::::::::::::::

To decode a file without inverse text normalization, please use:

.. code-block:: bash

  ./build/bin/sherpa-ncnn-offline \
    --tokens=./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt \
    --sense-voice-model-dir=./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2024-07-17 \
    --num-threads=1 \
    ./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2024-07-17/test_wavs/zh.wav

You should see the following output:

.. literalinclude:: ./code/2024-07-17-no-itn.txt

With inverse text normalization
:::::::::::::::::::::::::::::::

To decode a file with inverse text normalization, please use:

.. code-block:: bash

  ./build/bin/sherpa-ncnn-offline \
    --tokens=./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt \
    --sense-voice-model-dir=./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2024-07-17 \
    --sense-voice-use-itn=1 \
    --num-threads=1 \
    ./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2024-07-17/test_wavs/zh.wav

You should see the following output:

.. literalinclude:: ./code/2024-07-17-with-itn.txt

.. hint::

   When inverse text normalziation is enabled, the results contain
   punctuations.

Real-time Speech recognition from a microphone
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, download a :ref:`sherpa_ncnn_vad` model

.. code-block:: bash

   cd /path/to/sherpa-ncnn

   wget https://github.com/k2-fsa/sherpa-ncnn/releases/download/models/sherpa-ncnn-silero-vad.tar.bz2
   tar xvf sherpa-ncnn-silero-vad.tar.bz2
   rm sherpa-ncnn-silero-vad.tar.bz2

Now, run it:

.. code-block:: bash

  ./build/bin/sherpa-ncnn-vad-microphone-simulated-streaming-asr \
    --tokens=./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt \
    --sense-voice-model-dir=./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2024-07-17 \
    --silero-vad-model-dir=./sherpa-ncnn-silero-vad \
    --num-threads=1

.. hint::

   You can use ``./build/bin/sherpa-ncnn-pa-devs`` to list all microphone devices.

   The output of the command::

    ./build/bin/sherpa-ncnn-pa-devs

   is given below:

   .. literalinclude:: ./code/all-devices.txt

.. hint::

  If you want to use ``device #2`` with sample rate ``48000``, please run::

    ./build/bin/sherpa-ncnn-vad-microphone-simulated-streaming-asr \
      --mic-device-index=2 \
      --mic-sample-rate=48000 \
      --tokens=./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt \
      --sense-voice-model-dir=./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2024-07-17 \
      --silero-vad-model-dir=./sherpa-ncnn-silero-vad \
      --num-threads=1


Speed test on RK3588 CPU
^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::

 * - | RTF of SenseVoice
     | in sherpa-ncnn
   - 1 thread
   - 2 threads
   - 3 threads
   - 4 threads
 * - Cortex A55
   - 0.584
   - 0.320
   - 0.231
   - 0.188
 * - Cortex A76
   - 0.142
   - 0.079
   - 0.063
   - 0.049

Cortex A55
:::::::::::

.. code-block::

  # 1 cortex A55 CPU
  taskset 0x01 ./build/bin/sherpa-ncnn-offline \
    --num-threads=1 \
    --tokens=./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt \
    --sense-voice-model-dir=./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2024-07-17 \
    ./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2024-07-17/test_wavs/zh.wav

  # 2 cortex A55 CPUs
  taskset 0x03 ./build/bin/sherpa-ncnn-offline \
    --num-threads=2 \
    --tokens=./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt \
    --sense-voice-model-dir=./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2024-07-17 \
    ./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2024-07-17/test_wavs/zh.wav

  # 3 cortex A55 CPUs
  taskset 0x07 ./build/bin/sherpa-ncnn-offline \
    --num-threads=3 \
    --tokens=./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt \
    --sense-voice-model-dir=./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2024-07-17 \
    ./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2024-07-17/test_wavs/zh.wav

  # 4 cortex A55 CPUs
  taskset 0x0f ./build/bin/sherpa-ncnn-offline \
    --num-threads=4 \
    --tokens=./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt \
    --sense-voice-model-dir=./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2024-07-17 \
    ./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2024-07-17/test_wavs/zh.wav

Cortex A76
:::::::::::

.. code-block::

  # 1 cortex A76 CPU
  taskset 0x10 ./build/bin/sherpa-ncnn-offline \
    --num-threads=1 \
    --tokens=./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt \
    --sense-voice-model-dir=./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2024-07-17 \
    ./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2024-07-17/test_wavs/zh.wav

  # 2 cortex A76 CPUs
  taskset 0x30 ./build/bin/sherpa-ncnn-offline \
    --num-threads=2 \
    --tokens=./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt \
    --sense-voice-model-dir=./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2024-07-17 \
    ./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2024-07-17/test_wavs/zh.wav

  # 3 cortex A76 CPUs
  taskset 0x70 ./build/bin/sherpa-ncnn-offline \
    --num-threads=3 \
    --tokens=./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt \
    --sense-voice-model-dir=./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2024-07-17 \
    ./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2024-07-17/test_wavs/zh.wav

  # 4 cortex A76 CPUs
  taskset 0xf0 ./build/bin/sherpa-ncnn-offline \
    --num-threads=4 \
    --tokens=./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt \
    --sense-voice-model-dir=./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2024-07-17 \
    ./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2024-07-17/test_wavs/zh.wav

.. _sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2025-09-09:

sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2025-09-09 (Chinese, English, Japanese, Korean, Cantonese, 中英日韩粤语)
------------------------------------------------------------------------------------------------------------------------------

This model is converted from

  `<https://huggingface.co/ASLP-lab/WSYue-ASR/tree/main/sensevoice_small_yue>`_

It is fine-tuned on :ref:`sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2024-07-17` with 21.8k hours of ``Cantonese`` data.

It supports the following 5 languages:

  - Chinese (Mandarin, 普通话)
  - Cantonese (粤语, 广东话)
  - English
  - Japanese
  - Korean

.. hint::

   If you want a ``Cantonese`` ASR model, please choose this model.

.. hint::

   For ``RKNN`` users, please refer to :ref:`sherpa-onnx-rk3588-20-seconds-sense-voice-zh-en-ja-ko-yue-2025-09-09`.

   For ``onnxruntime`` users, please refer to :ref:`sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09`.

In the following, we describe how to use it.

Download
^^^^^^^^

Please use the following commands to download it::

  wget https://github.com/k2-fsa/sherpa-ncnn/releases/download/asr-models/sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2025-09-09.tar.bz2
  tar xvf sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2025-09-09.tar.bz2
  rm sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2025-09-09.tar.bz2

After downloading, you should find the following files:

.. code-block:: bash

  ls -lh sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2025-09-09/

  total 918672
  -rw-r--r--   1 fangjun  staff   131B Sep 13 19:17 README.md
  -rw-r--r--   1 fangjun  staff   443M Sep 13 19:17 model.ncnn.bin
  -rw-r--r--   1 fangjun  staff   162K Sep 13 19:17 model.ncnn.param
  drwxr-xr-x  23 fangjun  staff   736B Sep 13 19:17 test_wavs
  -rw-r--r--   1 fangjun  staff   308K Sep 13 19:17 tokens.txt

.. code-block:: bash

  ls sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2025-09-09/test_wavs/

  en.wav     yue-1.wav  yue-11.wav yue-13.wav yue-15.wav yue-17.wav yue-3.wav  yue-5.wav  yue-7.wav  yue-9.wav  zh.wav
  yue-0.wav  yue-10.wav yue-12.wav yue-14.wav yue-16.wav yue-2.wav  yue-4.wav  yue-6.wav  yue-8.wav  yue.wav

In the following, we show how to decode the files ``sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2025-09-09/test_wavs/yue-*.wav``.

yue-0.wav
^^^^^^^^^

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Ground truth</th>
    </tr>
    <tr>
      <td>yue-0.wav</td>
      <td>
       <audio title="yue-0.wav" controls="controls">
             <source src="/sherpa/_static/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/yue-0.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
      两只小企鹅都有嘢食
      </td>
    </tr>
  </table>

.. code-block:: bash

  ./build/bin/sherpa-ncnn-offline \
    --tokens=./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2025-09-09/tokens.txt \
    --sense-voice-model-dir=./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2025-09-09 \
    --num-threads=1 \
    sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2025-09-09/test_wavs/yue-0.wav

.. literalinclude:: ./code/yue-0.txt

yue-1.wav
^^^^^^^^^

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Ground truth</th>
    </tr>
    <tr>
      <td>yue-1.wav</td>
      <td>
       <audio title="yue-1.wav" controls="controls">
             <source src="/sherpa/_static/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/yue-1.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
      叫做诶诶直入式你个脑部里边咧记得呢一个嘅以前香港有一个广告好出名嘅佢乜嘢都冇噶净系影住喺弥敦道佢哋间铺头嘅啫但系就不停有人嗌啦平平吧平吧
      </td>
    </tr>
  </table>

.. code-block:: bash

  ./build/bin/sherpa-ncnn-offline \
    --tokens=./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2025-09-09/tokens.txt \
    --sense-voice-model-dir=./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2025-09-09 \
    --num-threads=1 \
    sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2025-09-09/test_wavs/yue-1.wav

.. literalinclude:: ./code/yue-1.txt

yue-2.wav
^^^^^^^^^

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Ground truth</th>
    </tr>
    <tr>
      <td>yue-2.wav</td>
      <td>
       <audio title="yue-2.wav" controls="controls">
             <source src="/sherpa/_static/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/yue-2.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
      忽然从光线死角嘅阴影度窜出一只大猫
      </td>
    </tr>
  </table>

.. code-block:: bash

  ./build/bin/sherpa-ncnn-offline \
    --tokens=./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2025-09-09/tokens.txt \
    --sense-voice-model-dir=./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2025-09-09 \
    --num-threads=1 \
    sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2025-09-09/test_wavs/yue-2.wav

.. literalinclude:: ./code/yue-2.txt

yue-3.wav
^^^^^^^^^

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Ground truth</th>
    </tr>
    <tr>
      <td>yue-3.wav</td>
      <td>
       <audio title="yue-3.wav" controls="controls">
             <source src="/sherpa/_static/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/yue-3.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
      今日我带大家去见识一位九零后嘅靓仔咧
      </td>
    </tr>
  </table>

.. code-block:: bash

  ./build/bin/sherpa-ncnn-offline \
    --tokens=./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2025-09-09/tokens.txt \
    --sense-voice-model-dir=./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2025-09-09 \
    --num-threads=1 \
    sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2025-09-09/test_wavs/yue-3.wav

.. literalinclude:: ./code/yue-3.txt

yue-4.wav
^^^^^^^^^

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Ground truth</th>
    </tr>
    <tr>
      <td>yue-4.wav</td>
      <td>
       <audio title="yue-4.wav" controls="controls">
             <source src="/sherpa/_static/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/yue-4.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
      香港嘅消费市场从此不一样
      </td>
    </tr>
  </table>

.. code-block:: bash

  ./build/bin/sherpa-ncnn-offline \
    --tokens=./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2025-09-09/tokens.txt \
    --sense-voice-model-dir=./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2025-09-09 \
    --num-threads=1 \
    sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2025-09-09/test_wavs/yue-4.wav

.. literalinclude:: ./code/yue-4.txt

yue-5.wav
^^^^^^^^^

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Ground truth</th>
    </tr>
    <tr>
      <td>yue-5.wav</td>
      <td>
       <audio title="yue-5.wav" controls="controls">
             <source src="/sherpa/_static/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/yue-5.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
      景天谂唔到呢个守门嘅弟子竟然咁无礼霎时间面色都变埋
      </td>
    </tr>
  </table>

.. code-block:: bash

  ./build/bin/sherpa-ncnn-offline \
    --tokens=./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2025-09-09/tokens.txt \
    --sense-voice-model-dir=./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2025-09-09 \
    --num-threads=1 \
    sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2025-09-09/test_wavs/yue-5.wav

.. literalinclude:: ./code/yue-5.txt

yue-6.wav
^^^^^^^^^

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Ground truth</th>
    </tr>
    <tr>
      <td>yue-6.wav</td>
      <td>
       <audio title="yue-6.wav" controls="controls">
             <source src="/sherpa/_static/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/yue-6.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
      六个星期嘅课程包括六堂课同两个测验你唔掌握到基本嘅十九个声母五十六个韵母同九个声调我哋仲针对咗广东话学习者会遇到嘅大樽颈啊以国语为母语人士最难掌握嘅五大韵母教课书唔会教你嘅七种变音同十种变调说话生硬唔自然嘅根本性问题提供全新嘅学习方向等你突破难关
      </td>
    </tr>
  </table>

.. code-block:: bash

  ./build/bin/sherpa-ncnn-offline \
    --tokens=./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2025-09-09/tokens.txt \
    --sense-voice-model-dir=./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2025-09-09 \
    --num-threads=1 \
    sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2025-09-09/test_wavs/yue-6.wav

.. literalinclude:: ./code/yue-6.txt

yue-7.wav
^^^^^^^^^

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Ground truth</th>
    </tr>
    <tr>
      <td>yue-7.wav</td>
      <td>
       <audio title="yue-7.wav" controls="controls">
             <source src="/sherpa/_static/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/yue-7.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
      同意嘅累积唔系阴同阳嘅累积可以讲三既融合咗一同意融合咗阴同阳
      </td>
    </tr>
  </table>

.. code-block:: bash

  ./build/bin/sherpa-ncnn-offline \
    --tokens=./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2025-09-09/tokens.txt \
    --sense-voice-model-dir=./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2025-09-09 \
    --num-threads=1 \
    sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2025-09-09/test_wavs/yue-7.wav

.. literalinclude:: ./code/yue-7.txt

yue-8.wav
^^^^^^^^^

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Ground truth</th>
    </tr>
    <tr>
      <td>yue-8.wav</td>
      <td>
       <audio title="yue-8.wav" controls="controls">
             <source src="/sherpa/_static/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/yue-8.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
      而较早前已经复航嘅氹仔北安码头星期五开始增设夜间航班不过两个码头暂时都冇凌晨班次有旅客希望尽快恢复可以留喺澳门长啲时间
      </td>
    </tr>
  </table>

.. code-block:: bash

  ./build/bin/sherpa-ncnn-offline \
    --tokens=./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2025-09-09/tokens.txt \
    --sense-voice-model-dir=./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2025-09-09 \
    --num-threads=1 \
    sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2025-09-09/test_wavs/yue-8.wav

.. literalinclude:: ./code/yue-8.txt

yue-9.wav
^^^^^^^^^

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Ground truth</th>
    </tr>
    <tr>
      <td>yue-9.wav</td>
      <td>
       <audio title="yue-9.wav" controls="controls">
             <source src="/sherpa/_static/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/yue-9.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
      刘备仲马鞭一指蜀兵一齐掩杀过去打到吴兵大败唉刘备八路兵马以雷霆万钧之势啊杀到吴兵啊尸横遍野血流成河
      </td>
    </tr>
  </table>

.. code-block:: bash

  ./build/bin/sherpa-ncnn-offline \
    --tokens=./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2025-09-09/tokens.txt \
    --sense-voice-model-dir=./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2025-09-09 \
    --num-threads=1 \
    sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2025-09-09/test_wavs/yue-9.wav

.. literalinclude:: ./code/yue-9.txt

yue-10.wav
^^^^^^^^^^

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Ground truth</th>
    </tr>
    <tr>
      <td>yue-10.wav</td>
      <td>
       <audio title="yue-10.wav" controls="controls">
             <source src="/sherpa/_static/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/yue-10.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
      原来王力宏咧系佢家中里面咧成就最低个吓哇
      </td>
    </tr>
  </table>

.. code-block:: bash

  ./build/bin/sherpa-ncnn-offline \
    --tokens=./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2025-09-09/tokens.txt \
    --sense-voice-model-dir=./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2025-09-09 \
    --num-threads=1 \
    sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2025-09-09/test_wavs/yue-10.wav

.. literalinclude:: ./code/yue-10.txt

yue-11.wav
^^^^^^^^^^

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Ground truth</th>
    </tr>
    <tr>
      <td>yue-11.wav</td>
      <td>
       <audio title="yue-11.wav" controls="controls">
             <source src="/sherpa/_static/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/yue-11.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
      无论你提出任何嘅要求
      </td>
    </tr>
  </table>

.. code-block:: bash

  ./build/bin/sherpa-ncnn-offline \
    --tokens=./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2025-09-09/tokens.txt \
    --sense-voice-model-dir=./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2025-09-09 \
    --num-threads=1 \
    sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2025-09-09/test_wavs/yue-11.wav

.. literalinclude:: ./code/yue-11.txt

yue-12.wav
^^^^^^^^^^

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Ground truth</th>
    </tr>
    <tr>
      <td>yue-12.wav</td>
      <td>
       <audio title="yue-12.wav" controls="controls">
             <source src="/sherpa/_static/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/yue-12.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
      咁咁多样材料咁我哋首先第一步处理咗一件
      </td>
    </tr>
  </table>

.. code-block:: bash

  ./build/bin/sherpa-ncnn-offline \
    --tokens=./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2025-09-09/tokens.txt \
    --sense-voice-model-dir=./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2025-09-09 \
    --num-threads=1 \
    sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2025-09-09/test_wavs/yue-12.wav

.. literalinclude:: ./code/yue-12.txt

yue-13.wav
^^^^^^^^^^

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Ground truth</th>
    </tr>
    <tr>
      <td>yue-13.wav</td>
      <td>
       <audio title="yue-13.wav" controls="controls">
             <source src="/sherpa/_static/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/yue-13.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
      啲点样对于佢哋嘅服务态度啊不透过呢一年左右嘅时间啦其实大家都静一静啦咁你就会见到香港嘅经济其实
      </td>
    </tr>
  </table>

.. code-block:: bash

  ./build/bin/sherpa-ncnn-offline \
    --tokens=./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2025-09-09/tokens.txt \
    --sense-voice-model-dir=./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2025-09-09 \
    --num-threads=1 \
    sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2025-09-09/test_wavs/yue-13.wav

.. literalinclude:: ./code/yue-13.txt

yue-14.wav
^^^^^^^^^^

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Ground truth</th>
    </tr>
    <tr>
      <td>yue-14.wav</td>
      <td>
       <audio title="yue-14.wav" controls="controls">
             <source src="/sherpa/_static/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/yue-14.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
      就即刻会同贵正两位八代长老带埋五名七代弟子前啲灵蛇岛想话生擒谢信抢咗屠龙宝刀翻嚟献俾帮主嘅
      </td>
    </tr>
  </table>

.. code-block:: bash

  ./build/bin/sherpa-ncnn-offline \
    --tokens=./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2025-09-09/tokens.txt \
    --sense-voice-model-dir=./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2025-09-09 \
    --num-threads=1 \
    sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2025-09-09/test_wavs/yue-14.wav

.. literalinclude:: ./code/yue-14.txt

yue-15.wav
^^^^^^^^^^

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Ground truth</th>
    </tr>
    <tr>
      <td>yue-15.wav</td>
      <td>
       <audio title="yue-15.wav" controls="controls">
             <source src="/sherpa/_static/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/yue-15.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
      我知道我的观众大部分都是对广东话有兴趣想学广东话的人
      </td>
    </tr>
  </table>

.. code-block:: bash

  ./build/bin/sherpa-ncnn-offline \
    --tokens=./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2025-09-09/tokens.txt \
    --sense-voice-model-dir=./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2025-09-09 \
    --num-threads=1 \
    sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2025-09-09/test_wavs/yue-15.wav

.. literalinclude:: ./code/yue-15.txt

yue-16.wav
^^^^^^^^^^

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Ground truth</th>
    </tr>
    <tr>
      <td>yue-16.wav</td>
      <td>
       <audio title="yue-16.wav" controls="controls">
             <source src="/sherpa/_static/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/yue-16.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
      诶原来啊我哋中国人呢讲究物极必反
      </td>
    </tr>
  </table>

.. code-block:: bash

  ./build/bin/sherpa-ncnn-offline \
    --tokens=./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2025-09-09/tokens.txt \
    --sense-voice-model-dir=./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2025-09-09 \
    --num-threads=1 \
    sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2025-09-09/test_wavs/yue-16.wav

.. literalinclude:: ./code/yue-16.txt

yue-17.wav
^^^^^^^^^^

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Ground truth</th>
    </tr>
    <tr>
      <td>yue-17.wav</td>
      <td>
       <audio title="yue-17.wav" controls="controls">
             <source src="/sherpa/_static/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/yue-17.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
      如果东边道建成咁丹东呢就会成为最近嘅出海港同埋经过哈大线出海相比绥分河则会减少运渠三百五十六公里
      </td>
    </tr>
  </table>

.. code-block:: bash

  ./build/bin/sherpa-ncnn-offline \
    --tokens=./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2025-09-09/tokens.txt \
    --sense-voice-model-dir=./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2025-09-09 \
    --num-threads=1 \
    sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2025-09-09/test_wavs/yue-17.wav

.. literalinclude:: ./code/yue-17.txt

