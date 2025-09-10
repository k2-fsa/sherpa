.. _wenet-offline-ctc:

WeNet CTC-based models
======================

This page lists all offline CTC models from `WeNet`_.


.. _sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10:

sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10 (Cantonese, 粤语)
-------------------------------------------------------------------------------------------------------------

This model is converted from

  `<https://huggingface.co/ASLP-lab/WSYue-ASR/tree/main/u2pp_conformer_yue>`_

It uses 21.8k hours of training data.

.. hint::

   If you want a ``Cantonese`` ASR model, please choose this model
   or :ref:`sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09`

In the following, we describe how to use it.

Huggingface space
^^^^^^^^^^^^^^^^^

You can visit

  `<https://huggingface.co/spaces/k2-fsa/automatic-speech-recognition>`_

to try this model in your browser.

.. hint::

   You need to first select the language ``Cantonese``
   and then select the model  ``csukuangfj/sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10``.

Android APKs
^^^^^^^^^^^^

Real-time speech recognition Android APKs can be found at

  `<https://k2-fsa.github.io/sherpa/onnx/android/apk-simulate-streaming-asr.html>`_

Please always download the latest version.

.. hint::

   Please search for ``wenetspeech_yue_u2pconformer_ctc_2025_09_10_int8.apk`` in the above page, e.g.,
   ``sherpa-onnx-1.12.11-arm64-v8a-simulated_streaming_asr-zh_en_yue-wenetspeech_yue_u2pconformer_ctc_2025_09_10_int8.apk``.

.. hint::

   For Chinese users, you can also visit `<https://k2-fsa.github.io/sherpa/onnx/android/apk-simulate-streaming-asr-cn.html>`_

Download
^^^^^^^^

Please use the following commands to download it::

  cd /path/to/sherpa-onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10.tar.bz2
  tar xf sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10.tar.bz2
  rm sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10.tar.bz2

After downloading, you should find the following files::

  ls -lh sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10/

  total 263264
  -rw-r--r--   1 fangjun  staff   129B Sep 10 14:18 README.md
  -rw-r--r--   1 fangjun  staff   128M Sep 10 14:18 model.int8.onnx
  drwxr-xr-x  22 fangjun  staff   704B Sep 10 14:18 test_wavs
  -rw-r--r--   1 fangjun  staff    83K Sep 10 14:18 tokens.txt

.. code-block::

  ls  sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10/test_wavs/

  en.wav     yue-1.wav  yue-11.wav yue-13.wav yue-15.wav yue-17.wav yue-3.wav  yue-5.wav  yue-7.wav  yue-9.wav
  yue-0.wav  yue-10.wav yue-12.wav yue-14.wav yue-16.wav yue-2.wav  yue-4.wav  yue-6.wav  yue-8.wav  zh.wav

In the following, we show how to decode the files ``sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10/test_wavs/yue-*.wav``.

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

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10/tokens.txt \
    --wenet-ctc-model=./sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10/model.int8.onnx \
    --num-threads=1 \
    sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10/test_wavs/yue-0.wav

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

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10/tokens.txt \
    --wenet-ctc-model=./sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10/model.int8.onnx \
    --num-threads=1 \
    sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10/test_wavs/yue-1.wav

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

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10/tokens.txt \
    --wenet-ctc-model=./sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10/model.int8.onnx \
    --num-threads=1 \
    sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10/test_wavs/yue-2.wav

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

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10/tokens.txt \
    --wenet-ctc-model=./sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10/model.int8.onnx \
    --num-threads=1 \
    sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10/test_wavs/yue-3.wav

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

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10/tokens.txt \
    --wenet-ctc-model=./sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10/model.int8.onnx \
    --num-threads=1 \
    sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10/test_wavs/yue-4.wav

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

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10/tokens.txt \
    --wenet-ctc-model=./sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10/model.int8.onnx \
    --num-threads=1 \
    sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10/test_wavs/yue-5.wav

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

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10/tokens.txt \
    --wenet-ctc-model=./sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10/model.int8.onnx \
    --num-threads=1 \
    sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10/test_wavs/yue-6.wav

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

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10/tokens.txt \
    --wenet-ctc-model=./sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10/model.int8.onnx \
    --num-threads=1 \
    sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10/test_wavs/yue-7.wav

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

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10/tokens.txt \
    --wenet-ctc-model=./sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10/model.int8.onnx \
    --num-threads=1 \
    sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10/test_wavs/yue-8.wav

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

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10/tokens.txt \
    --wenet-ctc-model=./sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10/model.int8.onnx \
    --num-threads=1 \
    sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10/test_wavs/yue-9.wav

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

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10/tokens.txt \
    --wenet-ctc-model=./sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10/model.int8.onnx \
    --num-threads=1 \
    sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10/test_wavs/yue-10.wav

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

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10/tokens.txt \
    --wenet-ctc-model=./sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10/model.int8.onnx \
    --num-threads=1 \
    sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10/test_wavs/yue-11.wav

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

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10/tokens.txt \
    --wenet-ctc-model=./sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10/model.int8.onnx \
    --num-threads=1 \
    sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10/test_wavs/yue-12.wav

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

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10/tokens.txt \
    --wenet-ctc-model=./sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10/model.int8.onnx \
    --num-threads=1 \
    sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10/test_wavs/yue-13.wav

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

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10/tokens.txt \
    --wenet-ctc-model=./sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10/model.int8.onnx \
    --num-threads=1 \
    sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10/test_wavs/yue-14.wav

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

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10/tokens.txt \
    --wenet-ctc-model=./sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10/model.int8.onnx \
    --num-threads=1 \
    sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10/test_wavs/yue-15.wav

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

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10/tokens.txt \
    --wenet-ctc-model=./sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10/model.int8.onnx \
    --num-threads=1 \
    sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10/test_wavs/yue-16.wav

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

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10/tokens.txt \
    --wenet-ctc-model=./sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10/model.int8.onnx \
    --num-threads=1 \
    sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10/test_wavs/yue-17.wav

.. literalinclude:: ./code/yue-17.txt
