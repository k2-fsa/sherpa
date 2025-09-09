Pre-trained Models
==================

This page describes how to download pre-trained `SenseVoice`_ models.


.. _sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17:

sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17 (Chinese, English, Japanese, Korean, Cantonese, 中英日韩粤语)
------------------------------------------------------------------------------------------------------------------------

This model is converted from `<https://www.modelscope.cn/models/iic/SenseVoiceSmall>`_
using the script `export-onnx.py <https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/sense-voice/export-onnx.py>`_.

It supports the following 5 languages:

  - Chinese (Mandarin, 普通话)
  - Cantonese (粤语, 广东话)
  - English
  - Japanese
  - Korean

In the following, we describe how to use it.



Download
^^^^^^^^

Please use the following commands to download it::

  cd /path/to/sherpa-onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2

  tar xvf sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
  rm sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2

After downloading, you should find the following files::

  ls -lh sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17

  total 1.1G
  -rw-r--r-- 1 runner docker   71 Jul 18 13:06 LICENSE
  -rw-r--r-- 1 runner docker  104 Jul 18 13:06 README.md
  -rwxr-xr-x 1 runner docker 5.8K Jul 18 13:06 export-onnx.py
  -rw-r--r-- 1 runner docker 229M Jul 18 13:06 model.int8.onnx
  -rw-r--r-- 1 runner docker 895M Jul 18 13:06 model.onnx
  drwxr-xr-x 2 runner docker 4.0K Jul 18 13:06 test_wavs
  -rw-r--r-- 1 runner docker 309K Jul 18 13:06 tokens.txt

  ls -lh sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/test_wavs

  total 940K
  -rw-r--r-- 1 runner docker 224K Jul 18 13:06 en.wav
  -rw-r--r-- 1 runner docker 226K Jul 18 13:06 ja.wav
  -rw-r--r-- 1 runner docker 145K Jul 18 13:06 ko.wav
  -rw-r--r-- 1 runner docker 161K Jul 18 13:06 yue.wav
  -rw-r--r-- 1 runner docker 175K Jul 18 13:06 zh.wav

.. hint::

   If you only need the ``int8`` model file, please use::

     wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2
     tar xvf sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2
     rm sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2

     ls -lh sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17

  It prints::

    total 229M
    -rwxr-xr-x 1 1001 118 5.8K Jul 18  2024 export-onnx.py
    -rw-r--r-- 1 1001 118   71 Jul 18  2024 LICENSE
    -rw-r--r-- 1 1001 118 229M Jul 18  2024 model.int8.onnx
    -rw-r--r-- 1 1001 118  104 Jul 18  2024 README.md
    drwxr-xr-x 2 1001 118 4.0K Jul 18  2024 test_wavs
    -rw-r--r-- 1 1001 118 309K Jul 18  2024 tokens.txt

Decode a file with model.onnx
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Without inverse text normalization
::::::::::::::::::::::::::::::::::

To decode a file without inverse text normalization, please use:

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt \
    --sense-voice-model=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.onnx \
    --num-threads=1 \
    --debug=0 \
    ./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/test_wavs/zh.wav \
    ./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/test_wavs/en.wav

You should see the following output:

.. literalinclude:: ./code/2024-07-17.txt

With inverse text normalization
:::::::::::::::::::::::::::::::

To decode a file with inverse text normalization, please use:

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt \
    --sense-voice-model=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.onnx \
    --num-threads=1 \
    --sense-voice-use-itn=1 \
    --debug=0 \
    ./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/test_wavs/zh.wav \
    ./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/test_wavs/en.wav

You should see the following output:

.. literalinclude:: ./code/2024-07-17-itn.txt

.. hint::

   When inverse text normalziation is enabled, the results also
   punctuations.

Specify a language
::::::::::::::::::

If you don't provide a language when decoding, it uses ``auto``.

To specify the language when decoding, please use:

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt \
    --sense-voice-model=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.onnx \
    --num-threads=1 \
    --sense-voice-language=zh \
    --debug=0 \
    ./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/test_wavs/zh.wav

You should see the following output:

.. literalinclude:: ./code/2024-07-17-lang.txt

.. hint::

   Valid values for ``--sense-voice-language`` are ``auto``, ``zh``, ``en``, ``ko``, ``ja``, and ``yue``.
   where ``zh`` is for Chinese, ``en`` for English, ``ko`` for Korean, ``ja`` for Japanese, and
   ``yue`` for ``Cantonese``.


Speech recognition from a microphone
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  ./build/bin/sherpa-onnx-microphone-offline \
    --tokens=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt \
    --sense-voice-model=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.int8.onnx

Speech recognition from a microphone with VAD
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

  ./build/bin/sherpa-onnx-vad-microphone-offline-asr \
    --silero-vad-model=./silero_vad.onnx \
    --tokens=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt \
    --sense-voice-model=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.int8.onnx

sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09 (Chinese, English, Japanese, Korean, Cantonese, 中英日韩粤语)
----------------------------------------------------------------------------------------------------------------------------------------

This model is converted from

  `<https://huggingface.co/ASLP-lab/WSYue-ASR/tree/main/sensevoice_small_yue>`_

It is fine-tuned on :ref:`sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17` with 218k hours ``Cantonese`` data.

It supports the following 5 languages:

  - Chinese (Mandarin, 普通话)
  - Cantonese (粤语, 广东话)
  - English
  - Japanese
  - Korean

.. hint::

   If you want a ``Cantonese`` ASR model, please choose this model.

In the following, we describe how to use it.

Huggingface space
^^^^^^^^^^^^^^^^^

You can visit

  `<https://huggingface.co/spaces/k2-fsa/automatic-speech-recognition>`_

to try this model in your browser.

.. hint::

   You need to first select the language ``Chinese+English+Cantonese+Japanese+Korean``
   and then select the model  ``csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09``.


Android APKs
^^^^^^^^^^^^

Real-time speech recognition Android APKs can be found at

  `<https://k2-fsa.github.io/sherpa/onnx/android/apk-simulate-streaming-asr.html>`_

Please always download the latest version.

.. hint::

   Please search for ``zh_en_ko_ja_yue-sense_voice_2025_09_09_int8.apk`` in the above page, e.g.,
   ``sherpa-onnx-1.12.11-arm64-v8a-simulated_streaming_asr-zh_en_ko_ja_yue-sense_voice_2025_09_09_int8.apk``.

.. hint::

   For Chinese users, you can also visit `<https://k2-fsa.github.io/sherpa/onnx/android/apk-simulate-streaming-asr-cn.html>`_

Download
^^^^^^^^

Please use the following commands to download it::

  cd /path/to/sherpa-onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09.tar.bz2
  tar xvf sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09.tar.bz2
  rm sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09.tar.bz2

After downloading, you should find the following files::

  ls -lh sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09

  total 492952
  -rw-r--r--   1 fangjun  staff   131B Sep  9 21:12 README.md
  -rw-r--r--   1 fangjun  staff   226M Sep  9 21:12 model.int8.onnx
  drwxr-xr-x  25 fangjun  staff   800B Sep  9 21:12 test_wavs
  -rw-r--r--   1 fangjun  staff   308K Sep  9 21:12 tokens.txt

.. code-block::

  ls  sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/test_wavs/

  en.wav     ko.wav     yue-1.wav  yue-11.wav yue-13.wav yue-15.wav yue-17.wav yue-3.wav  yue-5.wav  yue-7.wav  yue-9.wav  zh.wav
  ja.wav     yue-0.wav  yue-10.wav yue-12.wav yue-14.wav yue-16.wav yue-2.wav  yue-4.wav  yue-6.wav  yue-8.wav  yue.wav

In the following, we show how to decode the ``sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/test_wavs/yue-*.wav``.


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
    --tokens=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/tokens.txt \
    --sense-voice-model=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/model.int8.onnx \
    --num-threads=1 \
    sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/test_wavs/yue-0.wav

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
    --tokens=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/tokens.txt \
    --sense-voice-model=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/model.int8.onnx \
    --num-threads=1 \
    sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/test_wavs/yue-1.wav

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
    --tokens=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/tokens.txt \
    --sense-voice-model=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/model.int8.onnx \
    --num-threads=1 \
    sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/test_wavs/yue-2.wav

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
    --tokens=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/tokens.txt \
    --sense-voice-model=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/model.int8.onnx \
    --num-threads=1 \
    sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/test_wavs/yue-3.wav

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
    --tokens=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/tokens.txt \
    --sense-voice-model=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/model.int8.onnx \
    --num-threads=1 \
    sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/test_wavs/yue-4.wav

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
    --tokens=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/tokens.txt \
    --sense-voice-model=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/model.int8.onnx \
    --num-threads=1 \
    sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/test_wavs/yue-5.wav

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
    --tokens=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/tokens.txt \
    --sense-voice-model=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/model.int8.onnx \
    --num-threads=1 \
    sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/test_wavs/yue-6.wav

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
    --tokens=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/tokens.txt \
    --sense-voice-model=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/model.int8.onnx \
    --num-threads=1 \
    sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/test_wavs/yue-7.wav

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
    --tokens=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/tokens.txt \
    --sense-voice-model=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/model.int8.onnx \
    --num-threads=1 \
    sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/test_wavs/yue-8.wav

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
    --tokens=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/tokens.txt \
    --sense-voice-model=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/model.int8.onnx \
    --num-threads=1 \
    sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/test_wavs/yue-9.wav

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
    --tokens=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/tokens.txt \
    --sense-voice-model=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/model.int8.onnx \
    --num-threads=1 \
    sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/test_wavs/yue-10.wav

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
    --tokens=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/tokens.txt \
    --sense-voice-model=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/model.int8.onnx \
    --num-threads=1 \
    sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/test_wavs/yue-11.wav

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
    --tokens=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/tokens.txt \
    --sense-voice-model=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/model.int8.onnx \
    --num-threads=1 \
    sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/test_wavs/yue-12.wav

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
    --tokens=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/tokens.txt \
    --sense-voice-model=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/model.int8.onnx \
    --num-threads=1 \
    sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/test_wavs/yue-13.wav

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
    --tokens=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/tokens.txt \
    --sense-voice-model=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/model.int8.onnx \
    --num-threads=1 \
    sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/test_wavs/yue-14.wav

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
    --tokens=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/tokens.txt \
    --sense-voice-model=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/model.int8.onnx \
    --num-threads=1 \
    sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/test_wavs/yue-15.wav

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
    --tokens=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/tokens.txt \
    --sense-voice-model=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/model.int8.onnx \
    --num-threads=1 \
    sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/test_wavs/yue-16.wav

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
    --tokens=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/tokens.txt \
    --sense-voice-model=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/model.int8.onnx \
    --num-threads=1 \
    sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/test_wavs/yue-17.wav

.. literalinclude:: ./code/yue-17.txt
