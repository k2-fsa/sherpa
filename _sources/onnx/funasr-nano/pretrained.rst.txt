Pre-trained Models
==================

This page describes how to download pre-trained `Fun-ASR-Nano-2512`_ models.

.. _sherpa-onnx-funasr-nano-int8-2025-12-30:

sherpa-onnx-funasr-nano-int8-2025-12-30 (Chinese, English, Japanese)
--------------------------------------------------------------------

This model is converted from `Fun-ASR-Nano-2512`_
using scripts from `<https://github.com/Wasser1462/FunASR-nano-onnx>`_.

It supports the following 3 languages:

  - Chinese
  - English
  - Japanese

.. hint::

   中文包括 7 种方言（吴语、粤语、闽语、客家话、赣语、湘语、晋语）和
   26 种地方口音（河南、山西、湖北、四川、重庆、云南、贵州、广东、广西
   及其他 20 多个地区）。

   英文和日文涵盖多种地方口音。

   此外还支持歌词识别和说唱语音识别。

In the following, we describe how to use it.

Huggingface space
^^^^^^^^^^^^^^^^^

You can visit

  `<https://huggingface.co/spaces/k2-fsa/automatic-speech-recognition>`_

to try this model in your browser.

.. hint::

   You need to first select the language ``31 languages (FunASR Nano)``
   and then select the model  ``csukuangfj/sherpa-onnx-funasr-nano-int8-2025-12-30``.

Android APKs
^^^^^^^^^^^^

Real-time speech recognition Android APKs can be found at

  `<https://k2-fsa.github.io/sherpa/onnx/vad/apk-asr.html>`_

Please always download the latest version.

.. hint::

   Please search for ``multi-funasr_nano_int8_2025_12_30.apk`` in the above page, e.g.,
   ``sherpa-onnx-1.12.21-arm64-v8a-vad_asr-multi-funasr_nano_int8_2025_12_30.apk``.

.. hint::

   For Chinese users, you can also visit `<https://k2-fsa.github.io/sherpa/onnx/vad/apk-asr-cn.html>`_

Download
^^^^^^^^

Please use the following commands to download it::

  cd /path/to/sherpa-onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2

  # For Chinese users, you can also use
  # wget https://modelscope.cn/models/csukuangfj/asr-models/resolve/master/sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2

  tar xvf sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2
  rm sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2

After downloading, you should find the following files::

  ls -lh sherpa-onnx-funasr-nano-int8-2025-12-30/

  total 948M
  drwxr-xr-x  5 kuangfangjun root    0 Jan  7 19:28 Qwen3-0.6B
  -rw-r--r--  1 kuangfangjun root  253 Jan  7 19:33 README.md
  -rw-r--r--  1 kuangfangjun root 149M Jan  7 19:33 embedding.int8.onnx
  -rw-r--r--  1 kuangfangjun root 227M Jan  7 19:34 encoder_adaptor.int8.onnx
  -rw-r--r--  1 kuangfangjun root 573M Jan  7 19:34 llm.int8.onnx
  drwxr-xr-x 27 kuangfangjun root    0 Jan  7 19:28 test_wavs

.. code-block::

  ls -lh sherpa-onnx-funasr-nano-int8-2025-12-30/Qwen3-0.6B/
  total 16M
  -rw-r--r-- 1 kuangfangjun root 1.6M Jan  7 19:34 merges.txt
  -rw-r--r-- 1 kuangfangjun root  11M Jan  7 19:34 tokenizer.json
  -rw-r--r-- 1 kuangfangjun root 2.7M Jan  7 19:34 vocab.json

.. code-block::

  ls -lh sherpa-onnx-funasr-nano-int8-2025-12-30/test_wavs/
  total 9.7M
  -rw-r--r-- 1 kuangfangjun root 6.9K Jan  7 19:33 README.md
  -rw-r--r-- 1 kuangfangjun root 220K Jan  7 19:33 dia_hunan.wav
  -rw-r--r-- 1 kuangfangjun root 253K Jan  7 19:33 dia_minnan.wav
  -rw-r--r-- 1 kuangfangjun root 229K Jan  7 19:33 dia_sh.wav
  -rw-r--r-- 1 kuangfangjun root 297K Jan  7 19:33 dia_yue.wav
  -rw-r--r-- 1 kuangfangjun root 215K Jan  7 19:33 far_2.wav
  -rw-r--r-- 1 kuangfangjun root 682K Jan  7 19:33 far_3.wav
  -rw-r--r-- 1 kuangfangjun root 284K Jan  7 19:33 far_4.wav
  -rw-r--r-- 1 kuangfangjun root 279K Jan  7 19:33 far_5.wav
  -rw-r--r-- 1 kuangfangjun root 254K Jan  7 19:33 ja.wav
  -rw-r--r-- 1 kuangfangjun root 255K Jan  7 19:33 ja_en_codeswitch.wav
  -rw-r--r-- 1 kuangfangjun root 259K Jan  7 19:33 lyrics.wav
  -rw-r--r-- 1 kuangfangjun root 431K Jan  7 19:33 lyrics_2.wav
  -rw-r--r-- 1 kuangfangjun root 546K Jan  7 19:33 lyrics_3.wav
  -rw-r--r-- 1 kuangfangjun root 1.3M Jan  7 19:33 lyrics_en_1.wav
  -rw-r--r-- 1 kuangfangjun root 679K Jan  7 19:33 lyrics_en_2.wav
  -rw-r--r-- 1 kuangfangjun root 1.7M Jan  7 19:33 lyrics_en_3.wav
  -rw-r--r-- 1 kuangfangjun root 331K Jan  7 19:33 noise_en.wav
  -rw-r--r-- 1 kuangfangjun root 267K Jan  7 19:33 rag_biochemistry.wav
  -rw-r--r-- 1 kuangfangjun root 214K Jan  7 19:33 rag_chemistry.wav
  -rw-r--r-- 1 kuangfangjun root 248K Jan  7 19:33 rag_history.wav
  -rw-r--r-- 1 kuangfangjun root 173K Jan  7 19:33 rag_math.wav
  -rw-r--r-- 1 kuangfangjun root 192K Jan  7 19:33 rag_medical.wav
  -rw-r--r-- 1 kuangfangjun root 379K Jan  7 19:33 rag_physics.wav
  -rw-r--r-- 1 kuangfangjun root 224K Jan  7 19:33 vietnamese.wav

.. hint::

   If you need the ``float32`` model file, please use::

     wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-funasr-nano-2025-12-30.tar.bz2

     # For Chinese users, you can also use
     # wget https://modelscope.cn/models/csukuangfj/asr-models/resolve/master/sherpa-onnx-funasr-nano-2025-12-30.tar.bz2

     tar xvf sherpa-onnx-funasr-nano-2025-12-30.tar.bz2
     rm sherpa-onnx-funasr-nano-2025-12-30.tar.bz2

   .. code-block::

      ls -lh sherpa-onnx-funasr-nano-2025-12-30

      total 3.7G
      drwxr-xr-x  5 kuangfangjun root     0 Jan  7 19:27 Qwen3-0.6B
      -rw-r--r--  1 kuangfangjun root   253 Jan 13 12:19 README.md
      -rw-r--r--  1 kuangfangjun root  594M Jan 13 12:20 embedding.onnx
      -rw-r--r--  1 kuangfangjun root  888M Jan 13 12:19 encoder_adaptor.onnx
      -rw-r--r--  1 kuangfangjun root  2.3G Jan 13 12:22 llm.fp32.data
      -rw-r--r--  1 kuangfangjun root 1011K Jan 13 12:20 llm.fp32.onnx
      drwxr-xr-x 27 kuangfangjun root     0 Jan  7 19:27 test_wavs

   If you need the ``float16`` model file, please use::

      wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-funasr-nano-fp16-2025-12-30.tar.bz2

      # For Chinese users, you can also use
      # wget https://modelscope.cn/models/csukuangfj/asr-models/resolve/master/sherpa-onnx-funasr-nano-fp16-2025-12-30.tar.bz2

      tar xvf sherpa-onnx-funasr-nano-fp16-2025-12-30.tar.bz2
      rm sherpa-onnx-funasr-nano-fp16-2025-12-30.tar.bz2


   .. code-block::

      ls -lh sherpa-onnx-funasr-nano-fp16-2025-12-30/

      total 1.5G
      drwxr-xr-x  5 kuangfangjun root    0 Jan  7 19:24 Qwen3-0.6B
      -rw-r--r--  1 kuangfangjun root  253 Jan 13 12:26 README.md
      -rw-r--r--  1 kuangfangjun root 149M Jan 13 12:26 embedding.int8.onnx
      -rw-r--r--  1 kuangfangjun root 227M Jan 13 12:27 encoder_adaptor.int8.onnx
      -rw-r--r--  1 kuangfangjun root 1.2G Jan 13 12:27 llm.fp16.onnx
      drwxr-xr-x 27 kuangfangjun root    0 Jan  7 19:24 test_wavs

.. hint::

   The test wave files are from

    `<https://funaudiollm.github.io/funasr/>`_

dia_hunan.wav (湖南方言)
^^^^^^^^^^^^^^^^^^^^^^^^

To decode the test file ``./sherpa-onnx-funasr-nano-int8-2025-12-30/test_wavs/dia_hunan.wav``:

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline \
    --funasr-nano-encoder-adaptor=./sherpa-onnx-funasr-nano-int8-2025-12-30/encoder_adaptor.int8.onnx \
    --funasr-nano-llm=./sherpa-onnx-funasr-nano-int8-2025-12-30/llm.int8.onnx \
    --funasr-nano-tokenizer=./sherpa-onnx-funasr-nano-int8-2025-12-30/Qwen3-0.6B \
    --funasr-nano-embedding=./sherpa-onnx-funasr-nano-int8-2025-12-30/embedding.int8.onnx \
    ./sherpa-onnx-funasr-nano-int8-2025-12-30/test_wavs/dia_hunan.wav

You should see the following output:

.. literalinclude:: ./code/dia_hunan.txt

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Ground truth</th>
    </tr>
    <tr>
      <td>dia_hunan.wav</td>
      <td>
       <audio title="dia_hunan.wav" controls="controls">
             <source src="/sherpa/_static/fun-asr-nano-2025-12-30/dia_hunan.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
        但总来讲孙膑对兵法的理解运用比庞涓略胜一筹。
      </td>
    </tr>
  </table>


dia_minnan.wav (闽南语)
^^^^^^^^^^^^^^^^^^^^^^^^

To decode the test file ``./sherpa-onnx-funasr-nano-int8-2025-12-30/test_wavs/dia_minnan.wav``:

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline \
    --funasr-nano-encoder-adaptor=./sherpa-onnx-funasr-nano-int8-2025-12-30/encoder_adaptor.int8.onnx \
    --funasr-nano-llm=./sherpa-onnx-funasr-nano-int8-2025-12-30/llm.int8.onnx \
    --funasr-nano-tokenizer=./sherpa-onnx-funasr-nano-int8-2025-12-30/Qwen3-0.6B \
    --funasr-nano-embedding=./sherpa-onnx-funasr-nano-int8-2025-12-30/embedding.int8.onnx \
    ./sherpa-onnx-funasr-nano-int8-2025-12-30/test_wavs/dia_minnan.wav

You should see the following output:

.. literalinclude:: ./code/dia_minnan.txt

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Ground truth</th>
    </tr>
    <tr>
      <td>dia_minnan.wav</td>
      <td>
       <audio title="dia_minnan.wav" controls="controls">
             <source src="/sherpa/_static/fun-asr-nano-2025-12-30/dia_minnan.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
        嗯，下摆若有机会吧，因为即久吼开了吼卷啊遮厉害，会倒贴钱啊。
      </td>
    </tr>
  </table>

dia_sh.wav (上海话)
^^^^^^^^^^^^^^^^^^^^^^^^

To decode the test file ``./sherpa-onnx-funasr-nano-int8-2025-12-30/test_wavs/dia_sh.wav``:

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline \
    --funasr-nano-encoder-adaptor=./sherpa-onnx-funasr-nano-int8-2025-12-30/encoder_adaptor.int8.onnx \
    --funasr-nano-llm=./sherpa-onnx-funasr-nano-int8-2025-12-30/llm.int8.onnx \
    --funasr-nano-tokenizer=./sherpa-onnx-funasr-nano-int8-2025-12-30/Qwen3-0.6B \
    --funasr-nano-embedding=./sherpa-onnx-funasr-nano-int8-2025-12-30/embedding.int8.onnx \
    ./sherpa-onnx-funasr-nano-int8-2025-12-30/test_wavs/dia_sh.wav

You should see the following output:

.. literalinclude:: ./code/dia_sh.txt

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Ground truth</th>
    </tr>
    <tr>
      <td>dia_sh.wav</td>
      <td>
       <audio title="dia_sh.wav" controls="controls">
             <source src="/sherpa/_static/fun-asr-nano-2025-12-30/dia_sh.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
        人跟狗，包括人跟动物接触长了，全有感情。葛末随了阿拉社会个富裕。
      </td>
    </tr>
  </table>

dia_yue.wav (粤语，广东话)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To decode the test file ``./sherpa-onnx-funasr-nano-int8-2025-12-30/test_wavs/dia_yue.wav``:

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline \
    --funasr-nano-encoder-adaptor=./sherpa-onnx-funasr-nano-int8-2025-12-30/encoder_adaptor.int8.onnx \
    --funasr-nano-llm=./sherpa-onnx-funasr-nano-int8-2025-12-30/llm.int8.onnx \
    --funasr-nano-tokenizer=./sherpa-onnx-funasr-nano-int8-2025-12-30/Qwen3-0.6B \
    --funasr-nano-embedding=./sherpa-onnx-funasr-nano-int8-2025-12-30/embedding.int8.onnx \
    ./sherpa-onnx-funasr-nano-int8-2025-12-30/test_wavs/dia_yue.wav

You should see the following output:

.. literalinclude:: ./code/dia_yue.txt

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Ground truth</th>
    </tr>
    <tr>
      <td>dia_yue.wav</td>
      <td>
       <audio title="dia_yue.wav" controls="controls">
             <source src="/sherpa/_static/fun-asr-nano-2025-12-30/dia_yue.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
        啲身体好劲啊，跟住咧佢哋有一个人咧就突然可能就有高原反应啦，突然间就啊窒息咗，即系晕晕咗。
      </td>
    </tr>
  </table>

lyrics.wav (中文歌曲-1)
^^^^^^^^^^^^^^^^^^^^^^^

To decode the test file ``./sherpa-onnx-funasr-nano-int8-2025-12-30/test_wavs/lyrics.wav``:

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline \
    --funasr-nano-encoder-adaptor=./sherpa-onnx-funasr-nano-int8-2025-12-30/encoder_adaptor.int8.onnx \
    --funasr-nano-llm=./sherpa-onnx-funasr-nano-int8-2025-12-30/llm.int8.onnx \
    --funasr-nano-tokenizer=./sherpa-onnx-funasr-nano-int8-2025-12-30/Qwen3-0.6B \
    --funasr-nano-embedding=./sherpa-onnx-funasr-nano-int8-2025-12-30/embedding.int8.onnx \
    ./sherpa-onnx-funasr-nano-int8-2025-12-30/test_wavs/lyrics.wav

You should see the following output:

.. literalinclude:: ./code/lyrics.txt

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Ground truth</th>
    </tr>
    <tr>
      <td>lyrics.wav</td>
      <td>
       <audio title="lyrics.wav" controls="controls">
             <source src="/sherpa/_static/fun-asr-nano-2025-12-30/lyrics.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
      我看到我的身后盯着我的人群，喜欢或恨不一样的神情，我知道这可能就是所谓的成名，我知道必须往前一步也不能停。
      </td>
    </tr>
  </table>

lyrics_2.wav (中文歌曲-2)
^^^^^^^^^^^^^^^^^^^^^^^^^

To decode the test file ``./sherpa-onnx-funasr-nano-int8-2025-12-30/test_wavs/lyrics_2.wav``:

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline \
    --funasr-nano-encoder-adaptor=./sherpa-onnx-funasr-nano-int8-2025-12-30/encoder_adaptor.int8.onnx \
    --funasr-nano-llm=./sherpa-onnx-funasr-nano-int8-2025-12-30/llm.int8.onnx \
    --funasr-nano-tokenizer=./sherpa-onnx-funasr-nano-int8-2025-12-30/Qwen3-0.6B \
    --funasr-nano-embedding=./sherpa-onnx-funasr-nano-int8-2025-12-30/embedding.int8.onnx \
    ./sherpa-onnx-funasr-nano-int8-2025-12-30/test_wavs/lyrics_2.wav

You should see the following output:

.. literalinclude:: ./code/lyrics_2.txt

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Ground truth</th>
    </tr>
    <tr>
      <td>lyrics_2.wav</td>
      <td>
       <audio title="lyrics_2.wav" controls="controls">
             <source src="/sherpa/_static/fun-asr-nano-2025-12-30/lyrics_2.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
      明明那么远，为何却感觉离他那么近？闭上眼，你甚至能背出他所有押韵。虽然不听说唱了，但你已学会自信。我代表所有中文说唱歌手向你致敬。如今面对困难的你，早已不再抱怨。
      </td>
    </tr>
  </table>


lyrics_3.wav (中文歌曲-3)
^^^^^^^^^^^^^^^^^^^^^^^^^

To decode the test file ``./sherpa-onnx-funasr-nano-int8-2025-12-30/test_wavs/lyrics_3.wav``:

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline \
    --funasr-nano-encoder-adaptor=./sherpa-onnx-funasr-nano-int8-2025-12-30/encoder_adaptor.int8.onnx \
    --funasr-nano-llm=./sherpa-onnx-funasr-nano-int8-2025-12-30/llm.int8.onnx \
    --funasr-nano-tokenizer=./sherpa-onnx-funasr-nano-int8-2025-12-30/Qwen3-0.6B \
    --funasr-nano-embedding=./sherpa-onnx-funasr-nano-int8-2025-12-30/embedding.int8.onnx \
    ./sherpa-onnx-funasr-nano-int8-2025-12-30/test_wavs/lyrics_3.wav

You should see the following output:

.. literalinclude:: ./code/lyrics_3.txt

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Ground truth</th>
    </tr>
    <tr>
      <td>lyrics_3.wav</td>
      <td>
       <audio title="lyrics_3.wav" controls="controls">
             <source src="/sherpa/_static/fun-asr-nano-2025-12-30/lyrics_3.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
      你听啊秋末的落叶，你听它叹息着离别，只剩我独自领略海与山风和月，你听啊。
      </td>
    </tr>
  </table>

lyrics_en_1.wav (英文歌曲-1)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To decode the test file ``./sherpa-onnx-funasr-nano-int8-2025-12-30/test_wavs/lyrics_en_1.wav``:

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline \
    --funasr-nano-encoder-adaptor=./sherpa-onnx-funasr-nano-int8-2025-12-30/encoder_adaptor.int8.onnx \
    --funasr-nano-llm=./sherpa-onnx-funasr-nano-int8-2025-12-30/llm.int8.onnx \
    --funasr-nano-tokenizer=./sherpa-onnx-funasr-nano-int8-2025-12-30/Qwen3-0.6B \
    --funasr-nano-embedding=./sherpa-onnx-funasr-nano-int8-2025-12-30/embedding.int8.onnx \
    ./sherpa-onnx-funasr-nano-int8-2025-12-30/test_wavs/lyrics_en_1.wav

You should see the following output:

.. literalinclude:: ./code/lyrics_en_1.txt

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Ground truth</th>
    </tr>
    <tr>
      <td>lyrics_en_1.wav</td>
      <td>
       <audio title="lyrics_en_1.wav" controls="controls">
             <source src="/sherpa/_static/fun-asr-nano-2025-12-30/lyrics_en_1.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
      When I was young I'd listen to the radio. Waiting for my favorite songs. When they played I'd sing along. It made me smile.
      </td>
    </tr>
  </table>


lyrics_en_2.wav (英文歌曲-2)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To decode the test file ``./sherpa-onnx-funasr-nano-int8-2025-12-30/test_wavs/lyrics_en_2.wav``:

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline \
    --funasr-nano-encoder-adaptor=./sherpa-onnx-funasr-nano-int8-2025-12-30/encoder_adaptor.int8.onnx \
    --funasr-nano-llm=./sherpa-onnx-funasr-nano-int8-2025-12-30/llm.int8.onnx \
    --funasr-nano-tokenizer=./sherpa-onnx-funasr-nano-int8-2025-12-30/Qwen3-0.6B \
    --funasr-nano-embedding=./sherpa-onnx-funasr-nano-int8-2025-12-30/embedding.int8.onnx \
    ./sherpa-onnx-funasr-nano-int8-2025-12-30/test_wavs/lyrics_en_2.wav

You should see the following output:

.. literalinclude:: ./code/lyrics_en_2.txt

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Ground truth</th>
    </tr>
    <tr>
      <td>lyrics_en_2.wav</td>
      <td>
       <audio title="lyrics_en_2.wav" controls="controls">
             <source src="/sherpa/_static/fun-asr-nano-2025-12-30/lyrics_en_2.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
      I see your monsters. I see your pain. Tell me your problems; I'll chase them away. I'll be your lighthouse. I'll make it okay. When I see your monsters, I'll stand there so brave and chase them all away.
      </td>
    </tr>
  </table>


lyrics_en_3.wav (英文歌曲-3)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To decode the test file ``./sherpa-onnx-funasr-nano-int8-2025-12-30/test_wavs/lyrics_en_3.wav``:

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline \
    --funasr-nano-encoder-adaptor=./sherpa-onnx-funasr-nano-int8-2025-12-30/encoder_adaptor.int8.onnx \
    --funasr-nano-llm=./sherpa-onnx-funasr-nano-int8-2025-12-30/llm.int8.onnx \
    --funasr-nano-tokenizer=./sherpa-onnx-funasr-nano-int8-2025-12-30/Qwen3-0.6B \
    --funasr-nano-embedding=./sherpa-onnx-funasr-nano-int8-2025-12-30/embedding.int8.onnx \
    ./sherpa-onnx-funasr-nano-int8-2025-12-30/test_wavs/lyrics_en_3.wav

You should see the following output:

.. literalinclude:: ./code/lyrics_en_3.txt

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Ground truth</th>
    </tr>
    <tr>
      <td>lyrics_en_3.wav</td>
      <td>
       <audio title="lyrics_en_3.wav" controls="controls">
             <source src="/sherpa/_static/fun-asr-nano-2025-12-30/lyrics_en_3.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
      An empty street, an empty house, a hole inside my heart. I'm all alone and the rooms are getting smaller. I wonder how, I wonder why, I wonder where they are. The days we had, the songs we sang together.
      </td>
    </tr>
  </table>


noise_en.wav (英文歌曲-3)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To decode the test file ``./sherpa-onnx-funasr-nano-int8-2025-12-30/test_wavs/noise_en.wav``:

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline \
    --funasr-nano-encoder-adaptor=./sherpa-onnx-funasr-nano-int8-2025-12-30/encoder_adaptor.int8.onnx \
    --funasr-nano-llm=./sherpa-onnx-funasr-nano-int8-2025-12-30/llm.int8.onnx \
    --funasr-nano-tokenizer=./sherpa-onnx-funasr-nano-int8-2025-12-30/Qwen3-0.6B \
    --funasr-nano-embedding=./sherpa-onnx-funasr-nano-int8-2025-12-30/embedding.int8.onnx \
    ./sherpa-onnx-funasr-nano-int8-2025-12-30/test_wavs/noise_en.wav

You should see the following output:

.. literalinclude:: ./code/noise_en.txt

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Ground truth</th>
    </tr>
    <tr>
      <td>noise_en.wav</td>
      <td>
       <audio title="noise_en.wav" controls="controls">
             <source src="/sherpa/_static/fun-asr-nano-2025-12-30/noise_en.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
        So what's interesting here is I feel that you know brands knowing this when people sort of speak to the voice assistance at home and if you want to be the brand.
      </td>
    </tr>
  </table>


far_2.wav
^^^^^^^^^

To decode the test file ``./sherpa-onnx-funasr-nano-int8-2025-12-30/test_wavs/far_2.wav``:

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline \
    --funasr-nano-encoder-adaptor=./sherpa-onnx-funasr-nano-int8-2025-12-30/encoder_adaptor.int8.onnx \
    --funasr-nano-llm=./sherpa-onnx-funasr-nano-int8-2025-12-30/llm.int8.onnx \
    --funasr-nano-tokenizer=./sherpa-onnx-funasr-nano-int8-2025-12-30/Qwen3-0.6B \
    --funasr-nano-embedding=./sherpa-onnx-funasr-nano-int8-2025-12-30/embedding.int8.onnx \
    ./sherpa-onnx-funasr-nano-int8-2025-12-30/test_wavs/far_2.wav

You should see the following output:

.. literalinclude:: ./code/far_2.txt

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Ground truth</th>
    </tr>
    <tr>
      <td>far_2.wav</td>
      <td>
       <audio title="far_2.wav" controls="controls">
             <source src="/sherpa/_static/fun-asr-nano-2025-12-30/far_2.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
      然后被冠以了渣男线的称号，好了，不管这个，那么前方即将到达沈杜公路站，左边是8号线。
      </td>
    </tr>
  </table>

far_3.wav
^^^^^^^^^

To decode the test file ``./sherpa-onnx-funasr-nano-int8-2025-12-30/test_wavs/far_3.wav``:

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline \
    --funasr-nano-encoder-adaptor=./sherpa-onnx-funasr-nano-int8-2025-12-30/encoder_adaptor.int8.onnx \
    --funasr-nano-llm=./sherpa-onnx-funasr-nano-int8-2025-12-30/llm.int8.onnx \
    --funasr-nano-tokenizer=./sherpa-onnx-funasr-nano-int8-2025-12-30/Qwen3-0.6B \
    --funasr-nano-embedding=./sherpa-onnx-funasr-nano-int8-2025-12-30/embedding.int8.onnx \
    ./sherpa-onnx-funasr-nano-int8-2025-12-30/test_wavs/far_3.wav

You should see the following output:

.. literalinclude:: ./code/far_3.txt

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Ground truth</th>
    </tr>
    <tr>
      <td>far_3.wav</td>
      <td>
       <audio title="far_3.wav" controls="controls">
             <source src="/sherpa/_static/fun-asr-nano-2025-12-30/far_3.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
      周末要不要去露营，最近天气超舒服，露营？我怕虫子咬，而且晚上睡帐篷会不会很冷啊？放心，我借了专业装备还有暖宝宝，再带点火锅食材，边吃边看星星超惬意。
      </td>
    </tr>
  </table>

far_4.wav
^^^^^^^^^

To decode the test file ``./sherpa-onnx-funasr-nano-int8-2025-12-30/test_wavs/far_4.wav``:

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline \
    --funasr-nano-encoder-adaptor=./sherpa-onnx-funasr-nano-int8-2025-12-30/encoder_adaptor.int8.onnx \
    --funasr-nano-llm=./sherpa-onnx-funasr-nano-int8-2025-12-30/llm.int8.onnx \
    --funasr-nano-tokenizer=./sherpa-onnx-funasr-nano-int8-2025-12-30/Qwen3-0.6B \
    --funasr-nano-embedding=./sherpa-onnx-funasr-nano-int8-2025-12-30/embedding.int8.onnx \
    ./sherpa-onnx-funasr-nano-int8-2025-12-30/test_wavs/far_4.wav

You should see the following output:

.. literalinclude:: ./code/far_4.txt

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Ground truth</th>
    </tr>
    <tr>
      <td>far_4.wav</td>
      <td>
       <audio title="far_4.wav" controls="controls">
             <source src="/sherpa/_static/fun-asr-nano-2025-12-30/far_4.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
        唯一的遗憾就是他那个八宝鸭还有烤鸭都没吃上, 估计得提前预定吧, 只能怪我自己没有做好功课.
      </td>
    </tr>
  </table>

far_5.wav
^^^^^^^^^

To decode the test file ``./sherpa-onnx-funasr-nano-int8-2025-12-30/test_wavs/far_5.wav``:

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline \
    --funasr-nano-encoder-adaptor=./sherpa-onnx-funasr-nano-int8-2025-12-30/encoder_adaptor.int8.onnx \
    --funasr-nano-llm=./sherpa-onnx-funasr-nano-int8-2025-12-30/llm.int8.onnx \
    --funasr-nano-tokenizer=./sherpa-onnx-funasr-nano-int8-2025-12-30/Qwen3-0.6B \
    --funasr-nano-embedding=./sherpa-onnx-funasr-nano-int8-2025-12-30/embedding.int8.onnx \
    ./sherpa-onnx-funasr-nano-int8-2025-12-30/test_wavs/far_5.wav

You should see the following output:

.. literalinclude:: ./code/far_5.txt

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Ground truth</th>
    </tr>
    <tr>
      <td>far_5.wav</td>
      <td>
       <audio title="far_5.wav" controls="controls">
             <source src="/sherpa/_static/fun-asr-nano-2025-12-30/far_5.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
      别紧张, 我只是我是在这边逛街, 然后看到你们在这边拍照, 想跟你交个朋友, 认识一下.

      </td>
    </tr>
  </table>


rag_chemistry.wav
^^^^^^^^^^^^^^^^^^^

To decode the test file ``./sherpa-onnx-funasr-nano-int8-2025-12-30/test_wavs/rag_chemistry.wav``:

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline \
    --funasr-nano-encoder-adaptor=./sherpa-onnx-funasr-nano-int8-2025-12-30/encoder_adaptor.int8.onnx \
    --funasr-nano-llm=./sherpa-onnx-funasr-nano-int8-2025-12-30/llm.int8.onnx \
    --funasr-nano-tokenizer=./sherpa-onnx-funasr-nano-int8-2025-12-30/Qwen3-0.6B \
    --funasr-nano-embedding=./sherpa-onnx-funasr-nano-int8-2025-12-30/embedding.int8.onnx \
    ./sherpa-onnx-funasr-nano-int8-2025-12-30/test_wavs/rag_chemistry.wav

You should see the following output:

.. literalinclude:: ./code/rag_chemistry.txt

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Ground truth</th>
    </tr>
    <tr>
      <td>rag_chemistry.wav</td>
      <td>
       <audio title="rag_chemistry.wav" controls="controls">
             <source src="/sherpa/_static/fun-asr-nano-2025-12-30/rag_chemistry.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
      比如说酯在当时被认为是一种含氧酸盐
      </td>
    </tr>
  </table>

rag_history.wav
^^^^^^^^^^^^^^^^^^^

To decode the test file ``./sherpa-onnx-funasr-nano-int8-2025-12-30/test_wavs/rag_history.wav``:

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline \
    --funasr-nano-encoder-adaptor=./sherpa-onnx-funasr-nano-int8-2025-12-30/encoder_adaptor.int8.onnx \
    --funasr-nano-llm=./sherpa-onnx-funasr-nano-int8-2025-12-30/llm.int8.onnx \
    --funasr-nano-tokenizer=./sherpa-onnx-funasr-nano-int8-2025-12-30/Qwen3-0.6B \
    --funasr-nano-embedding=./sherpa-onnx-funasr-nano-int8-2025-12-30/embedding.int8.onnx \
    ./sherpa-onnx-funasr-nano-int8-2025-12-30/test_wavs/rag_history.wav

You should see the following output:

.. literalinclude:: ./code/rag_history.txt

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Ground truth</th>
    </tr>
    <tr>
      <td>rag_history.wav</td>
      <td>
       <audio title="rag_history.wav" controls="controls">
             <source src="/sherpa/_static/fun-asr-nano-2025-12-30/rag_history.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
      由罗马皇帝钦点的犹地亚王大希律王统治期间
      </td>
    </tr>
  </table>


rag_math.wav
^^^^^^^^^^^^^^^^^^^

To decode the test file ``./sherpa-onnx-funasr-nano-int8-2025-12-30/test_wavs/rag_math.wav``:

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline \
    --funasr-nano-encoder-adaptor=./sherpa-onnx-funasr-nano-int8-2025-12-30/encoder_adaptor.int8.onnx \
    --funasr-nano-llm=./sherpa-onnx-funasr-nano-int8-2025-12-30/llm.int8.onnx \
    --funasr-nano-tokenizer=./sherpa-onnx-funasr-nano-int8-2025-12-30/Qwen3-0.6B \
    --funasr-nano-embedding=./sherpa-onnx-funasr-nano-int8-2025-12-30/embedding.int8.onnx \
    ./sherpa-onnx-funasr-nano-int8-2025-12-30/test_wavs/rag_math.wav

You should see the following output:

.. literalinclude:: ./code/rag_math.txt

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Ground truth</th>
    </tr>
    <tr>
      <td>rag_math.wav</td>
      <td>
       <audio title="rag_math.wav" controls="controls">
             <source src="/sherpa/_static/fun-asr-nano-2025-12-30/rag_math.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
      对微分形式的积分是微分几何中的基本概念
      </td>
    </tr>
  </table>

rag_medical.wav
^^^^^^^^^^^^^^^^^^^

To decode the test file ``./sherpa-onnx-funasr-nano-int8-2025-12-30/test_wavs/rag_medical.wav``:

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline \
    --funasr-nano-encoder-adaptor=./sherpa-onnx-funasr-nano-int8-2025-12-30/encoder_adaptor.int8.onnx \
    --funasr-nano-llm=./sherpa-onnx-funasr-nano-int8-2025-12-30/llm.int8.onnx \
    --funasr-nano-tokenizer=./sherpa-onnx-funasr-nano-int8-2025-12-30/Qwen3-0.6B \
    --funasr-nano-embedding=./sherpa-onnx-funasr-nano-int8-2025-12-30/embedding.int8.onnx \
    ./sherpa-onnx-funasr-nano-int8-2025-12-30/test_wavs/rag_medical.wav

You should see the following output:

.. literalinclude:: ./code/rag_medical.txt

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Ground truth</th>
    </tr>
    <tr>
      <td>rag_medical.wav</td>
      <td>
       <audio title="rag_medical.wav" controls="controls">
             <source src="/sherpa/_static/fun-asr-nano-2025-12-30/rag_medical.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
      肾脏中肾小球囊上的细胞膜孔隙很小
      </td>
    </tr>
  </table>


rag_physics.wav
^^^^^^^^^^^^^^^^^^^

To decode the test file ``./sherpa-onnx-funasr-nano-int8-2025-12-30/test_wavs/rag_physics.wav``:

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline \
    --funasr-nano-encoder-adaptor=./sherpa-onnx-funasr-nano-int8-2025-12-30/encoder_adaptor.int8.onnx \
    --funasr-nano-llm=./sherpa-onnx-funasr-nano-int8-2025-12-30/llm.int8.onnx \
    --funasr-nano-tokenizer=./sherpa-onnx-funasr-nano-int8-2025-12-30/Qwen3-0.6B \
    --funasr-nano-embedding=./sherpa-onnx-funasr-nano-int8-2025-12-30/embedding.int8.onnx \
    ./sherpa-onnx-funasr-nano-int8-2025-12-30/test_wavs/rag_physics.wav

You should see the following output:

.. literalinclude:: ./code/rag_physics.txt

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Ground truth</th>
    </tr>
    <tr>
      <td>rag_physics.wav</td>
      <td>
       <audio title="rag_physics.wav" controls="controls">
             <source src="/sherpa/_static/fun-asr-nano-2025-12-30/rag_physics.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
      根据碰撞理论月面样本缺少挥发性物质
      </td>
    </tr>
  </table>

