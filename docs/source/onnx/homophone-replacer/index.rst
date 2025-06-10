拼音词组替换
============

.. hint::

   For English-speaking users, this page is for users using Chinese ASR models.

本文描述如何使用预定义的拼音规则，对匹配规则的汉字，进行替换。

.. hint::

   本文所描述的方法，与 ``hotwrods`` （热词）是不同的技术。它们互相独立、毫不相关。

使用场景
----------

举个例子，如果一个人说了下面 4 个字::

  xuán jiè xīn piàn

我们如何让模型输出 ``玄戒芯片``, 而不是 ``玄界芯片``，或者 ``悬界心骗`` 呢？

又举个例子，如果某人 ``f`` 和 ``h`` 分不清，把 ``湖南人`` 说成了::

 fú nán rén

我们如何让模型输出 ``湖南人`` 呢？

又双叒叕举一个例子，对于一些专有名词，比如，``弓头安装`` ，``机载传感器`` 等，如何准确的识别出来，
而不是识别成 ``公投安装`` ，``基载传感器`` 等词组呢？

使用限制
----------

只支持对汉字进行替换。

只支持对汉字进行替换。

只支持对汉字进行替换。

.. hint::

   重要的事情，重复三遍。

.. note::

   目前没计划实现对非汉字的字符进行替换。


支持 `sherpa-onnx`_ 里面，所有能输出中文的语音识别模型。不管是流式还是非流式，
不管采用何种解码方法，都支持。例如，非流式的 SenseVoice，流式的 Zipformer 等等。

.. hint::

   输出中英文的模型，也支持。但只能替换识别结果中的中文。非中文字符，会原样保留，不做任何替换。

规则文件，需要提前生成好。目前不支持动态修改规则文件。

使用方法
--------

需要用到3个文件。


.. list-table::

 * - 文件名
   - 说明
   - 下载地址
 * - ``dict.tar.bz2``
   - | 用于jieba 分词
     | ``通用``
     | 请解压，得到 dict 文件夹
   - `<https://github.com/k2-fsa/sherpa-onnx/releases/tag/hr-files>`_
 * - ``lexicon.txt``
   - | 用于汉字转拼音
     | ``通用``
   - `<https://github.com/k2-fsa/sherpa-onnx/releases/tag/hr-files>`_
 * - ``replace.fst``
   - | ``不通用``
     | 用户自己提供
   - 如何生成，请见本文后半部分

生成replace.fst
~~~~~~~~~~~~~~~~~~~~

接下来，我们描述如何生成 ``replace.fst`` 。

为了方便用户输入规则，我们不采用 ``xuán jiè xīn piàn``, 而是使用
``xuan2 jie4 xin1 pian4`` 。即用 1，2，3，4 来代替第一声、第二声、第三声和第四声。

.. hint::

   如果是轻声，请用第一声代替。比如 ``ma1 ma1``。

首先，我们需要安装 `pynini <https://github.com/kylebgorman/pynini>`_。

.. hint::

   对于 Linux 用户，请使用::

      pip install --only-binary :all: pynini

   对于非 Linux 用户，请找一台 Linux 系统的电脑，变成 Linux 用户。

   任何有关 `pynini <https://github.com/kylebgorman/pynini>`_ 的安装问题，请去
   `<https://github.com/kylebgorman/pynini/issues>`_ 提问。

   友情提示：我们还提供一个 `colab <https://colab.research.google.com/drive/1jEaS3s8FbRJIcVQJv2EQx19EM_mnuARi?usp=sharing>`_ ，供在线生成规则文件。


安装好 ``pynini`` 后，我们用下面的代码，生成针对本文开始部分提及的3个问题的规则文件:

.. literalinclude:: ./code/test2.py
   :language: python

测试
------

我们把生成的 ``replace.fst`` 重名为了 `hr-xuan-jie-replace.fst <https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/hr-xuan-jie-replace.fst>`_ 。 你可以在 `<https://github.com/k2-fsa/sherpa-onnx/releases/tag/hr-files>`_ 下载它。在同一个页面，你可以下载下面的测试音频 `hr-xuan-jie-test.wav <https://github.com/k2-fsa/sherpa-onnx/releases/tag/hr-files>`_

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
    </tr>
    <tr>
      <td>hr-xuan-jie-test.wav</td>
      <td>
       <audio title="hr-xuan-jie-test.wav" controls="controls">
             <source src="/sherpa/_static/hr/hr-xuan-jie-test.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
    </tr>
  </table>

下面我们使用 SenseVoice 模型 去识别这个音频。分两种情况:

  - (1) 不使用本文的方法
  - (2) 使用本文的方法

请大家注意比较两种方法识别的结果。

(1) 不使用本文的方法
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt \
    --sense-voice-model=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.int8.onnx \
    --debug=0 \
    ./hr-xuan-jie-test.wav

输出的 log 如下:

.. literalinclude:: ./code/without.txt

(2) 使用本文的方法
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt \
    --sense-voice-model=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.int8.onnx \
    --debug=0 \
    --hr-lexicon=./lexicon.txt \
    --hr-dict-dir=./dict \
    --hr-rule-fsts=./hr-xuan-jie-replace.fst \
    ./hr-xuan-jie-test.wav

.. hint::

   上述命令行工具，指定了3个参数:

     - ``--hr-lexicon``: 通用。`下载地址 <https://github.com/k2-fsa/sherpa-onnx/releases/tag/hr-files>`_
     - ``--hr-dict-dir``: 通用。`下载地址 <https://github.com/k2-fsa/sherpa-onnx/releases/tag/hr-files>`_
     - ``--hr-rule-fsts``: 为我们自己生成的规则文件。不通用。

   如果你是通过调用 API 的方式使用，请设置 ``OfflineRecongizerConfig`` 或者
   ``OnlineRecognizerConfig`` 里面的成员 ``hr`` 。


输出的 log 如下:

.. literalinclude:: ./code/with.txt

(3) 结果比较
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::

 * - 方法
   - 识别结果
 * - 不用本文的方法
   - 下面是一个测试 ``悬界芯片`` 湖南人 ``工投安装`` ``基载传感器``
 * - 用本文的方法
   - 下面是一个测试 ``玄戒芯片`` 湖南人 ``弓头安装`` ``机载传感器``

调试
------

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt \
    --sense-voice-model=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.int8.onnx \
    --debug=1 \
    --hr-lexicon=./lexicon.txt \
    --hr-dict-dir=./dict \
    --hr-rule-fsts=./hr-xuan-jie-replace.fst \
    ./hr-xuan-jie-test.wav

设置 ``--deubg=1`` 可以输出如下调试信息

.. code-block::

  Started
  /Users/fangjun/open-source/sherpa-onnx/sherpa-onnx/csrc/homophone-replacer.cc:Apply:157 Input text: '下面是一个测试悬界芯片湖南人工投安装基载传感器'
  /Users/fangjun/open-source/sherpa-onnx/sherpa-onnx/csrc/homophone-replacer.cc:Apply:165 After jieba: 下面_是_一个_测试_悬界_芯片_湖南_人工_投_安装_基载_传感器
  /Users/fangjun/open-source/sherpa-onnx/sherpa-onnx/csrc/homophone-replacer.cc:Apply:186 下面 xia4mian4
  /Users/fangjun/open-source/sherpa-onnx/sherpa-onnx/csrc/homophone-replacer.cc:Apply:186 是 shi4
  /Users/fangjun/open-source/sherpa-onnx/sherpa-onnx/csrc/homophone-replacer.cc:Apply:186 一个 yi2ge4
  /Users/fangjun/open-source/sherpa-onnx/sherpa-onnx/csrc/homophone-replacer.cc:Apply:186 测试 ce4shi4
  /Users/fangjun/open-source/sherpa-onnx/sherpa-onnx/csrc/homophone-replacer.cc:Apply:186 悬界 xuan2jie4
  /Users/fangjun/open-source/sherpa-onnx/sherpa-onnx/csrc/homophone-replacer.cc:Apply:186 芯片 xin1pian4
  /Users/fangjun/open-source/sherpa-onnx/sherpa-onnx/csrc/homophone-replacer.cc:Apply:186 湖南 hu2nan2
  /Users/fangjun/open-source/sherpa-onnx/sherpa-onnx/csrc/homophone-replacer.cc:Apply:186 人工 ren2gong1
  /Users/fangjun/open-source/sherpa-onnx/sherpa-onnx/csrc/homophone-replacer.cc:Apply:186 投 tou2
  /Users/fangjun/open-source/sherpa-onnx/sherpa-onnx/csrc/homophone-replacer.cc:Apply:186 安装 an1zhuang1
  /Users/fangjun/open-source/sherpa-onnx/sherpa-onnx/csrc/homophone-replacer.cc:Apply:186 基载 ji1zai4
  /Users/fangjun/open-source/sherpa-onnx/sherpa-onnx/csrc/homophone-replacer.cc:Apply:186 传感器 chuan2gan3qi4
  /Users/fangjun/open-source/sherpa-onnx/sherpa-onnx/csrc/homophone-replacer.cc:Apply:198 Output text: '下面是一个测试玄戒芯片湖南人头安装机载传感器'
  Done!

  ./hr-xuan-jie-test.wav
  {"lang": "<|zh|>", "emotion": "<|NEUTRAL|>", "event": "<|Speech|>", "text": "下面是一个测试玄戒芯片湖南人弓头安装机载传感器", "timestamps": [0.60, 0.84, 1.08, 1.26, 1.38, 1.56, 1.80, 2.88, 3.00, 3.48, 3.66, 5.34, 5.46, 5.64, 7.14, 7.26, 7.62, 7.80, 9.24, 9.42, 9.78, 9.90, 10.14], "tokens":["下", "面", "是", "一", "个", "测", "试", "悬", "界", "芯", "片", "湖", "南", "人", "工", "投", "安", "装", "基", "载", "传", "感", "器"], "words": []}
  ----
  num threads: 2
  decoding method: greedy_search
  Elapsed seconds: 0.583 s
  Real time factor (RTF): 0.583 / 11.673 = 0.050

大家可以根据调试信息中的拼音，去调整对应的规则。对于一条规则，只有它其中的拼音全部匹配上时，才会替换。
比如 ``xuan2jie4xin1pian4`` 这条规则，无法匹配 ``玄界`` 或者 ``新片`` 。只有 ``玄界新片`` 一起出现时，
才会匹配成功，替换成 ``玄戒芯片``。

.. warning::

   对于上面的例子 ``玄界新片`` ， 如果没有替换成 ``玄戒芯片`` ，那么需要看调试信息中，``片``
   这个字的发音，是 ``pian1`` 还是 ``pian4`` 。如果是 ``pian1``, 那么，你还需要加一条规则，
   把 ``xuan2jie4xin1pian1`` 变成 ``玄戒芯片`` 。 添加的规则如下::

      rule7 = pynini.cross("xuan2jie4xin1pian1", "玄戒芯片")
