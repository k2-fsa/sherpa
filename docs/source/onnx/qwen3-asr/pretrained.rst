Pre-trained Models
==================

This page describes how to download pre-trained `Qwen3-ASR`_ models.

.. _sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25:

sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25
--------------------------------------------------------------------

This model is converted from `Qwen3-ASR`_
using scripts from `<https://github.com/Wasser1462/Qwen3-ASR-onnx>`_.

It supports the following languages:

  - Chinese (zh), English (en), Cantonese (yue), Arabic (ar), German (de)
  - French (fr), Spanish (es), Portuguese (pt), Indonesian (id)
  - Italian (it), Korean (ko), Russian (ru), Thai (th)
  - Vietnamese (vi), Japanese (ja), Turkish (tr), Hindi (hi)
  - Malay (ms), Dutch (nl), Swedish (sv), Danish (da), Finnish (fi)
  - Polish (pl), Czech (cs), Filipino (fil), Persian (fa), Greek (el)
  - Hungarian (hu), Macedonian (mk), Romanian (ro)

It also supports the following Chinese dialects:

  - Anhui, Dongbei, Fujian, Gansu, Guizhou, Hebei, Henan, Hubei, Hunan
  - Jiangxi, Ningxia, Shandong, Shaanxi, Shanxi, Sichuan, Tianjin, Yunnan
  - Zhejiang, Cantonese (Hong Kong accent), Cantonese (Guangdong accent)
  - Wu language, Minnan language.

.. hint::

   支持的语言有:

    - 中文， 英语，粤语，阿拉伯语，德语，法语，西班牙语
    - 葡萄牙语，印尼语，意大利语，韩语，俄语，泰语，越南语
    - 日语，土耳其语，印地语，马来语，荷兰语，瑞典语，丹麦语
    - 芬兰语，波兰语，捷克语，菲律宾语，波斯语，希腊语，匈牙利语
    - 马其顿语，罗马尼亚语

   支持的中文方言有:

    - 安徽，东北，福建，甘肃，贵州，河北，河南，湖北，湖南，江西
    - 宁夏，山东，山西，陕西，四川，天津，云南，浙江，粤语（香港口音）
    - 粤语（广东口音）, 吴语, 闽南语


   此外还支持歌词识别和说唱语音识别。


The sections below show how to use it.

Download
^^^^^^^^

Please use the following commands to download it::

  cd /path/to/sherpa-onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25.tar.bz2
  tar xvf sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25.tar.bz2
  rm sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25.tar.bz2

After downloading, you should find the following files::

  ls -lh sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25

  total 1954552
  -rw-r--r--@  1 fangjun  staff    42M  7 Apr 17:45 conv_frontend.onnx
  -rw-r--r--@  1 fangjun  staff   721M  7 Apr 17:50 decoder.int8.onnx
  -rw-r--r--@  1 fangjun  staff   174M  7 Apr 17:50 encoder.int8.onnx
  -rw-r--r--@  1 fangjun  staff   328B  7 Apr 17:51 README.md
  drwxr-xr-x@ 19 fangjun  staff   608B  7 Apr 17:45 test_wavs
  drwxr-xr-x@  5 fangjun  staff   160B  7 Apr 17:51 tokenizer

.. code-block::

  ls -lh sherpa-onnx-funasr-nano-int8-2025-12-30/Qwen3-0.6B/
  total 16M
  -rw-r--r-- 1 kuangfangjun root 1.6M Jan  7 19:34 merges.txt
  -rw-r--r-- 1 kuangfangjun root  11M Jan  7 19:34 tokenizer.json
  -rw-r--r-- 1 kuangfangjun root 2.7M Jan  7 19:34 vocab.json

.. code-block::

  ls -lh sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/tokenizer

  total 8728
  -rw-r--r--@ 1 fangjun  staff   1.6M  7 Apr 17:50 merges.txt
  -rw-r--r--@ 1 fangjun  staff    12K  7 Apr 17:51 tokenizer_config.json
  -rw-r--r--@ 1 fangjun  staff   2.6M  7 Apr 17:51 vocab.json

.. code-block::

  ls -lh sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/test_wavs

  total 25624
  -rw-r--r--@ 1 fangjun  staff   164K  7 Apr 17:45 ar1.wav
  -rw-r--r--@ 1 fangjun  staff   514K  7 Apr 17:45 cantonese.wav
  -rw-r--r--@ 1 fangjun  staff   537K  7 Apr 17:45 codeswitch.wav
  -rw-r--r--@ 1 fangjun  staff   210K  7 Apr 17:45 de.wav
  -rw-r--r--@ 1 fangjun  staff   161K  7 Apr 17:45 es1.wav
  -rw-r--r--@ 1 fangjun  staff   1.6M  7 Apr 17:45 f1_noise.wav
  -rw-r--r--@ 1 fangjun  staff   980K  7 Apr 17:45 fast1.wav
  -rw-r--r--@ 1 fangjun  staff   187K  7 Apr 17:45 fr1.wav
  -rw-r--r--@ 1 fangjun  staff   438K  7 Apr 17:45 ja1.wav
  -rw-r--r--@ 1 fangjun  staff   2.7M  7 Apr 17:45 noise1-en.wav
  -rw-r--r--@ 1 fangjun  staff   724K  7 Apr 17:45 noise2.wav
  -rw-r--r--@ 1 fangjun  staff   1.6M  7 Apr 17:45 qiqiu1.wav
  -rw-r--r--@ 1 fangjun  staff   1.7M  7 Apr 17:45 raokouling.wav
  -rw-r--r--@ 1 fangjun  staff   914K  7 Apr 17:45 rap1.wav
  -rw-r--r--@ 1 fangjun  staff    76B  7 Apr 17:45 README.md
  -rw-r--r--@ 1 fangjun  staff   149K  7 Apr 17:45 ru1.wav
  -rw-r--r--@ 1 fangjun  staff   5.3K  7 Apr 17:45 transcript.txt

.. code-block::

  cat sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/test_wavs/transcript.txt

  fast1.wav 蹦出来之后，左手、右手接一个慢动作，右边再直接拉到这上面之后，直接拉到这个轮胎上，上边再接过去之后，然后上边再直接拉到这个位置了之后，右边再直接这个位置接倒过去的之后，再倒一下，然后右边再直接抓住这个上边了之后，直接从这边上边过去了之后，直接抓住这个树杈，然后这个位置直接倒到这个树杈
  raokouling.wav 广西壮族自治区爱吃红鲤鱼与绿鲤鱼与驴的出租车司机，拉着苗族土家族自治州爱喝自制的刘奶奶榴莲牛奶的骨质疏松症患者，遇见别着喇叭的哑巴，打败咬死山前四十四棵紫色柿子树的四十四只石狮子之后，碰到年年恋牛娘的牛郎，念着灰黑灰化肥发黑会挥发，走出香港官方网站设置组，到广西壮族自治区首府南宁市民族医院就医。
  noise2.wav 拨号，请再说一次，请说出您要拨打的号码。幺三五八幺八八七五七。一三五八二八八八幺八八。纠正纠正。九六九。纠正纠正，不是九六。
  qiqiu1.wav 黑的白的红的粉的紫的绿的蓝的灰的，你的我的他的她的大的小的圆的扁的，好的坏的美的丑的新的旧的，各种款式各种花式，让我选择。飞得高喽，越远越好，天都沦陷，他就死掉，说明多能高兴就好，喜欢就好，没大不了，越变越小，越来越小，快要死掉也很骄傲。你不想说就别再说，我不想听不想再听，就把一切誓言当作气球一般随它而去，我不在意不会在意，随它而去随它而去。气球飘进眼里，飘进风里，结束生命。气球飘进爱里，飘进心里，慢慢死去。
  cantonese.wav 今次寻寻觅觅，终于揾到my princess，肯借个场俾我哋玩。你知啦，喺香港地喺繁忙时间要揾个场嚟拍嘢系非常之难嘅。再一次多谢你哋，亦都好多谢片入边嘅每一个人。下一次我哋斗啲咩好？喺下面留言话我哋知啦。拜拜。
  f1_noise.wav Okay, Charles. It looks like we have a problem with the radio. What happened? Yeah, someone spilled water on their machine. I uh, yeah. Charles, can you hear us? Mamma mia.
  noise1-en.wav My girls, my girls, my girls, my girls. Ready? Hey, babe. Hey, babe. Where are you? I'm actually crazy traffic right now. Oh, really? Yeah. It's crazy. The freeway is completely stopped. Oh, you're still coming to my parents' house, right? Um, I can't really hear you, babe. What? Mariachi band playing live music, yeah. Babe. I can't. They're really loud, and I can't hear. Babe. What? Yeah, they're being really loud right now. I'm sorry. What were you saying? You're still coming to my parents' house, right? It actually started raining like crazy. What? It's raining and thundering like crazy, man. I can't hear shit. Where are you? Out of nowhere, it's just pouring. It's pouring. It's pouring like crazy. What? Insane. I don't think I can get anywhere today. It's crazy day. Babe, that's crazy. Where are you? Oh my God! Someone just hit my car. Come on, get in the car, Gabe. Oh my God! He's getting out of his car. He's getting out of his car. What? Hey, Mari, you just hit my car! Oh, babe, this guy's crazy. What the fuck? Out, out, out, out, out! How you like that? Oh, babe, he's beating the shit out of you. Hold on a sec, babe. Oh, you know I'm gonna get my gun. Here's a gun. Oh, babe, he shot me. He shot me in the leg. He shot. Oh, babe, oh fuck! Babe, I need a driveway. I need to get out of here. I need to get out of here. I'm driving away. Where are you? Babe, this is crazy. No, I'm okay. I'll be fine. I'm okay. I'm okay. I just had to drive. Oh shit, babe, I think I'm getting pulled over now. I think I'm getting pulled over. Pull over your vehicle. Oh my God, babe, hold on a sec. Baby, talk to me. Oh shit. License and registration, sir. Yeah, of course, officer. Of course. Ah, babe, this is the worst day of my life. I just got pulled over. Oh my God. Oh my God. Officer, police. Oh my God. I think a riot's breaking out. We got a crazy riot. People are going crazy. Get out of here! It's like a lotus matter. There's like a war going on or some shit.
  rap1.wav Sometimes I just feel like quitting. I still might. Why do I put up this fight? Why do I still write? Sometimes it's hard enough just dealing with real life. Sometimes I wanna jump a state and just kill mics. And so these people want my level of skills, like, but I'm still white. Sometimes I just hate life. Something ain't right. Hit the brake lights. Taste of the state's right. Drawing the blank line. It ain't my fault.
  codeswitch.wav I'm alone, all by myself. Je suis tout seul. Sono tutto. Estoy solo.
  ar1.wav إطلالات مكياج عيون ذهبي لسهرات صيف عشرين واحد وعشرين بأسلوب النجمات.
  de.wav Raptorium Bergbau scheint profitierter als Monroe als Reaktion auf die wirtschaftlichen Ausfälle zu sein.
  es1.wav Esta prenda es amplia, recomiendo elegir una talla menor a la habitual.
  fr1.wav Alice et moi sommes allés à Paris voyager en train au printemps, c'était très amusant.
  ru1.wav Барсук, живущий в киевском зоопарке, совершил побег из своего вольера.
  ja1.wav 抜群の運動神経を持ち合わせ、どんな要求にも応えてきた。



.. hint::

   If you need the ``float32`` model file, please visit `<https://modelscope.cn/models/zengshuishui/Qwen3-ASR-onnx/tree/master/model_0.6B>`_

   If you need the ``1.7B`` model, please visit `<https://modelscope.cn/models/zengshuishui/Qwen3-ASR-onnx/tree/master/model_1.7B>`_

.. hint::

   The test wave files are from

    `<https://qwen.ai/blog?id=qwen3asr>`_

raokouling.wav (绕口令, 中文)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To decode the test file

  ``./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/test_wavs/test_wavs/raokouling.wav``:

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline \
    --qwen3-asr-conv-frontend=./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/conv_frontend.onnx \
    --qwen3-asr-encoder=./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/encoder.int8.onnx \
    --qwen3-asr-decoder=./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/decoder.int8.onnx \
    --qwen3-asr-tokenizer=./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/tokenizer \
    --qwen3-asr-max-new-tokens=512 \
    --num-threads=2 \
    ./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/test_wavs/raokouling.wav

You should see the following output:

.. literalinclude:: ./code/raokouling.txt

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Ground truth</th>
    </tr>
    <tr>
      <td>raokouling.wav</td>
      <td>
       <audio title="raokouling.wav" controls="controls">
             <source src="/sherpa/_static/qwen3-asr-0.6b/raokouling.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
      广西壮族自治区爱吃红鲤鱼与绿鲤鱼与驴的出租车司机，拉着苗族土家族自治州爱喝自制的刘奶奶榴莲牛奶的骨质疏松症患者，遇见别着喇叭的哑巴，打败咬死山前四十四棵紫色柿子树的四十四只石狮子之后，碰到年年恋牛娘的牛郎，念着灰黑灰化肥发黑会挥发，走出香港官方网站设置组，到广西壮族自治区首府南宁市民族医院就医。
      </td>
    </tr>
  </table>

.. code-block:: bash

  # Use hotwords. Note you use , to separate multiple hotwords

  ./build/bin/sherpa-onnx-offline \
    --qwen3-asr-conv-frontend=./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/conv_frontend.onnx \
    --qwen3-asr-encoder=./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/encoder.int8.onnx \
    --qwen3-asr-decoder=./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/decoder.int8.onnx \
    --qwen3-asr-tokenizer=./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/tokenizer \
    --qwen3-asr-max-new-tokens=512 \
    --num-threads=2 \
    --qwen3-asr-hotwords="骨质疏松症患者,咬死山前,紫色柿子树,年年恋牛娘,灰黑灰化肥,走出香港" \
    ./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/test_wavs/raokouling.wav

You should see the following output:

.. literalinclude:: ./code/raokouling-hotwords.txt

Decode a long audio file with VAD (Example 1/2, English)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following examples show how to decode a very long audio file with the help
of VAD.

.. code-block:: bash

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/Obama.wav

  ./build/bin/sherpa-onnx-vad-with-offline-asr \
    --silero-vad-model=./silero_vad.onnx \
    --silero-vad-threshold=0.2 \
    --silero-vad-min-speech-duration=0.2 \
    --qwen3-asr-conv-frontend=./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/conv_frontend.onnx \
    --qwen3-asr-encoder=./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/encoder.int8.onnx \
    --qwen3-asr-decoder=./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/decoder.int8.onnx \
    --qwen3-asr-tokenizer=./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/tokenizer \
    --qwen3-asr-max-new-tokens=512 \
    --num-threads=2 \
    ./Obama.wav

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
    </tr>
    <tr>
      <td>Obama.wav</td>
      <td>
       <audio title="Obama.wav" controls="controls">
             <source src="/sherpa/_static/sense-voice/Obama.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
    </tr>
  </table>

You should see the following output:

.. literalinclude:: ./code/obama.txt

Decode a long audio file with VAD (Example 2/2, Chinese)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following examples show how to decode a very long audio file with the help
of VAD.

.. code-block:: bash

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/lei-jun-test.wav

  ./build/bin/sherpa-onnx-vad-with-offline-asr \
    --silero-vad-model=./silero_vad.onnx \
    --silero-vad-threshold=0.2 \
    --silero-vad-min-speech-duration=0.2 \
    --qwen3-asr-conv-frontend=./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/conv_frontend.onnx \
    --qwen3-asr-encoder=./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/encoder.int8.onnx \
    --qwen3-asr-decoder=./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/decoder.int8.onnx \
    --qwen3-asr-tokenizer=./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/tokenizer \
    --qwen3-asr-max-new-tokens=512 \
    --num-threads=2 \
    ./lei-jun-test.wav

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
    </tr>
    <tr>
      <td>lei-jun-test.wav</td>
      <td>
       <audio title="lei-jun-test.wav" controls="controls">
             <source src="/sherpa/_static/sense-voice/lei-jun-test.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
    </tr>
  </table>

You should see the following output:

.. literalinclude:: ./code/lei-jun.txt
