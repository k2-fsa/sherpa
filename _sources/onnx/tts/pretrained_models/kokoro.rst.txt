Kokoro
======

This page lists pre-trained models from `<https://huggingface.co/hexgrad/Kokoro-82M>`_.

.. _kokoro-multi-lang-v1_0:

kokoro-multi-lang-v1_1 (Chinese + English, 103 speakers)
---------------------------------------------------------

This model contains 103 speakers. Please see

  `<https://github.com/k2-fsa/sherpa-onnx/pull/1942>`_

for details.

See also `<https://k2-fsa.github.io/sherpa/onnx/tts/all/Chinese-English/kokoro-multi-lang-v1_1.html>`_

.. list-table::

 * - Model
   - Comment
 * - `kokoro-int8-multi-lang-v1_1.tar.bz2 <https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-int8-multi-lang-v1_1.tar.bz2>`_
   - ``int8`` quantization
 * - `kokoro-multi-lang-v1_1.tar.bz2 <https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-multi-lang-v1_1.tar.bz2>`_
   - No quantization

kokoro-multi-lang-v1_0 (Chinese + English, 53 speakers)
-------------------------------------------------------

This model contains 53 speakers. The ONNX model is from
`<https://github.com/taylorchu/kokoro-onnx/releases/tag/v0.2.0>`_

.. hint::

   If you want to convert the kokoro 1.0 onnx model to sherpa-onnx, please
   see `<https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/kokoro/v1.0/run.sh>`_

This model in sherpa-onnx supports both English and Chinese.

In the following, we describe how to download it and use it with `sherpa-onnx`_.

.. warning::

   It is a multi-lingual model, but we only add English and Chinese support for it.

See also `<https://k2-fsa.github.io/sherpa/onnx/tts/all/Chinese-English/kokoro-multi-lang-v1_0.html>`_

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

  cd /path/to/sherpa-onnx
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-multi-lang-v1_0.tar.bz2
  tar xf kokoro-multi-lang-v1_0.tar.bz2
  rm kokoro-multi-lang-v1_0.tar.bz2

Please check that the file sizes of the pre-trained models are correct. See
the file sizes of ``*.onnx`` files below.

.. code-block::

  ls -lh kokoro-multi-lang-v1_0/
  total 718872
  -rw-r--r--    1 fangjun  staff    11K Feb  7 10:16 LICENSE
  -rw-r--r--    1 fangjun  staff    50B Feb  7 10:18 README.md
  -rw-r--r--    1 fangjun  staff    58K Feb  7 10:18 date-zh.fst
  drwxr-xr-x    9 fangjun  staff   288B Apr 19  2024 dict
  drwxr-xr-x  122 fangjun  staff   3.8K Nov 28  2023 espeak-ng-data
  -rw-r--r--    1 fangjun  staff   6.0M Feb  7 10:18 lexicon-gb-en.txt
  -rw-r--r--    1 fangjun  staff   5.6M Feb  7 10:18 lexicon-us-en.txt
  -rw-r--r--    1 fangjun  staff   2.3M Feb  7 10:18 lexicon-zh.txt
  -rw-r--r--    1 fangjun  staff   310M Feb  7 10:18 model.onnx
  -rw-r--r--    1 fangjun  staff    63K Feb  7 10:18 number-zh.fst
  -rw-r--r--    1 fangjun  staff    87K Feb  7 10:18 phone-zh.fst
  -rw-r--r--    1 fangjun  staff   687B Feb  7 10:18 tokens.txt
  -rw-r--r--    1 fangjun  staff    26M Feb  7 10:18 voices.bin

Map between speaker ID and speaker name
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This model contains 53 speakers and we use integer IDs ``0-52`` to represent
each speaker.

Please visit `<https://github.com/k2-fsa/sherpa-onnx/pull/1795>`_ to listen to
audio samples from different speakers.

The map is given below:

 - **ID to Speaker**

   .. code-block::

        0->af_alloy, 1->af_aoede, 2->af_bella, 3->af_heart, 4->af_jessica,
        5->af_kore, 6->af_nicole, 7->af_nova, 8->af_river, 9->af_sarah,
        10->af_sky, 11->am_adam, 12->am_echo, 13->am_eric, 14->am_fenrir,
        15->am_liam, 16->am_michael, 17->am_onyx, 18->am_puck, 19->am_santa,
        20->bf_alice, 21->bf_emma, 22->bf_isabella, 23->bf_lily, 24->bm_daniel,
        25->bm_fable, 26->bm_george, 27->bm_lewis, 28->ef_dora, 29->em_alex,
        30->ff_siwis, 31->hf_alpha, 32->hf_beta, 33->hm_omega, 34->hm_psi,
        35->if_sara, 36->im_nicola, 37->jf_alpha, 38->jf_gongitsune,
        39->jf_nezumi, 40->jf_tebukuro, 41->jm_kumo,
        42->pf_dora, 43->pm_alex, 44->pm_santa, 45->zf_xiaobei, 46->zf_xiaoni,
        47->zf_xiaoxiao, 48->zf_xiaoyi,49->zm_yunjian, 50->zm_yunxi,
        51->zm_yunxia, 52->zm_yunyang,

 - **Speaker to ID**

   .. code-block::

        af_alloy->0, af_aoede->1, af_bella->2, af_heart->3, af_jessica->4,
        af_kore->5, af_nicole->6, af_nova->7, af_river->8, af_sarah->9,
        af_sky->10, am_adam->11, am_echo->12, am_eric->13, am_fenrir->14,
        am_liam->15, am_michael->16, am_onyx->17, am_puck->18, am_santa->19,
        bf_alice->20, bf_emma->21, bf_isabella->22, bf_lily->23, bm_daniel->24,
        bm_fable->25, bm_george->26, bm_lewis->27, ef_dora->28, em_alex->29,
        ff_siwis->30, hf_alpha->31, hf_beta->32, hm_omega->33, hm_psi->34,
        if_sara->35, im_nicola->36, jf_alpha->37, jf_gongitsune->38,
        jf_nezumi->39, jf_tebukuro->40, jm_kumo->41, pf_dora->42, pm_alex->43,
        pm_santa->44, zf_xiaobei->45, zf_xiaoni->46, zf_xiaoxiao->47,
        zf_xiaoyi->48, zm_yunjian->49, zm_yunxi->50, zm_yunxia->51,
        zm_yunyang->52

Generate speech with executables compiled from C++
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container:: toggle

    .. container:: header

      Click ▶ to see it.

    .. code-block:: bash

      cd /path/to/sherpa-onnx

      for sid in $(seq 0 19); do
        build/bin/sherpa-onnx-offline-tts \
          --debug=0 \
          --kokoro-model=./kokoro-multi-lang-v1_0/model.onnx \
          --kokoro-voices=./kokoro-multi-lang-v1_0/voices.bin \
          --kokoro-tokens=./kokoro-multi-lang-v1_0/tokens.txt \
          --kokoro-data-dir=./kokoro-multi-lang-v1_0/espeak-ng-data \
          --kokoro-dict-dir=./kokoro-multi-lang-v1_0/dict \
          --kokoro-lexicon=./kokoro-multi-lang-v1_0/lexicon-us-en.txt,./kokoro-multi-lang-v1_0/lexicon-zh.txt \
          --num-threads=2 \
          --sid=$sid \
          --output-filename="./kokoro-1.0-sid-$sid-en-us.wav" \
          "Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone."
      done

      for sid in $(seq 20 27); do
        build/bin/sherpa-onnx-offline-tts \
          --debug=0 \
          --kokoro-model=./kokoro-multi-lang-v1_0/model.onnx \
          --kokoro-voices=./kokoro-multi-lang-v1_0/voices.bin \
          --kokoro-tokens=./kokoro-multi-lang-v1_0/tokens.txt \
          --kokoro-data-dir=./kokoro-multi-lang-v1_0/espeak-ng-data \
          --kokoro-dict-dir=./kokoro-multi-lang-v1_0/dict \
          --kokoro-lexicon=./kokoro-multi-lang-v1_0/lexicon-us-en.txt,./kokoro-multi-lang-v1_0/lexicon-zh.txt \
          --num-threads=2 \
          --sid=$sid \
          --output-filename="./kokoro-1.0-sid-$sid-en-gb.wav" \
          "Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone."
      done


      build/bin/sherpa-onnx-offline-tts \
        --debug=0 \
        --kokoro-model=./kokoro-multi-lang-v1_0/model.onnx \
        --kokoro-voices=./kokoro-multi-lang-v1_0/voices.bin \
        --kokoro-tokens=./kokoro-multi-lang-v1_0/tokens.txt \
        --kokoro-data-dir=./kokoro-multi-lang-v1_0/espeak-ng-data \
        --kokoro-dict-dir=./kokoro-multi-lang-v1_0/dict \
        --kokoro-lexicon=./kokoro-multi-lang-v1_0/lexicon-us-en.txt,./kokoro-multi-lang-v1_0/lexicon-zh.txt \
        --num-threads=2 \
        --sid=23 \
        --output-filename="./kokoro-1.0-sid-23-en-gb.wav" \
        "Liliana, the most beautiful and lovely assistant of our team"

      build/bin/sherpa-onnx-offline-tts \
        --debug=0 \
        --kokoro-model=./kokoro-multi-lang-v1_0/model.onnx \
        --kokoro-voices=./kokoro-multi-lang-v1_0/voices.bin \
        --kokoro-tokens=./kokoro-multi-lang-v1_0/tokens.txt \
        --kokoro-data-dir=./kokoro-multi-lang-v1_0/espeak-ng-data \
        --kokoro-dict-dir=./kokoro-multi-lang-v1_0/dict \
        --kokoro-lexicon=./kokoro-multi-lang-v1_0/lexicon-us-en.txt,./kokoro-multi-lang-v1_0/lexicon-zh.txt \
        --num-threads=2 \
        --sid=24 \
        --output-filename="./kokoro-1.0-sid-24-en-gb.wav" \
        "Liliana, the most beautiful and lovely assistant of our team"

      build/bin/sherpa-onnx-offline-tts \
        --debug=0 \
        --kokoro-model=./kokoro-multi-lang-v1_0/model.onnx \
        --kokoro-voices=./kokoro-multi-lang-v1_0/voices.bin \
        --kokoro-tokens=./kokoro-multi-lang-v1_0/tokens.txt \
        --kokoro-data-dir=./kokoro-multi-lang-v1_0/espeak-ng-data \
        --kokoro-dict-dir=./kokoro-multi-lang-v1_0/dict \
        --kokoro-lexicon=./kokoro-multi-lang-v1_0/lexicon-us-en.txt,./kokoro-multi-lang-v1_0/lexicon-zh.txt \
        --num-threads=2 \
        --sid=45 \
        --output-filename="./kokoro-1.0-sid-45-zh.wav" \
        "小米的核心价值观是什么？答案是真诚热爱！"

      build/bin/sherpa-onnx-offline-tts \
        --debug=0 \
        --kokoro-model=./kokoro-multi-lang-v1_0/model.onnx \
        --kokoro-voices=./kokoro-multi-lang-v1_0/voices.bin \
        --kokoro-tokens=./kokoro-multi-lang-v1_0/tokens.txt \
        --kokoro-data-dir=./kokoro-multi-lang-v1_0/espeak-ng-data \
        --kokoro-dict-dir=./kokoro-multi-lang-v1_0/dict \
        --kokoro-lexicon=./kokoro-multi-lang-v1_0/lexicon-us-en.txt,./kokoro-multi-lang-v1_0/lexicon-zh.txt \
        --num-threads=2 \
        --sid=45 \
        --output-filename="./kokoro-1.0-sid-45-zh-1.wav" \
        "当夜幕降临，星光点点，伴随着微风拂面，我在静谧中感受着时光的流转，思念如涟漪荡漾，梦境如画卷展开，我与自然融为一体，沉静在这片宁静的美丽之中，感受着生命的奇迹与温柔."

      build/bin/sherpa-onnx-offline-tts \
        --debug=0 \
        --kokoro-model=./kokoro-multi-lang-v1_0/model.onnx \
        --kokoro-voices=./kokoro-multi-lang-v1_0/voices.bin \
        --kokoro-tokens=./kokoro-multi-lang-v1_0/tokens.txt \
        --kokoro-data-dir=./kokoro-multi-lang-v1_0/espeak-ng-data \
        --kokoro-dict-dir=./kokoro-multi-lang-v1_0/dict \
        --kokoro-lexicon=./kokoro-multi-lang-v1_0/lexicon-us-en.txt,./kokoro-multi-lang-v1_0/lexicon-zh.txt \
        --num-threads=2 \
        --sid=46 \
        --output-filename="./kokoro-1.0-sid-46-zh.wav" \
        "小米的使命是，始终坚持做感动人心、价格厚道的好产品，让全球每个人都能享受科技带来的美好生活。"

      build/bin/sherpa-onnx-offline-tts \
        --debug=0 \
        --kokoro-model=./kokoro-multi-lang-v1_0/model.onnx \
        --kokoro-voices=./kokoro-multi-lang-v1_0/voices.bin \
        --kokoro-tokens=./kokoro-multi-lang-v1_0/tokens.txt \
        --kokoro-data-dir=./kokoro-multi-lang-v1_0/espeak-ng-data \
        --kokoro-dict-dir=./kokoro-multi-lang-v1_0/dict \
        --kokoro-lexicon=./kokoro-multi-lang-v1_0/lexicon-us-en.txt,./kokoro-multi-lang-v1_0/lexicon-zh.txt \
        --num-threads=2 \
        --sid=46 \
        --output-filename="./kokoro-1.0-sid-46-zh-1.wav" \
        "当夜幕降临，星光点点，伴随着微风拂面，我在静谧中感受着时光的流转，思念如涟漪荡漾，梦境如画卷展开，我与自然融为一体，沉静在这片宁静的美丽之中，感受着生命的奇迹与温柔."

      build/bin/sherpa-onnx-offline-tts \
        --debug=0 \
        --kokoro-model=./kokoro-multi-lang-v1_0/model.onnx \
        --kokoro-voices=./kokoro-multi-lang-v1_0/voices.bin \
        --kokoro-tokens=./kokoro-multi-lang-v1_0/tokens.txt \
        --kokoro-data-dir=./kokoro-multi-lang-v1_0/espeak-ng-data \
        --kokoro-dict-dir=./kokoro-multi-lang-v1_0/dict \
        --kokoro-lexicon=./kokoro-multi-lang-v1_0/lexicon-us-en.txt,./kokoro-multi-lang-v1_0/lexicon-zh.txt \
        --tts-rule-fsts=./kokoro-multi-lang-v1_0/number-zh.fst \
        --num-threads=2 \
        --sid=47 \
        --output-filename="./kokoro-1.0-sid-47-zh.wav" \
        "35年前，他于长沙出生, 在长白山长大。9年前他当上了银行的领导，主管行政。"

      build/bin/sherpa-onnx-offline-tts \
        --debug=0 \
        --kokoro-model=./kokoro-multi-lang-v1_0/model.onnx \
        --kokoro-voices=./kokoro-multi-lang-v1_0/voices.bin \
        --kokoro-tokens=./kokoro-multi-lang-v1_0/tokens.txt \
        --kokoro-data-dir=./kokoro-multi-lang-v1_0/espeak-ng-data \
        --kokoro-dict-dir=./kokoro-multi-lang-v1_0/dict \
        --kokoro-lexicon=./kokoro-multi-lang-v1_0/lexicon-us-en.txt,./kokoro-multi-lang-v1_0/lexicon-zh.txt \
        --num-threads=2 \
        --sid=47 \
        --output-filename="./kokoro-1.0-sid-47-zh-1.wav" \
        "当夜幕降临，星光点点，伴随着微风拂面，我在静谧中感受着时光的流转，思念如涟漪荡漾，梦境如画卷展开，我与自然融为一体，沉静在这片宁静的美丽之中，感受着生命的奇迹与温柔."


      build/bin/sherpa-onnx-offline-tts \
        --debug=0 \
        --kokoro-model=./kokoro-multi-lang-v1_0/model.onnx \
        --kokoro-voices=./kokoro-multi-lang-v1_0/voices.bin \
        --kokoro-tokens=./kokoro-multi-lang-v1_0/tokens.txt \
        --kokoro-data-dir=./kokoro-multi-lang-v1_0/espeak-ng-data \
        --kokoro-dict-dir=./kokoro-multi-lang-v1_0/dict \
        --kokoro-lexicon=./kokoro-multi-lang-v1_0/lexicon-us-en.txt,./kokoro-multi-lang-v1_0/lexicon-zh.txt \
        --tts-rule-fsts=./kokoro-multi-lang-v1_0/phone-zh.fst,./kokoro-multi-lang-v1_0/number-zh.fst \
        --num-threads=2 \
        --sid=48 \
        --output-filename="./kokoro-1.0-sid-48-zh-1.wav" \
        "有困难，请拨打110 或者18601200909"

      build/bin/sherpa-onnx-offline-tts \
        --debug=0 \
        --kokoro-model=./kokoro-multi-lang-v1_0/model.onnx \
        --kokoro-voices=./kokoro-multi-lang-v1_0/voices.bin \
        --kokoro-tokens=./kokoro-multi-lang-v1_0/tokens.txt \
        --kokoro-data-dir=./kokoro-multi-lang-v1_0/espeak-ng-data \
        --kokoro-dict-dir=./kokoro-multi-lang-v1_0/dict \
        --kokoro-lexicon=./kokoro-multi-lang-v1_0/lexicon-us-en.txt,./kokoro-multi-lang-v1_0/lexicon-zh.txt \
        --num-threads=2 \
        --sid=48 \
        --output-filename="./kokoro-1.0-sid-48-zh-2.wav" \
        "当夜幕降临，星光点点，伴随着微风拂面，我在静谧中感受着时光的流转，思念如涟漪荡漾，梦境如画卷展开，我与自然融为一体，沉静在这片宁静的美丽之中，感受着生命的奇迹与温柔."


      build/bin/sherpa-onnx-offline-tts \
        --debug=0 \
        --kokoro-model=./kokoro-multi-lang-v1_0/model.onnx \
        --kokoro-voices=./kokoro-multi-lang-v1_0/voices.bin \
        --kokoro-tokens=./kokoro-multi-lang-v1_0/tokens.txt \
        --kokoro-data-dir=./kokoro-multi-lang-v1_0/espeak-ng-data \
        --kokoro-dict-dir=./kokoro-multi-lang-v1_0/dict \
        --kokoro-lexicon=./kokoro-multi-lang-v1_0/lexicon-us-en.txt,./kokoro-multi-lang-v1_0/lexicon-zh.txt \
        --tts-rule-fsts=./kokoro-multi-lang-v1_0/date-zh.fst,./kokoro-multi-lang-v1_0/number-zh.fst \
        --num-threads=2 \
        --sid=48 \
        --output-filename="./kokoro-1.0-sid-48-zh.wav" \
        "现在是2025年12点55分, 星期5。明天是周6，不用上班, 太棒啦！"

      build/bin/sherpa-onnx-offline-tts \
        --debug=0 \
        --kokoro-model=./kokoro-multi-lang-v1_0/model.onnx \
        --kokoro-voices=./kokoro-multi-lang-v1_0/voices.bin \
        --kokoro-tokens=./kokoro-multi-lang-v1_0/tokens.txt \
        --kokoro-data-dir=./kokoro-multi-lang-v1_0/espeak-ng-data \
        --kokoro-dict-dir=./kokoro-multi-lang-v1_0/dict \
        --kokoro-lexicon=./kokoro-multi-lang-v1_0/lexicon-us-en.txt,./kokoro-multi-lang-v1_0/lexicon-zh.txt \
        --tts-rule-fsts=./kokoro-multi-lang-v1_0/date-zh.fst,./kokoro-multi-lang-v1_0/phone-zh.fst,./kokoro-multi-lang-v1_0/number-zh.fst \
        --num-threads=2 \
        --sid=49 \
        --output-filename="./kokoro-1.0-sid-49-zh.wav" \
        "根据第7次全国人口普查结果表明，我国总人口有1443497378人。普查登记的大陆31个省、自治区、直辖市和现役军人的人口共1411778724人。电话号码是110。手机号是13812345678"

      build/bin/sherpa-onnx-offline-tts \
        --debug=0 \
        --kokoro-model=./kokoro-multi-lang-v1_0/model.onnx \
        --kokoro-voices=./kokoro-multi-lang-v1_0/voices.bin \
        --kokoro-tokens=./kokoro-multi-lang-v1_0/tokens.txt \
        --kokoro-data-dir=./kokoro-multi-lang-v1_0/espeak-ng-data \
        --kokoro-dict-dir=./kokoro-multi-lang-v1_0/dict \
        --kokoro-lexicon=./kokoro-multi-lang-v1_0/lexicon-us-en.txt,./kokoro-multi-lang-v1_0/lexicon-zh.txt \
        --num-threads=2 \
        --sid=49 \
        --output-filename="./kokoro-1.0-sid-49-zh-1.wav" \
        "当夜幕降临，星光点点，伴随着微风拂面，我在静谧中感受着时光的流转，思念如涟漪荡漾，梦境如画卷展开，我与自然融为一体，沉静在这片宁静的美丽之中，感受着生命的奇迹与温柔."


      build/bin/sherpa-onnx-offline-tts \
        --debug=0 \
        --kokoro-model=./kokoro-multi-lang-v1_0/model.onnx \
        --kokoro-voices=./kokoro-multi-lang-v1_0/voices.bin \
        --kokoro-tokens=./kokoro-multi-lang-v1_0/tokens.txt \
        --kokoro-data-dir=./kokoro-multi-lang-v1_0/espeak-ng-data \
        --kokoro-dict-dir=./kokoro-multi-lang-v1_0/dict \
        --kokoro-lexicon=./kokoro-multi-lang-v1_0/lexicon-us-en.txt,./kokoro-multi-lang-v1_0/lexicon-zh.txt \
        --num-threads=2 \
        --sid=50 \
        --output-filename="./kokoro-1.0-sid-50-zh.wav" \
        "林美丽最美丽、最漂亮、最可爱！"

      build/bin/sherpa-onnx-offline-tts \
        --debug=0 \
        --kokoro-model=./kokoro-multi-lang-v1_0/model.onnx \
        --kokoro-voices=./kokoro-multi-lang-v1_0/voices.bin \
        --kokoro-tokens=./kokoro-multi-lang-v1_0/tokens.txt \
        --kokoro-data-dir=./kokoro-multi-lang-v1_0/espeak-ng-data \
        --kokoro-dict-dir=./kokoro-multi-lang-v1_0/dict \
        --kokoro-lexicon=./kokoro-multi-lang-v1_0/lexicon-us-en.txt,./kokoro-multi-lang-v1_0/lexicon-zh.txt \
        --num-threads=2 \
        --sid=50 \
        --output-filename="./kokoro-1.0-sid-50-zh-1.wav" \
        "当夜幕降临，星光点点，伴随着微风拂面，我在静谧中感受着时光的流转，思念如涟漪荡漾，梦境如画卷展开，我与自然融为一体，沉静在这片宁静的美丽之中，感受着生命的奇迹与温柔."

      build/bin/sherpa-onnx-offline-tts \
        --debug=0 \
        --kokoro-model=./kokoro-multi-lang-v1_0/model.onnx \
        --kokoro-voices=./kokoro-multi-lang-v1_0/voices.bin \
        --kokoro-tokens=./kokoro-multi-lang-v1_0/tokens.txt \
        --kokoro-data-dir=./kokoro-multi-lang-v1_0/espeak-ng-data \
        --kokoro-dict-dir=./kokoro-multi-lang-v1_0/dict \
        --kokoro-lexicon=./kokoro-multi-lang-v1_0/lexicon-us-en.txt,./kokoro-multi-lang-v1_0/lexicon-zh.txt \
        --num-threads=2 \
        --sid=51 \
        --output-filename="./kokoro-1.0-sid-51-zh.wav" \
        "当夜幕降临，星光点点，伴随着微风拂面，我在静谧中感受着时光的流转，思念如涟漪荡漾，梦境如画卷展开，我与自然融为一体，沉静在这片宁静的美丽之中，感受着生命的奇迹与温柔."

      build/bin/sherpa-onnx-offline-tts \
        --debug=0 \
        --kokoro-model=./kokoro-multi-lang-v1_0/model.onnx \
        --kokoro-voices=./kokoro-multi-lang-v1_0/voices.bin \
        --kokoro-tokens=./kokoro-multi-lang-v1_0/tokens.txt \
        --kokoro-data-dir=./kokoro-multi-lang-v1_0/espeak-ng-data \
        --kokoro-dict-dir=./kokoro-multi-lang-v1_0/dict \
        --kokoro-lexicon=./kokoro-multi-lang-v1_0/lexicon-us-en.txt,./kokoro-multi-lang-v1_0/lexicon-zh.txt \
        --num-threads=2 \
        --sid=52 \
        --output-filename="./kokoro-1.0-sid-52-zh.wav" \
        "当夜幕降临，星光点点，伴随着微风拂面，我在静谧中感受着时光的流转，思念如涟漪荡漾，梦境如画卷展开，我与自然融为一体，沉静在这片宁静的美丽之中，感受着生命的奇迹与温柔."

      build/bin/sherpa-onnx-offline-tts \
        --debug=0 \
        --kokoro-model=./kokoro-multi-lang-v1_0/model.onnx \
        --kokoro-voices=./kokoro-multi-lang-v1_0/voices.bin \
        --kokoro-tokens=./kokoro-multi-lang-v1_0/tokens.txt \
        --kokoro-data-dir=./kokoro-multi-lang-v1_0/espeak-ng-data \
        --kokoro-dict-dir=./kokoro-multi-lang-v1_0/dict \
        --kokoro-lexicon=./kokoro-multi-lang-v1_0/lexicon-us-en.txt,./kokoro-multi-lang-v1_0/lexicon-zh.txt \
        --tts-rule-fsts=./kokoro-multi-lang-v1_0/date-zh.fst,./kokoro-multi-lang-v1_0/number-zh.fst \
        --num-threads=2 \
        --sid=52 \
        --output-filename="./kokoro-1.0-sid-52-zh-en.wav" \
        "Are you ok 是雷军2015年4月小米在印度举行新品发布会时说的。他还说过, I am very happy to be in China. 雷军事后在微博上表示 “万万没想到，视频火速传到国内，全国人民都笑了”. 现在国际米粉越来越多，我的确应该把英文学好，不让大家失望！加油！"

      build/bin/sherpa-onnx-offline-tts \
        --debug=0 \
        --kokoro-model=./kokoro-multi-lang-v1_0/model.onnx \
        --kokoro-voices=./kokoro-multi-lang-v1_0/voices.bin \
        --kokoro-tokens=./kokoro-multi-lang-v1_0/tokens.txt \
        --kokoro-data-dir=./kokoro-multi-lang-v1_0/espeak-ng-data \
        --kokoro-dict-dir=./kokoro-multi-lang-v1_0/dict \
        --kokoro-lexicon=./kokoro-multi-lang-v1_0/lexicon-us-en.txt,./kokoro-multi-lang-v1_0/lexicon-zh.txt \
        --tts-rule-fsts=./kokoro-multi-lang-v1_0/date-zh.fst,./kokoro-multi-lang-v1_0/number-zh.fst \
        --num-threads=2 \
        --sid=1 \
        --output-filename="./kokoro-1.0-sid-1-zh-en.wav" \
        "Are you ok 是雷军2015年4月小米在印度举行新品发布会时说的。他还说过, I am very happy to be in China. 雷军事后在微博上表示 “万万没想到，视频火速传到国内，全国人民都笑了”. 现在国际米粉越来越多，我的确应该把英文学好，不让大家失望！加油！"

      build/bin/sherpa-onnx-offline-tts \
        --debug=0 \
        --kokoro-model=./kokoro-multi-lang-v1_0/model.onnx \
        --kokoro-voices=./kokoro-multi-lang-v1_0/voices.bin \
        --kokoro-tokens=./kokoro-multi-lang-v1_0/tokens.txt \
        --kokoro-data-dir=./kokoro-multi-lang-v1_0/espeak-ng-data \
        --kokoro-dict-dir=./kokoro-multi-lang-v1_0/dict \
        --kokoro-lexicon=./kokoro-multi-lang-v1_0/lexicon-us-en.txt,./kokoro-multi-lang-v1_0/lexicon-zh.txt \
        --tts-rule-fsts=./kokoro-multi-lang-v1_0/date-zh.fst,./kokoro-multi-lang-v1_0/number-zh.fst \
        --num-threads=2 \
        --sid=18 \
        --output-filename="./kokoro-1.0-sid-18-zh-en.wav" \
        "Are you ok 是雷军2015年4月小米在印度举行新品发布会时说的。他还说过, I am very happy to be in China. 雷军事后在微博上表示 “万万没想到，视频火速传到国内，全国人民都笑了”. 现在国际米粉越来越多，我的确应该把英文学好，不让大家失望！加油！"

After running, it will generate many ``.wav`` files in the
current directory.

Audio samples
:::::::::::::

An example is given below:

.. container:: toggle

    .. container:: header

      Click ▶ to see it.

    .. code-block::

      soxi ./kokoro-1.0-sid-1-zh-en.wav

      Input File     : './kokoro-1.0-sid-1-zh-en.wav'
      Channels       : 1
      Sample Rate    : 24000
      Precision      : 16-bit
      Duration       : 00:00:26.00 = 624008 samples ~ 1950.02 CDDA sectors
      File Size      : 1.25M
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
          <td>kokoro-1.0-sid-0-en-us.wav</td>
          <td>
           <audio title="Generated ./kokoro-1.0-sid-0-en-us.wav" controls="controls">
                 <source src="/sherpa/_static/kokoro-multi-lang-v1_0/kokoro-1.0-sid-0-en-us.wav" type="audio/wav">
                 Your browser does not support the <code>audio</code> element.
           </audio>
          </td>
          <td>
            "Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone."
          </td>
        </tr>

        <tr>
          <td>kokoro-1.0-sid-1-en-us.wav</td>
          <td>
           <audio title="Generated ./kokoro-1.0-sid-1-en-us.wav" controls="controls">
                 <source src="/sherpa/_static/kokoro-multi-lang-v1_0/kokoro-1.0-sid-1-en-us.wav" type="audio/wav">
                 Your browser does not support the <code>audio</code> element.
           </audio>
          </td>
          <td>
            "Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone."
          </td>
        </tr>

        <tr>
          <td>kokoro-1.0-sid-2-en-us.wav</td>
          <td>
           <audio title="Generated ./kokoro-1.0-sid-2-en-us.wav" controls="controls">
                 <source src="/sherpa/_static/kokoro-multi-lang-v1_0/kokoro-1.0-sid-2-en-us.wav" type="audio/wav">
                 Your browser does not support the <code>audio</code> element.
           </audio>
          </td>
          <td>
            "Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone."
          </td>
        </tr>

        <tr>
          <td>kokoro-1.0-sid-3-en-us.wav</td>
          <td>
           <audio title="Generated ./kokoro-1.0-sid-3-en-us.wav" controls="controls">
                 <source src="/sherpa/_static/kokoro-multi-lang-v1_0/kokoro-1.0-sid-3-en-us.wav" type="audio/wav">
                 Your browser does not support the <code>audio</code> element.
           </audio>
          </td>
          <td>
            "Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone."
          </td>
        </tr>

        <tr>
          <td>kokoro-1.0-sid-4-en-us.wav</td>
          <td>
           <audio title="Generated ./kokoro-1.0-sid-4-en-us.wav" controls="controls">
                 <source src="/sherpa/_static/kokoro-multi-lang-v1_0/kokoro-1.0-sid-4-en-us.wav" type="audio/wav">
                 Your browser does not support the <code>audio</code> element.
           </audio>
          </td>
          <td>
            "Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone."
          </td>
        </tr>

        <tr>
          <td>kokoro-1.0-sid-5-en-us.wav</td>
          <td>
           <audio title="Generated ./kokoro-1.0-sid-5-en-us.wav" controls="controls">
                 <source src="/sherpa/_static/kokoro-multi-lang-v1_0/kokoro-1.0-sid-5-en-us.wav" type="audio/wav">
                 Your browser does not support the <code>audio</code> element.
           </audio>
          </td>
          <td>
            "Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone."
          </td>
        </tr>

        <tr>
          <td>kokoro-1.0-sid-6-en-us.wav</td>
          <td>
           <audio title="Generated ./kokoro-1.0-sid-6-en-us.wav" controls="controls">
                 <source src="/sherpa/_static/kokoro-multi-lang-v1_0/kokoro-1.0-sid-6-en-us.wav" type="audio/wav">
                 Your browser does not support the <code>audio</code> element.
           </audio>
          </td>
          <td>
            "Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone."
          </td>
        </tr>

        <tr>
          <td>kokoro-1.0-sid-7-en-us.wav</td>
          <td>
           <audio title="Generated ./kokoro-1.0-sid-7-en-us.wav" controls="controls">
                 <source src="/sherpa/_static/kokoro-multi-lang-v1_0/kokoro-1.0-sid-7-en-us.wav" type="audio/wav">
                 Your browser does not support the <code>audio</code> element.
           </audio>
          </td>
          <td>
            "Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone."
          </td>
        </tr>

        <tr>
          <td>kokoro-1.0-sid-8-en-us.wav</td>
          <td>
           <audio title="Generated ./kokoro-1.0-sid-8-en-us.wav" controls="controls">
                 <source src="/sherpa/_static/kokoro-multi-lang-v1_0/kokoro-1.0-sid-8-en-us.wav" type="audio/wav">
                 Your browser does not support the <code>audio</code> element.
           </audio>
          </td>
          <td>
            "Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone."
          </td>
        </tr>

        <tr>
          <td>kokoro-1.0-sid-9-en-us.wav</td>
          <td>
           <audio title="Generated ./kokoro-1.0-sid-9-en-us.wav" controls="controls">
                 <source src="/sherpa/_static/kokoro-multi-lang-v1_0/kokoro-1.0-sid-9-en-us.wav" type="audio/wav">
                 Your browser does not support the <code>audio</code> element.
           </audio>
          </td>
          <td>
            "Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone."
          </td>
        </tr>

        <tr>
          <td>kokoro-1.0-sid-10-en-us.wav</td>
          <td>
           <audio title="Generated ./kokoro-1.0-sid-10-en-us.wav" controls="controls">
                 <source src="/sherpa/_static/kokoro-multi-lang-v1_0/kokoro-1.0-sid-10-en-us.wav" type="audio/wav">
                 Your browser does not support the <code>audio</code> element.
           </audio>
          </td>
          <td>
            "Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone."
          </td>
        </tr>

        <tr>
          <td>kokoro-1.0-sid-11-en-us.wav</td>
          <td>
           <audio title="Generated ./kokoro-1.0-sid-11-en-us.wav" controls="controls">
                 <source src="/sherpa/_static/kokoro-multi-lang-v1_0/kokoro-1.0-sid-11-en-us.wav" type="audio/wav">
                 Your browser does not support the <code>audio</code> element.
           </audio>
          </td>
          <td>
            "Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone."
          </td>
        </tr>

        <tr>
          <td>kokoro-1.0-sid-12-en-us.wav</td>
          <td>
           <audio title="Generated ./kokoro-1.0-sid-12-en-us.wav" controls="controls">
                 <source src="/sherpa/_static/kokoro-multi-lang-v1_0/kokoro-1.0-sid-12-en-us.wav" type="audio/wav">
                 Your browser does not support the <code>audio</code> element.
           </audio>
          </td>
          <td>
            "Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone."
          </td>
        </tr>

        <tr>
          <td>kokoro-1.0-sid-13-en-us.wav</td>
          <td>
           <audio title="Generated ./kokoro-1.0-sid-13-en-us.wav" controls="controls">
                 <source src="/sherpa/_static/kokoro-multi-lang-v1_0/kokoro-1.0-sid-13-en-us.wav" type="audio/wav">
                 Your browser does not support the <code>audio</code> element.
           </audio>
          </td>
          <td>
            "Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone."
          </td>
        </tr>

        <tr>
          <td>kokoro-1.0-sid-14-en-us.wav</td>
          <td>
           <audio title="Generated ./kokoro-1.0-sid-14-en-us.wav" controls="controls">
                 <source src="/sherpa/_static/kokoro-multi-lang-v1_0/kokoro-1.0-sid-14-en-us.wav" type="audio/wav">
                 Your browser does not support the <code>audio</code> element.
           </audio>
          </td>
          <td>
            "Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone."
          </td>
        </tr>

        <tr>
          <td>kokoro-1.0-sid-15-en-us.wav</td>
          <td>
           <audio title="Generated ./kokoro-1.0-sid-15-en-us.wav" controls="controls">
                 <source src="/sherpa/_static/kokoro-multi-lang-v1_0/kokoro-1.0-sid-15-en-us.wav" type="audio/wav">
                 Your browser does not support the <code>audio</code> element.
           </audio>
          </td>
          <td>
            "Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone."
          </td>
        </tr>

        <tr>
          <td>kokoro-1.0-sid-16-en-us.wav</td>
          <td>
           <audio title="Generated ./kokoro-1.0-sid-16-en-us.wav" controls="controls">
                 <source src="/sherpa/_static/kokoro-multi-lang-v1_0/kokoro-1.0-sid-16-en-us.wav" type="audio/wav">
                 Your browser does not support the <code>audio</code> element.
           </audio>
          </td>
          <td>
            "Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone."
          </td>
        </tr>

        <tr>
          <td>kokoro-1.0-sid-17-en-us.wav</td>
          <td>
           <audio title="Generated ./kokoro-1.0-sid-17-en-us.wav" controls="controls">
                 <source src="/sherpa/_static/kokoro-multi-lang-v1_0/kokoro-1.0-sid-17-en-us.wav" type="audio/wav">
                 Your browser does not support the <code>audio</code> element.
           </audio>
          </td>
          <td>
            "Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone."
          </td>
        </tr>

        <tr>
          <td>kokoro-1.0-sid-18-en-us.wav</td>
          <td>
           <audio title="Generated ./kokoro-1.0-sid-18-en-us.wav" controls="controls">
                 <source src="/sherpa/_static/kokoro-multi-lang-v1_0/kokoro-1.0-sid-18-en-us.wav" type="audio/wav">
                 Your browser does not support the <code>audio</code> element.
           </audio>
          </td>
          <td>
            "Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone."
          </td>
        </tr>

        <tr>
          <td>kokoro-1.0-sid-19-en-us.wav</td>
          <td>
           <audio title="Generated ./kokoro-1.0-sid-19-en-us.wav" controls="controls">
                 <source src="/sherpa/_static/kokoro-multi-lang-v1_0/kokoro-1.0-sid-19-en-us.wav" type="audio/wav">
                 Your browser does not support the <code>audio</code> element.
           </audio>
          </td>
          <td>
            "Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone."
          </td>
        </tr>

        <tr>
          <td>kokoro-1.0-sid-20-en-gb.wav</td>
          <td>
           <audio title="Generated ./kokoro-1.0-sid-20-en-gb.wav" controls="controls">
                 <source src="/sherpa/_static/kokoro-multi-lang-v1_0/kokoro-1.0-sid-20-en-gb.wav" type="audio/wav">
                 Your browser does not support the <code>audio</code> element.
           </audio>
          </td>
          <td>
            "Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone."
          </td>
        </tr>

        <tr>
          <td>kokoro-1.0-sid-21-en-gb.wav</td>
          <td>
           <audio title="Generated ./kokoro-1.0-sid-21-en-gb.wav" controls="controls">
                 <source src="/sherpa/_static/kokoro-multi-lang-v1_0/kokoro-1.0-sid-21-en-gb.wav" type="audio/wav">
                 Your browser does not support the <code>audio</code> element.
           </audio>
          </td>
          <td>
            "Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone."
          </td>
        </tr>

        <tr>
          <td>kokoro-1.0-sid-22-en-gb.wav</td>
          <td>
           <audio title="Generated ./kokoro-1.0-sid-22-en-gb.wav" controls="controls">
                 <source src="/sherpa/_static/kokoro-multi-lang-v1_0/kokoro-1.0-sid-22-en-gb.wav" type="audio/wav">
                 Your browser does not support the <code>audio</code> element.
           </audio>
          </td>
          <td>
            "Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone."
          </td>
        </tr>

        <tr>
          <td>kokoro-1.0-sid-23-en-gb.wav</td>
          <td>
           <audio title="Generated ./kokoro-1.0-sid-23-en-gb.wav" controls="controls">
                 <source src="/sherpa/_static/kokoro-multi-lang-v1_0/kokoro-1.0-sid-23-en-gb.wav" type="audio/wav">
                 Your browser does not support the <code>audio</code> element.
           </audio>
          </td>
          <td>
            "Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone."
          </td>
        </tr>

        <tr>
          <td>kokoro-1.0-sid-24-en-gb.wav</td>
          <td>
           <audio title="Generated ./kokoro-1.0-sid-24-en-gb.wav" controls="controls">
                 <source src="/sherpa/_static/kokoro-multi-lang-v1_0/kokoro-1.0-sid-24-en-gb.wav" type="audio/wav">
                 Your browser does not support the <code>audio</code> element.
           </audio>
          </td>
          <td>
            "Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone."
          </td>
        </tr>

        <tr>
          <td>kokoro-1.0-sid-25-en-gb.wav</td>
          <td>
           <audio title="Generated ./kokoro-1.0-sid-25-en-gb.wav" controls="controls">
                 <source src="/sherpa/_static/kokoro-multi-lang-v1_0/kokoro-1.0-sid-25-en-gb.wav" type="audio/wav">
                 Your browser does not support the <code>audio</code> element.
           </audio>
          </td>
          <td>
            "Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone."
          </td>
        </tr>

        <tr>
          <td>kokoro-1.0-sid-23-en-gb.wav</td>
          <td>
           <audio title="Generated ./kokoro-1.0-sid-23-en-gb.wav" controls="controls">
                 <source src="/sherpa/_static/kokoro-multi-lang-v1_0/kokoro-1.0-sid-23-en-gb.wav" type="audio/wav">
                 Your browser does not support the <code>audio</code> element.
           </audio>
          </td>
          <td>
            "Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone."
          </td>
        </tr>

        <tr>
          <td>kokoro-1.0-sid-24-en-gb.wav</td>
          <td>
           <audio title="Generated ./kokoro-1.0-sid-24-en-gb.wav" controls="controls">
                 <source src="/sherpa/_static/kokoro-multi-lang-v1_0/kokoro-1.0-sid-24-en-gb.wav" type="audio/wav">
                 Your browser does not support the <code>audio</code> element.
           </audio>
          </td>
          <td>
            "Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone."
          </td>
        </tr>

        <tr>
          <td>kokoro-1.0-sid-25-en-gb.wav</td>
          <td>
           <audio title="Generated ./kokoro-1.0-sid-25-en-gb.wav" controls="controls">
                 <source src="/sherpa/_static/kokoro-multi-lang-v1_0/kokoro-1.0-sid-25-en-gb.wav" type="audio/wav">
                 Your browser does not support the <code>audio</code> element.
           </audio>
          </td>
          <td>
            "Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone."
          </td>
        </tr>

        <tr>
          <td>kokoro-1.0-sid-26-en-gb.wav</td>
          <td>
           <audio title="Generated ./kokoro-1.0-sid-26-en-gb.wav" controls="controls">
                 <source src="/sherpa/_static/kokoro-multi-lang-v1_0/kokoro-1.0-sid-26-en-gb.wav" type="audio/wav">
                 Your browser does not support the <code>audio</code> element.
           </audio>
          </td>
          <td>
            "Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone."
          </td>
        </tr>

        <tr>
          <td>kokoro-1.0-sid-27-en-gb.wav</td>
          <td>
           <audio title="Generated ./kokoro-1.0-sid-27-en-gb.wav" controls="controls">
                 <source src="/sherpa/_static/kokoro-multi-lang-v1_0/kokoro-1.0-sid-27-en-gb.wav" type="audio/wav">
                 Your browser does not support the <code>audio</code> element.
           </audio>
          </td>
          <td>
            "Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone."
          </td>
        </tr>

        <tr>
          <td>kokoro-1.0-sid-45-zh.wav</td>
          <td>
           <audio title="Generated ./kokoro-1.0-sid-45-zh.wav" controls="controls">
                 <source src="/sherpa/_static/kokoro-multi-lang-v1_0/kokoro-1.0-sid-45-zh.wav" type="audio/wav">
                 Your browser does not support the <code>audio</code> element.
           </audio>
          </td>
          <td>
            "小米的核心价值观是什么？答案是真诚热爱！"
          </td>
        </tr>

        <tr>
          <td>kokoro-1.0-sid-45-zh-1.wav</td>
          <td>
           <audio title="Generated ./kokoro-1.0-sid-45-zh-1.wav" controls="controls">
                 <source src="/sherpa/_static/kokoro-multi-lang-v1_0/kokoro-1.0-sid-45-zh-1.wav" type="audio/wav">
                 Your browser does not support the <code>audio</code> element.
           </audio>
          </td>
          <td>
            "当夜幕降临，星光点点，伴随着微风拂面，我在静谧中感受着时光的流转，思念如涟漪荡漾，梦境如画卷展开，我与自然融为一体，沉静在这片宁静的美丽之中，感受着生命的奇迹与温柔."
          </td>
        </tr>

        <tr>
          <td>kokoro-1.0-sid-46-zh.wav</td>
          <td>
           <audio title="Generated ./kokoro-1.0-sid-46-zh.wav" controls="controls">
                 <source src="/sherpa/_static/kokoro-multi-lang-v1_0/kokoro-1.0-sid-46-zh.wav" type="audio/wav">
                 Your browser does not support the <code>audio</code> element.
           </audio>
          </td>
          <td>
             "小米的使命是，始终坚持做感动人心、价格厚道的好产品，让全球每个人都能享受科技带来的美好生活。"
          </td>
        </tr>

        <tr>
          <td>kokoro-1.0-sid-46-zh-1.wav</td>
          <td>
           <audio title="Generated ./kokoro-1.0-sid-46-zh-1.wav" controls="controls">
                 <source src="/sherpa/_static/kokoro-multi-lang-v1_0/kokoro-1.0-sid-46-zh-1.wav" type="audio/wav">
                 Your browser does not support the <code>audio</code> element.
           </audio>
          </td>
          <td>
            "当夜幕降临，星光点点，伴随着微风拂面，我在静谧中感受着时光的流转，思念如涟漪荡漾，梦境如画卷展开，我与自然融为一体，沉静在这片宁静的美丽之中，感受着生命的奇迹与温柔."
          </td>
        </tr>

        <tr>
          <td>kokoro-1.0-sid-47-zh.wav</td>
          <td>
           <audio title="Generated ./kokoro-1.0-sid-47-zh.wav" controls="controls">
                 <source src="/sherpa/_static/kokoro-multi-lang-v1_0/kokoro-1.0-sid-47-zh.wav" type="audio/wav">
                 Your browser does not support the <code>audio</code> element.
           </audio>
          </td>
          <td>
            "35年前，他于长沙出生, 在长白山长大。9年前他当上了银行的领导，主管行政。"
          </td>
        </tr>

        <tr>
          <td>kokoro-1.0-sid-47-zh-1.wav</td>
          <td>
           <audio title="Generated ./kokoro-1.0-sid-47-zh-1.wav" controls="controls">
                 <source src="/sherpa/_static/kokoro-multi-lang-v1_0/kokoro-1.0-sid-47-zh-1.wav" type="audio/wav">
                 Your browser does not support the <code>audio</code> element.
           </audio>
          </td>
          <td>
            "当夜幕降临，星光点点，伴随着微风拂面，我在静谧中感受着时光的流转，思念如涟漪荡漾，梦境如画卷展开，我与自然融为一体，沉静在这片宁静的美丽之中，感受着生命的奇迹与温柔."
          </td>
        </tr>

        <tr>
          <td>kokoro-1.0-sid-48-zh-1.wav</td>
          <td>
           <audio title="Generated ./kokoro-1.0-sid-48-zh-1.wav" controls="controls">
                 <source src="/sherpa/_static/kokoro-multi-lang-v1_0/kokoro-1.0-sid-48-zh-1.wav" type="audio/wav">
                 Your browser does not support the <code>audio</code> element.
           </audio>
          </td>
          <td>
            "有困难，请拨打110 或者18601200909"
          </td>
        </tr>

        <tr>
          <td>kokoro-1.0-sid-48-zh-2.wav</td>
          <td>
           <audio title="Generated ./kokoro-1.0-sid-48-zh-2.wav" controls="controls">
                 <source src="/sherpa/_static/kokoro-multi-lang-v1_0/kokoro-1.0-sid-48-zh-2.wav" type="audio/wav">
                 Your browser does not support the <code>audio</code> element.
           </audio>
          </td>
          <td>
            "当夜幕降临，星光点点，伴随着微风拂面，我在静谧中感受着时光的流转，思念如涟漪荡漾，梦境如画卷展开，我与自然融为一体，沉静在这片宁静的美丽之中，感受着生命的奇迹与温柔."
          </td>
        </tr>

        <tr>
          <td>kokoro-1.0-sid-48-zh.wav</td>
          <td>
           <audio title="Generated ./kokoro-1.0-sid-48-zh.wav" controls="controls">
                 <source src="/sherpa/_static/kokoro-multi-lang-v1_0/kokoro-1.0-sid-48-zh.wav" type="audio/wav">
                 Your browser does not support the <code>audio</code> element.
           </audio>
          </td>
          <td>
            "现在是2025年12点55分, 星期5。明天是周6，不用上班, 太棒啦！"
          </td>
        </tr>

        <tr>
          <td>kokoro-1.0-sid-49-zh.wav</td>
          <td>
           <audio title="Generated ./kokoro-1.0-sid-49-zh.wav" controls="controls">
                 <source src="/sherpa/_static/kokoro-multi-lang-v1_0/kokoro-1.0-sid-49-zh.wav" type="audio/wav">
                 Your browser does not support the <code>audio</code> element.
           </audio>
          </td>
          <td>
            "根据第7次全国人口普查结果表明，我国总人口有1443497378人。普查登记的大陆31个省、自治区、直辖市和现役军人的人口共1411778724人。电话号码是110。手机号是13812345678"

          </td>
        </tr>

        <tr>
          <td>kokoro-1.0-sid-49-zh-1.wav</td>
          <td>
           <audio title="Generated ./kokoro-1.0-sid-49-zh-1.wav" controls="controls">
                 <source src="/sherpa/_static/kokoro-multi-lang-v1_0/kokoro-1.0-sid-49-zh-1.wav" type="audio/wav">
                 Your browser does not support the <code>audio</code> element.
           </audio>
          </td>
          <td>
            "当夜幕降临，星光点点，伴随着微风拂面，我在静谧中感受着时光的流转，思念如涟漪荡漾，梦境如画卷展开，我与自然融为一体，沉静在这片宁静的美丽之中，感受着生命的奇迹与温柔."
          </td>
        </tr>

        <tr>
          <td>kokoro-1.0-sid-50-zh.wav</td>
          <td>
           <audio title="Generated ./kokoro-1.0-sid-50-zh.wav" controls="controls">
                 <source src="/sherpa/_static/kokoro-multi-lang-v1_0/kokoro-1.0-sid-50-zh.wav" type="audio/wav">
                 Your browser does not support the <code>audio</code> element.
           </audio>
          </td>
          <td>
            "林美丽最美丽、最漂亮、最可爱！"
          </td>
        </tr>

        <tr>
          <td>kokoro-1.0-sid-50-zh-1.wav</td>
          <td>
           <audio title="Generated ./kokoro-1.0-sid-50-zh-1.wav" controls="controls">
                 <source src="/sherpa/_static/kokoro-multi-lang-v1_0/kokoro-1.0-sid-50-zh-1.wav" type="audio/wav">
                 Your browser does not support the <code>audio</code> element.
           </audio>
          </td>
          <td>
            "当夜幕降临，星光点点，伴随着微风拂面，我在静谧中感受着时光的流转，思念如涟漪荡漾，梦境如画卷展开，我与自然融为一体，沉静在这片宁静的美丽之中，感受着生命的奇迹与温柔."
          </td>
        </tr>

        <tr>
          <td>kokoro-1.0-sid-51-zh.wav</td>
          <td>
           <audio title="Generated ./kokoro-1.0-sid-51-zh.wav" controls="controls">
                 <source src="/sherpa/_static/kokoro-multi-lang-v1_0/kokoro-1.0-sid-51-zh.wav" type="audio/wav">
                 Your browser does not support the <code>audio</code> element.
           </audio>
          </td>
          <td>
            "当夜幕降临，星光点点，伴随着微风拂面，我在静谧中感受着时光的流转，思念如涟漪荡漾，梦境如画卷展开，我与自然融为一体，沉静在这片宁静的美丽之中，感受着生命的奇迹与温柔."
          </td>
        </tr>

        <tr>
          <td>kokoro-1.0-sid-52-zh.wav</td>
          <td>
           <audio title="Generated ./kokoro-1.0-sid-52-zh.wav" controls="controls">
                 <source src="/sherpa/_static/kokoro-multi-lang-v1_0/kokoro-1.0-sid-52-zh.wav" type="audio/wav">
                 Your browser does not support the <code>audio</code> element.
           </audio>
          </td>
          <td>
            "当夜幕降临，星光点点，伴随着微风拂面，我在静谧中感受着时光的流转，思念如涟漪荡漾，梦境如画卷展开，我与自然融为一体，沉静在这片宁静的美丽之中，感受着生命的奇迹与温柔."
          </td>
        </tr>

        <tr>
          <td>kokoro-1.0-sid-52-zh-en.wav</td>
          <td>
           <audio title="Generated ./kokoro-1.0-sid-52-zh-en.wav" controls="controls">
                 <source src="/sherpa/_static/kokoro-multi-lang-v1_0/kokoro-1.0-sid-52-zh-en.wav" type="audio/wav">
                 Your browser does not support the <code>audio</code> element.
           </audio>
          </td>
          <td>
           "Are you ok 是雷军2015年4月小米在印度举行新品发布会时说的。他还说过, I am very happy to be in China. 雷军事后在微博上表示 “万万没想到，视频火速传到国内，全国人民都笑了”. 现在国际米粉越来越多，我的确应该把英文学好，不让大家失望！加油！"
          </td>
        </tr>

        <tr>
          <td>kokoro-1.0-sid-1-zh-en.wav</td>
          <td>
           <audio title="Generated ./kokoro-1.0-sid-1-zh-en.wav" controls="controls">
                 <source src="/sherpa/_static/kokoro-multi-lang-v1_0/kokoro-1.0-sid-1-zh-en.wav" type="audio/wav">
                 Your browser does not support the <code>audio</code> element.
           </audio>
          </td>
          <td>
           "Are you ok 是雷军2015年4月小米在印度举行新品发布会时说的。他还说过, I am very happy to be in China. 雷军事后在微博上表示 “万万没想到，视频火速传到国内，全国人民都笑了”. 现在国际米粉越来越多，我的确应该把英文学好，不让大家失望！加油！"

          </td>
        </tr>

        <tr>
          <td>kokoro-1.0-sid-18-zh-en.wav</td>
          <td>
           <audio title="Generated ./kokoro-1.0-sid-18-zh-en.wav" controls="controls">
                 <source src="/sherpa/_static/kokoro-multi-lang-v1_0/kokoro-1.0-sid-18-zh-en.wav" type="audio/wav">
                 Your browser does not support the <code>audio</code> element.
           </audio>
          </td>
          <td>
           "Are you ok 是雷军2015年4月小米在印度举行新品发布会时说的。他还说过, I am very happy to be in China. 雷军事后在微博上表示 “万万没想到，视频火速传到国内，全国人民都笑了”. 现在国际米粉越来越多，我的确应该把英文学好，不让大家失望！加油！"

          </td>
        </tr>

      </table>

Generate speech with Python script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Please replace ``build/bin/sherpa-onnx-offline-tts`` in the above examples
with ``python3 ./python-api-examples/offline-tts.py``.
or with ``python3 ./python-api-examples/offline-tts-play.py``.

.. hint::

   - Download `offline-tts.py <https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/offline-tts.py>`_
   - Download `offline-tts-play.py <https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/offline-tts-play.py>`_

RTF on Raspberry Pi 4 Model B Rev 1.5
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We use the following command to test the RTF of this model on Raspberry Pi 4 Model B Rev 1.5:

.. code-block:: bash


   for t in 1 2 3 4; do
    build/bin/sherpa-onnx-offline-tts \
      --num-threads=$t \
      --kokoro-model=./kokoro-multi-lang-v1_0/model.onnx \
      --kokoro-voices=./kokoro-multi-lang-v1_0/voices.bin \
      --kokoro-tokens=./kokoro-multi-lang-v1_0/tokens.txt \
      --kokoro-data-dir=./kokoro-multi-lang-v1_0/espeak-ng-data \
      --kokoro-dict-dir=./kokoro-multi-lang-v1_0/dict \
      --kokoro-lexicon=./kokoro-multi-lang-v1_0/lexicon-us-en.txt,./kokoro-multi-lang-v1_0/lexicon-zh.txt \
      --tts-rule-fsts=./kokoro-multi-lang-v1_0/date-zh.fst,./kokoro-multi-lang-v1_0/number-zh.fst \
      --sid=1 \
      --output-filename="./kokoro-1.0-sid-1-en.wav" \
      "你好吗？Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone."
   done

The results are given below:

  +-------------+-------+-------+-------+-------+
  | num_threads | 1     | 2     | 3     | 4     |
  +=============+=======+=======+=======+=======+
  | RTF         | 7.635 | 4.470 | 3.430 | 3.191 |
  +-------------+-------+-------+-------+-------+

.. _kokoro-en-v0_19:

kokoro-en-v0_19 (English, 11 speakers)
--------------------------------------

This model contains 11 speakers. The ONNX model is from
`<https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files>`_

The script for adding meta data to the ONNX model can be found at
`<https://github.com/k2-fsa/sherpa-onnx/tree/master/scripts/kokoro>`_

In the following, we describe how to download it and use it with `sherpa-onnx`_.

See also `<https://k2-fsa.github.io/sherpa/onnx/tts/all/English/kokoro-en-v0_19.html>`_

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
