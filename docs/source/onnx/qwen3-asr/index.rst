.. _onnx-qwen3-asr:

Qwen3-ASR
=========

This section describes how to use models from `<https://github.com/QwenLM/Qwen3-ASR>`_.

Qwen3-ASR-0.6B
---------------

A single model from `Qwen3-ASR`_ supports the following languages:

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

We have converted `Qwen3-ASR`_ to onnx and provided APIs for the following programming languages

  - 1. C++
  - 2. C
  - 3. Python
  - 4. C#
  - 5. Go
  - 6. Kotlin
  - 7. Java
  - 8. JavaScript
  - 9. Swift
  - 10. `Dart`_ (Support `Flutter`_)
  - 11. Rust
  - 12. Pascal

You can find the onnx export script at `<https://github.com/Wasser1462/Qwen3-ASR-onnx>`_

Note that you can use `Qwen3-ASR`_ with `sherpa-onnx`_ on the following platforms:

  - Linux (x64, aarch64, arm, riscv64)
  - macOS (x64, arm64)
  - Windows (x64, x86, arm64)
  - Android (arm64-v8a, armv7-eabi, x86, x86_64)
  - iOS (arm64)

In the following, we describe how to download pre-trained `Qwen3-ASR`_ models
and use them in `sherpa-onnx`_.

.. toctree::
   :maxdepth: 6

   ./huggingface-space.rst
   ./export.rst
   ./pretrained.rst
