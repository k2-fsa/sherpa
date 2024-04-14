Pre-trained models
==================

This section lists pre-trained models for adding punctuations to text.

You can find all models at the following URL:

  `<https://github.com/k2-fsa/sherpa-onnx/releases/tag/punctuation-models>`_

sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12
-------------------------------------------------------------

This model is converted from

  `<https://modelscope.cn/models/iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/summary>`_

and it supports both Chinese and English.

.. hint::

   If you want to know how the model is converted to `sherpa-onnx`_, please download
   it and you can find related scripts in the downloaded model directory.

In the following, we describe how to download and use it with `sherpa-onnx`_.

Download the model
^^^^^^^^^^^^^^^^^^

Please use the following commands to download it::

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2

  # For Chinese users, you can aso use the following mirror:
  # wget https://hub.nuaa.cf/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2

  tar xvf sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2
  rm sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2

You will find the following files after unzipping::

    -rw-r--r--  1 fangjun  staff   1.4K Apr 12 12:32 README.md
    -rwxr-xr-x  1 fangjun  staff   1.6K Apr 12 14:40 add-model-metadata.py
    -rw-r--r--  1 fangjun  staff   810B Apr 12 11:56 config.yaml
    -rw-r--r--  1 fangjun  staff    42B Apr 12 11:45 configuration.json
    -rw-r--r--  1 fangjun  staff   281M Apr 12 14:40 model.onnx
    -rwxr-xr-x  1 fangjun  staff   745B Apr 12 11:53 show-model-input-output.py
    -rwxr-xr-x  1 fangjun  staff   4.9K Apr 13 18:45 test.py
    -rw-r--r--  1 fangjun  staff   4.0M Apr 12 11:56 tokens.json

Only ``model.onnx`` is needed in `sherpa-onnx`_. All other files are for your information about
how the model is converted to `sherpa-onnx`_.

C++ binary examples
^^^^^^^^^^^^^^^^^^^

After installing `sherpa-onnx`_, you can use the following command to add punctuations to text::

   ./bin/sherpa-onnx-offline-punctuation \
     --ct-transformer=./sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12/model.onnx \
     "我们都是木头人不会说话不会动"

The output is given below::

  /Users/fangjun/open-source/sherpa-onnx/sherpa-onnx/csrc/parse-options.cc:Read:361 ./bin/sherpa-onnx-offline-punctuation --ct-transformer=./sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12/model.onnx '我们都是木头人不会说话不会动'

  OfflinePunctuationConfig(model=OfflinePunctuationModelConfig(ct_transformer="./sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12/model.onnx", num_threads=1, debug=False, provider="cpu"))
  Creating OfflinePunctuation ...
  Started
  Done
  Num threads: 1
  Elapsed seconds: 0.007 s
  Input text: 我们都是木头人不会说话不会动
  Output text: 我们都是木头人，不会说话不会动。

The second example is for text containing both Chinese and English::

  ./bin/sherpa-onnx-offline-punctuation \
    --ct-transformer=./sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12/model.onnx \
    "这是一个测试你好吗How are you我很好thank you are you ok谢谢你"

Its output is given below::

  /Users/fangjun/open-source/sherpa-onnx/sherpa-onnx/csrc/parse-options.cc:Read:361 ./bin/sherpa-onnx-offline-punctuation --ct-transformer=./sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12/model.onnx '这是一个测试你好吗How are you我很好thank you are you ok谢谢你'

  OfflinePunctuationConfig(model=OfflinePunctuationModelConfig(ct_transformer="./sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12/model.onnx", num_threads=1, debug=False, provider="cpu"))
  Creating OfflinePunctuation ...
  Started
  Done
  Num threads: 1
  Elapsed seconds: 0.005 s
  Input text: 这是一个测试你好吗How are you我很好thank you are you ok谢谢你
  Output text: 这是一个测试，你好吗？How are you？我很好？thank you，are you ok，谢谢你。

The last example is for text containing only English::

  ./bin/sherpa-onnx-offline-punctuation \
    --ct-transformer=./sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12/model.onnx \
    "The African blogosphere is rapidly expanding bringing more voices online in the form of commentaries opinions analyses rants and poetry"

Its output is given below::

  /Users/fangjun/open-source/sherpa-onnx/sherpa-onnx/csrc/parse-options.cc:Read:361 ./bin/sherpa-onnx-offline-punctuation --ct-transformer=./sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12/model.onnx 'The African blogosphere is rapidly expanding bringing more voices online in the form of commentaries opinions analyses rants and poetry'

  OfflinePunctuationConfig(model=OfflinePunctuationModelConfig(ct_transformer="./sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12/model.onnx", num_threads=1, debug=False, provider="cpu"))
  Creating OfflinePunctuation ...
  Started
  Done
  Num threads: 1
  Elapsed seconds: 0.003 s
  Input text: The African blogosphere is rapidly expanding bringing more voices online in the form of commentaries opinions analyses rants and poetry
  Output text: The African blogosphere is rapidly expanding，bringing more voices online in the form of commentaries，opinions，analyses，rants and poetry。

Python API examples
^^^^^^^^^^^^^^^^^^^

Please see

  `<https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/add-punctuation.py>`_

Huggingface space examples
^^^^^^^^^^^^^^^^^^^^^^^^^^

Please see

  - `<https://huggingface.co/spaces/k2-fsa/generate-subtitles-for-videos>`_
  - `<https://huggingface.co/spaces/k2-fsa/automatic-speech-recognition>`_

.. hint::

    For Chinese users, please visit the following mirrors:

      - `<https://hf-mirror.com/spaces/k2-fsa/generate-subtitles-for-videos>`_
      - `<https://hf-mirror.com/spaces/k2-fsa/automatic-speech-recognition>`_

Video demos
^^^^^^^^^^^

The following `video <https://www.bilibili.com/video/BV1Tm421j7K3/>`_ is in Chinese.

.. raw:: html

  <iframe src="//player.bilibili.com/player.html?bvid=BV1Tm421j7K3&page=1" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true" width="600" height="600"> </iframe>
