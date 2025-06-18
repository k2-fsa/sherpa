Pre-trained models
==================

This section lists pre-trained models for adding punctuations to text.

You can find all models at the following URL:

  `<https://github.com/k2-fsa/sherpa-onnx/releases/tag/punctuation-models>`_

sherpa-onnx-online-punct-en-2024-08-06
----------------------------------------

This model is from `<https://github.com/frankyoujian/Edge-Punct-Casing>`_
and it supports only English.

Please see its paper at `<https://arxiv.org/abs/2407.13142>`_ for more details.

In the following, we describe how to download and use it with `sherpa-onnx`_.

Download the model
^^^^^^^^^^^^^^^^^^

Please use the following commands to download it::

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-online-punct-en-2024-08-06.tar.bz2
  tar xvf sherpa-onnx-online-punct-en-2024-08-06.tar.bz2
  rm sherpa-onnx-online-punct-en-2024-08-06.tar.bz2

You will find the following files after unzipping::

  ls -lh sherpa-onnx-online-punct-en-2024-08-06/
  total 74416
  -rw-r--r--  1 fangjun  staff   244B Aug  6  2024 README.md
  -rw-r--r--  1 fangjun  staff   146K Aug  5  2024 bpe.vocab
  -rw-r--r--  1 fangjun  staff   7.1M Aug  5  2024 model.int8.onnx
  -rw-r--r--  1 fangjun  staff    28M Aug  5  2024 model.onnx

Note you only need two files:

  - ``model.onnx`` + ``bpe.vocab``
  - or ``model.int8.onnx`` + ``bpe.vocab``

C++ binary examples
^^^^^^^^^^^^^^^^^^^

After installing `sherpa-onnx`_, you can use the following command to add punctuations to text
with the ``model.onnx``::

  ./bin/sherpa-onnx-online-punctuation \
    --cnn-bilstm=./sherpa-onnx-online-punct-en-2024-08-06/model.onnx \
    --bpe-vocab=./sherpa-onnx-online-punct-en-2024-08-06/bpe.vocab \
    "how are you i am fine thank you"

The output is given below::

  OnlinePunctuationConfig(model=OnlinePunctuationModelConfig(cnn_bilstm="./sherpa-onnx-online-punct-en-2024-08-06/model.onnx", bpe_vocab="./sherpa-onnx-online-punct-en-2024-08-06/bpe.vocab", num_threads=1, debug=False, provider="cpu"))
  Creating OnlinePunctuation ...
  Started
  Done
  Num threads: 1
  Elapsed seconds: 0.030 s
  Input text: how are you i am fine thank you
  Output text: How are you? I am fine. Thank you.

To use the ``model.int8.onnx``, you can run::

  ./bin/sherpa-onnx-online-punctuation \
    --cnn-bilstm=./sherpa-onnx-online-punct-en-2024-08-06/model.int8.onnx \
    --bpe-vocab=./sherpa-onnx-online-punct-en-2024-08-06/bpe.vocab \
    "how are you i am fine thank you"

The output is given below::

  OnlinePunctuationConfig(model=OnlinePunctuationModelConfig(cnn_bilstm="./sherpa-onnx-online-punct-en-2024-08-06/model.int8.onnx", bpe_vocab="./sherpa-onnx-online-punct-en-2024-08-06/bpe.vocab", num_threads=1, debug=False, provider="cpu"))
  Creating OnlinePunctuation ...
  Started
  Done
  Num threads: 1
  Elapsed seconds: 0.013 s
  Input text: how are you i am fine thank you
  Output text: How are you? I am fine. Thank you.



sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8
------------------------------------------------------------------

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

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8.tar.bz2

  tar xvf sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8.tar.bz2
  rm sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8.tar.bz2

You will find the following files after unzipping::

  total 155776
  -rw-r--r--  1 fangjun  staff   1.5K Jun 18 10:34 README.md
  -rwxr-xr-x  1 fangjun  staff   1.6K Apr 12  2024 add-model-metadata.py
  -rw-r--r--  1 fangjun  staff   810B Apr 12  2024 config.yaml
  -rw-r--r--  1 fangjun  staff    72M Jun 18 10:33 model.int8.onnx
  -rwxr-xr-x  1 fangjun  staff   745B Apr 12  2024 show-model-input-output.py
  -rwxr-xr-x  1 fangjun  staff   4.6K Apr 12  2024 test.py
  -rw-r--r--  1 fangjun  staff   4.0M Apr 12  2024 tokens.json

Only ``model.int8.onnx`` is needed in `sherpa-onnx`_. All other files are for your information about
how the model is converted to `sherpa-onnx`_.

C++ binary examples
^^^^^^^^^^^^^^^^^^^

After installing `sherpa-onnx`_, you can use the following command to add punctuations to text::

   ./bin/sherpa-onnx-offline-punctuation \
     --ct-transformer=./sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8/model.int8.onnx \
     "我们都是木头人不会说话不会动"

The output is given below::

  /Users/fangjun/open-source/sherpa-onnx/sherpa-onnx/csrc/parse-options.cc:Read:372 ./bin/sherpa-onnx-offline-punctuation --ct-transformer=./sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8/model.int8.onnx '我们都是木头人不会说话不会动'

  OfflinePunctuationConfig(model=OfflinePunctuationModelConfig(ct_transformer="./sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8/model.int8.onnx", num_threads=1, debug=False, provider="cpu"))
  Creating OfflinePunctuation ...
  Started
  Done
  Num threads: 1
  Elapsed seconds: 0.014 s
  Input text: 我们都是木头人不会说话不会动
  Output text: 我们都是木头人，不会说话，不会动。

The second example is for text containing both Chinese and English::

  ./bin/sherpa-onnx-offline-punctuation \
    --ct-transformer=./sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8/model.int8.onnx \
    "这是一个测试你好吗How are you我很好thank you are you ok谢谢你"


Its output is given below::

  OfflinePunctuationConfig(model=OfflinePunctuationModelConfig(ct_transformer="./sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8/model.int8.onnx", num_threads=1, debug=False, provider="cpu"))
  Creating OfflinePunctuation ...
  Started
  Done
  Num threads: 1
  Elapsed seconds: 0.010 s
  Input text: 这是一个测试你好吗How are you我很好thank you are you ok谢谢你
  Output text: 这是一个测试你好吗？How are you？我很好？thank you，are you ok，谢谢你。

The last example is for text containing only English::

  ./bin/sherpa-onnx-offline-punctuation \
    --ct-transformer=./sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8/model.int8.onnx \
    "The African blogosphere is rapidly expanding bringing more voices online in the form of commentaries opinions analyses rants and poetry"

The last example is for text containing only English::

  OfflinePunctuationConfig(model=OfflinePunctuationModelConfig(ct_transformer="./sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8/model.int8.onnx", num_threads=1, debug=False, provider="cpu"))
  Creating OfflinePunctuation ...
  Started
  Done
  Num threads: 1
  Elapsed seconds: 0.007 s
  Input text: The African blogosphere is rapidly expanding bringing more voices online in the form of commentaries opinions analyses rants and poetry
  Output text: The African blogosphere is rapidly expanding，bringing more voices online in the form of commentaries，opinions，analyses，rants and poetry。

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
