.. _sherpa-onnx-kws-pre-trained-models:

Pre-trained models
==================

In this section, we describe how to download and use all
available keyword spotting pre-trained models.

.. hint::

  Please install `git-lfs <https://git-lfs.com/>`_ before you continue.

  Otherwise, you will be ``SAD`` later.


.. hint::

   Please refer to :ref:`install_sherpa_onnx` to install `sherpa-onnx`_
   before you read this section.


sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01 (Chinese)
------------------------------------------------------------------

Training code for this model can be found at `<https://github.com/k2-fsa/icefall/pull/1428>`_.
The model is trained on WenetSpeech L subset (10000 hours), it supports only Chinese.

In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. tabs::

   .. tab:: Github

      .. code-block:: bash

        cd /path/to/sherpa-onnx
        wget -qq https://github.com/pkufool/keyword-spotting-models/releases/download/v0.1/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz 
        tar xf sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz
        ls -lh sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01

   .. tab:: ModelScope

      .. code-block:: bash

        cd /path/to/sherpa-onnx
        git lfs install
        git clone https://www.modelscope.cn/pkufool/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.git
        ls -lh sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01

The output is given below:

.. code-block::

    $ ls -lh sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01
    total 18M
    -rw-r--r--  1 kangwei root   48 Jan  1 21:45 configuration.json
    -rw-r--r--  1 kangwei root 177K Jan 17 11:38 decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx
    -rw-r--r--  1 kangwei root 660K Jan  1 21:45 decoder-epoch-12-avg-2-chunk-16-left-64.onnx
    -rw-r--r--  1 kangwei root 4.6M Jan 17 11:38 encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx
    -rw-r--r--  1 kangwei root  12M Jan  1 21:45 encoder-epoch-12-avg-2-chunk-16-left-64.onnx
    -rw-r--r--  1 kangwei root  64K Jan 17 11:38 joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx
    -rw-r--r--  1 kangwei root 248K Jan  1 21:45 joiner-epoch-12-avg-2-chunk-16-left-64.onnx
    -rw-r--r--  1 kangwei root  101 Jan  8 17:14 keywords_raw.txt
    -rw-r--r--  1 kangwei root  286 Jan  8 17:14 keywords.txt
    -rw-r--r--  1 kangwei root  750 Jan  8 17:14 README.md
    drwxr-xr-x 10 kangwei root    0 Jan 15 22:52 test_wavs
    -rw-r--r--  1 kangwei root 1.6K Jan  1 21:45 tokens.txt

Test the model
~~~~~~~~~~~~~~

.. hint::

   It supports decoding only wave files of a single channel with 16-bit
   encoded samples, while the sampling rate does not need to be 16 kHz.

fp32
^^^^

The following code shows how to use ``fp32`` models:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-keyword-spotter \
    --encoder=sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/encoder-epoch-12-avg-2-chunk-16-left-64.onnx \
    --decoder=sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/decoder-epoch-12-avg-2-chunk-16-left-64.onnx \
    --joiner=sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/joiner-epoch-12-avg-2-chunk-16-left-64.onnx \
    --tokens=sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/tokens.txt \
    --keywords-file=sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/test_wavs/test_keywords.txt  \
    sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/test_wavs/3.wav \
    sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/test_wavs/4.wav \
    sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/test_wavs/5.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-keyword-spotter.exe`` for Windows.

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

You should see the following output:

.. code-block::

  KeywordSpotterConfig(feat_config=FeatureExtractorConfig(sampling_rate=16000, feature_dim=80), model_config=OnlineModelConfig(transducer=OnlineTransducerModelConfig(encoder="sherpa-on$x-kws-zipformer-wenetspeech-3.3M-2024-01-01/encoder-epoch-12-avg-2-chunk-16-left-64.onnx", decoder="sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/decoder-epoch-12-avg-2-chunk$16-left-64.onnx", joiner="sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/joiner-epoch-12-avg-2-chunk-16-left-64.onnx"), paraformer=OnlineParaformerModelConfig(encoder="", deco$er=""), wenet_ctc=OnlineWenetCtcModelConfig(model="", chunk_size=16, num_left_chunks=4), zipformer2_ctc=OnlineZipformer2CtcModelConfig(model=""), tokens="sherpa-onnx-kws-zipformer-we$etspeech-3.3M-2024-01-01/tokens.txt", num_threads=1, debug=False, provider="cpu", model_type=""), max_active_paths=4, num_trailing_blanks=1, keywords_score=1, keywords_threshold=0.25 keywords_file="sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/test_wavs/test_keywords.txt")
  
  2024-01-19 12:32:29.983790275 [E:onnxruntime:, env.cc:254 ThreadMain] pthread_setaffinity_np failed for thread: 3385848, index: 15, mask: {16, 52, }, error code: 22 error msg: Invali$
   argument. Specify the number of threads explicitly so the affinity is not set.
  2024-01-19 12:32:29.983792055 [E:onnxruntime:, env.cc:254 ThreadMain] pthread_setaffinity_np failed for thread: 3385849, index: 16, mask: {17, 53, }, error code: 22 error msg: Invali$
   argument. Specify the number of threads explicitly so the affinity is not set.
  sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/test_wavs/4.wav
  {"start_time":0.00, "keyword": "蒋友伯", "timestamps": [0.64, 0.68, 0.84, 0.96, 1.12, 1.16], "tokens":["j", "iǎng", "y", "ǒu", "b", "ó"]}
  
  sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/test_wavs/5.wav
  {"start_time":0.00, "keyword": "周望军", "timestamps": [0.64, 0.68, 0.76, 0.84, 1.00, 1.04], "tokens":["zh", "ōu", "w", "àng", "j", "ūn"]}
  
  sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/test_wavs/3.wav
  {"start_time":0.00, "keyword": "文森特卡索", "timestamps": [0.32, 0.72, 0.96, 1.00, 1.20, 1.32, 1.48, 1.60, 1.88, 1.92], "tokens":["w", "én", "s", "ēn", "t", "è", "k", "ǎ", "s", "uǒ"$
  }
  
  sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/test_wavs/5.wav
  {"start_time":0.00, "keyword": "落实", "timestamps": [1.76, 1.92, 2.12, 2.20], "tokens":["l", "uò", "sh", "í"]}
  
  sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/test_wavs/4.wav
  {"start_time":0.00, "keyword": "女儿", "timestamps": [3.08, 3.20, 3.24], "tokens":["n", "ǚ", "ér"]}
  
  sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/test_wavs/3.wav
  {"start_time":0.00, "keyword": "法国", "timestamps": [4.56, 4.64, 4.80, 4.88], "tokens":["f", "ǎ", "g", "uó"]}
  

int8
^^^^

The following code shows how to use ``int8`` models:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-keyword-spotter \
    --encoder=sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx \
    --decoder=sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx \
    --joiner=sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx \
    --tokens=sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/tokens.txt \
    --keywords-file=sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/test_wavs/test_keywords.txt  \
    sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/test_wavs/3.wav \
    sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/test_wavs/4.wav \
    sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/test_wavs/5.wav


.. code-block::

    KeywordSpotterConfig(feat_config=FeatureExtractorConfig(sampling_rate=16000, feature_dim=80), model_config=OnlineModelConfig(transducer=OnlineTransducerModelConfig(encoder="sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx", decoder="sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx", joiner="sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx"), paraformer=OnlineParaformerModelConfig(encoder="", decoder=""), wenet_ctc=OnlineWenetCtcModelConfig(model="", chunk_size=16, num_left_chunks=4), zipformer2_ctc=OnlineZipformer2CtcModelConfig(model=""), tokens="sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/tokens.txt", num_threads=1, debug=False, provider="cpu", model_type=""), max_active_paths=4, num_trailing_blanks=1, keywords_score=1, keywords_threshold=0.25, keywords_file="sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/test_wavs/test_keywords.txt")
    
    2024-01-19 12:36:44.635979490 [E:onnxruntime:, env.cc:254 ThreadMain] pthread_setaffinity_np failed for thread: 3391918, index: 15, mask: {16, 52, }, error code: 22 error msg: Invalid argument. Specify the number of threads explicitly so the affinity is not set.
    2024-01-19 12:36:44.635981379 [E:onnxruntime:, env.cc:254 ThreadMain] pthread_setaffinity_np failed for thread: 3391919, index: 16, mask: {17, 53, }, error code: 22 error msg: Invalid argument. Specify the number of threads explicitly so the affinity is not set.
    sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/test_wavs/4.wav
    {"start_time":0.00, "keyword": "蒋友伯", "timestamps": [0.64, 0.68, 0.84, 0.96, 1.12, 1.16], "tokens":["j", "iǎng", "y", "ǒu", "b", "ó"]}
    
    sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/test_wavs/5.wav
    {"start_time":0.00, "keyword": "周望军", "timestamps": [0.64, 0.68, 0.76, 0.84, 1.00, 1.08], "tokens":["zh", "ōu", "w", "àng", "j", "ūn"]}
    
    sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/test_wavs/3.wav
    {"start_time":0.00, "keyword": "文森特卡索", "timestamps": [0.32, 0.72, 0.96, 1.04, 1.28, 1.32, 1.52, 1.60, 1.92, 1.96], "tokens":["w", "én", "s", "ēn", "t", "è", "k", "ǎ", "s", "uǒ"]}
    
    sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/test_wavs/5.wav
    {"start_time":0.00, "keyword": "落实", "timestamps": [1.80, 1.92, 2.12, 2.20], "tokens":["l", "uò", "sh", "í"]}
    
    sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/test_wavs/4.wav
    {"start_time":0.00, "keyword": "女儿", "timestamps": [3.08, 3.20, 3.24], "tokens":["n", "ǚ", "ér"]}
    
    sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/test_wavs/3.wav
    {"start_time":0.00, "keyword": "法国", "timestamps": [4.56, 4.64, 4.80, 4.88], "tokens":["f", "ǎ", "g", "uó"]}


Customize your own keywords
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To customize your own keywords, the only thing you need to do is replacing the ``--keywords-file``. The keywords file is generated as follows:

For example your keywords are (keywords_raw.txt):

.. code-block::

   你好军哥 @你好军哥
   你好问问 @你好问问
   小爱同学 @小爱同学

Run the following command:

.. code-block::

   sherpa-onnx-cli text2token \
     --tokens sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/tokens.txt \
     --tokens-type ppinyin \
     keywords_raw.txt keywords.txt

The ``keywords.txt`` looks like:

.. code-block::

   n ǐ h ǎo j ūn g ē @你好军哥
   n ǐ h ǎo w èn w èn @你好问问
   x iǎo ài t óng x ué @小爱同学

.. note::

   If you install sherpa-onnx from sources (i.e. not by pip), you can use the
   alternative script in `scripts`, the usage is almost the same as the command
   line tool, read the help information by:

   .. code-block::

     python3 scripts/text2token.py --help


sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01 (English)
------------------------------------------------------------------

Training code for this model can be found at `<https://github.com/k2-fsa/icefall/pull/1428>`_.
The model is trained on GigaSpeech XL subset (10000 hours), it supports only English.

In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. tabs::

   .. tab:: Github

      .. code-block:: bash

        cd /path/to/sherpa-onnx
        wget -qq https://github.com/pkufool/keyword-spotting-models/releases/download/v0.1/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.tar.bz 
        tar xf sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.tar.bz
        ls -lh sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01

   .. tab:: ModelScope

      .. code-block:: bash

        cd /path/to/sherpa-onnx
        git lfs install
        git clone https://www.modelscope.cn/pkufool/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.git
        ls -lh sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01

The output is given below:

.. code-block::

    $ ls -lh sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01
    total 19M
    -rw-r--r-- 1 kangwei root 240K Jan 19 15:25 bpe.model
    -rw-r--r-- 1 kangwei root   48 Jan 19 15:25 configuration.json
    -rw-r--r-- 1 kangwei root 272K Jan 19 15:25 decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx
    -rw-r--r-- 1 kangwei root 1.1M Jan 19 15:25 decoder-epoch-12-avg-2-chunk-16-left-64.onnx
    -rw-r--r-- 1 kangwei root 4.6M Jan 19 15:25 encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx
    -rw-r--r-- 1 kangwei root  12M Jan 19 15:25 encoder-epoch-12-avg-2-chunk-16-left-64.onnx
    -rw-r--r-- 1 kangwei root 160K Jan 19 15:25 joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx
    -rw-r--r-- 1 kangwei root 628K Jan 19 15:25 joiner-epoch-12-avg-2-chunk-16-left-64.onnx
    -rw-r--r-- 1 kangwei root  102 Jan 19 15:25 keywords_raw.txt
    -rw-r--r-- 1 kangwei root  184 Jan 19 15:25 keywords.txt
    -rw-r--r-- 1 kangwei root  743 Jan 19 15:25 README.md
    drwxr-xr-x 6 kangwei root    0 Jan 19 15:25 test_wavs
    -rw-r--r-- 1 kangwei root 4.9K Jan 19 15:25 tokens.txt

Test the model
~~~~~~~~~~~~~~

.. hint::

   It supports decoding only wave files of a single channel with 16-bit
   encoded samples, while the sampling rate does not need to be 16 kHz.

fp32
^^^^

The following code shows how to use ``fp32`` models:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-keyword-spotter \
    --encoder=sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01/encoder-epoch-12-avg-2-chunk-16-left-64.onnx \
    --decoder=sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01/decoder-epoch-12-avg-2-chunk-16-left-64.onnx \
    --joiner=sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01/joiner-epoch-12-avg-2-chunk-16-left-64.onnx \
    --tokens=sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01/tokens.txt \
    --keywords-file=sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01/test_wavs/test_keywords.txt  \
    sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01/test_wavs/0.wav \
    sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01/test_wavs/1.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-keyword-spotter.exe`` for Windows.

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

You should see the following output:

.. code-block::

    KeywordSpotterConfig(feat_config=FeatureExtractorConfig(sampling_rate=16000, feature_dim=80), model_config=OnlineModelConfig(transducer=OnlineTransducerModelConfig(encoder="sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01/encoder-epoch-12-avg-2-chunk-16-left-64.onnx", decoder="sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01/decoder-epoch-12-avg-2-chunk-16-left-64.onnx", joiner="sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01/joiner-epoch-12-avg-2-chunk-16-left-64.onnx"), paraformer=OnlineParaformerModelConfig(encoder="", decoder=""), wenet_ctc=OnlineWenetCtcModelConfig(model="", chunk_size=16, num_left_chunks=4), zipformer2_ctc=OnlineZipformer2CtcModelConfig(model=""), tokens="sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01/tokens.txt", num_threads=1, debug=False, provider="cpu", model_type=""), max_active_paths=4, num_trailing_blanks=1, keywords_score=1, keywords_threshold=0.25, keywords_file="sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01/test_wavs/test_keywords.txt")
    2024-01-19 15:32:46.420331393 [E:onnxruntime:, env.cc:254 ThreadMain] pthread_setaffinity_np failed for thread: 3492733, index: 16, mask: {17, 53, }, error code: 22 error msg: Invalid argument. Specify the number of threads explicitly so the affinity is not set.
    2024-01-19 15:32:46.420332978 [E:onnxruntime:, env.cc:254 ThreadMain] pthread_setaffinity_np failed for thread: 3492732, index: 15, mask: {16, 52, }, error code: 22 error msg: Invalid argument. Specify the number of threads explicitly so the affinity is not set.
    sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01/test_wavs/0.wav
    {"start_time":0.00, "keyword": "LIGHT UP", "timestamps": [3.04, 3.08, 3.12, 3.20], "tokens":[" ", "L", "IGHT", " UP"]}
    
    sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01/test_wavs/1.wav
    {"start_time":0.00, "keyword": "LOVELY CHILD", "timestamps": [5.44, 5.56, 5.84, 6.00, 6.04], "tokens":[" LOVE", "LY", " CHI", "L", "D"]}
    
    sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01/test_wavs/1.wav
    {"start_time":0.00, "keyword": "FOREVER", "timestamps": [10.88, 11.04, 11.08], "tokens":[" FOR", "E", "VER"]}


int8
^^^^

The following code shows how to use ``int8`` models:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-keyword-spotter \
    --encoder=sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01/encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx \
    --decoder=sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01/decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx \
    --joiner=sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01/joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx \
    --tokens=sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01/tokens.txt \
    --keywords-file=sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01/test_wavs/test_keywords.txt  \
    sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01/test_wavs/0.wav \
    sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01/test_wavs/1.wav


.. code-block::

    KeywordSpotterConfig(feat_config=FeatureExtractorConfig(sampling_rate=16000, feature_dim=80), model_config=OnlineModelConfig(transducer=OnlineTransducerModelConfig(encoder="sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01/encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx", decoder="sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01/decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx", joiner="sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01/joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx"), paraformer=OnlineParaformerModelConfig(encoder="", decoder=""), wenet_ctc=OnlineWenetCtcModelConfig(model="", chunk_size=16, num_left_chunks=4), zipformer2_ctc=OnlineZipformer2CtcModelConfig(model=""), tokens="sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01/tokens.txt", num_threads=1, debug=False, provider="cpu", model_type=""), max_active_paths=4, num_trailing_blanks=1, keywords_score=1, keywords_threshold=0.25, keywords_file="sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01/test_wavs/test_keywords.txt")
    2024-01-19 15:31:39.743344642 [E:onnxruntime:, env.cc:254 ThreadMain] pthread_setaffinity_np failed for thread: 3492115, index: 15, mask: {16, 52, }, error code: 22 error msg: Invalid argument. Specify the number of threads explicitly so the affinity is not set.
    2024-01-19 15:31:39.743346583 [E:onnxruntime:, env.cc:254 ThreadMain] pthread_setaffinity_np failed for thread: 3492116, index: 16, mask: {17, 53, }, error code: 22 error msg: Invalid argument. Specify the number of threads explicitly so the affinity is not set.
    sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01/test_wavs/0.wav
    {"start_time":0.00, "keyword": "LIGHT UP", "timestamps": [3.04, 3.08, 3.12, 3.16], "tokens":[" ", "L", "IGHT", " UP"]}
    
    sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01/test_wavs/1.wav
    {"start_time":0.00, "keyword": "LOVELY CHILD", "timestamps": [5.36, 5.60, 5.84, 6.00, 6.04], "tokens":[" LOVE", "LY", " CHI", "L", "D"]}
    
    sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01/test_wavs/1.wav
    {"start_time":0.00, "keyword": "FOREVER", "timestamps": [10.88, 11.04, 11.08], "tokens":[" FOR", "E", "VER"]}


Customize your own keywords
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To customize your own keywords, the only thing you need to do is replacing the ``--keywords-file``. The keywords file is generated as follows:

For example your keywords are (keywords_raw.txt):

.. code-block::

   HELLO WORLD
   HI GOOGLE
   HEY SIRI

Run the following command:

.. code-block::

   sherpa-onnx-cli text2token \
     --tokens sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01/tokens.txt \
     --tokens-type bpe \
     --bpe-model sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01/bpe.model \
     keywords_raw.txt keywords.txt

The ``keywords.txt`` looks like:

.. code-block::

    ▁HE LL O ▁WORLD
    ▁HI ▁GO O G LE
    ▁HE Y ▁S I RI

.. note::

   If you install sherpa-onnx from sources (i.e. not by pip), you can use the
   alternative script in `scripts`, the usage is almost the same as the command
   line tool, read the help information by:

   .. code-block::

     python3 scripts/text2token.py --help
