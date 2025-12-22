.. _sherpa-onnx-kws-pre-trained-models:

In this section, we describe how to download and use all
available keyword spotting pre-trained models.

.. hint::

   Please refer to :ref:`install_sherpa_onnx` to install `sherpa-onnx`_
   before you read this section.


sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20 (Chinese & English)
-----------------------------------------------------------------

In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

  cd /path/to/sherpa-onnx
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20.tar.bz2
  tar xf sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20.tar.bz2
  rm sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20.tar.bz2
  ls -lh sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20


The output is given below:

.. code-block::

    $ ls -lh sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20
    total 38M
    -rw-r--r--  1 kangwei root 743K 12月 22 17:14 decoder-epoch-13-avg-2-chunk-16-left-64.onnx
    -rw-r--r--  1 kangwei root 743K 12月 22 17:14 decoder-epoch-13-avg-2-chunk-8-left-64.onnx
    -rw-r--r--  1 kangwei root 4.4M 12月 22 17:14 encoder-epoch-13-avg-2-chunk-16-left-64.int8.onnx
    -rw-r--r--  1 kangwei root  12M 12月 22 17:14 encoder-epoch-13-avg-2-chunk-16-left-64.onnx
    -rw-r--r--  1 kangwei root 4.4M 12月 22 17:14 encoder-epoch-13-avg-2-chunk-8-left-64.int8.onnx
    -rw-r--r--  1 kangwei root  12M 12月 22 17:14 encoder-epoch-13-avg-2-chunk-8-left-64.onnx
    -rw-r--r--  1 kangwei root 3.2M 12月 22 17:15 en.phone
    -rw-r--r--  1 kangwei root  85K 12月 22 17:14 joiner-epoch-13-avg-2-chunk-16-left-64.int8.onnx
    -rw-r--r--  1 kangwei root 331K 12月 22 17:14 joiner-epoch-13-avg-2-chunk-16-left-64.onnx
    -rw-r--r--  1 kangwei root  85K 12月 22 17:14 joiner-epoch-13-avg-2-chunk-8-left-64.int8.onnx
    -rw-r--r--  1 kangwei root 331K 12月 22 17:14 joiner-epoch-13-avg-2-chunk-8-left-64.onnx
    drwxr-xr-x 14 kangwei root    0 12月 22 18:46 test_wavs
    -rw-r--r--  1 kangwei root 1.9K 12月 22 17:15 tokens.txt

.. hint::

    The models with ``chunk-16`` in their names are exported for
    ``chunk_size=16``, while those with ``chunk-8`` in their names are for
    ``chunk_size=8``. The ``chunk_size=8`` model has a latency of 160ms, while
    the ``chunk_size=16`` model has a latency of 320ms, please choose the right
    models according to your needs (lower latency usually means lower accuracy).

    The ``int8`` models are quantized from ``fp32`` models using ONNX Runtime, decoder
    does not benefit much from quantization, so we only provide ``fp32`` versions, you
    can use ``int8`` encoder and joiner and ``fp32`` decoder together.


Test the model
~~~~~~~~~~~~~~

.. hint::

   It supports decoding only wave files of a single channel with 16-bit
   encoded samples, while the sampling rate does not need to be 16 kHz.


The following code shows how to use ``fp32`` and ``chunk=16`` models:

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-keyword-spotter \
    --encoder=sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20/encoder-epoch-13-avg-2-chunk-16-left-64.onnx \
    --decoder=sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20/decoder-epoch-13-avg-2-chunk-16-left-64.onnx \
    --joiner=sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20/joiner-epoch-13-avg-2-chunk-16-left-64.onnx \
    --tokens=sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20/tokens.txt \
    --keywords-file=sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20/test_wavs/keywords.txt  \
    sherpa-onnx-kws-zipformer-zh-en-3M-2025-12.20/test_wavs/zh_3.wav \
    sherpa-onnx-kws-zipformer-zh-en-3M-2025-12.20/test_wavs/zh_4.wav \
    sherpa-onnx-kws-zipformer-zh-en-3M-2025-12.20/test_wavs/zh_5.wav \
    sherpa-onnx-kws-zipformer-zh-en-3M-2025-12.20/test_wavs/en_0.wav \
    sherpa-onnx-kws-zipformer-zh-en-3M-2025-12.20/test_wavs/en_1.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-keyword-spotter.exe`` for Windows.

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

You should see the following output:

.. code-block::

    KeywordSpotterConfig(feat_config=FeatureExtractorConfig(sampling_rate=16000, feature_dim=80, low_freq=20, high_freq=-400, dither=0, normalize_samples=True, snip_edges=False), model_config=OnlineModelConfig(transducer=OnlineTransducerModelConfig(encoder="sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20/encoder-epoch-13-avg-2-chunk-16-left-64.onnx", decoder="sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20/decoder-epoch-13-avg-2-chunk-16-left-64.onnx", joiner="sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20/joiner-epoch-13-avg-2-chunk-16-left-64.onnx"), paraformer=OnlineParaformerModelConfig(encoder="", decoder=""), wenet_ctc=OnlineWenetCtcModelConfig(model="", chunk_size=16, num_left_chunks=4), zipformer2_ctc=OnlineZipformer2CtcModelConfig(model=""), nemo_ctc=OnlineNeMoCtcModelConfig(model=""), t_one_ctc=OnlineToneCtcModelConfig(model=""), provider_config=ProviderConfig(device=0, provider="cpu", cuda_config=CudaConfig(cudnn_conv_algo_search=1), trt_config=TensorrtConfig(trt_max_workspace_size=2147483647, trt_max_partition_iterations=10, trt_min_subgraph_size=5, trt_fp16_enable="True", trt_detailed_build_log="False", trt_engine_cache_enable="True", trt_engine_cache_path=".", trt_timing_cache_enable="True", trt_timing_cache_path=".",trt_dump_subgraphs="False" )), tokens="sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20/tokens.txt", num_threads=1, warm_up=0, debug=False, model_type="", modeling_unit="cjkchar", bpe_vocab=""), max_active_paths=4, num_trailing_blanks=1, keywords_score=1, keywords_threshold=0.25, keywords_file="sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20/test_wavs/keywords.txt")

    sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20/test_wavs/zh_5.wav
    {"start_time":0.00, "keyword": "周望军", "timestamps": [0.64, 0.72, 0.80, 0.84, 1.04, 1.08], "tokens":["zh", "ōu", "w", "àng", "j", "ūn"]}

    sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20/test_wavs/zh_4.wav
    {"start_time":0.00, "keyword": "蒋友伯", "timestamps": [0.64, 0.72, 0.84, 1.00, 1.12, 1.20], "tokens":["j", "iǎng", "y", "ǒu", "b", "ó"]}

    sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20/test_wavs/zh_3.wav
    {"start_time":0.00, "keyword": "文森特卡索", "timestamps": [0.64, 0.76, 0.96, 1.04, 1.28, 1.36, 1.52, 1.64, 1.84, 1.96], "tokens":["w", "én", "s", "ēn", "t", "è", "k", "ǎ", "s", "uǒ"]}

    sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20/test_wavs/zh_5.wav
    {"start_time":0.00, "keyword": "落实", "timestamps": [1.80, 1.92, 2.12, 2.20], "tokens":["l", "uò", "sh", "í"]}

    sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20/test_wavs/zh_4.wav
    {"start_time":0.00, "keyword": "女儿", "timestamps": [3.04, 3.20, 3.24], "tokens":["n", "ǚ", "ér"]}

    sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20/test_wavs/en_0.wav
    {"start_time":0.00, "keyword": "LIGHT_UP", "timestamps": [2.92, 2.96, 3.04, 3.20, 3.28], "tokens":["L", "AY1", "T", "AH1", "P"]}

    sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20/test_wavs/zh_3.wav
    {"start_time":0.00, "keyword": "法国", "timestamps": [4.52, 4.68, 4.84, 4.92], "tokens":["f", "ǎ", "g", "uó"]}

    sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20/test_wavs/en_1.wav
    {"start_time":0.00, "keyword": "LOVELY_CHILD", "timestamps": [5.24, 5.32, 5.40, 5.48, 5.56, 5.68, 5.84, 5.92, 5.96], "tokens":["L", "AH1", "V", "L", "IY0", "CH", "AY1", "L", "D"]}


Customize your own keywords
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To customize your own keywords, the only thing you need to do is replacing the ``--keywords-file``. The keywords file is generated as follows:

For example your keywords are (keywords_raw.txt):

.. note::

   Each line contains a keyword, you MUST provide the original keyword (starting with ``@``)
   and the original keyword CAN NOT contain spaces, please replace spaces with underscores (``_``).

.. code-block::

   LIGHT UP @LIGHT_UP
   LOVELY CHILD @LOVELY_CHILD
   文森特卡索 @文森特卡索
   周望军 @周望军
   朱丽楠 @朱丽楠
   蒋友伯 @蒋友伯
   女儿 @女儿
   法国 @法国
   见面会 @见面会
   落实 @落实

Run the following command:

.. code-block::

   sherpa-onnx-cli text2token \
     --tokens sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20/tokens.txt \
     --tokens-type phone+ppinyin \
     --lexicon sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20/en.phone \
     keywords_raw.txt keywords.txt

The ``keywords.txt`` looks like:

.. code-block::

   L AY1 T AH1 P @LIGHT_UP
   L AH1 V L IY0 CH AY1 L D @LOVELY_CHILD
   w én s ēn t è k ǎ s uǒ @文森特卡索
   zh ōu w àng j ūn @周望军
   zh ū l ì n án @朱丽楠
   j iǎng y ǒu b ó @蒋友伯
   n ǚ ér @女儿
   f ǎ g uó @法国
   j iàn m iàn h uì @见面会
   l uò sh í @落实

.. note::

   If you install sherpa-onnx from sources (i.e. not by pip), you can use the
   alternative script in `scripts`, the usage is almost the same as the command
   line tool, read the help information by:

   .. code-block::

     python3 scripts/text2token.py --help



sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01 (Chinese)
---------------------------------------------------------------

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
        wget https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2
        tar xf sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2
        rm sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2
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
        wget https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.tar.bz2
        tar xvf sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.tar.bz2
        rm sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.tar.bz2
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
