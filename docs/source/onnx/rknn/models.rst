Pre-trained models
==================


You can download pre-trained models for RKNPU from `<https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models>`_.

In the following, we use models for ``rk3588`` as an example. You can replace
``rk3588`` with ``rk3576``, ``rk3568``, ``rk3566`` or ``rk3562``.


Before you continue, we assume you have followed :ref:`sherpa-onnx-rknn-install`
to install `sherpa-onnx`_. The following is an example of installing
`sherpa-onnx`_ with RKNN support on OrangePi 5 max.

.. code-block::

  (py310) orangepi@orangepi5max:~/t$ uname -a
  Linux orangepi5max 6.1.43-rockchip-rk3588 #1.0.0 SMP Mon Jul  8 11:54:40 CST 2024 aarch64 aarch64 aarch64 GNU/Linux
  (py310) orangepi@orangepi5max:~/t$ ls -lh sherpa_onnx-1.11.2-cp310-cp310-linux_aarch64.whl
  -rw-r--r-- 1 orangepi orangepi 17M Mar  8 00:20 sherpa_onnx-1.11.2-cp310-cp310-linux_aarch64.whl
  (py310) orangepi@orangepi5max:~/t$ pip install ./sherpa_onnx-1.11.2-cp310-cp310-linux_aarch64.whl
  Processing ./sherpa_onnx-1.11.2-cp310-cp310-linux_aarch64.whl
  Installing collected packages: sherpa-onnx
  Successfully installed sherpa-onnx-1.11.2

  (py310) orangepi@orangepi5max:~/t$ which sherpa-onnx
  /home/orangepi/py310/bin/sherpa-onnx

  (py310) orangepi@orangepi5max:~/t$ ldd $(which sherpa-onnx)
    linux-vdso.so.1 (0x0000007f9fd93000)
    librknnrt.so => /lib/librknnrt.so (0x0000007f9f480000)
    libonnxruntime.so => /home/orangepi/py310/bin/../lib/python3.10/site-packages/sherpa_onnx/lib/libonnxruntime.so (0x0000007f9e7f0000)
    libm.so.6 => /lib/aarch64-linux-gnu/libm.so.6 (0x0000007f9e750000)
    libstdc++.so.6 => /lib/aarch64-linux-gnu/libstdc++.so.6 (0x0000007f9e520000)
    libgcc_s.so.1 => /lib/aarch64-linux-gnu/libgcc_s.so.1 (0x0000007f9e4f0000)
    libc.so.6 => /lib/aarch64-linux-gnu/libc.so.6 (0x0000007f9e340000)
    /lib/ld-linux-aarch64.so.1 (0x0000007f9fd5a000)
    libpthread.so.0 => /lib/aarch64-linux-gnu/libpthread.so.0 (0x0000007f9e320000)
    libdl.so.2 => /lib/aarch64-linux-gnu/libdl.so.2 (0x0000007f9e300000)
    librt.so.1 => /lib/aarch64-linux-gnu/librt.so.1 (0x0000007f9e2e0000)

  (py310) orangepi@orangepi5max:~/t$ strings /lib/librknnrt.so | grep "librknnrt version"
  librknnrt version: 2.1.0 (967d001cc8@2024-08-07T19:28:19)


sherpa-onnx-rk3588-streaming-zipformer-small-bilingual-zh-en-2023-02-16
-----------------------------------------------------------------------

This model is converted from :ref:`sherpa_onnx_streaming_zipformer_small_bilingual_zh_en_2023_02_16`.

Please use the following commands to download it.

.. code-block:: bash

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-rk3588-streaming-zipformer-small-bilingual-zh-en-2023-02-16.tar.bz2
   tar xvf sherpa-onnx-rk3588-streaming-zipformer-small-bilingual-zh-en-2023-02-16.tar.
   rm sherpa-onnx-rk3588-streaming-zipformer-small-bilingual-zh-en-2023-02-16.tar.

After downloading, you can check the file size::

  ls -lh sherpa-onnx-rk3588-streaming-zipformer-small-bilingual-zh-en-2023-02-16/
  total 58M
  -rw-r--r-- 1 orangepi orangepi 7.7M Mar 19  2025 decoder.rknn
  -rw-r--r-- 1 orangepi orangepi  44M Mar 19  2025 encoder.rknn
  -rw-r--r-- 1 orangepi orangepi 6.2M Mar 19  2025 joiner.rknn
  drwxr-xr-x 2 orangepi orangepi 4.0K Mar 19  2025 test_wavs
  -rw-r--r-- 1 orangepi orangepi  55K Mar 19  2025 tokens.txt

Decode files
~~~~~~~~~~~~

You can use the following command to decode files with the downloaded model files::

  sherpa-onnx \
    --provider=rknn \
    --encoder=./sherpa-onnx-rk3588-streaming-zipformer-small-bilingual-zh-en-2023-02-16/encoder.rknn \
    --decoder=./sherpa-onnx-rk3588-streaming-zipformer-small-bilingual-zh-en-2023-02-16/decoder.rknn \
    --joiner=./sherpa-onnx-rk3588-streaming-zipformer-small-bilingual-zh-en-2023-02-16/joiner.rknn \
    --tokens=./sherpa-onnx-rk3588-streaming-zipformer-small-bilingual-zh-en-2023-02-16/tokens.txt \
    ./sherpa-onnx-rk3588-streaming-zipformer-small-bilingual-zh-en-2023-02-16/test_wavs/4.wav

The output is given below::

  OnlineRecognizerConfig(feat_config=FeatureExtractorConfig(sampling_rate=16000, feature_dim=80, low_freq=20, high_freq=-400, dither=0, normalize_samples=True, snip_edges=False), model_config=OnlineModelConfig(transducer=OnlineTransducerModelConfig(encoder="./sherpa-onnx-rk3588-streaming-zipformer-small-bilingual-zh-en-2023-02-16/encoder.rknn", decoder="./sherpa-onnx-rk3588-streaming-zipformer-small-bilingual-zh-en-2023-02-16/decoder.rknn", joiner="./sherpa-onnx-rk3588-streaming-zipformer-small-bilingual-zh-en-2023-02-16/joiner.rknn"), paraformer=OnlineParaformerModelConfig(encoder="", decoder=""), wenet_ctc=OnlineWenetCtcModelConfig(model="", chunk_size=16, num_left_chunks=4), zipformer2_ctc=OnlineZipformer2CtcModelConfig(model=""), nemo_ctc=OnlineNeMoCtcModelConfig(model=""), provider_config=ProviderConfig(device=0, provider="rknn", cuda_config=CudaConfig(cudnn_conv_algo_search=1), trt_config=TensorrtConfig(trt_max_workspace_size=2147483647, trt_max_partition_iterations=10, trt_min_subgraph_size=5, trt_fp16_enable="True", trt_detailed_build_log="False", trt_engine_cache_enable="True", trt_engine_cache_path=".", trt_timing_cache_enable="True", trt_timing_cache_path=".",trt_dump_subgraphs="False" )), tokens="./sherpa-onnx-rk3588-streaming-zipformer-small-bilingual-zh-en-2023-02-16/tokens.txt", num_threads=1, warm_up=0, debug=False, model_type="", modeling_unit="cjkchar", bpe_vocab=""), lm_config=OnlineLMConfig(model="", scale=0.5, shallow_fusion=True), endpoint_config=EndpointConfig(rule1=EndpointRule(must_contain_nonsilence=False, min_trailing_silence=2.4, min_utterance_length=0), rule2=EndpointRule(must_contain_nonsilence=True, min_trailing_silence=1.2, min_utterance_length=0), rule3=EndpointRule(must_contain_nonsilence=False, min_trailing_silence=0, min_utterance_length=20)), ctc_fst_decoder_config=OnlineCtcFstDecoderConfig(graph="", max_active=3000), enable_endpoint=True, max_active_paths=4, hotwords_score=1.5, hotwords_file="", decoding_method="greedy_search", blank_penalty=0, temperature_scale=2, rule_fsts="", rule_fars="")
  ./sherpa-onnx-rk3588-streaming-zipformer-small-bilingual-zh-en-2023-02-16/test_wavs/4.wav
  Number of threads: 1, Elapsed seconds: 3.5, Audio duration (s): 18, Real time factor (RTF) = 3.5/18 = 0.2
  嗯 ON TIME比较准时 IN TIME是及时叫他总是准时教他的作业那用一般现在时是没有什么感情色彩的陈述一个事实下一句话为什么要用现在进行时它的意思并不是说说他现在正在教他的
  { "text": "嗯 ON TIME比较准时 IN TIME是及时叫他总是准时教他的作业那用一般现在时是没有什么感情色彩的陈述一个事实下一句话为什么要用现在进行时它的意思并不是说说他现在正在教他的", "tokens": ["嗯", " ON", " TIME", "比", "较", "准", "时", " IN", " TIME", "是", "及", "时", "叫", "他", "总", "是", "准", "时", "教", "他", "的", "作", "业", "那", "用", "一", "般", "现", "在", "时", "是", "没", "有", "什", "么", "感", "情", "色", "彩", "的", "陈", "述", "一", "个", "事", "实", "下", "一", "句", "话", "为", "什", "么", "要", "用", "现", "在", "进", "行", "时", "它", "的", "意", "思", "并", "不", "是", "说", "说", "他", "现", "在", "正", "在", "教", "他", "的"], "timestamps": [0.00, 0.64, 0.80, 1.12, 1.16, 1.36, 1.64, 2.00, 2.16, 2.52, 2.80, 2.92, 3.28, 3.64, 3.92, 4.16, 4.48, 4.60, 4.84, 5.12, 5.28, 5.52, 5.72, 6.20, 6.52, 6.80, 7.04, 7.28, 7.52, 7.72, 7.84, 8.08, 8.24, 8.40, 8.44, 8.68, 8.92, 9.00, 9.24, 9.48, 9.80, 9.92, 10.16, 10.32, 10.56, 10.80, 11.52, 11.60, 11.80, 11.96, 12.20, 12.32, 12.40, 12.56, 12.80, 13.12, 13.32, 13.56, 13.76, 13.92, 14.24, 14.36, 14.52, 14.68, 14.92, 15.04, 15.16, 15.32, 15.72, 16.12, 16.36, 16.48, 16.68, 16.88, 17.08, 17.24, 17.84], "ys_probs": [], "lm_probs": [], "context_scores": [], "segment": 0, "words": [], "start_time": 0.00, "is_final": false}

.. hint::

  If you get the following errors::

    E RKNN: [01:24:27.170] 6, 1
    E RKNN: [01:24:27.170] Invalid RKNN model version 6
    E RKNN: [01:24:27.171] rknn_init, load model failed!
    /home/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/rknn/online-zipformer-transducer-model-rknn.cc:InitEncoder:330 Return code is: -1
    /home/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/rknn/online-zipformer-transducer-model-rknn.cc:InitEncoder:330 Failed to init encoder './sherpa-onnx-rk3588-streaming-zipformer-small-bilingual-zh-en-2023-02-16/encoder.rknn'

  Please update your ``/lib/librknnrt.so`` or ``/usr/lib/librknnrt.so`` with the
  one from `<https://github.com/airockchip/rknn-toolkit2/blob/master/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so>`_.

  Note that you can locate where your ``librknnrt.so`` is by::

      ldd $(which sherpa-onnx)

.. note::

   You can use::

    watch -n 0.5 cat /sys/kernel/debug/rknpu/load

   to watch the usage of NPU.

   For the RK3588 board, you can use:

    - ``--num-threads=1`` to select ``RKNN_NPU_CORE_AUTO``
    - ``--num-threads=0`` to select ``RKNN_NPU_CORE_0``
    - ``--num-threads=-1`` to select ``RKNN_NPU_CORE_1``
    - ``--num-threads=-2`` to select ``RKNN_NPU_CORE_2``
    - ``--num-threads=-3`` to select ``RKNN_NPU_CORE_0_1``
    - ``--num-threads=-4`` to select ``RKNN_NPU_CORE_0_1_2``

Real-time speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, we need to get the name of the microphone on the board::

  arecord -l
  **** List of CAPTURE Hardware Devices ****
  card 2: rockchipes8388 [rockchip,es8388], device 0: dailink-multicodecs ES8323 HiFi-0 [dailink-multicodecs ES8323 HiFi-0]
    Subdevices: 1/1
    Subdevice #0: subdevice #0
  card 3: UACDemoV10 [UACDemoV1.0], device 0: USB Audio [USB Audio]
    Subdevices: 1/1
    Subdevice #0: subdevice #0

We will use ``card 3`` ``device 0``, so the name is ``plughw:3,0``.

.. code-block::

  sherpa-onnx-alsa \
    --provider=rknn \
    --encoder=./sherpa-onnx-rk3588-streaming-zipformer-small-bilingual-zh-en-2023-02-16/encoder.rknn \
    --decoder=./sherpa-onnx-rk3588-streaming-zipformer-small-bilingual-zh-en-2023-02-16/decoder.rknn \
    --joiner=./sherpa-onnx-rk3588-streaming-zipformer-small-bilingual-zh-en-2023-02-16/joiner.rknn \
    --tokens=./sherpa-onnx-rk3588-streaming-zipformer-small-bilingual-zh-en-2023-02-16/tokens.txt \
    plughw:3,0

You should see the following output::

  /home/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/parse-options.cc:Read:375 sherpa-onnx-alsa --provider=rknn --encoder=./sherpa-onnx-rk3588-streaming-zipformer-small-bilingual-zh-en-2023-02-16/encoder.rknn --decoder=./sherpa-onnx-rk3588-streaming-zipformer-small-bilingual-zh-en-2023-02-16/decoder.rknn --joiner=./sherpa-onnx-rk3588-streaming-zipformer-small-bilingual-zh-en-2023-02-16/joiner.rknn --tokens=./sherpa-onnx-rk3588-streaming-zipformer-small-bilingual-zh-en-2023-02-16/tokens.txt plughw:3,0

  OnlineRecognizerConfig(feat_config=FeatureExtractorConfig(sampling_rate=16000, feature_dim=80, low_freq=20, high_freq=-400, dither=0, normalize_samples=True, snip_edges=False), model_config=OnlineModelConfig(transducer=OnlineTransducerModelConfig(encoder="./sherpa-onnx-rk3588-streaming-zipformer-small-bilingual-zh-en-2023-02-16/encoder.rknn", decoder="./sherpa-onnx-rk3588-streaming-zipformer-small-bilingual-zh-en-2023-02-16/decoder.rknn", joiner="./sherpa-onnx-rk3588-streaming-zipformer-small-bilingual-zh-en-2023-02-16/joiner.rknn"), paraformer=OnlineParaformerModelConfig(encoder="", decoder=""), wenet_ctc=OnlineWenetCtcModelConfig(model="", chunk_size=16, num_left_chunks=4), zipformer2_ctc=OnlineZipformer2CtcModelConfig(model=""), nemo_ctc=OnlineNeMoCtcModelConfig(model=""), provider_config=ProviderConfig(device=0, provider="rknn", cuda_config=CudaConfig(cudnn_conv_algo_search=1), trt_config=TensorrtConfig(trt_max_workspace_size=2147483647, trt_max_partition_iterations=10, trt_min_subgraph_size=5, trt_fp16_enable="True", trt_detailed_build_log="False", trt_engine_cache_enable="True", trt_engine_cache_path=".", trt_timing_cache_enable="True", trt_timing_cache_path=".",trt_dump_subgraphs="False" )), tokens="./sherpa-onnx-rk3588-streaming-zipformer-small-bilingual-zh-en-2023-02-16/tokens.txt", num_threads=1, warm_up=0, debug=False, model_type="", modeling_unit="cjkchar", bpe_vocab=""), lm_config=OnlineLMConfig(model="", scale=0.5, shallow_fusion=True), endpoint_config=EndpointConfig(rule1=EndpointRule(must_contain_nonsilence=False, min_trailing_silence=2.4, min_utterance_length=0), rule2=EndpointRule(must_contain_nonsilence=True, min_trailing_silence=1.2, min_utterance_length=0), rule3=EndpointRule(must_contain_nonsilence=False, min_trailing_silence=0, min_utterance_length=20)), ctc_fst_decoder_config=OnlineCtcFstDecoderConfig(graph="", max_active=3000), enable_endpoint=True, max_active_paths=4, hotwords_score=1.5, hotwords_file="", decoding_method="greedy_search", blank_penalty=0, temperature_scale=2, rule_fsts="", rule_fars="")
  Current sample rate: 16000
  Recording started!
  Use recording device: plughw:3,0
  Started! Please speak
  0:这是一个实时的语音识别
  1:今天是二零二五年三月二十二号

sherpa-onnx-rk3588-streaming-zipformer-bilingual-zh-en-2023-02-20
-----------------------------------------------------------------

This model is converted from :ref:`sherpa_onnx_streaming_zipformer_small_bilingual_zh_en_2023_02_16`.

Please use the following commands to download it.

.. code-block:: bash

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-rk3588-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
   tar xvf sherpa-onnx-rk3588-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
   rm sherpa-onnx-rk3588-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2

After downloading, you can check the file size::

  ls -lh sherpa-onnx-rk3588-streaming-zipformer-bilingual-zh-en-2023-02-20/
  total 146M
  -rw-r--r-- 1 orangepi orangepi 7.7M Mar 19  2025 decoder.rknn
  -rw-r--r-- 1 orangepi orangepi 132M Mar 19  2025 encoder.rknn
  -rw-r--r-- 1 orangepi orangepi 6.2M Mar 19  2025 joiner.rknn
  drwxr-xr-x 2 orangepi orangepi 4.0K Mar 19  2025 test_wavs
  -rw-r--r-- 1 orangepi orangepi  55K Mar 19  2025 tokens.txt

Decode files
~~~~~~~~~~~~

You can use the following command to decode files with the downloaded model files::

  sherpa-onnx \
    --provider=rknn \
    --encoder=./sherpa-onnx-rk3588-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder.rknn \
    --decoder=./sherpa-onnx-rk3588-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder.rknn \
    --joiner=./sherpa-onnx-rk3588-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner.rknn \
    --tokens=./sherpa-onnx-rk3588-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt \
    ./sherpa-onnx-rk3588-streaming-zipformer-bilingual-zh-en-2023-02-20/test_wavs/4.wav

The output is given below::

  OnlineRecognizerConfig(feat_config=FeatureExtractorConfig(sampling_rate=16000, feature_dim=80, low_freq=20, high_freq=-400, dither=0, normalize_samples=True, snip_edges=False), model_config=OnlineModelConfig(transducer=OnlineTransducerModelConfig(encoder="./sherpa-onnx-rk3588-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder.rknn", decoder="./sherpa-onnx-rk3588-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder.rknn", joiner="./sherpa-onnx-rk3588-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner.rknn"), paraformer=OnlineParaformerModelConfig(encoder="", decoder=""), wenet_ctc=OnlineWenetCtcModelConfig(model="", chunk_size=16, num_left_chunks=4), zipformer2_ctc=OnlineZipformer2CtcModelConfig(model=""), nemo_ctc=OnlineNeMoCtcModelConfig(model=""), provider_config=ProviderConfig(device=0, provider="rknn", cuda_config=CudaConfig(cudnn_conv_algo_search=1), trt_config=TensorrtConfig(trt_max_workspace_size=2147483647, trt_max_partition_iterations=10, trt_min_subgraph_size=5, trt_fp16_enable="True", trt_detailed_build_log="False", trt_engine_cache_enable="True", trt_engine_cache_path=".", trt_timing_cache_enable="True", trt_timing_cache_path=".",trt_dump_subgraphs="False" )), tokens="./sherpa-onnx-rk3588-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt", num_threads=1, warm_up=0, debug=False, model_type="", modeling_unit="cjkchar", bpe_vocab=""), lm_config=OnlineLMConfig(model="", scale=0.5, shallow_fusion=True), endpoint_config=EndpointConfig(rule1=EndpointRule(must_contain_nonsilence=False, min_trailing_silence=2.4, min_utterance_length=0), rule2=EndpointRule(must_contain_nonsilence=True, min_trailing_silence=1.2, min_utterance_length=0), rule3=EndpointRule(must_contain_nonsilence=False, min_trailing_silence=0, min_utterance_length=20)), ctc_fst_decoder_config=OnlineCtcFstDecoderConfig(graph="", max_active=3000), enable_endpoint=True, max_active_paths=4, hotwords_score=1.5, hotwords_file="", decoding_method="greedy_search", blank_penalty=0, temperature_scale=2, rule_fsts="", rule_fars="")
  ./sherpa-onnx-rk3588-streaming-zipformer-bilingual-zh-en-2023-02-20/test_wavs/2.wav
  Number of threads: 1, Elapsed seconds: 1.8, Audio duration (s): 4.7, Real time factor (RTF) = 1.8/4.7 = 0.38
  这个是频繁的啊不认识记下来 FREQUENTLY频繁的
  { "text": "这个是频繁的啊不认识记下来 FREQUENTLY频繁的", "tokens": ["这", "个", "是", "频", "繁", "的", "啊", "不", "认", "识", "记", "下", "来", " F", "RE", "QU", "ENT", "LY", "频", "繁", "的"], "timestamps": [0.00, 0.36, 0.52, 0.80, 1.00, 1.16, 1.44, 1.64, 1.92, 2.00, 2.20, 2.36, 2.52, 2.64, 2.88, 2.96, 3.08, 3.32, 3.60, 3.80, 4.40], "ys_probs": [], "lm_probs": [], "context_scores": [], "segment": 0, "words": [], "start_time": 0.00, "is_final": false}

.. hint::

  If you get the following errors::

    E RKNN: [01:24:27.170] 6, 1
    E RKNN: [01:24:27.170] Invalid RKNN model version 6
    E RKNN: [01:24:27.171] rknn_init, load model failed!
    /home/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/rknn/online-zipformer-transducer-model-rknn.cc:InitEncoder:330 Return code is: -1
    /home/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/rknn/online-zipformer-transducer-model-rknn.cc:InitEncoder:330 Failed to init encoder './sherpa-onnx-rk3588-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder.rknn'

  Please update your ``/lib/librknnrt.so`` or ``/usr/lib/librknnrt.so`` with the
  one from `<https://github.com/airockchip/rknn-toolkit2/blob/master/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so>`_.

  Note that you can locate where your ``librknnrt.so`` is by::

      ldd $(which sherpa-onnx)

.. note::

   You can use::

    watch -n 0.5 cat /sys/kernel/debug/rknpu/load

   to watch the usage of NPU.

   For the RK3588 board, you can use:

    - ``--num-threads=1`` to select ``RKNN_NPU_CORE_AUTO``
    - ``--num-threads=0`` to select ``RKNN_NPU_CORE_0``
    - ``--num-threads=-1`` to select ``RKNN_NPU_CORE_1``
    - ``--num-threads=-2`` to select ``RKNN_NPU_CORE_2``
    - ``--num-threads=-3`` to select ``RKNN_NPU_CORE_0_1``
    - ``--num-threads=-4`` to select ``RKNN_NPU_CORE_0_1_2``

Real-time speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, we need to get the name of the microphone on the board::

  arecord -l
  **** List of CAPTURE Hardware Devices ****
  card 2: rockchipes8388 [rockchip,es8388], device 0: dailink-multicodecs ES8323 HiFi-0 [dailink-multicodecs ES8323 HiFi-0]
    Subdevices: 1/1
    Subdevice #0: subdevice #0
  card 3: UACDemoV10 [UACDemoV1.0], device 0: USB Audio [USB Audio]
    Subdevices: 1/1
    Subdevice #0: subdevice #0

We will use ``card 3`` ``device 0``, so the name is ``plughw:3,0``.

.. code-block::

  sherpa-onnx-alsa \
    --provider=rknn \
    --encoder=./sherpa-onnx-rk3588-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder.rknn \
    --decoder=./sherpa-onnx-rk3588-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder.rknn \
    --joiner=./sherpa-onnx-rk3588-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner.rknn \
    --tokens=./sherpa-onnx-rk3588-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt \
    plughw:3,0

You should see the following output::

  OnlineRecognizerConfig(feat_config=FeatureExtractorConfig(sampling_rate=16000, feature_dim=80, low_freq=20, high_freq=-400, dither=0, normalize_samples=True, snip_edges=False), model_config=OnlineModelConfig(transducer=OnlineTransducerModelConfig(encoder="./sherpa-onnx-rk3588-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder.rknn", decoder="./sherpa-onnx-rk3588-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder.rknn", joiner="./sherpa-onnx-rk3588-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner.rknn"), paraformer=OnlineParaformerModelConfig(encoder="", decoder=""), wenet_ctc=OnlineWenetCtcModelConfig(model="", chunk_size=16, num_left_chunks=4), zipformer2_ctc=OnlineZipformer2CtcModelConfig(model=""), nemo_ctc=OnlineNeMoCtcModelConfig(model=""), provider_config=ProviderConfig(device=0, provider="rknn", cuda_config=CudaConfig(cudnn_conv_algo_search=1), trt_config=TensorrtConfig(trt_max_workspace_size=2147483647, trt_max_partition_iterations=10, trt_min_subgraph_size=5, trt_fp16_enable="True", trt_detailed_build_log="False", trt_engine_cache_enable="True", trt_engine_cache_path=".", trt_timing_cache_enable="True", trt_timing_cache_path=".",trt_dump_subgraphs="False" )), tokens="./sherpa-onnx-rk3588-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt", num_threads=1, warm_up=0, debug=False, model_type="", modeling_unit="cjkchar", bpe_vocab=""), lm_config=OnlineLMConfig(model="", scale=0.5, shallow_fusion=True), endpoint_config=EndpointConfig(rule1=EndpointRule(must_contain_nonsilence=False, min_trailing_silence=2.4, min_utterance_length=0), rule2=EndpointRule(must_contain_nonsilence=True, min_trailing_silence=1.2, min_utterance_length=0), rule3=EndpointRule(must_contain_nonsilence=False, min_trailing_silence=0, min_utterance_length=20)), ctc_fst_decoder_config=OnlineCtcFstDecoderConfig(graph="", max_active=3000), enable_endpoint=True, max_active_paths=4, hotwords_score=1.5, hotwords_file="", decoding_method="greedy_search", blank_penalty=0, temperature_scale=2, rule_fsts="", rule_fars="")
  Current sample rate: 16000
  Recording started!
  Use recording device: plughw:3,0
  Started! Please speak
  0:现在开始测试
  1:现在是星期六
  2:二零二五年三月二十二号
  3:下午六点四十四分
