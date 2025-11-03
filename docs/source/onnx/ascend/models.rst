Pre-trained models
==================

You can download pre-trained models for Ascend NPU from `<https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models>`_.

We provide exported ``*.om`` models for 910B, 910B2, and 310P3 with CANN 8.0.0 on Linux aarch64.

If you need models for other types of NPU or for a different version of CANN, please
see :ref:`export-models-to-ascend-npu-onnx`.

.. _sherpa-onnx-ascend-910B-sense-voice-zh-en-ja-ko-yue-2024-07-17:

sherpa-onnx-ascend-910B-sense-voice-zh-en-ja-ko-yue-2024-07-17 (Chinese, English, Japanese, Korean, Cantonese, 中英日韩粤语)
-------------------------------------------------------------------------------------------------------------------------------------

This model is converted from :ref:`sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17` using code from the following URL:

  `<https://github.com/k2-fsa/sherpa-onnx/tree/master/scripts/sense-voice/ascend-npu>`_

.. hint::

   You can find how to run the export code at

      `<https://github.com/k2-fsa/sherpa-onnx/blob/master/.github/workflows/export-sense-voice-to-ascend-npu.yaml>`_

The original PyTorch checkpoint is available at

  `<https://huggingface.co/FunAudioLLM/SenseVoiceSmall>`_

.. hint::

   It supports dynamic input shapes, but the batch size is fixed to 1 at present.

Decode long files with a VAD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following example demonstrates how to use the model to decode a long wave file.

.. code-block::

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/lei-jun-test.wav

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-ascend-910B-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
   tar xvf sherpa-onnx-ascend-910B-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
   rm sherpa-onnx-ascend-910B-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2

   ls -lh sherpa-onnx-ascend-910B-sense-voice-zh-en-ja-ko-yue-2024-07-17

You should see the following output::

  ls -lh sherpa-onnx-ascend-910B-sense-voice-zh-en-ja-ko-yue-2024-07-17/
  total 999M
  -rw-r--r-- 1 root root 204K Oct 23 21:43 features.bin
  -rw-r--r-- 1 root root   71 Oct 23 13:52 LICENSE
  -rw------- 1 root root 998M Oct 23 13:52 model.om
  -rw-r--r-- 1 root root  104 Oct 23 13:52 README.md
  -rwxr-xr-x 1 root root 3.6K Oct 23 21:43 test_om.py
  drwxr-xr-x 2 root root 4.0K Oct 23 13:52 test_wavs
  -rw-r--r-- 1 root root 309K Oct 23 13:52 tokens.txt

.. hint::

   The above ``test_om.py`` uses `ais_bench <https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench>`_
   Python API to run ``model.om`` without `sherpa-onnx`_

Then run:

.. code-block:: bash

  cd /path/to/sherpa-onnx/build

  ./bin/sherpa-onnx-vad-with-offline-asr \
    --provider=ascend \
    --silero-vad-model=./silero_vad.onnx \
    --silero-vad-threshold=0.4 \
    --sense-voice-model=./sherpa-onnx-ascend-910B-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.om \
    --tokens=./sherpa-onnx-ascend-910B-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt \
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

The output is given below:


.. container:: toggle

    .. container:: header

      Click ▶ to see the output

    .. literalinclude:: ./code-sense-voice-2024-04-17/lei-jun-test.txt

Decode a short file
~~~~~~~~~~~~~~~~~~~

The following example demonstrates how to use the model to decode a short wave file.

.. code-block:: bash

  cd /path/to/sherpa-onnx/build

  ./bin/sherpa-onnx-offline \
    --provider=ascend \
    --sense-voice-model=./sherpa-onnx-ascend-910B-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.om \
    --tokens=./sherpa-onnx-ascend-910B-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt \
    ./sherpa-onnx-ascend-910B-sense-voice-zh-en-ja-ko-yue-2024-07-17/test_wavs/zh.wav

The output is given below:

.. literalinclude:: ./code-sense-voice-2024-04-17/short.txt

.. _sherpa-onnx-ascend-910B-sense-voice-zh-en-ja-ko-yue-2025-09-09:

sherpa-onnx-ascend-910B-sense-voice-zh-en-ja-ko-yue-2025-09-09 (Chinese, English, Japanese, Korean, Cantonese, 中英日韩粤语)
----------------------------------------------------------------------------------------------------------------------------------

This model is converted from :ref:`sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09` using code from the following URL:

  `<https://github.com/k2-fsa/sherpa-onnx/tree/master/scripts/sense-voice/ascend-npu>`_

.. hint::

   You can find how to run the export code at

      `<https://github.com/k2-fsa/sherpa-onnx/blob/master/.github/workflows/export-sense-voice-to-ascend-npu.yaml>`_

The original PyTorch checkpoint is available at

  `<https://huggingface.co/ASLP-lab/WSYue-ASR/tree/main/sensevoice_small_yue>`_

Please refer to :ref:`sherpa-onnx-ascend-910B-sense-voice-zh-en-ja-ko-yue-2024-07-17` for how to use this model.

.. _sherpa-onnx-ascend-910B-paraformer-zh-2023-03-28:

sherpa-onnx-ascend-910B-paraformer-zh-2023-03-28 (Chinese + English)
--------------------------------------------------------------------------------

This model is converted from :ref:`sherpa_onnx_offline_paraformer_zh_2023_03_28_chinese` using code from the following URL:

  `<https://github.com/k2-fsa/sherpa-onnx/tree/master/scripts/paraformer/ascend-npu>`_

.. hint::

   You can find how to run the export code at

      `<https://github.com/k2-fsa/sherpa-onnx/blob/master/.github/workflows/export-paraformer-to-ascend-npu.yaml>`_

.. hint::

   It supports dynamic input shapes, but the batch size is fixed to 1 at present.

Decode long files with a VAD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following example demonstrates how to use the model to decode a long wave file.

.. code-block::

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/lei-jun-test.wav

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-ascend-910B-paraformer-zh-2023-03-28.tar.bz2
   tar xvf sherpa-onnx-ascend-910B-paraformer-zh-2023-03-28.tar.bz2
   rm sherpa-onnx-ascend-910B-paraformer-zh-2023-03-28.tar.bz2

   ls -lh sherpa-onnx-ascend-910B-paraformer-zh-2023-03-28


You should see the following output::

  ls -lh sherpa-onnx-ascend-910B-paraformer-zh-2023-03-28
  total 1.1G
  -rw------- 1 root root 291M Oct 17 23:39 decoder.om
  -rw------- 1 root root 701M Oct 17 23:39 encoder.om
  -rw------- 1 root root  52M Oct 17 23:39 predictor.om
  -rw-r--r-- 1 root root  379 Oct 17 23:39 README.md
  -rwxr-xr-x 1 root root 5.5K Nov  3 09:37 test_om.py
  drwxr-xr-x 2 root root 4.0K Oct 17 23:39 test_wavs
  -rw-r--r-- 1 root root  74K Oct 17 23:39 tokens.txt

.. hint::

   The above ``test_om.py`` uses `ais_bench <https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench>`_
   Python API to run ``model.om`` without `sherpa-onnx`_

Then run:

.. code-block:: bash

  cd /path/to/sherpa-onnx/build

  ./bin/sherpa-onnx-vad-with-offline-asr \
    --provider=ascend \
    --silero-vad-model=./silero_vad.onnx \
    --silero-vad-threshold=0.4 \
    --paraformer="sherpa-onnx-ascend-910B-paraformer-zh-2023-03-28/encoder.om,sherpa-onnx-ascend-910B-paraformer-zh-2023-03-28/predictor.om,sherpa-onnx-ascend-910B-paraformer-zh-2023-03-28/decoder.om" \
    --tokens=sherpa-onnx-ascend-910B-paraformer-zh-2023-03-28/tokens.txt \
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

The output is given below:


.. container:: toggle

    .. container:: header

      Click ▶ to see the output

    .. literalinclude:: ./code-paraformer-2023-03-28/lei-jun-test.txt

Decode a short file
~~~~~~~~~~~~~~~~~~~

The following example demonstrates how to use the model to decode a short wave file.

.. code-block:: bash

  cd /path/to/sherpa-onnx/build

  ./bin/sherpa-onnx-offline \
    --provider=ascend \
    --paraformer="sherpa-onnx-ascend-910B-paraformer-zh-2023-03-28/encoder.om,sherpa-onnx-ascend-910B-paraformer-zh-2023-03-28/predictor.om,sherpa-onnx-ascend-910B-paraformer-zh-2023-03-28/decoder.om" \
    --tokens=sherpa-onnx-ascend-910B-paraformer-zh-2023-03-28/tokens.txt \
    sherpa-onnx-ascend-910B-paraformer-zh-2023-03-28/test_wavs/1.wav \

The output is given below:

.. literalinclude:: ./code-paraformer-2023-03-28/short.txt


.. _sherpa-onnx-ascend-910B2-paraformer-zh-2025-10-07:

sherpa-onnx-ascend-910B2-paraformer-zh-2025-10-07 (四川话、重庆话、川渝方言)
----------------------------------------------------------------------------------------------------

This model is converted from :ref:`sherpa-onnx-paraformer-zh-int8-2025-10-07` using code from the following URL:

  `<https://github.com/k2-fsa/sherpa-onnx/tree/master/scripts/paraformer/ascend-npu>`_

.. hint::

   You can find how to run the export code at

      `<https://github.com/k2-fsa/sherpa-onnx/blob/master/.github/workflows/export-paraformer-to-ascend-npu.yaml>`_

The original PyTorch checkpoint is available at

  `<https://huggingface.co/ASLP-lab/WSChuan-ASR/tree/main/Paraformer-large-Chuan>`_

.. hint::

   It supports dynamic input shapes, but the batch size is fixed to 1 at present.


Please refer to :ref:`sherpa-onnx-ascend-910B-paraformer-zh-2023-03-28` for how to use this model.
