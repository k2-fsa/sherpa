
raokouling.wav (绕口令, 中文)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To decode the test file

  ``./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/test_wavs/raokouling.wav``

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
