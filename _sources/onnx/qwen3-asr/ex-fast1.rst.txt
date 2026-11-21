fast1.wav (中文, 极快语速)
^^^^^^^^^^^^^^^^^^^^^^^^^^

To decode the test file

  ``./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/test_wavs/fast1.wav``

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline \
    --qwen3-asr-conv-frontend=./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/conv_frontend.onnx \
    --qwen3-asr-encoder=./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/encoder.int8.onnx \
    --qwen3-asr-decoder=./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/decoder.int8.onnx \
    --qwen3-asr-tokenizer=./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/tokenizer \
    --qwen3-asr-max-new-tokens=512 \
    --num-threads=2 \
    ./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/test_wavs/fast1.wav

You should see the following output:

.. literalinclude:: ./code/fast1.txt

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Ground truth</th>
    </tr>
    <tr>
      <td>fast1.wav</td>
      <td>
       <audio title="fast1.wav" controls="controls">
             <source src="/sherpa/_static/qwen3-asr-0.6b/fast1.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
        蹦出来之后，左手、右手接一个慢动作，右边再直接拉到这上面之后，直接拉到这个轮胎上，上边再接过去之后，然后上边再直接拉到这个位置了之后，右边再直接这个位置接倒过去的之后，再倒一下，然后右边再直接抓住这个上边了之后，直接从这边上边过去了之后，直接抓住这个树杈，然后这个位置直接倒到这个树杈
      </td>
    </tr>
  </table>
