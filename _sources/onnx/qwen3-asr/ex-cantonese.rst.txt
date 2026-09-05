
cantonese.wav (粤语, 中文)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To decode the test file

  ``./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/test_wavs/cantonese.wav``

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline \
    --qwen3-asr-conv-frontend=./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/conv_frontend.onnx \
    --qwen3-asr-encoder=./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/encoder.int8.onnx \
    --qwen3-asr-decoder=./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/decoder.int8.onnx \
    --qwen3-asr-tokenizer=./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/tokenizer \
    --qwen3-asr-max-new-tokens=512 \
    --num-threads=2 \
    ./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/test_wavs/cantonese.wav

You should see the following output:

.. literalinclude:: ./code/cantonese.txt

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Ground truth</th>
    </tr>
    <tr>
      <td>cantonese.wav</td>
      <td>
       <audio title="cantonese.wav" controls="controls">
             <source src="/sherpa/_static/qwen3-asr-0.6b/cantonese.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
      今次寻寻觅觅，终于揾到my princess，肯借个场俾我哋玩。你知啦，喺香港地喺繁忙时间要揾个场嚟拍嘢系非常之难嘅。再一次多谢你哋，亦都好多谢片入边嘅每一个人。下一次我哋斗啲咩好？喺下面留言话我哋知啦。拜拜。
      </td>
    </tr>
  </table>
