qiqiu1.wav (中文, 说唱音乐, 语速极快)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To decode the test file

  ``./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/test_wavs/qiqiu1.wav``

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline \
    --qwen3-asr-conv-frontend=./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/conv_frontend.onnx \
    --qwen3-asr-encoder=./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/encoder.int8.onnx \
    --qwen3-asr-decoder=./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/decoder.int8.onnx \
    --qwen3-asr-tokenizer=./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/tokenizer \
    --qwen3-asr-max-total-len=1024 \
    --qwen3-asr-max-new-tokens=1024 \
    --num-threads=2 \
    ./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/test_wavs/qiqiu1.wav

You should see the following output:

.. literalinclude:: ./code/qiqiu1.txt

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Ground truth</th>
    </tr>
    <tr>
      <td>qiqiu1.wav</td>
      <td>
       <audio title="qiqiu1.wav" controls="controls">
             <source src="/sherpa/_static/qwen3-asr-0.6b/qiqiu1.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
        黑的白的红的粉的紫的绿的蓝的灰的，你的我的他的她的大的小的圆的扁的，好的坏的美的丑的新的旧的，各种款式各种花式，让我选择。飞得高喽，越远越好，天都沦陷，他就死掉，说明多能高兴就好，喜欢就好，没大不了，越变越小，越来越小，快要死掉也很骄傲。你不想说就别再说，我不想听不想再听，就把一切誓言当作气球一般随它而去，我不在意不会在意，随它而去随它而去。气球飘进眼里，飘进风里，结束生命。气球飘进爱里，飘进心里，慢慢死去。
      </td>
    </tr>
  </table>
