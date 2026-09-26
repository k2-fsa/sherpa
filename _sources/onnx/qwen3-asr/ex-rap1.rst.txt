rap1.wav (English, Rap)
^^^^^^^^^^^^^^^^^^^^^^^

To decode the test file

  ``./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/test_wavs/rap1.wav``

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline \
    --qwen3-asr-conv-frontend=./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/conv_frontend.onnx \
    --qwen3-asr-encoder=./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/encoder.int8.onnx \
    --qwen3-asr-decoder=./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/decoder.int8.onnx \
    --qwen3-asr-tokenizer=./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/tokenizer \
    --qwen3-asr-max-total-len=1024 \
    --qwen3-asr-max-new-tokens=1024 \
    --num-threads=2 \
    ./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/test_wavs/rap1.wav

You should see the following output:

.. literalinclude:: ./code/rap1.txt

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Ground truth</th>
    </tr>
    <tr>
      <td>rap1.wav</td>
      <td>
       <audio title="rap1.wav" controls="controls">
             <source src="/sherpa/_static/qwen3-asr-0.6b/rap1.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
        Sometimes I just feel like quitting. I still might. Why do I put up this fight? Why do I still write? Sometimes it's hard enough just dealing with real life. Sometimes I wanna jump a state and just kill mics. And so these people want my level of skills, like, but I'm still white. Sometimes I just hate life. Something ain't right. Hit the brake lights. Taste of the state's right. Drawing the blank line. It ain't my fault.
      </td>
    </tr>
  </table>
