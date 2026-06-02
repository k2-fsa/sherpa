de.wav (German, 德语)
^^^^^^^^^^^^^^^^^^^^^^^^

To decode the test file

  ``./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/test_wavs/de.wav``

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline \
    --qwen3-asr-conv-frontend=./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/conv_frontend.onnx \
    --qwen3-asr-encoder=./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/encoder.int8.onnx \
    --qwen3-asr-decoder=./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/decoder.int8.onnx \
    --qwen3-asr-tokenizer=./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/tokenizer \
    --qwen3-asr-max-new-tokens=512 \
    --num-threads=2 \
    ./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/test_wavs/de.wav

You should see the following output:

.. literalinclude:: ./code/de.txt

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Ground truth</th>
    </tr>
    <tr>
      <td>de.wav</td>
      <td>
       <audio title="de.wav" controls="controls">
             <source src="/sherpa/_static/qwen3-asr-0.6b/de.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
      Raptorium Bergbau scheint profitierter als Monroe als Reaktion auf die wirtschaftlichen Ausfälle zu sein.
      </td>
    </tr>
  </table>
