Decode a long audio file with VAD (Example 2/2, Chinese)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following examples show how to decode a very long audio file with the help
of VAD.

.. code-block:: bash

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/lei-jun-test.wav

  ./build/bin/sherpa-onnx-vad-with-offline-asr \
    --silero-vad-model=./silero_vad.onnx \
    --silero-vad-threshold=0.2 \
    --silero-vad-min-speech-duration=0.2 \
    --qwen3-asr-conv-frontend=./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/conv_frontend.onnx \
    --qwen3-asr-encoder=./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/encoder.int8.onnx \
    --qwen3-asr-decoder=./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/decoder.int8.onnx \
    --qwen3-asr-tokenizer=./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/tokenizer \
    --qwen3-asr-max-new-tokens=512 \
    --num-threads=2 \
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

You should see the following output:

.. literalinclude:: ./code/lei-jun.txt

