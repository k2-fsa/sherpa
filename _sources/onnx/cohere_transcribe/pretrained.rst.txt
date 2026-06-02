Pre-trained model
=================

This page describes how to download a pre-trained `Cohere Transcribe`_ model
for `sherpa-onnx`_.

Download the released archive from
`<https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models>`_:

.. code-block:: bash

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01.tar.bz2
   tar xvf sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01.tar.bz2
   rm sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01.tar.bz2

   ls -lh sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01

You should see the following output::

  -rw-r--r--@  1 fangjun  staff   146M  1 Apr 19:00 decoder.int8.onnx
  -rw-r--r--@  1 fangjun  staff   2.9M  1 Apr 19:01 encoder.int8.onnx
  -rw-r--r--@  1 fangjun  staff   2.5G  1 Apr 19:01 encoder.int8.onnx.data
  -rw-r--r--@  1 fangjun  staff   294B  1 Apr 19:00 README.md
  drwxr-xr-x@ 11 fangjun  staff   352B  2 Apr 19:14 test_wavs
  -rw-r--r--@  1 fangjun  staff   203K  2 Apr 14:16 tokens.txt

Decode a short audio file
--------------------------

The following example shows how to decode a ``wav`` file:

.. code-block:: bash

   ./build/bin/sherpa-onnx-offline \
     --cohere-transcribe-encoder=./sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01/encoder.int8.onnx \
     --cohere-transcribe-decoder=./sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01/decoder.int8.onnx \
     --cohere-transcribe-language=en \
     --cohere-transcribe-use-punct=1 \
     --cohere-transcribe-use-itn=1 \
     --tokens=./sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01/tokens.txt \
     --num-threads=2 \
     ./sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01/test_wavs/en.wav

.. note::

   Cohere Transcribe requires the input language to be set explicitly. Please
   replace ``en`` with the language of your audio, such as ``de`` or ``zh``.

The output logs are given below:

.. literalinclude:: ./code/2026-04-01.txt

Decode a long audio file with VAD (Example 1/2, English)
----------------------------------------------------------

The following examples show how to decode a very long audio file with the help
of VAD.

.. code-block:: bash

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/Obama.wav

  ./build/bin/sherpa-onnx-vad-with-offline-asr \
    --silero-vad-model=./silero_vad.onnx \
    --silero-vad-threshold=0.2 \
    --silero-vad-min-speech-duration=0.2 \
    --cohere-transcribe-encoder=./sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01/encoder.int8.onnx \
    --cohere-transcribe-decoder=./sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01/decoder.int8.onnx \
     --tokens=./sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01/tokens.txt \
    --cohere-transcribe-language=en \
    --cohere-transcribe-use-punct=1 \
    --cohere-transcribe-use-itn=1 \
    --num-threads=2 \
    ./Obama.wav

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
    </tr>
    <tr>
      <td>Obama.wav</td>
      <td>
       <audio title="Obama.wav" controls="controls">
             <source src="/sherpa/_static/sense-voice/Obama.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
    </tr>
  </table>

You should see the following output:

.. literalinclude:: ./code/obama.txt

Decode a long audio file with VAD (Example 2/2, Chinese)
--------------------------------------------------------

The following examples show how to decode a very long audio file with the help
of VAD.

.. code-block:: bash

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/lei-jun-test.wav

  ./build/bin/sherpa-onnx-vad-with-offline-asr \
    --silero-vad-model=./silero_vad.onnx \
    --silero-vad-threshold=0.2 \
    --silero-vad-min-speech-duration=0.2 \
    --cohere-transcribe-encoder=./sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01/encoder.int8.onnx \
    --cohere-transcribe-decoder=./sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01/decoder.int8.onnx \
     --tokens=./sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01/tokens.txt \
    --cohere-transcribe-language=zh \
    --cohere-transcribe-use-punct=1 \
    --cohere-transcribe-use-itn=1 \
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
