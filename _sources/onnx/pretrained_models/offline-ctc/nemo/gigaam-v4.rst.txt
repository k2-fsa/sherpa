.. _gigaam-v4-ctc-e2e:

GigaAM-v4 CTC E2E (Russian and English)
=======================================

`GigaAM-v4 from Sber <https://github.com/salute-developers/GigaAM>`_ recognizes
Russian and English. This example uses an INT8 ONNX export of Sber's
`original CTC checkpoint <https://huggingface.co/ai-sage/GigaAM-v4/tree/5b81216afa29c8b5a1a3e64c424c8416e18c1e5b>`_
with the existing offline NeMo CTC loader in `sherpa-onnx`_. The E2E output
includes capitalization and punctuation.

Download the model
------------------

Download ``model.int8.onnx`` (approximately 242 MB) and its matching
``tokens.txt`` from the same pinned revision. The commands use Bash, curl,
and shasum.

.. code-block:: bash

   (
     set -e
     mkdir -p gigaam-v4-ctc-e2e
     cd gigaam-v4-ctc-e2e
     weights=https://huggingface.co/ai-babai/gigaam-v4-onnx/resolve/d39e2a200f596315acef7003d246b65a73e9a327/ctc-e2e
     curl -fsSLO "$weights/model.int8.onnx"
     curl -fsSLO "$weights/tokens.txt"
     shasum -a 256 -c <<'EOF'
   0f61ec8ca69517cb68e24531c5000053b80ab84cf7071cc294571edfae493462  model.int8.onnx
   c6a37fab667d53c37e66835c90083808daa116c3993c5e93ce518b5a3896de6e  tokens.txt
   EOF
   )

Decode a WAV file
-----------------

Use a Python 3.11 or 3.12 virtual environment. The example decodes one
16 kHz mono, 16-bit PCM WAV on the CPU with 64 mel bins and three threads.

.. code-block:: bash

   python3 -m pip install sherpa-onnx==2.0.3 numpy==2.4.6
   curl -fsSL https://raw.githubusercontent.com/k2-fsa/sherpa-onnx/38016c6e0622e90bf417c35e8b47e0eb805af417/python-api-examples/offline-decode-files.py \
     -o offline-decode-files.py
   curl -fsSL https://raw.githubusercontent.com/k2-fsa/sherpa/0643dfb4e1e9851bccf766bfd23d335283c6b6f4/docs/source/_static/onnx/riscv64/a-test.wav \
     -o example.wav
   python3 offline-decode-files.py \
     --nemo-ctc=gigaam-v4-ctc-e2e/model.int8.onnx \
     --tokens=gigaam-v4-ctc-e2e/tokens.txt \
     --sample-rate=16000 --feature-dim=64 --num-threads=3 \
     example.wav

The transcription starts with:

.. code-block:: text

   Friends fell out often because life was changing so fast.

For your own audio, pass one 16 kHz mono, 16-bit PCM WAV per run. Use clips
containing one utterance of 20 ms to 30 s and split longer recordings at
speech boundaries.
