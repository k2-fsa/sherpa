Uzbek
=====

.. hint::

   See :ref:`install_sherpa_onnx` to install `sherpa-onnx`_
   before you read this section.

This page describes an offline CTC model for Uzbek that uses the existing
NeMo CTC backend in `sherpa-onnx`_.

.. _sherpa-onnx-gigaam-uzbek-600m:

sherpa-onnx-gigaam-uzbek-600m
-----------------------------

`Alex13/sherpa-onnx-gigaam-uzbek-600m <https://huggingface.co/Alex13/sherpa-onnx-gigaam-uzbek-600m>`_
is an INT8 ONNX export of the ``large_full_600m`` checkpoint from
`rustam1221/uzbek-asr-gigaam <https://huggingface.co/rustam1221/uzbek-asr-gigaam>`_,
an Uzbek fine-tune of
`ai-sage/GigaAM-Multilingual <https://huggingface.co/ai-sage/GigaAM-Multilingual>`_.
Training is credited to rustam1221; the ONNX export is by Alex13.

The export uses the NeMo CTC configuration, with GigaAM preprocessing selected
by the ONNX metadata ``is_giga_am=1``. No language flag is required.
The model card describes lowercase output with punctuation.

Model licensing
~~~~~~~~~~~~~~~

.. warning::

   The export's model card declares MIT. However, the
   `review of the model example <https://github.com/k2-fsa/sherpa-onnx/pull/4072>`_
   reports that the source Uzbek fine-tune declares ``uzbek-asr-commercial``,
   allowing evaluation and non-commercial research and requiring a written
   license for commercial use.

   The export's MIT label does not establish unrestricted rights to the
   underlying weights. Permission to redistribute or commercially license
   the derived weights has not been verified here. Consult the source model's
   terms and obtain any required permission from its rights holders before
   redistribution or commercial use. This page links to the third-party export
   and does not distribute weights or grant additional rights to them.

Download the model
~~~~~~~~~~~~~~~~~~

From your ``sherpa-onnx`` working directory, download the model and its matching
vocabulary. The model download is approximately 592 MB. Both URLs below are
pinned to the same Hugging Face revision.

.. code-block:: bash

   mkdir -p sherpa-onnx-gigaam-uzbek-600m
   base=https://huggingface.co/Alex13/sherpa-onnx-gigaam-uzbek-600m/resolve/b53434a0b06418831dbc1cec73d6b851054f593d
   curl -fL "$base/model.int8.onnx" \
     -o sherpa-onnx-gigaam-uzbek-600m/model.int8.onnx
   curl -fL "$base/tokens.txt" \
     -o sherpa-onnx-gigaam-uzbek-600m/tokens.txt

The model repository does not include test audio. Use your own Uzbek recording.

Decode wave files
~~~~~~~~~~~~~~~~~

.. hint::

   The command-line example expects a single-channel wave file with 16-bit
   encoded samples. The sampling rate does not need to be 16 kHz;
   `sherpa-onnx`_ resamples the audio when necessary.

.. code-block:: bash

   ./build/bin/sherpa-onnx-offline \
     --nemo-ctc-model=./sherpa-onnx-gigaam-uzbek-600m/model.int8.onnx \
     --tokens=./sherpa-onnx-gigaam-uzbek-600m/tokens.txt \
     --num-threads=2 \
     recording.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.
   If you encounter encoding issues, run ``CHCP 65001`` in your command line.

Use short utterances or split long recordings into speech segments with VAD
before decoding.

Python example
~~~~~~~~~~~~~~

Install the dependencies in your Python environment:

.. code-block:: bash

   python3 -m pip install sherpa-onnx soundfile

Save the following as ``decode_uzbek.py`` next to the downloaded model directory.
This example reads the first channel of ``recording.wav`` and prints its
transcript.

.. code-block:: python

   import sherpa_onnx
   import soundfile as sf

   recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
       model="./sherpa-onnx-gigaam-uzbek-600m/model.int8.onnx",
       tokens="./sherpa-onnx-gigaam-uzbek-600m/tokens.txt",
       num_threads=2,
       sample_rate=16000,
       provider="cpu",
   )
   samples, sample_rate = sf.read(
       "recording.wav", dtype="float32", always_2d=True
   )
   stream = recognizer.create_stream()
   stream.accept_waveform(sample_rate, samples[:, 0].copy())
   recognizer.decode_stream(stream)
   print(stream.result.text)

Run it with:

.. code-block:: bash

   python3 decode_uzbek.py

Speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   ./build/bin/sherpa-onnx-microphone-offline \
     --nemo-ctc-model=./sherpa-onnx-gigaam-uzbek-600m/model.int8.onnx \
     --tokens=./sherpa-onnx-gigaam-uzbek-600m/tokens.txt

Speech recognition from a microphone with VAD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   curl -fL https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx \
     -o silero_vad.onnx

   ./build/bin/sherpa-onnx-vad-microphone-offline-asr \
     --silero-vad-model=./silero_vad.onnx \
     --nemo-ctc-model=./sherpa-onnx-gigaam-uzbek-600m/model.int8.onnx \
     --tokens=./sherpa-onnx-gigaam-uzbek-600m/tokens.txt
