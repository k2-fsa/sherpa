How to export models from Tele-AI/TeleSpeech-ASR to sherpa-onnx
===============================================================

This section describes how to export CTC models from `Tele-AI/TeleSpeech-ASR` to `sherpa-onnx`_.

Step 1: Export model.onnx
-------------------------

The first step is to obtain ``model.onnx``.

Please see `<https://github.com/lovemefan/telespeech-asr-python/blob/main/telespeechasr/onnx/onnx_export.py>`_
for details.

Step 2: Add metadata
--------------------

To be usable in `sherpa-onnx`_, we have to use `add-metadata.py <https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/tele-speech/add-metadata.py>`_ to add metadata to ``model.onnx``.

Please see `<https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/tele-speech/run.sh>`_
for details.


Step 3: Obtain tokens.txt
-------------------------

Please also see `<https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/tele-speech/add-metadata.py>`_
