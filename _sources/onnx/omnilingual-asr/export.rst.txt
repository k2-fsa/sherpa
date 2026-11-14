Export to ONNX
==============

Where to find the export code
-----------------------------

You can find how we export models from `Omnilingual ASR`_ to ONNX at

  `<https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/omnilingual-asr/export-onnx.py>`_

Where to find test code
-----------------------

For testing the exported ONNX model in Python, please see

  `<https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/omnilingual-asr/test.py>`_

GitHub Actions
--------------

We use GitHub actions to export and upload the models.

The entrypoint is at

  `<https://github.com/k2-fsa/sherpa-onnx/blob/master/.github/workflows/export-omnilingual-asr-to-onnx.yaml>`_
