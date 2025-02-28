How to export models from NeMo to sherpa-onnx
=============================================

This section describes how to export CTC models from NeMo to `sherpa-onnx`_.

.. hint::

   Please refer to `<https://catalog.ngc.nvidia.com/orgs/nvidia/collections/nemo_asr>`_
   for a list of pre-trained NeMo models.

   You can use method described in this section to convert more models
   to `sherpa-onnx`_.

Let us take the following model as an example:

`<https://ngc.nvidia.com/models/nvidia:nemo:stt_en_conformer_ctc_small>`_.

.. hint::

    You can find the exported files in this example by visiting

      `<https://huggingface.co/csukuangfj/sherpa-onnx-nemo-ctc-en-conformer-small>`_

The steps to export it to `sherpa-onnx`_ are given below.

Step 1: Export model.onnx
-------------------------

The first step is to obtain ``model.onnx``.

.. code-block:: python

  import nemo.collections.asr as nemo_asr
  m = nemo_asr.models.EncDecCTCModelBPE.from_pretrained('stt_en_conformer_ctc_small')
  m.export('model.onnx')

Step 2: Add metadata
--------------------

To be usable in `sherpa-onnx`_, we have to use `add-model-metadata.py <https://huggingface.co/csukuangfj/sherpa-onnx-nemo-ctc-en-conformer-small/blob/main/add-model-metadata.py>`_ to add metadata to ``model.onnx``.

.. code-block:: bash

   wget https://huggingface.co/csukuangfj/sherpa-onnx-nemo-ctc-en-conformer-small/resolve/main/add-model-metadata.py

   # The following command changes model.onnx in-place
   python3 add-model-metadata.py


Step 3: Obtain model.int8.onnx
------------------------------

We can use `quantize-model.py <https://huggingface.co/csukuangfj/sherpa-onnx-nemo-ctc-en-conformer-small/blob/main/model.int8.onnx>`_ to obtain a quantized version of ``model.onnx``:

.. code-block:: bash

   wget https://huggingface.co/csukuangfj/sherpa-onnx-nemo-ctc-en-conformer-small/resolve/main/quantize-model.py

   # The following command will generate model.int8.onnx
   python3 ./quantize-model.py

Step 4: Obtain tokens.txt
-------------------------

Use the following command to obtain ``tokens.txt``:

.. code-block:: python

  import nemo.collections.asr as nemo_asr
  m = nemo_asr.models.EncDecCTCModelBPE.from_pretrained('stt_en_conformer_ctc_small')

  with open('tokens.txt', 'w') as f:
    for i, s in enumerate(m.decoder.vocabulary):
      f.write(f"{s} {i}\n")
    f.write(f"<blk> {i+1}\n")
