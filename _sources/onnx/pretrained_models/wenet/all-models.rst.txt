All models from WeNet
=====================

`<https://github.com/wenet-e2e/wenet/blob/main/docs/pretrained_models.en.md>`_
lists all pre-trained models from `WeNet`_ and we have converted all of them
to `sherpa-onnx`_ using the following script:

  `<https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/wenet/run.sh>`_.

We have uploaded the exported models to huggingface and you can find them from
the following figure:

  .. figure:: ./pic/wenet-models-onnx-list.jpg
     :alt: All pretrained models from `WeNet`
     :width: 600

     All pre-trained models from `WeNet`_.

To make it easier to copy the links, we list them below:

  - `<https://huggingface.co/csukuangfj/sherpa-onnx-zh-wenet-aishell>`_
  - `<https://huggingface.co/csukuangfj/sherpa-onnx-zh-wenet-aishell2>`_
  - `<https://huggingface.co/csukuangfj/sherpa-onnx-en-wenet-gigaspeech>`_
  - `<https://huggingface.co/csukuangfj/sherpa-onnx-en-wenet-librispeech>`_
  - `<https://huggingface.co/csukuangfj/sherpa-onnx-zh-wenet-multi-cn>`_
  - `<https://huggingface.co/csukuangfj/sherpa-onnx-zh-wenet-wenetspeech>`_

Colab
-----

We provide a colab notebook
|Sherpa-onnx wenet ctc colab notebook|
for you to try the exported `WeNet`_ models with `sherpa-onnx`_.

.. |Sherpa-onnx wenet ctc colab notebook| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://github.com/k2-fsa/colab/blob/master/sherpa-onnx/sherpa_onnx_with_models_from_wenet.ipynb
